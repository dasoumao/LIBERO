"""Action chunks that preserve robosuite's per-action control and task hooks."""

import math

import numpy as np
from robosuite.utils.observables import NO_CORRUPTION, NO_DELAY, NO_FILTER


_CAMERA_SENSORS = {
    "RobotEnv._create_camera_sensors.<locals>.camera_rgb",
    "RobotEnv._create_camera_sensors.<locals>.camera_depth",
    "RobotEnv._create_segementation_sensor.<locals>.camera_segmentation",
}
_STATE_SENSOR_MODULES = {
    "robosuite.robots.robot",
    "robosuite.robots.manipulator",
    "robosuite.robots.single_arm",
    "libero.libero.envs.bddl_base_domain",
}


class ChunkStepMixin:
    """Skip unused observations without changing the physics stepping loop.

    This mixin must precede the robosuite environment in the MRO. The optimized
    path targets robosuite 1.4.0's standard, stateless camera observables. Camera
    clocks still advance through Observable.update; no force-update is added at
    a chunk boundary.
    """

    def step_chunk(self, actions, *, stop_on_success=True, render_intermediate=False):
        """Execute a nonempty [K, action_dim] chunk and return a Gym-style tuple.

        Returns (last_obs, reward_sum, done, info). ``info`` contains
        ``steps_executed``, ``rewards``, ``successes``, ``step_infos``,
        ``horizon_reached``, ``observation_available``, and
        ``camera_sampling_deferred``. Success is latched
        across the chunk, even when stop_on_success=False. The environment's
        horizon always stops execution when ignore_done=False.

        If success stops an optimized chunk before its planned last action,
        last_obs is None: its unused cameras were not rendered. Set
        render_intermediate=True when exact early-terminal observations or all
        camera samples are needed. This flag still returns only the last obs.
        """
        actions = np.asarray(actions, dtype=float)
        valid_dims = (4, 7) if self.action_dim == 4 else (self.action_dim,)
        if (
            actions.ndim != 2
            or len(actions) == 0
            or actions.shape[1] not in valid_dims
            or not np.isfinite(actions).all()
        ):
            raise ValueError("actions must be a finite, nonempty [K, action_dim] array")
        if getattr(self, "_chunk_running", False):
            raise RuntimeError("step_chunk cannot be nested on the same environment")

        # Non-MuJoCo viewers can bypass _get_observations; keep their original
        # stepping behavior rather than silently returning incomplete images.
        optimize = (
            not render_intermediate
            and self.renderer == "mujoco"
            and not self.viewer_get_obs
        )
        cameras = {
            name: observable
            for name, observable in self._observables.items()
            if observable.is_enabled() and observable.modality == "image"
        }
        # RGB, depth, and segmentation may share cache entries. If one camera
        # observable needs its real intermediate input, keep the entire group.
        defer_cameras = (
            optimize
            and all(self._can_defer_camera(obs) for obs in cameras.values())
            and all(
                self._is_standard_state_sensor(obs)
                for obs in self._observables.values()
                if obs.is_enabled() and obs.modality != "image"
            )
        )
        deferred_cameras = frozenset(cameras) if defer_cameras else frozenset()
        rewards, successes, step_infos = [], [], []
        obs = None
        horizon_reached = False
        self._chunk_running = True
        self._chunk_deferred_cameras = deferred_cameras
        try:
            for index, action in enumerate(actions):
                planned_last = index == len(actions) - 1 or (
                    not self.ignore_done and self.timestep + 1 >= self.horizon
                )
                self._chunk_collect_observations = not optimize or planned_last
                obs, reward, success, step_info = self.step(action)
                rewards.append(reward)
                successes.append(bool(success))
                step_infos.append(step_info)
                horizon_reached = bool(self.done)
                if horizon_reached or (stop_on_success and success):
                    break
        finally:
            self._chunk_collect_observations = True
            self._chunk_running = False
            self._chunk_deferred_cameras = frozenset()

        info = {
            "steps_executed": len(rewards),
            "rewards": np.asarray(rewards),
            "successes": np.asarray(successes, dtype=bool),
            "step_infos": step_infos,
            "horizon_reached": horizon_reached,
            "observation_available": obs is not None,
            "camera_sampling_deferred": bool(deferred_cameras),
        }
        return obs, sum(rewards), any(successes) or horizon_reached, info

    def _get_observations(self, force_update=False):
        if not getattr(self, "_chunk_collect_observations", True) and not force_update:
            return None
        return super()._get_observations(force_update=force_update)

    def _update_observables(self, force=False):
        if getattr(self, "_chunk_collect_observations", True) or force:
            return super()._update_observables(force=force)

        for name, observable in self._observables.items():
            if name not in self._chunk_deferred_cameras:
                observable.update(
                    timestep=self.model_timestep, obs_cache=self._obs_cache
                )
                continue

            # Keep robosuite's sampling phase and cache bookkeeping exactly as
            # in ordinary step(). Reuse the old pixel value only internally;
            # no observation is returned for this action. Restoring the sensor
            # before the final action lets it sample at its original time.
            sensor = observable._sensor
            cached_value = observable._current_observed_value
            observable._sensor = lambda obs_cache, value=cached_value: value
            try:
                observable.update(
                    timestep=self.model_timestep, obs_cache=self._obs_cache
                )
            finally:
                observable._sensor = sensor

    def _can_defer_camera(self, observable):
        # Other sample rates might need a frame captured before the last action.
        # Stateful filters/noise/delay and custom sensors must run normally.
        return (
            observable.modality == "image"
            and getattr(observable._sensor, "__qualname__", None) in _CAMERA_SENSORS
            and getattr(observable._sensor, "__module__", None)
            == "robosuite.environments.robot_env"
            and observable._filter is NO_FILTER
            and observable._corrupter is NO_CORRUPTION
            and observable._delayer is NO_DELAY
            and math.isclose(
                observable._sampling_timestep,
                self.control_timestep,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )

    @staticmethod
    def _is_standard_state_sensor(observable):
        # A custom state sensor or filter can consume camera cache entries.
        # Do not feed it old images just because its modality is not "image".
        return (
            getattr(observable._sensor, "__module__", None) in _STATE_SENSOR_MODULES
            and observable._filter is NO_FILTER
            and observable._corrupter is NO_CORRUPTION
            and observable._delayer is NO_DELAY
        )
