"""Contract tests using robosuite's real sampling clock, without MuJoCo scenes.

Run in a configured LIBERO environment: python -m pytest tests/test_chunk_step.py
The file loader avoids LIBERO's first-import interactive configuration prompt.
"""

import importlib.util
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from robosuite.utils.observables import Observable, sensor


source = Path(__file__).resolve().parents[1] / "libero/libero/envs/chunk_step.py"
spec = importlib.util.spec_from_file_location("chunk_step_under_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ChunkStepMixin = module.ChunkStepMixin


class SamplingEnv:
    """A changing state observed through the real robosuite Observable class."""

    action_dim = 1
    renderer = "mujoco"
    viewer_get_obs = False
    model_timestep = 0.002
    control_timestep = 0.05
    ignore_done = False

    def __init__(self, success_at=None, horizon=100, fail_at=None):
        self.timestep = 0
        self.ticks = 0
        self.done = False
        self.horizon = horizon
        self.success_at = success_at
        self.fail_at = fail_at
        self.render_calls = 0
        self.observation_calls = 0
        self._obs_cache = {}

        @sensor(modality="image")
        def camera_rgb(cache):
            self.render_calls += 1
            cache["camera_depth"] = np.full((2, 2, 1), self.ticks)
            return np.full((2, 2, 3), self.ticks)

        @sensor(modality="image")
        def camera_depth(cache):
            return cache.get("camera_depth", np.zeros((2, 2, 1)))

        @sensor(modality="state")
        def state(cache):
            return np.array([self.ticks])

        # Mark these synthetic cameras as built-ins to exercise the optimized
        # branch; separate tests verify custom sensors fall back.
        camera_rgb.__module__ = "robosuite.environments.robot_env"
        camera_depth.__module__ = "robosuite.environments.robot_env"
        camera_rgb.__qualname__ = "RobotEnv._create_camera_sensors.<locals>.camera_rgb"
        camera_depth.__qualname__ = "RobotEnv._create_camera_sensors.<locals>.camera_depth"
        state.__module__ = "robosuite.robots.robot"
        self._observables = OrderedDict(
            (name, Observable(name, callback, sampling_rate=20))
            for name, callback in (
                ("camera_image", camera_rgb),
                ("camera_depth", camera_depth),
                ("state", state),
            )
        )
        self._update_observables(force=True)
        self.render_calls = 0

    def _update_observables(self, force=False):
        for observable in self._observables.values():
            observable.update(self.model_timestep, self._obs_cache, force=force)

    def _get_observations(self, force_update=False):
        self.observation_calls += 1
        if force_update:
            self._update_observables(force=True)
        return {name: obs.obs.copy() for name, obs in self._observables.items()}

    def step(self, action):
        if self.done:
            raise ValueError("executing action in terminated episode")
        self.timestep += 1
        for _ in range(25):
            self.ticks += 1
            self._update_observables()
            if self.ticks == self.fail_at:
                raise RuntimeError("injected failure")
        self.done = self.timestep >= self.horizon
        success = self.timestep == self.success_at
        return self._get_observations(), float(success), success, {"t": self.timestep}


class ChunkEnv(ChunkStepMixin, SamplingEnv):
    pass


def assert_observations_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])


@pytest.mark.parametrize("length", [1, 2, 16, 32])
def test_chunk_preserves_boundary_samples_across_repeated_chunks(length):
    reference, optimized = SamplingEnv(horizon=1000), ChunkEnv(horizon=1000)
    for _ in range(3):
        for action in np.zeros((length, 1)):
            expected, _, _, _ = reference.step(action)
        actual, reward, done, info = optimized.step_chunk(np.zeros((length, 1)))
        assert_observations_equal(expected, actual)
        assert optimized.ticks == reference.ticks
        assert reward == 0 and not done
        assert info["steps_executed"] == length
        for name in reference._observables:
            a, b = reference._observables[name], optimized._observables[name]
            assert a._time_since_last_sample == b._time_since_last_sample
            assert a._sampled == b._sampled
    assert optimized.observation_calls == 3
    assert optimized.render_calls == 3
    assert reference.render_calls == 3 * length


def test_early_success_has_no_stale_terminal_observation():
    env = ChunkEnv(success_at=2)
    obs, reward, done, info = env.step_chunk(np.zeros((16, 1)))
    assert obs is None and not info["observation_available"]
    assert reward == 1 and done and info["steps_executed"] == 2
    np.testing.assert_array_equal(info["successes"], [False, True])
    assert env.render_calls == 0
    # A subsequent ordinary step takes a fresh camera sample.
    obs, _, _, _ = env.step([0])
    assert obs["camera_image"].min() > 0
    assert env.render_calls == 1


def test_compatibility_mode_returns_exact_early_terminal_observation():
    reference, optimized = SamplingEnv(success_at=2), ChunkEnv(success_at=2)
    for _ in range(2):
        expected, _, _, _ = reference.step([0])
    actual, _, done, info = optimized.step_chunk(
        np.zeros((16, 1)), render_intermediate=True
    )
    assert_observations_equal(expected, actual)
    assert done and info["observation_available"]
    assert not info["camera_sampling_deferred"]


def test_success_is_latched_when_execution_continues():
    env = ChunkEnv(success_at=2)
    obs, reward, done, info = env.step_chunk(np.zeros((4, 1)), stop_on_success=False)
    assert obs is not None and reward == 1 and done
    assert info["steps_executed"] == 4
    np.testing.assert_array_equal(info["successes"], [False, True, False, False])


def test_horizon_renders_last_action_and_never_steps_past_it():
    reference, optimized = SamplingEnv(horizon=3), ChunkEnv(horizon=3)
    for _ in range(3):
        expected, _, _, _ = reference.step([0])
    actual, _, done, info = optimized.step_chunk(np.zeros((8, 1)))
    assert_observations_equal(expected, actual)
    assert done and info["horizon_reached"] and info["steps_executed"] == 3


@pytest.mark.parametrize("customization", ["rate", "filter", "sensor"])
def test_camera_group_falls_back_for_nonstandard_observable(customization):
    env = ChunkEnv()
    camera = env._observables["camera_depth"]
    if customization == "rate":
        camera.set_sampling_rate(10)
    elif customization == "filter":
        camera.set_filter(lambda value: value)
    else:
        camera._sensor.__module__ = "custom_camera"
    _, _, _, info = env.step_chunk(np.zeros((4, 1)))
    assert not info["camera_sampling_deferred"]
    assert env.render_calls == 4


def test_custom_state_sensor_keeps_real_camera_cache():
    env = ChunkEnv()
    env._observables["state"]._sensor.__module__ = "custom_state_sensor"
    _, _, _, info = env.step_chunk(np.zeros((4, 1)))
    assert not info["camera_sampling_deferred"]
    assert env.render_calls == 4


def test_failure_restores_sensor_and_normal_step_mode():
    env = ChunkEnv(fail_at=30)
    original = env._observables["camera_image"]._sensor
    with pytest.raises(RuntimeError, match="injected failure"):
        env.step_chunk(np.zeros((4, 1)))
    assert env._observables["camera_image"]._sensor is original
    assert env._chunk_collect_observations and not env._chunk_running
    assert env._get_observations() is not None


@pytest.mark.parametrize("actions", [[], [0], [[0, 1]], [[np.nan]], [[np.inf]]])
def test_invalid_chunk_does_not_advance_state(actions):
    env = ChunkEnv()
    with pytest.raises(ValueError):
        env.step_chunk(actions)
    assert env.timestep == 0 and env.ticks == 0
