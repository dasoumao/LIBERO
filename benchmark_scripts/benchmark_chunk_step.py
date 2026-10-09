"""Compare ordinary and chunk stepping on a machine with LIBERO installed.

This script never loads a policy. It replays identical seeded actions and checks
chunk-boundary simulator state, observations, rewards, and success flags before
reporting timings. Configure LIBERO's paths before running it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def reset_env(env, seed, initial_state):
    env.seed(seed)
    env.reset()
    if initial_state is not None:
        env.set_init_state(initial_state)
    for _ in range(5):
        env.step(np.zeros(env.env.action_dim))


def simulator_state(env):
    return {
        name: np.array(getattr(env.sim.data, name), copy=True)
        for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
    }


def compare_arrays(expected, actual, label, exact=False):
    if expected.keys() != actual.keys():
        raise AssertionError("{} keys differ".format(label))
    for name in expected:
        if exact or name.endswith("_image") or "segmentation" in name:
            np.testing.assert_array_equal(
                expected[name], actual[name], err_msg="{}: {}".format(label, name)
            )
        else:
            np.testing.assert_allclose(
                expected[name], actual[name], rtol=0, atol=1e-10,
                err_msg="{}: {}".format(label, name),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl-file", required=True)
    parser.add_argument("--init-state-file", help="Optional LIBERO .pruned_init file")
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--segmentation", choices=["instance", "class", "element"])
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if min(args.chunk_size, args.chunks, args.repeats, args.image_size) <= 0:
        parser.error("sizes, chunks, and repeats must be positive")

    # Delay heavyweight imports until after --help and argument validation.
    import robosuite
    import mujoco
    from libero.libero.envs import OffScreenRenderEnv

    initial_state = None
    if args.init_state_file:
        import torch
        initial_state = torch.load(args.init_state_file, map_location="cpu")[
            args.init_state_index
        ]
    kwargs = dict(
        bddl_file_name=str(Path(args.bddl_file).resolve()),
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        camera_depths=args.depth,
        camera_segmentations=args.segmentation,
        horizon=5 + args.chunk_size * args.chunks + 1,
        hard_reset=True,
    )
    env = None
    rows = []
    try:
        # Reuse one environment so measurements do not depend on switching
        # between two OpenGL contexts in the same process.
        env = OffScreenRenderEnv(**kwargs)
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            rng = np.random.RandomState(seed)
            chunks = rng.uniform(-0.03, 0.03, (args.chunks, args.chunk_size, 7))
            chunks[:, :, -1] = -1.0
            order = ("step", "chunk") if repeat % 2 == 0 else ("chunk", "step")
            outputs, reset_states = {}, {}
            for mode in order:
                reset_env(env, seed, initial_state)
                # Warm up compilation / renderer paths equally; do not time it.
                env.step(np.zeros(7))
                reset_states[mode] = simulator_state(env)
                outputs[mode] = []
                for actions in chunks:
                    rewards, successes = [], []
                    start = time.perf_counter()
                    if mode == "step":
                        for action in actions:
                            obs, reward, success, _ = env.step(action)
                            rewards.append(reward)
                            successes.append(success)
                        reward_sum = sum(rewards)
                        done = bool(any(successes) or env.env.done)
                        deferred = False
                    else:
                        obs, reward_sum, done, info = env.step_chunk(
                            actions, stop_on_success=False
                        )
                        rewards, successes = info["rewards"], info["successes"]
                        deferred = info["camera_sampling_deferred"]
                        assert info["steps_executed"] == args.chunk_size
                    elapsed = time.perf_counter() - start
                    outputs[mode].append({
                        "observation": {k: np.array(v, copy=True) for k, v in obs.items()},
                        "state": simulator_state(env),
                        "rewards": np.asarray(rewards),
                        "successes": np.asarray(successes),
                        "reward_sum": reward_sum, "done": done,
                        "seconds": elapsed, "deferred": deferred,
                    })
            compare_arrays(reset_states["step"], reset_states["chunk"], "reset")
            for index, (reference, optimized) in enumerate(zip(outputs["step"], outputs["chunk"])):
                compare_arrays(reference["state"], optimized["state"], "state")
                compare_arrays(reference["observation"], optimized["observation"], "observation")
                np.testing.assert_array_equal(reference["rewards"], optimized["rewards"])
                np.testing.assert_array_equal(reference["successes"], optimized["successes"])
                assert reference["reward_sum"] == optimized["reward_sum"]
                assert reference["done"] == optimized["done"]
                rows.append({
                    "repeat": repeat, "chunk": index,
                    "step_seconds": reference["seconds"],
                    "chunk_seconds": optimized["seconds"],
                    "camera_sampling_deferred": optimized["deferred"],
                })
        baseline = sum(row["step_seconds"] for row in rows)
        chunked = sum(row["chunk_seconds"] for row in rows)
        report = {
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
            "arguments": vars(args),
            "boundary_checks_passed": True,
            "step_seconds": baseline,
            "chunk_seconds": chunked,
            "speedup": baseline / chunked,
            "measurements": rows,
        }
        result = json.dumps(report, indent=2)
        print(result)
        if args.output:
            Path(args.output).write_text(result + "\n")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
