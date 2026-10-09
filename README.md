<div align="center">
<img src="https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images/libero_logo.png" width="360">


<p align="center">
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/Lifelong-Robot-Learning/LIBERO?color=0088ff" />

## **Benchmarking Knowledge Transfer for Lifelong Robot Learning**

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

[[Website]](https://libero-project.github.io)
[[Paper]](https://arxiv.org/pdf/2306.03310.pdf)
[[Docs]](https://lifelong-robot-learning.github.io/LIBERO/)
______________________________________________________________________
![pull_figure](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images//fig1.png)
</div>

**LIBERO** is designed for studying knowledge transfer in multitask and lifelong robot learning problems. Successfully resolving these problems require both declarative knowledge about objects/spatial relationships and procedural knowledge about motion/behaviors. **LIBERO** provides:
- a procedural generation pipeline that could in principle generate an infinite number of manipulation tasks.
- 130 tasks grouped into four task suites: **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-100**. The first three task suites have controlled distribution shifts, meaning that they require the transfer of a specific type of knowledge. In contrast, **LIBERO-100** consists of 100 manipulation tasks that require the transfer of entangled knowledge. **LIBERO-100** is further splitted into **LIBERO-90** for pretraining a policy and **LIBERO-10** for testing the agent's downstream lifelong learning performance.
- five research topics.
- three visuomotor policy network architectures.
- three lifelong learning algorithms with the sequential finetuning and multitask learning baselines.

---


# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Task](#Task)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Citation](#Citation)
- [License](#License)

## Chunked simulation

`OffScreenRenderEnv.step_chunk` accepts an entire action chunk. It keeps the
original physics substeps, OSC controller updates, rewards, object-state updates,
and success checks for every action. The default path skips unused intermediate
camera renders / pixel readbacks and observation dictionary construction.
Ordinary `env.step(action)` is unchanged.

```python
obs, reward_sum, done, info = env.step_chunk(actions)  # actions: [K, 7]
if not done:
    actions = policy(obs)
```

The return value is `(last_obs, reward_sum, done, info)`:

- `done` is true if any action succeeded or the environment horizon was reached.
- `info["steps_executed"]` is the number of actions actually executed.
- `info["rewards"]`, `info["successes"]`, and `info["step_infos"]` preserve the
  per-action results. Reward is summed; success is latched across the chunk.
- The default `stop_on_success=True` stops on the first successful action. If
  that action precedes the planned final action, `last_obs` is **None** and
  `info["observation_available"]` is false. No stale terminal image is returned.
  Use `render_intermediate=True` if you need the exact early-terminal observation.
- `stop_on_success=False` executes the full chunk unless the environment horizon
  is reached. `info["horizon_reached"]` distinguishes that limit from success.
- `info["camera_sampling_deferred"]` reports whether camera suppression was
  enabled for the chunk; this does not imply that any intermediate action ran.

Camera suppression targets the standard robosuite **1.4.0** observation graph,
with stateless cameras sampled at the control frequency. It advances the original
Observable sampling clocks using cached pixel values internally, restores the
real camera sensors for the last action, and adds no forced boundary render.
Intermediate pixel-array copies still occur; this first implementation removes
rendering/readback rather than replacing robosuite's sampler. It uses private
Observable fields from the pinned version and needs revalidation on upgrades.

Custom camera sensors, nonstandard camera sampling rates, custom observation
filters/noise/delays, and custom state-sensor modules cause the whole camera group
to fall back to normal sampling. This protects shared RGB/depth/segmentation
caches. Intermediate observation dictionaries are still omitted on that path.
Use `render_intermediate=True` to retain all sampling and observation construction,
including for custom sensors that depend on camera caches. No global robosuite
functions or installed dependency files are patched.

This API is currently provided on individual environments. It does not add a
chunk RPC to `SubprocVectorEnv`, change model inference scheduling, change physics
timesteps, or merge OSC updates into constant-torque MuJoCo steps.

### Validation on a simulation machine

The following checks require a configured LIBERO installation and a working
rendering backend. They were **not run on the development machine**; only static
source checks were performed. No speedup or trajectory-equivalence measurement
is claimed yet.

```bash
python -m pytest tests/test_chunk_step.py
MUJOCO_GL=egl python benchmark_scripts/benchmark_chunk_step.py \
  --bddl-file /path/to/task.bddl \
  --init-state-file /path/to/task.pruned_init \
  --chunk-size 32 --output chunk-benchmark.json
```

The benchmark replays the same seeded actions without a policy. It checks
simulator state and observations at each chunk boundary, per-action rewards and
success flags, then reports wall-clock timings. Use `--depth` and
`--segmentation element` to check additional camera outputs. Unit tests cover
sampling phases across repeated chunks, early success, horizon truncation,
custom-observable fallbacks, and exception recovery. These checks do not replace
full-policy success-rate validation before using the optimized path in a report.


# Installtion
Please run the following commands in the given order to install the dependency for **LIBERO**.
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the `libero` package:
```
pip install -e .
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

**NEW!!!**

Alternatively, you can download the dataset from HuggingFace by using:
```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

This option can also be combined with the specific dataset selection:
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).


# Getting Started

For a detailed walk-through, please either refer to the documentation or the notebook examples provided under the `notebooks` folder. In the following, we provide example scripts for retrieving a task, training and evaluation.

## Task

The following is a minimal example of retrieving a specific task from a specific task suite.
```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()
```
Currently, we only support sparse reward function (i.e., the agent receives `+1` when the task is finished). As sparse-reward RL is extremely hard to learn, currently we mainly focus on lifelong imitation learning.

## Training
To start a lifelong learning experiment, please choose:
- `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]`
- `POLICY` from `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]`
- `ALGO` from `[base, er, ewc, packnet, multitask]`

then run the following:

```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```
Please see the documentation for the details of reproducing the study results.

## Evaluation

By default the policies will be evaluated on the fly during training. If you have limited computing resource of GPUs, we offer an evaluation script for you to evaluate models separately.

```shell
python libero/lifelong/evaluate.py --benchmark BENCHMARK_NAME \
                                   --task_id TASK_ID \ 
                                   --algo ALGO_NAME \
                                   --policy POLICY_NAME \
                                   --seed SEED \
                                   --ep EPOCH \
                                   --load_task LOAD_TASK \
                                   --device_id CUDA_ID
```

# Citation
If you find **LIBERO** to be useful in your own research, please consider citing our paper:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

# License
| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                                      |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |
