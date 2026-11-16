# Unified Automatic Control of Vehicular Systems With Reinforcement Learning
This repo contains the code and model checkpoints for our IEEE T-ASE (presented at IROS 2022) paper *Unified Automatic Control of Vehicular Systems with Reinforcement Learning*. Videos of results can be found on our [project website](https://mit-wu-lab.github.io/automatic_vehicular_control).

# Relevant Links
You may find this project at: [Project Website](https://mit-wu-lab.github.io/automatic_vehicular_control), [IEEE Website](https://ieeexplore.ieee.org/document/9765650), [arXiv](https://arxiv.org/abs/2208.00268).

```
@article{yan2022unified,
  title={Unified Automatic Control of Vehicular Systems With Reinforcement Learning},
  author={Yan, Zhongxia and Kreidieh, Abdul Rahman and Vinitsky, Eugene and Bayen, Alexandre M and Wu, Cathy},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2022},
  publisher={IEEE}
}
```

## Running in the i24-sim-22.04 container (Gymnasium + Stable-Baselines3)
This copy is ported to the container's stack: SUMO 1.27.1, gymnasium, stable-baselines3 / sb3-contrib and numpy 2 (see `requirements.txt`). No SUMO build, Ray, gym or lxml is needed. Put the repo root on the path instead of `pip install -e .`:
```
cd /workspaces/i24motion_macro_micro/automatic_vehicular_control
export PYTHONPATH=$PWD F=$PWD/automatic_vehicular_control
```
The paper's `train_command.sh` / `eval_commands.sh` (from the results zip) then run unchanged from inside a run directory, e.g. the single-ring "Greedy" run:
```
python $F/ring.py . "worker_kwargs=[{'circumference':c} for c in range(230,270)]" "n_workers=40" "n_rollouts_per_step=40" \
    "warmup_steps=2000" "skip_stat_steps=5000" "horizon=5000" "n_steps=50" "alg='TRPO'" "use_critic=False" "gamma=0.999"
```
What changed:
- Training goes through `sb3_backend.py` (`backend='sb3'`, the default): sb3-contrib's `TRPO` over a `SubprocVecEnv` with one environment per rollout, configured from AVC's own hyperparameters and defaults. Its docstring lists the flag mapping and the accepted differences (SB3's state-independent Gaussian, SB3's line search). Checkpoints are `models/model-<iterations>.zip`, `train_results.csv` keeps AVC's columns, and an interrupted run resumes from the latest checkpoint. `backend='legacy'` keeps the original loop.
- `env.GymnasiumEnv` exposes an AVC environment to SB3; `NormEnv` reward normalization is unchanged and runs per worker.
- `episodes_per_update` (default 1, the paper's layout) makes a rollout that many episodes per worker instead of one. It exists to keep the paper's batch size when episodes are shortened -- e.g. `"horizon=380" "skip_stat_steps=0" "episodes_per_update=26"` gives ~380-step episodes, matching the I-24 corridor's bubble traversal, at ~395k transitions per update.
- SUMO ignores `sigma` under IDM, so the fork's IDM acceleration noise is emulated through TraCI (`TrafficState.apply_idm_noise`, disable with `emulate_idm_sigma=False`).
- The all-human ring runs about 10% slower under SUMO 1.27.1 than in the paper's evaluations (3.0 vs 3.36 m/s at 250 m), with or without noise.

## Installation
Clone this repo with
```
git clone https://github.com/mit-wu-lab/automatic_vehicular_control.git
```

Installation instructions are provided for MacOS and Ubuntu 14.04, 16.04, and 18.04. For microscopic traffic simulations, we use the SUMO simulator with version 1.1.0 (which unfortunately is quite outdated by now); the same code may work on newer SUMO versions (we didn't test it). We require Python 3.8+.
1. Run `bash setup/setup_sumo_<os_version>.sh` corresponding to your OS version to set up SUMO and add `~/sumo_binaries/bin` to your `SUMO_HOME` and `PATH` environment variables. Try running `sumo` and `sumo-gui` (if you'd like to use GUI). Note that GUI probably does not work on servers and may only work on local computers. For Mac installation issues, please refer to `setup/setup_issues_osx.md`. **Update**: due to MacOS `brew` updates, it could be very difficult to install the correct versions of packages for SUMO 1.1.0 on MacOS, so Ubuntu is recommended; for MacOS, you may consider installing `gdal` with `conda install -c conda-forge gdal=2.4.2` and download the `ffmpeg=4.4.1` library files (within the `tar.bz2`) from [conda-forge](https://anaconda.org/conda-forge/ffmpeg/files?version=4.4.1) directly instead of trying to use `brew`.
2. Note: the previous SUMO installation actually installs a SUMO version which does not support IDM with Gaussian noise. If you'd like to use Gaussian noise (which is what we use in the paper but does not significantly affect results), you can build the forked version of SUMO 1.1.0 at https://github.com/ZhongxiaYan/sumo.
3. If needed, follow instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install Miniconda, likely `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` followed by `bash Miniconda3-latest-Linux-x86_64.sh`.
4. If desired, create and activate a new conda environment following these [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
5. If needed, install PyTorch (1.7+) from [pytorch.org](pytorch.org).
6. If needed, install missing Python dependencies `pip install -r requirements.txt`.

Install this repo with
```
pip install -e .
```

For all training and evaluation commands, model checkpoints, and evaluation results, download and unzip https://www.dropbox.com/s/l83qkw8qvimqg62/results.zip?dl=0 (file size of around 4G).

Set the environmental variables
```
# Code directory
export F=automatic_vehicular_control/automatic_vehicular_control

# Results directory extracted from the zip file
export R=results
```

## Directory Structure

The code directory structure is
```
$F (automatic_vehicular_control)
 ├─ exp.py  # The base experiment class and helpers
 ├─ env.py  # The base environment class and helpers
 ├─ ut.py  # RL-related utility functions
 ├─ u.py  # General utility functions
 │  # Scenario specific experiment/environment classes
 └─ [ring, figure_eight, highway_bottleneck, highway_ramp, intersection].py
```

The results directory structure is
```
$R (results)
└─ [ring, figure_eight, highway_bottleneck, highway_ramp, intersection]
    ├─ plots  # Symbolic links to evaluations, organized for plotting
    │    │
    │    │  # Specific evaluation setting
    │    ├─ <e.g. Ours (DRL) horizontal_inflow=1000 vertical_inflow=1000>
    │    │    ├─ evaluation.csv  # Recorded stats
    │    │    ├─ trajectories.npz  # Recorded vehicle trajectories
    │    │    ├─ trajectories_agent.npz  # Recorded agent actions
    │    │    └─ trajectories.net.xml  # Recorded traffic network
    │    └─ ...
    ├─ baselines  # Evaluations of human driving baseline
    ├─ derived  # Evaluations of derived control policy
    │
    │  # Checkpoints and evaluations of RL-based policy
    ├─ <e.g. multiflow16_av0.333_horizon2000>
    │    ├─ train_command.sh  # Training command for RL-based policies
    │    ├─ eval_commands.sh  # All evaluation commands
    │    ├─ config.yaml  # Recorded parameters
    │    ├─ models/*.pth  # Saved PyTorch model checkpoints
    │    ├─ train_results.csv  # Training statistics
    │    ├─ evaluations/*.csv  # Recorded evaluation statistics
    │    │
    │    │  # Recorded evaluation trajectories, agent actions, and traffic network
    │    └─ trajectories/*.[npz,xml]
    └─ ...
```
**Note that the training and evaluation commands are included in the zipped results directory detailed above.**

# Training and Evaluation
Given any experiment directory `$EXP_DIR` in the zipped results directory, run the commands in `$EXP_DIR/train_command.sh` and `$EXP_DIR/eval_commands.sh`, for training and evaluation. Training uses around 45-48 parallel workers for rollout collection, which should be adjusted accordingly based on computational budget.

Note that Baseline and Derived are not learning-based, so there is no need to run training for them.
