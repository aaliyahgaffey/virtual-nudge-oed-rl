# Virtual-Nudge Inertial Identification

**Optimal Experimental Design (OED), Reinforcement Learning (RL), and Hybrid Action Selection**

This repository investigates how to select a sequence of **three virtual nudges** to identify an asteroid's inertial properties. A nudge applies a modeled impulse to the asteroid, and the resulting motion is observed to update estimates of its **mass, center of mass, and inertia tensor**. The previous implementation allowed for adaptive observation periods. However, to facilitate a consistent comparison across OED, RL, and hybrid approaches, the current implementation uses a fixed 30-second observation interval after each nudge.

Three action-selection approaches are considered:

- **OED:** Selects a nudge using predicted reduction in inertial-parameter uncertainty.
- **RL:** Uses a learned Soft Actor-Critic (SAC) policy to choose nudges from the current state and parameter estimates.
- **Hybrid OED–RL:** Generates one OED proposal and one RL proposal, then executes the one with the greater model-predicted information utility.

The primary training and planning environment is a **rigid-body surrogate with virtual impulses**. It is intended for algorithm development, not for demonstrating spacecraft/robot contact feasibility or certified collision avoidance.

## Start here: which file should I run?

| Goal | Entry point | What it does |
| --- | --- | --- |
| **Compare OED, RL, and hybrid policies** | [`notebooks/three_nudge_rl_v11_extended_training.ipynb`](notebooks/three_nudge_rl_v11_extended_training.ipynb) | Main notebook for training, evaluation, and comparisons across policies. **Recommended starting point for reviewing the full study.** |
| **Evaluate OED without training an RL model** | `scripts/train_virtual_nudge_rl.py` with `--mode baseline --baseline oed` | Runs the standalone OED baseline in the virtual-nudge environment. |
| **Train and evaluate an RL policy** | [`scripts/train_virtual_nudge_rl.py`](scripts/train_virtual_nudge_rl.py) | Checks the environment, trains SAC on the surrogate, and evaluates saved models. |
| **Inspect the virtual-nudge identification experiment** | [`notebooks/three_nudge_virtual_10param.ipynb`](notebooks/three_nudge_virtual_10param.ipynb) | A focused three-nudge, ten-parameter identification experiment. |
| **Inspect continuous estimation** | [`notebooks/three_nudge_virtual_10param_continuous.ipynb`](notebooks/three_nudge_virtual_10param_continuous.ipynb) | Examines measurement-by-measurement estimation during the observation intervals. |
| **Review the mathematical formulation** | [`MATHEMATICS.md`](MATHEMATICS.md) | Dynamics, identification, uncertainty, OED, RL reward, hybrid selection, and limitations. |

> **Important:** The standalone CLI and the extended comparison notebook use related but distinct experiment workflows. A SAC model trained using the CLI should **not** be assumed compatible with the notebook's comparison environment without verifying the observation and action spaces.

## Setup

Run commands from the **repository root** (the directory containing `src/`, `scripts/`, and `notebooks/`).

```bash
# For a fresh clone:
git clone https://github.com/aaliyahgaffey/virtual-nudge-oed-rl.git
cd virtual-nudge-oed-rl

# Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux

# Install the RL dependencies and notebook tools:
python -m pip install -r requirements-rl.txt
python -m pip install pandas matplotlib jupyterlab
```

The additional `pandas`, `matplotlib`, and `jupyterlab` packages support the workflow analysis and notebooks. For an existing local copy, simply navigate to that directory and activate its `.venv` instead of cloning again.

## 1. Evaluate OED (no RL training required)

The standalone script offers a quick comparison of OED-selected nudges against random actions. Use the **same seeds and episode count** for both runs:

```bash
# OED baseline
python -m scripts.train_virtual_nudge_rl --mode baseline --baseline oed --episodes 5 --seed 19

# Random baseline
python -m scripts.train_virtual_nudge_rl --mode baseline --baseline random --episodes 5 --seed 19
```

These commands use the default **surrogate** backend. They report each episode's return, number of nudges, identification-rank diagnostic, and failure status. Rollout JSON files are written under `results/rl_virtual_nudges/`.

For the **information-aware OED formulation used in the full OED/RL/hybrid study**, use the extended comparison notebook below; its evaluation functions are implemented in [`src/experiment_workflow.py`](src/experiment_workflow.py).

## 2. Train and evaluate RL

The standalone RL workflow uses **Soft Actor-Critic (SAC)**.

```bash
# Verify the environment implements the Gymnasium interface
python -m scripts.train_virtual_nudge_rl --mode check

# Train a SAC policy on the surrogate
python -m scripts.train_virtual_nudge_rl --mode train --timesteps 20000 --seed 19

# Evaluate the saved policy
python -m scripts.train_virtual_nudge_rl --mode evaluate --episodes 5 --seed 19
```

The default trained model is saved under `results/rl_virtual_nudges/sac_virtual_nudge.zip`. The evaluation command loads this model and writes rollout details under `results/rl_virtual_nudges/`. Increase training steps and the number of evaluation seeds for a substantive performance study; the commands above are a starting point, **not** a validated performance claim.

## 3. Compare OED, RL, and the hybrid approach

**Main notebook:** [`notebooks/three_nudge_rl_v11_extended_training.ipynb`](notebooks/three_nudge_rl_v11_extended_training.ipynb)

Open JupyterLab from the repository root:

```bash
jupyter lab
```

Then open `notebooks/three_nudge_rl_v11_extended_training.ipynb` and follow the notebook's configuration, training, and evaluation sections. This is the preferred route for comparing:

1. Random action selection (reference baseline).
2. Greedy, information-aware OED.
3. A trained SAC/RL policy.
4. Hybrid OED–RL selection (choosing between the OED and RL proposals using the same predicted-information score).

The supporting functions are in [`src/experiment_workflow.py`](src/experiment_workflow.py): `make_env`, `run_episode`, `evaluate`, `summary`, and `save`. The full workflow can require a **compatible trained SAC model**; training checkpoints and generated results are not bundled with the code.

For fair comparisons, evaluate policies on matched simulation configurations and held-out seeds. Compare **mass error, center-of-mass error, inertia error, episode success rate, and predicted versus observed covariance-proxy changes**. These distinguish the optimizer's information objective from actual estimation accuracy.

## Validation and testing

Run the automated tests from the repository root:

```bash
python -m pytest tests/ -q
```

The tests cover inertial-estimation mathematics, the virtual ten-parameter experiment, the RL environment, and the information-aware nudge workflow. Passing tests verify implemented behavior covered by those tests; they do not establish physical contact feasibility or flight safety.

## Repository organization

| Path | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | Interactive experiments, training, plots, and policy comparisons |
| [`scripts/train_virtual_nudge_rl.py`](scripts/train_virtual_nudge_rl.py) | Command-line baseline, SAC training, and evaluation |
| [`src/experiment_workflow.py`](src/experiment_workflow.py) | Matched-policy execution, metrics, and comparison summaries |
| [`src/information_gain.py`](src/information_gain.py) | Covariance and information-utility calculations |
| [`src/nudge_optimization.py`](src/nudge_optimization.py) | Nudge optimization routines |
| [`src/inertial_dynamics.py`](src/inertial_dynamics.py) | Rigid-body parameters, dynamics, and action mapping |
| [`src/nudge_environment.py`](src/nudge_environment.py) | Information-aware sequential-nudge environment |
| [`src/virtual_nudge_rl_env.py`](src/virtual_nudge_rl_env.py) | Standalone RL environment and optional MuJoCo backend |
| [`tests/`](tests/) | Automated validation |
| [`model/`](model/) and [`experiments/`](experiments/) | Meshes and supporting MuJoCo/spacecraft experiments |
| [`MATHEMATICS.md`](MATHEMATICS.md) | Mathematical details and modeling assumptions |

`results/` is intended for **local generated outputs** such as checkpoints, plots, and evaluation tables and is excluded from Git tracking. A fresh clone therefore needs to rerun training/evaluation to regenerate these artifacts.

## Scope and limitations

The principal surrogate models an asteroid responding to directly applied virtual impulses, **without an actuating spacecraft or robotic arm**. The separate MuJoCo backend includes supporting physical/spacecraft simulation infrastructure. The surrogate's action bounds limit quantities such as force, impulse, duration, and nominal torque, but do **not** by themselves guarantee asteroid drift containment, spacecraft separation, or control-barrier-function safety. The estimator's uncertainty is also an approximate covariance proxy, not a calibrated full Bayesian posterior. See [`MATHEMATICS.md`](MATHEMATICS.md) for details.
