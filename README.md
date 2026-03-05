# Safe CommonRoad RL Environment

This repository contains a specialized safety-constrained Reinforcement Learning (RL) environment for autonomous driving, built upon the **CommonRoad-RL** framework. It implements a formal supervisory control loop that filters agent actions through a safety verifier to prevent collisions and ensure kinematic feasibility.

## Overview

The core of this project is `safe_commonroad_env.py`, which introduces a `SafetyLayer` wrapper and a `SafetyVerifier`. Unlike standard RL environments where the agent's output is directly applied to the vehicle, this system treats agent outputs as "proposed actions" that are modified if they risk violating safety constraints.



## Key Technical Components

### 1. Safety Verifier (`SafetyVerifier`)
The mathematical engine responsible for spatial and temporal safety checks.
* **Curvilinear Coordinate Integration**: Utilizes `commonroad-clcs` to transform Cartesian coordinates into path-aligned curvilinear coordinates ($s, d$) for high-precision lane tracking.
* **Collision-Free Area Calculation**: Dynamically identifies safe gaps between obstacles in the current lane and potential successor lanes.
* **Kinematic Constraint Enforcement**:
    * **Curvature ($\kappa$) Control**: Computes required steering rates ($\ddot{\kappa}$) based on lateral acceleration limits.
    * **Velocity Scaling**: Automatically adjusts maximum longitudinal velocity based on local lane curvature: $$v_{max} = \sqrt{r \cdot a_{lat}}$$
* **Safe Distance Sets**: Implements longitudinal safety margins (RSS-inspired), generating sets of admissible (velocity, polygon) pairs.
* **Intersection & Successor Logic**: Features specialized handling for merging lanes and deep successor searches (up to 3 levels) to prevent collisions in complex junctions.

### 2. Safety Layer (`SafetyLayer`)
A Gymnasium wrapper that modifies the interaction between the agent and the environment.
* **Action Filtering:** Intercepts `jerk_dot` and `kappa_dot_dot` commands. It uses a PD controller for curvature ($K_p=4.0, K_d=2.0$) to maintain lane centering.
* **Observation Augmentation:** The agent receives additional safety context in its observation vector, including safe action bounds and distance to the end of the current lane.
* **Priority Management:** Specifically designed for intersection logic, identifying "conflict zones" and enforcing yielding behavior if high-priority vehicles are detected within a 30m range.

---

## Global Safety Constants

| Constant | Description | Value |
| :--- | :--- | :--- |
| `MAX_KAPPA_DOT_DOT` | Max change in curvature rate | 0.7 |
| `VELOCITY_MAX` | Global speed limit ($m/s$) | 40.0 |
| `SAFE_VELOCITY_TOLERANCE` | Buffer for velocity checks | 0.35 |
| `POLYGON_BUFFER` | Inflation/Deflation for road polygons | 0.2 |
| `MAX_SUCCESSOR_DEPTH` | Depth for Time-To-Collision (TTC) logic | 3 |

---

## Dataset Limitation

**Important:** This environment is explicitly configured and optimized to run **only on the inD dataset**. The lanelet densification, intersection priority logic, and curvilinear coordinate systems are tuned to the specific geometries and traffic behaviors found in the inD recordings.

## Usage

To use this environment, you must integrate it with the provided version of the CommonRoad-RL codebase.

1.  **Install Dependencies:**
    Install the `commonroad-rl` package and its dependencies (including `commonroad-io`, `commonroad-clcs`, and `gymnasium`) as you would for a standard CommonRoad-RL setup.

2.  **Environment Setup:**
    Ensure the `safe_commonroad_env.py` file is in your python path.

3.  **Running the Training/Simulation:**
    Run the provided `RLtut1.py` script. While this script follows the structure of a "vanilla" CommonRoad-RL tutorial, it has been modified to initialize the `SafetyLayer` and utilize the safety-verified step function.

```bash
# Run the modified tutorial script
python RLtut1.py
```

---

## 📚 Credits and Attribution

This project is an extension of the **CommonRoad-RL** framework. It integrates a custom safety-constrained layer into the existing reinforcement learning pipeline.

### Original Framework
* **CommonRoad-RL:** Developed by the [Cyber-Physical Systems Group](https://commonroad.in.tum.de/) at the Technical University of Munich (TUM).
* **Source Code:** [CommonRoad-RL GitHub Repository](https://gitlab.lrz.de/tum-cps/commonroad-rl)
* **Dataset:** This project specifically utilizes the [inD Dataset](https://www.ind-dataset.com/) (Intersection Dataset by fka GmbH).

### Academic Citation
If you use this codebase for research, please cite the original CommonRoad-RL paper as follows:

```bibtex
@inproceedings{wang2021commonroad,
  title={CommonRoad-RL: A Benchmark for Predictive Motion Planning with Deep Reinforcement Learning},
  author={Wang, Xiao and others},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  year={2021},
  organization={IEEE}
}
