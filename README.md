# DeePC for Unitree Go2 with Imitation-Learned Locomotion Execution

Research implementation accompanying a bachelor's thesis on **Data-enabled Predictive Control (DeePC)** for target-pose control of the Unitree Go2.

The project studies DeePC as a **high-level controller** that generates body-frame velocity commands

\[
u = [v_x,\; v_y,\; \omega_z]^T
\]

while locomotion is handled by a separate execution layer.

Two execution settings are evaluated:

1. **Physical Unitree Go2** — DeePC commands are executed through the native Unitree locomotion controller.
2. **Genesis simulation** — DeePC commands are executed by a frozen observable behaviour-cloning locomotion policy.

These are complementary experimental branches and should not be interpreted as the learned policy being deployed on the physical robot.

---

## Main Results

### Physical Unitree Go2

The final physical controller uses a three-input, body-frame **MIQP DeePC** formulation with motion-capture feedback and native Unitree locomotion.

**Final physical benchmark:**

- **17 / 17 retained runs** satisfied the `0.08 m` post-run position criterion.
- Experiments included:
  - forward motion,
  - backward motion,
  - left/right lateral motion,
  - diagonal motion,
  - one target-pose experiment with final-yaw regulation.
- The target-pose run achieved approximately:
  - final position error: **0.0126 m**
  - final yaw error: **0.0290 rad**
  - target hit time: **14.3 s**

The target-pose run additionally satisfied the `0.05 rad` final-yaw criterion.

These results apply only to the tested robot, target set, Hankel banks, tolerances, sensing setup, and operating conditions.

### Genesis + learned locomotion execution

The final simulation pipeline is:

```text
Target pose
    ↓
Constrained DeePC
    ↓
[vx, vy, yaw_rate]
    ↓
Observable BC locomotion policy
    ↓
12 raw joint actions
    ↓
Joint-target decoder + fixed PD
    ↓
Genesis physics
    ↓
Pose feedback to DeePC
```

The final frozen benchmark contains **43 sampled target poses**.

**Result: 24 / 43 targets reached.**

Notable regions of the demonstrated operating envelope include:

- **7 / 8** tested directions reached at `0.25 m`,
- all three forward distances reached through `1.20 m`,
- every tested heading from `-75°` to `+75°`,
- additional successful heading targets at `-90°` and `+120°`.

The remaining cases exposed direction-, distance-, solver-, actuator-, and stability-dependent limitations.

This benchmark is **simulation-only** and does not demonstrate sim-to-real transfer of the learned locomotion policy.

---

## System Architecture

### Physical execution

```text
Target pose
    ↓
Body-frame target error
    ↓
Three-input MIQP DeePC
    ↓
Safety / stale-pose / timeout checks
    ↓
[vx, vy, yaw_rate]
    ↓
Native Unitree locomotion controller
    ↓
Unitree Go2
    ↓
Motion-capture pose feedback
    └──────────────────────────────→ DeePC
```

The physical DeePC controller does not directly command joint positions or motor torques.

### Learned execution in Genesis

```text
Target pose
    ↓
Constrained DeePC
    ↓
Desired body-frame command
    ↓
Observable BC student
    ↓
raw_action[12]
    ↓
q_target = q_default + 0.5 * raw_action
    ↓
Joint-limit clipping + fixed PD
    ↓
Genesis Go2
    ↓
Measured state / pose feedback
```

At runtime the learned policy receives only observable robot state and the desired command. It does **not** receive a teacher, reference trajectory, motion ID, phase variable, or privileged state.

---

## Learned Locomotion Policy

The final student is a feed-forward behaviour-cloning network:

```text
48 → 256 ELU → 256 ELU → 12 linear
```

### Input

The 48-dimensional input consists of:

| Component | Dimensions |
|---|---:|
| Projected gravity | 3 |
| Body linear velocity | 3 |
| Body angular velocity | 3 |
| Relative joint positions | 12 |
| Joint velocities | 12 |
| Previous raw action | 12 |
| Desired `[vx, vy, yaw_rate]` | 3 |
| **Total** | **48** |

### Output

The network predicts 12 dimensionless raw actions:

```text
q_target = q_default + 0.5 * raw_action
```

The resulting joint targets are executed through fixed PD control in Genesis.

### Training data

The final dataset contains:

- **120 expert rollouts**
- **240,000 transitions**
- episode-level split:
  - 96 training episodes
  - 12 validation episodes
  - 12 test episodes

The final student was selected using validation MSE only.

Selected checkpoint test performance:

```text
Raw-action test MSE  ≈ 2.79746e-4
Raw-action test RMSE ≈ 0.01673
```

Low supervised error should not be interpreted as a guarantee of stable closed-loop locomotion; the integrated Genesis benchmark is therefore evaluated separately.

---

## DeePC Configuration

### Physical controller

The final physical controller uses:

```text
Sampling rate: 20 Hz
T_ini:         10
N:             20
Hankel columns: 900
Solver:        Gurobi MIQP
Time limit:    2.0 s
MIP gap:       0.08
Threads:       1
```

The resulting history and prediction windows correspond to:

```text
Past context:       0.5 s
Prediction horizon: 1.0 s
```

Final physical command alphabet:

```text
vx       ∈ {-0.10, 0.00, 0.10, 0.15} m/s
vy       ∈ {-0.10, 0.00, 0.10} m/s
yaw_rate ∈ {-0.30, -0.20, 0.00, 0.20, 0.30} rad/s
```

These values are empirical command levels selected from the tested physical pipeline and available Hankel-data support. They are **not maximum Unitree Go2 capabilities**.

Two final physical Hankel banks are included:

- backward-enriched bank: **3548 columns**
- balanced online bank: **900 columns**

Both stored input Hankel matrices have rank `90 / 90` at depth `T_ini + N = 30`.

### Genesis controller

The final Genesis benchmark uses:

```text
T_ini:          10
N:              20
Controller dt:  0.1 s
Reference gain: 2.75
Solver:         OSQP
```

The frozen optimisation includes:

```text
vy + yaw_rate <= 0.28800664310564905
-0.8 <= vx <= 0.8
```

The coupled lateral/yaw constraint is an **empirical numerical boundary**, not a formal physical safety set.

No post-solver command rewriting or projection is used in the final benchmark.

---

## Repository Structure

```text
.
├── configs/
│   ├── genesis/
│   └── imitation/
│
├── data/
│   ├── imitation/
│   └── physical/
│       ├── final_runs/
│       └── hankel/
│
├── docs/
│   └── repo_inventory/
│
├── models/
│
├── results/
│   ├── genesis/
│   └── physical/
│
├── scripts/
│   ├── genesis/
│   ├── imitation_learning/
│   └── physical/
│
├── src/
│   ├── deepc/
│   ├── learned_execution/
│   └── physical/
│
├── tests/
├── requirements.txt
└── README.md
```

Historical development branches, unsuccessful experimental checkpoints, debug logs, virtual environments, and unrelated intermediate artifacts are intentionally excluded from this research release.

---

## Physical Benchmark Data

The final physical benchmark contains **17 retained runs** covering:

```text
3 × forward
3 × backward
3 × left
3 × right
4 × diagonal
1 × target pose + final yaw
```

Raw final-run logs are stored under:

```text
data/physical/final_runs/
```

The benchmark manifest and derived metrics are stored separately from the raw logs.

The reported position results are evaluated in the **initial Go2 body frame**, rather than directly in the motion-capture world frame.

The online physical controller uses a `0.06 m` stopping tolerance, while the final stable-window evaluation uses a `0.08 m` position criterion. These have different purposes and should not be interpreted as conflicting thresholds.

---

## Genesis Benchmark Data

The canonical final Genesis benchmark is stored under:

```text
results/genesis/
```

The frozen benchmark contains:

```text
43 evaluated targets
24 reached
19 unsuccessful
```

The unsuccessful cases consisted of:

```text
10 requested-torque-limit terminations
4 falls
2 joint-limit violations
3 DeePC solver failures
```

The requested-torque criterion is checked before actuator clipping. Applied torque does not exceed the configured limit because execution is clipped by the runtime.

---

## Model Integrity

Final observable BC student:

```text
go2_observable_student_bc_linear_v1.pt
```

SHA-256:

```text
6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc
```

The hash can be used to verify that the checkpoint has not changed during transfer or reproduction.

For example:

```bash
sha256sum models/go2_observable_student_bc_linear_v1.pt
```

---

## Imitation Dataset

The canonical imitation-learning dataset is intentionally **not stored in normal Git history** because of its size.

Expected dataset:

```text
expert_imitation_dataset_v1.npz
```

Expected size:

```text
246,636,478 bytes
```

SHA-256:

```text
7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99
```

The dataset is distributed separately as a GitHub Release asset.

After downloading it, place it according to:

```text
data/imitation/README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/dylannwj/Go2-DeePC-imitationLearning.git
cd Go2-DeePC-imitationLearning
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Some parts of the project require additional platform-specific software:

### Genesis / learned execution

Requires the Genesis simulator and compatible PyTorch installation.

### Physical Unitree Go2

Physical deployment additionally requires the appropriate:

- Ubuntu / ROS2 environment,
- Unitree communication interface,
- motion-capture / VRPN interface,
- Gurobi installation and licence,
- access to the physical Go2 and associated network configuration.

See the files under:

```text
docs/
```

for additional environment and release information.

---

## Reproducibility Notes

The repository separates three evidence settings:

| Branch | Execution layer | Purpose |
|---|---|---|
| Ideal simulation | Direct body-motion model | Evaluate the DeePC formulation independently of locomotion |
| Genesis | Frozen observable BC policy + PD | Evaluate DeePC with imitation-learned locomotion execution |
| Physical Go2 | Native Unitree locomotion | Validate DeePC deployment on hardware |

Results from these branches should not be compared as though only simulator realism changed. They use different execution layers, datasets, target sets, and acceptance criteria.

In particular:

- physical success does **not** demonstrate deployment of the learned BC policy;
- Genesis success does **not** demonstrate sim-to-real transfer;
- full stored Hankel rank does **not** establish the stronger persistent-excitation assumptions for the complete nonlinear robot;
- empirical runtime constraints do **not** constitute formal safety guarantees.

---

## Thesis Scope

This project focuses on **high-level planar target-pose control**.

The demonstrated scope does not include:

- obstacle avoidance,
- global path planning,
- arbitrary terrain navigation,
- direct joint-level DeePC control,
- formal safety guarantees for the complete nonlinear quadruped,
- physical deployment of the imitation-learned locomotion controller,
- universal target-reaching guarantees.

The physical experiments support target-reaching claims only within the tested hardware configuration, targets, command limits, data banks, tolerances, and operating conditions.

---

## Related Work

The implementation builds on research in:

- Data-enabled Predictive Control and regularised DeePC,
- receding-horizon and model predictive control,
- behaviour cloning and DAgger,
- learned quadruped locomotion,
- offline imitation learning for Unitree Go2,
- Genesis-based robot simulation.

The thesis bibliography contains the complete references used in the accompanying research work.

---

## Citation

If you use this repository in academic work, please cite the accompanying thesis.

```bibtex
@thesis{neoh2026deepcgo2,
  author = {Dylan Neoh},
  title  = {Direct Data-Driven Prediction Control for Nonlinear System via Imitation Learning},
  school = {Technical University of Munich},
  year   = {2026}
}
```

> The thesis title above reflects the submitted thesis title. The repository itself focuses specifically on DeePC target-pose control for the Unitree Go2 and imitation-learned locomotion execution in Genesis.

---

## License

See [`LICENSE`](LICENSE) for the repository licence.

Third-party software, pretrained models, simulator assets, and robot interfaces remain subject to their respective licences.

---

## Repository

**GitHub:**  
https://github.com/dylannwj/Go2-DeePC-imitationLearning
