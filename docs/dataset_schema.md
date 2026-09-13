# Frozen expert imitation dataset schema

Schema version: **`expert-imitation-dataset-v1`**

The raw representation is one deterministic NPZ per episode. The processed
canonical representation is one deterministic, episode-flattened NPZ containing
only accepted episodes. All arrays use the first dimension for policy-step
samples unless explicitly noted.

## Core learning arrays

| Field | Shape | Description |
|---|---|---|
| `student_state` | `[N,45]` | Frozen observable student state, exact order in `student_observation_contract.md` |
| `desired_command` | `[N,3]` | Actual body-frame command passed to the expert: `[vx, vy, yaw_rate]` |
| `student_input` | `[N,48]` | Exact concatenation `student_state + desired_command`; no privileged fields |
| `teacher_obs` | `[N,45]` | Expert-only observation: 42-D teacher state with command inserted at indices 6:9 |
| `expert_action` | `[N,12]` | Raw frozen expert output in teacher policy order (`FL, FR, RL, RR`); never clipped in storage |
| `expert_action_student_order` | `[N,12]` | Same raw action after the explicit named permutation into canonical Genesis order |
| `q_target` | `[N,12]` | Unclipped target `q_default + 0.5 * expert_action` in teacher policy order |
| `q_target_student_order` | `[N,12]` | `q_target` after the explicit named permutation into canonical Genesis order |
| `previous_expert_action` | `[N,12]` | Previous raw action used in the teacher observation, in teacher policy order |
| `previous_expert_action_student_order` | `[N,12]` | Previous raw action in canonical Genesis order used by the frozen student state contract |

## Achieved-state and actuator diagnostics

| Field | Shape | Description |
|---|---|---|
| `base_position` | `[N,3]` | Post-action Genesis base position in world coordinates |
| `base_rpy` | `[N,3]` | Post-action roll, pitch, yaw in radians |
| `base_linear_velocity` | `[N,3]` | Post-action body-frame base linear velocity |
| `base_angular_velocity` | `[N,3]` | Post-action body-frame base angular velocity |
| `joint_position` | `[N,12]` | Post-action joint position in canonical Genesis order |
| `joint_velocity` | `[N,12]` | Post-action joint velocity in canonical Genesis order |
| `foot_contacts` | `[N,4]` | Post-action contact flags in `FR, FL, RR, RL` order |
| `q_target_applied` | `[N,12]` | Target after Genesis joint-limit safety clipping |
| `preclip_torque` | `[N,12]` | Fixed-PD torque before effort-limit clipping |
| `applied_torque` | `[N,12]` | Torque returned by the validated Genesis path |

## Identity, time, and safety arrays

| Field | Shape | Description |
|---|---|---|
| `episode_id` | `[N]` | Integer episode identifier |
| `step_index` | `[N]` | Zero-based policy step within the episode |
| `seed` | `[N]` | Episode schedule seed |
| `command_segment_id` | `[N]` | Segment index in the command schedule |
| `transition_window` | `[N]` | First 1.0 s after a command change, excluding episode start |
| `timestamp` | `[N]` | Monotonic pre-action policy-step timestamp in seconds |
| `fall_flag` | `[N]` | Roll/pitch/height fall validity flag |
| `base_contact` | `[N]` | Genesis base-contact flag |
| `actual_joint_limit_violation` | `[N]` | Measured position outside Genesis limits beyond tolerance |
| `q_target_clipping` | `[N]` | Any requested target clipped by the physical joint limit |
| `torque_limit_violation` | `[N]` | Any preclip fixed-PD torque beyond the frozen effort limit |
| `nan_inf_flag` | `[N]` | Any nonfinite teacher, target, torque, or measured state |

Raw episode metadata additionally records the schedule, termination reason,
acceptance decision, provenance hashes, and schema version. A failed episode is
kept as a whole raw episode and is never row-pruned into the accepted set.

## Command collection envelope

The initial command bank is conservative relative to the source deploy ranges:

```text
vx       ∈ [-0.30, +0.50] m/s
vy       ∈ [-0.20, +0.20] m/s
yaw_rate ∈ [-0.50, +0.50] rad/s
```

The previously validated gate covered the lower forward range, lateral
characterization, and both yaw signs. The collection report distinguishes
previously covered values from newly characterized values; it does not treat
achieved velocity as a command label.
