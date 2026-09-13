# Frozen student observation contract

Classification: **STUDENT_OBSERVATION_CONTRACT_PASS**

This contract is copied from the existing authoritative Genesis walking
interface in `src/deepc_go2/genesis_walking/contracts.py`. It is not derived
from the teacher observation. The student state is recorded independently from
the Genesis deployment snapshot at the beginning of every 50 Hz policy step.

## Future student input

```text
student_input[48] = student_state[45] + desired_command[3]
```

The desired command order is `[vx, vy, yaw_rate]`, in the body frame, with
units `[m/s, m/s, rad/s]`. The command is the exact command passed to the
expert for that row. Achieved velocity is never substituted for the desired
command.

## Exact 45-D state order

| Index | Name | Meaning | Units / frame |
|---:|---|---|---|
| 0 | `projected_gravity_x` | World gravity inverse-rotated into the base frame | unit vector, body |
| 1 | `projected_gravity_y` | World gravity inverse-rotated into the base frame | unit vector, body |
| 2 | `projected_gravity_z` | World gravity inverse-rotated into the base frame | unit vector, body |
| 3 | `base_linear_velocity_x` | Base linear velocity | m/s, body |
| 4 | `base_linear_velocity_y` | Base linear velocity | m/s, body |
| 5 | `base_linear_velocity_z` | Base linear velocity | m/s, body |
| 6 | `base_angular_velocity_x` | Base angular velocity | rad/s, body |
| 7 | `base_angular_velocity_y` | Base angular velocity | rad/s, body |
| 8 | `base_angular_velocity_z` | Base angular velocity | rad/s, body |
| 9 | `joint_pos_rel_FR_hip` | FR hip position minus the frozen default pose | rad, canonical Genesis order |
| 10 | `joint_pos_rel_FR_thigh` | FR thigh position minus the frozen default pose | rad, canonical Genesis order |
| 11 | `joint_pos_rel_FR_calf` | FR calf position minus the frozen default pose | rad, canonical Genesis order |
| 12 | `joint_pos_rel_FL_hip` | FL hip position minus the frozen default pose | rad, canonical Genesis order |
| 13 | `joint_pos_rel_FL_thigh` | FL thigh position minus the frozen default pose | rad, canonical Genesis order |
| 14 | `joint_pos_rel_FL_calf` | FL calf position minus the frozen default pose | rad, canonical Genesis order |
| 15 | `joint_pos_rel_RR_hip` | RR hip position minus the frozen default pose | rad, canonical Genesis order |
| 16 | `joint_pos_rel_RR_thigh` | RR thigh position minus the frozen default pose | rad, canonical Genesis order |
| 17 | `joint_pos_rel_RR_calf` | RR calf position minus the frozen default pose | rad, canonical Genesis order |
| 18 | `joint_pos_rel_RL_hip` | RL hip position minus the frozen default pose | rad, canonical Genesis order |
| 19 | `joint_pos_rel_RL_thigh` | RL thigh position minus the frozen default pose | rad, canonical Genesis order |
| 20 | `joint_pos_rel_RL_calf` | RL calf position minus the frozen default pose | rad, canonical Genesis order |
| 21 | `joint_vel_rel_FR_hip` | FR hip joint velocity | rad/s, canonical Genesis order |
| 22 | `joint_vel_rel_FR_thigh` | FR thigh joint velocity | rad/s, canonical Genesis order |
| 23 | `joint_vel_rel_FR_calf` | FR calf joint velocity | rad/s, canonical Genesis order |
| 24 | `joint_vel_rel_FL_hip` | FL hip joint velocity | rad/s, canonical Genesis order |
| 25 | `joint_vel_rel_FL_thigh` | FL thigh joint velocity | rad/s, canonical Genesis order |
| 26 | `joint_vel_rel_FL_calf` | FL calf joint velocity | rad/s, canonical Genesis order |
| 27 | `joint_vel_rel_RR_hip` | RR hip joint velocity | rad/s, canonical Genesis order |
| 28 | `joint_vel_rel_RR_thigh` | RR thigh joint velocity | rad/s, canonical Genesis order |
| 29 | `joint_vel_rel_RR_calf` | RR calf joint velocity | rad/s, canonical Genesis order |
| 30 | `joint_vel_rel_RL_hip` | RL hip joint velocity | rad/s, canonical Genesis order |
| 31 | `joint_vel_rel_RL_thigh` | RL thigh joint velocity | rad/s, canonical Genesis order |
| 32 | `joint_vel_rel_RL_calf` | RL calf joint velocity | rad/s, canonical Genesis order |
| 33 | `previous_action_FR_hip` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 34 | `previous_action_FR_thigh` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 35 | `previous_action_FR_calf` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 36 | `previous_action_FL_hip` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 37 | `previous_action_FL_thigh` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 38 | `previous_action_FL_calf` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 39 | `previous_action_RR_hip` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 40 | `previous_action_RR_thigh` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 41 | `previous_action_RR_calf` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 42 | `previous_action_RL_hip` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 43 | `previous_action_RL_thigh` | Previous normalized controller action | raw expert action slot, canonical Genesis order |
| 44 | `previous_action_RL_calf` | Previous normalized controller action | raw expert action slot, canonical Genesis order |

The previous-action values occupy the frozen observable history slot. They are
not appended as an additional teacher feature, and no teacher-only reference,
future state, achieved future velocity, contact force, hidden state, or expert
diagnostic is part of `student_input`.

## Joint and timing identity

- Canonical joint order: `FR, FL, RR, RL`, each as `hip, thigh, calf`.
- Frozen student default pose from the authoritative Genesis contract: `[0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]` rad in canonical Genesis order. The teacher has a separate deploy pose, recorded in the expert contract reference.
- One sample is recorded per policy step: 50 Hz (`0.02 s`); Genesis physics remains 200 Hz with decimation 4.

Authoritative source: [contracts.py](../../src/deepc_go2/genesis_walking/contracts.py).
