# Frozen expert contract reference

This dataset executes the exact artifacts that passed
`EXPERT_GENESIS_COMPATIBILITY_PASS`. The previous compatibility evidence is
read-only and is not copied over or rewritten by the dataset workflow.

## Exact artifacts

| Artifact | Path | SHA256 |
|---|---|---|
| Checkpoint | `expert_policy_validation/mjlab_genesis_visual_transfer/artifacts/model_500.pt` | `c5d125be84ecedf8c6d89ece8d12d7c1d6d8d081f304da3dd12f734944248eb3` |
| ONNX auxiliary | `expert_policy_validation/mjlab_genesis_visual_transfer/artifacts/policy.onnx` | `fbb8b61b12f2cd44dfc889f33566a8be2438124efee88b60c700b9605a94187a` |
| ONNX external data | `expert_policy_validation/mjlab_genesis_visual_transfer/artifacts/policy.onnx.data` | `40474dd235917877d657085f3f0a2d485d10e947e3599cad5328337e32222a74` |
| Deploy config | `expert_policy_validation/mjlab_genesis_visual_transfer/configs/deploy.yaml` | `a86582e599411ebaa011cb30f1711497a50a21230decf9c91bd02cb43d1e5300` |
| Environment config | `expert_policy_validation/mjlab_genesis_visual_transfer/configs/env.yaml` | `07aac8492b3119faea95577d9900c6dd1a6c57a2fb44454c3eda8c637a403c7c` |
| Agent config | `expert_policy_validation/mjlab_genesis_visual_transfer/configs/agent.yaml` | `4559c7cfa14933c896ffaea0af8a47f9692d6b63dde160126277a257ee5dfd69` |

The executed checkpoint is the frozen visual-transfer snapshot, not a later
checkpoint in the candidate directory. The compatibility gate's artifact
inventory and frozen contract are the authority for the hashes above.

## Teacher execution contract

- Teacher input: 42-D non-command state plus the exact desired command, making a 45-D input in the order `[base_ang_vel(3), projected_gravity(3), command(3), joint_pos_rel(12), joint_vel_rel(12), previous_raw_action(12)]`.
- Teacher output: 12-D raw normalized joint-position offset; no raw-action clipping.
- Decoding: `q_target = q_default + 0.5 * expert_action`.
- Fixed PD: `Kp=[20,20,40]` and `Kd=[1,1,2]` repeated across four legs.
- Joint mapping: policy order `FL, FR, RL, RR` maps to the Genesis named order `FR, FL, RR, RL` with `[3,4,5,0,1,2,9,10,11,6,7,8]`.
- Robot: Genesis 1.3.1 bundled Go2 URDF, flat plane, root initialization at 0.40 m, frozen default pose from the deploy config.
- Timing: policy 50 Hz, physics 200 Hz, decimation 4.
- Runtime safety: only the validated physical Genesis joint-target and effort-limit clips; requested and applied values are both recorded.

## Evidence references

- Compatibility result: `expert_policy_validation/mjlab_expert_compatibility_gate/reports/expert_genesis_compatibility_report.md`.
- Frozen contract: `expert_policy_validation/mjlab_expert_compatibility_gate/reports/frozen_contract.md`.
- Frozen artifact inventory: `expert_policy_validation/mjlab_expert_compatibility_gate/reports/frozen_artifacts.json`.
- Joint/actuator evidence: `expert_policy_validation/mjlab_expert_compatibility_gate/reports/joint_actuator_contract.md`.
- Reload evidence: `expert_policy_validation/mjlab_expert_compatibility_gate/reports/reload_determinism.md`.

