# Go2Project inventory for a clean thesis release

Inventory date: 2026-09-13  
Source inspected read-only: `/home/dylan/Desktop/Go2Project`  
Release target: `/home/dylan/Desktop/DeePC-Go2-Thesis`

## Result

The source tree contains two different final-looking experiments that must remain separate in the thesis release:

1. The final physical branch is the frozen 3-input MIQP DeePC with final-yaw UI evidence in `evidence_miqp_3input_final_yaw_2026_07_06_180804`. Its retained physical logs are the 17 category files at the repository root. Its two Hankel banks are not the Genesis student Hankel.
2. The final Genesis branch is `final_deepc_student/live_ui` plus the `C2_PLUS_VX` frozen-controller adapter in `catastrophic_command_safety_v2`. Its selected benchmark is the 43-case report generated on 2026-09-13.

Only the five reports in this directory were created. No file in `Go2Project` was copied, moved, deleted, renamed, or modified during this inventory pass. The target project remains a skeleton; all destinations below are proposals for a later reviewed release pass.

The companion CSV inventories release-relevant files individually and uses explicit glob/family rows for large historical groups. A glob row is one classification record, not a claim that every member should be copied.

## Decision table

| Area | Selected source | Proposed release location | Decision |
|---|---|---|---|
| Core DeePC library | `src/deepc_go2/{geometry,hankel,qp,receding,local_controller,miqp}.py` | `src/deepc/` | Keep as reusable source, after import/package cleanup. |
| Physical controller | `evidence_miqp_3input_final_yaw_2026_07_06_180804/source_files/` | `src/physical/` | Keep the exact final-yaw source files as the physical provenance baseline. |
| Physical Hankels | `evidence_miqp_3input_final_yaw_2026_07_06_180804/hankels/` | `data/physical/hankel/` | Keep exactly two named banks; do not substitute the Genesis Hankel. |
| Physical evidence | root category folders: forward, backward, left, right, diagonals, target yaw | `data/physical/final_runs/` | Keep 17 CSVs, with category names normalized only in the destination proposal. |
| Imitation dataset | `expert_imitation_dataset/` | `data/imitation/` | Keep the 120 accepted episodes, canonical processed dataset, split metadata, and contracts needed to reproduce BC. |
| Final learned student | `student_bc/linear_output_revision/checkpoints/final/` | `models/` | Keep `go2_observable_student_bc_linear_v1.pt` and its exact SHA-256. |
| Genesis controller | `final_deepc_student/target_suite_recovery/` + frozen champion adapter | `src/deepc/` and `src/learned_execution/` | Keep only the selected runtime chain; rewrite sibling-repository path assumptions. |
| Genesis evidence | `final_deepc_student/live_ui/reports/` | `results/genesis/` | Keep the CSV, JSON, Markdown report, and freeze manifest for the 43-case result. |
| Historical branches | U4/U5/B2/B3/S0-S7/P families, archives, older mode-switched runs | none | Exclude from the clean release unless a thesis chapter explicitly cites one. |

## Classification counts

Counts are computed from rows in `file_classification.csv`; family/glob rows count once.

| Classification | Rows |
|---|---:|
| CORE | 33 |
| REPRODUCIBILITY | 44 |
| EVIDENCE | 30 |
| EXCLUDE | 27 |
| **Total** | **134** |

## Final physical branch

The intended physical command input is `[vx, vy, yaw_rate]` with `Tini=10` and `N=20`. The final MIQP command alphabet found in the exact solver source is:

- `vx`: `{-0.10, 0, 0.10, 0.15}`
- `vy`: `{-0.10, 0, 0.10}`
- `yaw_rate`: `{-0.30, -0.20, 0, 0.20, 0.30}`

The two physical banks are:

| File | Bytes | SHA-256 | Hankel columns |
|---|---:|---|---:|
| `dog2_vxyyaw_hankel_Tini10_N20_BACKWARD_2026_07_06.npz` | 10,221,512 | `6928fa66425ddc65399080d27acac42042b70bdedd645bfa460b80b0538b3cf7` | 3,548 |
| `dog2_vxyyaw_hankel_Tini10_N20_BALANCED_YAW_BACK_2026_07_06.npz` | 2,595,272 | `0ad50305d200811eab41008f242630f8e4ead592f0a9bda2f5527678499073f3` | 900 |

Both contain `Up`, `Uf`, `Yp`, `Yf`, `H_u_all`, and `H_y_all` with 3 input channels, 3 output channels, `Tini=10`, and `N=20`. The balanced bank is the practical yaw-on-spot candidate referenced by the evidence README; the larger backward bank is retained because it is explicitly named as a final bank.

The 17 located physical evidence files are:

| Category | Count | Source folder |
|---|---:|---|
| forward | 3 | `foward 50cm/` (source spelling preserved) |
| backward | 3 | `back50cm/` |
| left | 3 | `left 30cm/` |
| right | 3 | `right30cm/` |
| diagonal | 4 | `diagonals/` |
| target yaw | 1 | `random_yawHeading/newFilerandomYaw.csv` |
| **Total** | **17** | |

Important provenance limitation: no single final physical manifest or independently recomputed metrics file tying these 17 root-level CSVs to a denominator was located. The `17/17` wording is preserved in the thesis review material, but this inventory does not independently recompute it. The target-yaw CSV contains 610 non-NaN `ui_target_yaw` samples while `controller_final_yaw` is entirely NaN, so it should be treated as raw target-yaw evidence rather than a self-sufficient final-yaw metrics table.

## Imitation-learning pipeline

The located final pipeline is:

`model_500.pt` expert → 120 accepted Genesis episodes → 240,000 transitions → episode-disjoint 80/10/10 split → linear-output BC → verified student checkpoint.

Observed artifacts and claims:

- Expert compatibility checkpoint: `expert_policy_validation/mjlab_genesis_visual_transfer/artifacts/model_500.pt`, 4,569,571 bytes, SHA-256 `c5d125be84ecedf8c6d89ece8d12d7c1d6d8d081f304da3dd12f734944248eb3`.
- Raw dataset: 120 `episode_*.npz` files, 248,955,375 bytes in aggregate.
- Canonical processed dataset: `expert_imitation_dataset_v1.npz`, 246,636,478 bytes, SHA-256 `7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99`.
- Split: 96 episodes / 192,000 rows train, 12 / 24,000 validation, 12 / 24,000 test.
- Student architecture: `48 → 256 ELU → 256 ELU → 12 linear`.
- Final checkpoint: `go2_observable_student_bc_linear_v1.pt`, 333,509 bytes, SHA-256 `6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc`.
- The dataset report records zero falls, base contacts, NaN/Inf flags, actual joint-limit violations, and torque-limit violations. The student report records the linear BC preflight, offline, reload, and closed-loop gates as passing.

## Final Genesis benchmark

The selected result is `C2_PLUS_VX`, not the historical C2-only variant:

- 43 total cases: 24 translation and 19 pure-heading cases.
- Target reaching: `24/43`.
- Solver failures: `3`.
- Hard safety events: `16` episode events.
- Completion: position error `≤0.06 m`, wrapped yaw error `≤0.1 rad`, three confirmations, maximum 100 replans.
- DeePC: 3 inputs `[vx, vy, yaw_rate]`, outputs `[forward_step_m, left_step_m, yaw_step_rad]`, `Tini=10`, `N=20`, sample period `0.1 s`, reference gain `2.75`.
- Solver-side constraints: `future_input[t, vy] + future_input[t, yaw_rate] <= 0.28800664310564905` and `-0.8 <= future_input[t, vx] <= 0.8` for every horizon step.
- The freeze explicitly records `post_solver_command_rewriting=false`.
- Genesis freeze runtime: Genesis `1.3.1`, Python `3.12.13`, NumPy `2.4.4`, PyTorch `2.10.0+cpu`, OSQP `1.1.1`, CVXPY `1.8.2`, physics timestep `0.005 s`, control timestep `0.02 s`, four physics steps per policy action, collision enabled, viewer disabled, seed `20260804`.

The Genesis benchmark Hankel is a separate file: `final_student_deepc_hankel_v2_stride2.npz`, 377,543 bytes, SHA-256 `5ece444b2d82a75df6b751809d2b5f5c1aacfe06cc69a0919fa21532804bfb7b`. It has 1,216 columns and must not replace either physical bank.

## Release blockers to resolve after review

1. The target skeleton calls the package `src/deepc`, while selected code imports `deepc_go2`; either preserve that package name in the target or perform a tested import rewrite.
2. The exact physical online controller contains a hardcoded `/home/dylan/Desktop/Go2Project/deadzoneQuantized` base path and the physical stack assumes ROS2, mocap topics, Tkinter, and a MIQP-capable solver. Convert this to configuration/CLI paths before release.
3. The Genesis runtime resolves Go2 and plane URDFs from an installed Genesis package inside a project-local recovery virtual environment. Those URDFs are not project-local release files; licensing and pinning must be decided.
4. Genesis runtime code dynamically loads files across `final_deepc_student`, `student_bc`, `catastrophic_command_safety_v2`, and root `scripts`. The proposed tree must collapse this into one importable package and preserve all freeze hashes.
5. Generated benchmark manifests contain absolute source paths and per-run log paths. Keep the selected summaries, but normalize paths in any copied release manifest.

## Stop point

This is an inventory and release-boundary proposal only. Do not copy or modify source files until the classification and open issues are reviewed.

## Review-stage decision

The follow-up review retained 84 of the original 107 proposed items and added 22 verified dependency-closure items. The resulting [final migration manifest](final_migration_manifest.csv) has 106 retained rows: 44 CORE, 40 REPRODUCIBILITY, and 22 EVIDENCE. Twenty-three original candidates were removed after review, principally duplicate generated reports, obsolete controller/data variants, raw imitation episodes, auxiliary ONNX files, and internal UI integration metadata.

The canonical imitation artifact is the processed 246,636,478-byte dataset. The review recommends distributing only that file as a GitHub Release asset, with the six small split files in the repository; the 120 raw episodes remain local/provenance-only. The final student checkpoint is hash-matched and normal-Git-sized.

The Genesis branch is ready for later migration after path/package cleanup. The physical branch has all 17 located CSVs and both final Hankels, but it is not publication-ready until a final run manifest and final metrics file are located or reconstructed. The target-yaw CSV's all-NaN `controller_final_yaw` column is an additional evidence caveat. See [final_release_tree.md](final_release_tree.md) for the exact reviewed tree.
