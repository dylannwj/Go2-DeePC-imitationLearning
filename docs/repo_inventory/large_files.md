# Large-file audit

Scope: release-relevant files selected in `file_classification.csv`, plus representative large files from historical branches. Source was inspected read-only. Thresholds below use decimal megabytes: 10 MB = 10,000,000 bytes, 50 MB = 50,000,000 bytes, 100 MB = 100,000,000 bytes.

## Selected artifacts

| Artifact | Bytes | Thresholds crossed | Decision |
|---|---:|---|---|
| `expert_imitation_dataset/processed/expert_imitation_dataset_v1.npz` | 246,636,478 | >10 MB, >50 MB, >100 MB | REPRODUCIBILITY; canonical processed dataset; SHA-256 `7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99`. |
| `evidence_miqp_3input_final_yaw_2026_07_06_180804/hankels/dog2_vxyyaw_hankel_Tini10_N20_BACKWARD_2026_07_06.npz` | 10,221,512 | >10 MB | CORE; one of two final physical banks; SHA-256 `6928fa66425ddc65399080d27acac42042b70bdedd645bfa460b80b0538b3cf7`. |
| Genesis 1.3.1 `assets/urdf/go2/dae/base.dae` | 10,727,399 | >10 MB | External Genesis asset referenced by the Go2 URDF; obtain from the pinned dependency rather than Git. |
| `expert_imitation_dataset/raw/episode_*.npz` | 248,955,375 aggregate | aggregate >10 MB, >50 MB, >100 MB | REPRODUCIBILITY; 120 individual files, each approximately 2.07 MB, so no individual raw episode crosses 10 MB. |

The final Genesis Hankel is only 377,543 bytes, the final student checkpoint is 333,509 bytes, the expert checkpoint is 4,569,571 bytes, and the balanced physical Hankel is 2,595,272 bytes. They do not cross the 10 MB individual-file threshold.

For the selected release rows, the individual-file counts are:

| Threshold | Count |
|---|---:|
| >10 MB | 3 |
| >50 MB | 1 |
| >100 MB | 1 |

The aggregate raw imitation set is reported separately because counting the 120 raw files as one file would obscure the per-file packaging behavior.

## Representative historical files that must stay out

The source tree contains many much larger experimental datasets, telemetry logs, rosbag databases, and archives. These are not evidence for the selected final branches and are classified `EXCLUDE` or represented by exclusion families in the CSV.

| File | Bytes | Reason to exclude |
|---|---:|---|
| `kine2go_work/genesis_walking_rebuild/phase3_final_ppo_curriculum/telemetry/C1_rollouts.csv` | 1,273,060,669 | Historical PPO telemetry. |
| `kine2go_dfki_dynamic_validation/dfki_extract/raw/l7/rosbag2_2024_10_01-08_34_37_0.db3` | 1,184,112,640 | Raw rosbag database; unrelated to selected thesis evidence. |
| `kine2go_work/genesis_walking_rebuild/phase3_t1_motion_recovery/telemetry/R5_parent_vx010_staggered100.csv` | 778,122,679 | Historical locomotion experiment. |
| `kine2go_work/genesis_imitation_learning/r4b_torque_gate_audit/telemetry/targeted_torque_trace.csv` | 654,132,671 | Historical safety audit telemetry. |
| `kine2go_work/deepc_lowlevel_kine2go/physics_pd_unified_start_long_sliding_hankel_Tini10_N120.npz` | 462,986,544 | Different low-level Hankel problem and horizon. |
| `datasets/U5V_dagger_dataset.npz` | 439,270,809 | Historical DAgger candidate. |
| `low_level_mujoco_deepc/lowlevel_hankel_Tini10_N400_multiseg.npz` | 386,833,536 | MuJoCo low-level branch, not final physical or Genesis bank. |
| `datasets/U5W_complete_replay_corpus.npz` | 305,969,982 | Historical replay corpus. |
| `datasets/BW_U5U-B_joint_dataset.npz` | 283,872,754 | Historical candidate dataset. |
| `datasets/BW_modified_U5V_joint_dataset_round2.npz` | 259,785,813 | Historical candidate dataset. |
| `datasets/final_canonical_locomotion_teacher_bank_v1.npz` | 213,675,615 | Earlier teacher-bank branch; not the accepted 120-episode processed dataset. |
| `parallel_closed_loop_il/datasets/D_parallel_v1/D_parallel_v1.npz` | 183,572,185 | Historical parallel/DAgger branch; not final linear BC. |
| `expert_imitation_dataset/processed/expert_imitation_dataset_v1.npz` | 246,636,478 | This one is retained despite size because it is the canonical final dataset. |

The exclusion list is intentionally broader than this table: `datasets/*.npz`, historical `U4/U5/B2/B3/S0-S7/P` families, generated telemetry, media, archives, virtual environments, and caches are all excluded unless a later thesis review identifies a specific citation.

## Packaging guidance

- Prefer Git LFS, a release asset, or an explicitly documented external artifact for the processed imitation dataset.
- Keep raw episodes optional but reproducible; if included, preserve all 120 filenames and the episode manifest rather than collapsing them into a new archive.
- Keep the 10.22 MB physical backward Hankel unmodified and hash-verify it after packaging.
- Do not package `.venv`, `deepdeeps_env`, recovery virtual environments, caches, rosbag databases, or generated per-run logs.
- The selected Genesis benchmark JSON references per-run JSONL paths, but those logs are not selected release artifacts. Normalize or remove those path references in a reviewed copy of the report; do not rewrite the source report during this inventory pass.
