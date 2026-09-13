# Final release tree after migration

This is the current clean-repository tree after the approved migration.
External-release assets are intentionally described in the notes below but
are not present in this working tree.

```text
DeePC-Go2-Thesis/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── deepc/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   ├── hankel.py
│   │   ├── qp.py
│   │   └── genesis_recovered_deepc.py
│   │
│   ├── physical/
│   │   ├── miqp_solver_final.py
│   │   ├── online_receding_controller_final.py
│   │   └── final_yaw_ui.py
│   │
│   └── learned_execution/
│       ├── contracts.py
│       ├── direct_student_runtime.py
│       ├── candidate_qp.py
│       ├── frozen_champion_controller.py
│       ├── genesis_guard.py
│       ├── genesis_deployment_scene.py
│       ├── genesis_window.py
│       ├── linear_bc_common.py
│       ├── ui_state.py
│       ├── ui_canvas.py
│       ├── ui_runtime_bridge.py
│       ├── final_deepc_ui.py
│       ├── ui/
│       │   └── __init__.py
│       ├── controllers/
│       │   └── __init__.py
│       └── champion/
│           └── __init__.py
│
├── scripts/
│   ├── physical/
│   │   ├── build_vxyyaw_hankel.py
│   │   └── collect_vxyyaw_dataset.py
│   │
│   ├── imitation_learning/
│   │   ├── collect_expert_dataset.py
│   │   ├── dataset_common.py
│   │   ├── validate_expert_dataset.py
│   │   ├── build_episode_splits.py
│   │   ├── run_expert_validation.py
│   │   ├── run_expert_compatibility_gate.py
│   │   ├── collect_command_teacher_rollouts.py
│   │   ├── train_command_bc.py
│   │   ├── train_linear_bc.py
│   │   ├── preflight_linear.py
│   │   ├── evaluate_student_offline.py
│   │   ├── reload_determinism.py
│   │   └── finalize_linear.py
│   │
│   └── genesis/
│       ├── final_champion_benchmark.py
│       ├── validate_direct_student_interface.py
│       └── run_live_product.py
│
├── configs/
│   ├── genesis/
│   │   ├── candidate_config.json
│   │   ├── deepc_final_recovered.json
│   │   ├── frozen_runtime_contract.json
│   │   └── ui_config.json
│   └── imitation/
│       ├── collection_config.json
│       ├── split_manifest.json
│       ├── dataset_hashes.json
│       ├── expert_artifact_manifest.json
│       ├── final_student_manifest.json
│       ├── selected_checkpoint.json
│       ├── offline_gate.json
│       ├── closed_loop_result.json
│       └── expert/
│           ├── agent.yaml
│           ├── deploy.yaml
│           └── env.yaml
│
├── models/
│   └── go2_observable_student_bc_linear_v1.pt
│
├── data/
│   ├── physical/
│   │   ├── PENDING_FILES.md
│   │   ├── hankel/
│   │   │   ├── dog2_vxyyaw_hankel_Tini10_N20_BACKWARD_2026_07_06.npz
│   │   │   └── dog2_vxyyaw_hankel_Tini10_N20_BALANCED_YAW_BACK_2026_07_06.npz
│   │   └── final_runs/
│   │       ├── forward/
│   │       │   ├── live_forward_validation_1785516534.csv
│   │       │   ├── live_forward_validation_1785516572.csv
│   │       │   └── live_forward_validation_1785516614.csv
│   │       ├── backward/
│   │       │   ├── live_forward_validation_1785518673.csv
│   │       │   ├── live_forward_validation_1785518702.csv
│   │       │   └── live_forward_validation_1785518730.csv
│   │       ├── left/
│   │       │   ├── live_forward_validation_1785517424.csv
│   │       │   ├── live_forward_validation_1785517457.csv
│   │       │   └── live_forward_validation_1785517481.csv
│   │       ├── right/
│   │       │   ├── live_forward_validation_1785517727.csv
│   │       │   ├── live_forward_validation_1785517751.csv
│   │       │   └── live_forward_validation_1785517771.csv
│   │       ├── diagonal/
│   │       │   ├── live_forward_validation_1785518575.csv
│   │       │   ├── live_forward_validation_1785519411.csv
│   │       │   ├── live_forward_validation_1785519870.csv
│   │       │   └── live_forward_validation_1785519929.csv
│   │       └── target_yaw/
│   │           └── newFilerandomYaw.csv
│   │
│   └── imitation/
│       ├── README.md
│       ├── hankel/
│       │   └── final_student_deepc_hankel_v2_stride2.npz
│       └── splits/
│           ├── train_indices.npz
│           ├── val_indices.npz
│           ├── test_indices.npz
│           ├── train_episode_ids.txt
│           ├── val_episode_ids.txt
│           └── test_episode_ids.txt
│
├── results/
│   ├── physical/
│   │   ├── README_EVIDENCE.md
│   │   └── go2_center_correction_report.json
│   └── genesis/
│       ├── final_champion_benchmark.csv
│       ├── final_genesis_champion_freeze.json
│       ├── final_dataset_report.md
│       ├── final_linear_student_bc_report.md
│       └── audits/
│           ├── candidate_decision.json
│           └── final_constraint_audit.json
│
├── docs/
│   ├── dataset_schema.md
│   ├── environment.md
│   ├── expert_contract_reference.md
│   ├── student_observation_contract.md
│   └── repo_inventory/
│       ├── inventory_summary.md
│       ├── file_classification.csv
│       ├── dependency_audit.md
│       ├── large_files.md
│       ├── proposed_release_tree.md
│       ├── final_migration_manifest.csv
│       ├── migration_report.md
│       └── final_release_tree.md
```

## Canonical artifact choices

- Genesis benchmark evidence: `results/genesis/final_champion_benchmark.csv` is the canonical per-case result. `final_genesis_champion_freeze.json` is retained separately because it defines the frozen protocol, hashes, runtime, and constraints. The duplicate benchmark JSON and Markdown outputs are omitted.
- Imitation dataset: `data/imitation/expert_imitation_dataset_v1.npz` is distributed as a GitHub Release asset and is not present in this working tree. The six split files are kept in normal Git because they are small and required by BC scripts. Raw episodes remain local or in a separate provenance archive.
- Final student: `models/go2_observable_student_bc_linear_v1.pt` is small enough for normal Git and is hash-verified.
- Genesis assets: `urdf/go2/urdf/go2.urdf`, its meshes `dae/base.dae`, `dae/calf.dae`, `dae/calf_mirror.dae`, `dae/foot.dae`, `dae/hip.dae`, `dae/thigh.dae`, and `dae/thigh_mirror.dae`, plus `urdf/plane/plane.urdf` and `urdf/plane/plane100.obj`, are obtained from the pinned Genesis 1.3.1 dependency and verified against the freeze/asset hashes; they are not blindly copied into this tree.

## Files that should not go to GitHub

### Keep local only

- Historical U4/U5/B2/B3/S0-S7/P branches and all candidate checkpoints.
- Obsolete physical controllers, the older 2-input center-correction script, and older mode-switched physical logs.
- Raw 120-episode files and the 999 KB raw episode manifest unless a separate provenance archive is published.
- Duplicate benchmark JSON/Markdown reports and internal UI integration audits.
- Generated telemetry, per-run JSONL logs, videos, plots, caches, virtual environments, rosbag databases, and archives.

### Distribute outside normal Git

- `expert_imitation_dataset_v1.npz` as one GitHub Release asset.
- `expert_model_500.pt` only if its upstream license permits redistribution; otherwise document the original source and require a separately licensed download. It is not present in this working tree.
- Genesis 1.3.1 and its Go2/plane URDF assets through the dependency/setup process.

### Normal Git

- Python source, configuration, contracts, small manifests, split files, final physical CSV evidence, the two physical Hankels, the Genesis Hankel, final benchmark CSV, freeze JSON, and the 333,509-byte final student checkpoint.

## Readiness

The approved files are migrated and the selected code paths import from the
clean repository. The canonical physical run manifest and metrics remain
pending transfer from the original Mac workspace; they were not reconstructed.
The 17 located CSVs remain unchanged evidence. Genesis runtime assets and the
expert checkpoint remain external dependencies.
