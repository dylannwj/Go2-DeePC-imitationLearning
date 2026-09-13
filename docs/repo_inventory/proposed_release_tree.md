# Proposed clean release tree

This is a destination proposal only. The target skeleton already exists, but no source or artifact listed below has been copied into it. Exact source-to-destination records, hashes, sizes, and classification decisions are in `file_classification.csv`.

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
│   │   ├── receding.py
│   │   ├── local_controller.py
│   │   ├── miqp.py
│   │   └── genesis_recovered_deepc.py
│   │
│   ├── physical/
│   │   ├── command_gate.py
│   │   ├── ros2_deepc_node.py
│   │   ├── miqp_solver_final.py
│   │   ├── online_receding_controller_final.py
│   │   └── final_yaw_ui.py
│   │
│   └── learned_execution/
│       ├── contracts.py
│       ├── linear_bc_common.py
│       ├── direct_student_runtime.py
│       ├── candidate_qp.py
│       ├── frozen_champion_controller.py
│       ├── genesis_guard.py
│       ├── genesis_deployment_scene.py
│       ├── genesis_window.py
│       ├── ui_state.py
│       ├── ui_runtime_bridge.py
│       └── final_deepc_ui.py
│
├── scripts/
│   ├── physical/
│   │   ├── build_vxyyaw_hankel.py
│   │   ├── correct_center_and_rebuild_hankel.py
│   │   └── collect_vxyyaw_dataset.py
│   │
│   ├── imitation_learning/
│   │   ├── collect_expert_dataset.py
│   │   ├── dataset_common.py
│   │   ├── validate_expert_dataset.py
│   │   ├── build_episode_splits.py
│   │   ├── run_expert_validation.py
│   │   ├── run_expert_compatibility_gate.py
│   │   ├── train_linear_bc.py
│   │   ├── preflight_linear.py
│   │   ├── evaluate_student_offline.py
│   │   ├── evaluate_student_genesis.py
│   │   ├── reload_determinism.py
│   │   ├── finalize_linear.py
│   │   └── rewrite_retention_report.py
│   │
│   └── genesis/
│       ├── evaluate_command_bc_genesis.py
│       ├── validate_direct_student_interface.py
│       ├── finalize_reports.py
│       ├── final_champion_benchmark.py
│       ├── run_ui_integration.py
│       ├── run_live_product.py
│       └── recovery_launcher.py
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
│       ├── episode_manifest.json
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
│   ├── go2_observable_student_bc_linear_v1.pt
│   ├── expert_model_500.pt
│   ├── expert_policy.onnx
│   └── expert_policy.onnx.data
│
├── data/
│   ├── physical/
│   │   ├── hankel/
│   │   │   ├── dog2_vxyyaw_hankel_Tini10_N20_BACKWARD_2026_07_06.npz
│   │   │   └── dog2_vxyyaw_hankel_Tini10_N20_BALANCED_YAW_BACK_2026_07_06.npz
│   │   └── final_runs/
│   │       ├── forward/
│   │       │   └── live_forward_validation_*.csv  (3 selected files)
│   │       ├── backward/
│   │       │   └── live_forward_validation_*.csv  (3 selected files)
│   │       ├── left/
│   │       │   └── live_forward_validation_*.csv  (3 selected files)
│   │       ├── right/
│   │       │   └── live_forward_validation_*.csv  (3 selected files)
│   │       ├── diagonal/
│   │       │   └── live_forward_validation_*.csv  (4 selected files)
│   │       └── target_yaw/
│   │           └── newFilerandomYaw.csv
│   │
│   └── imitation/
│       ├── expert_imitation_dataset_v1.npz
│       ├── hankel/
│       │   └── final_student_deepc_hankel_v2_stride2.npz
│       └── raw/
│           └── episode_*.npz  (120 selected files, optional large payload)
│
├── results/
│   ├── physical/
│   │   ├── README_EVIDENCE.md
│   │   └── go2_center_correction_report.json
│   │
│   └── genesis/
│       ├── final_champion_benchmark.csv
│       ├── final_champion_benchmark.json
│       ├── final_champion_benchmark.md
│       ├── final_genesis_champion_freeze.json
│       ├── final_dataset_report.md
│       ├── final_linear_student_bc_report.md
│       ├── Final_champion_Genesis_UI_integration.md
│       ├── Final_champion_Genesis_UI_integration_summary.json
│       └── audits/
│           ├── candidate_decision.json
│           ├── final_constraint_audit.json
│           ├── champion_constraint_audit.json
│           ├── integration_smoke_test.json
│           ├── ui_integration_decision.json
│           └── ui_vs_final_freeze_equivalence.json
│
├── docs/
│   ├── dataset_schema.md
│   ├── expert_contract_reference.md
│   ├── student_observation_contract.md
│   └── repo_inventory/
│       ├── inventory_summary.md
│       ├── file_classification.csv
│       ├── dependency_audit.md
│       ├── large_files.md
│       └── proposed_release_tree.md
│
└── tests/
    ├── test_geometry.py                 (to be selected from maintained tests)
    ├── test_hankel_schema.py            (to be selected from maintained tests)
    ├── test_student_contract.py         (to be added/selected after review)
    ├── test_frozen_controller_identity.py (to be added/selected after review)
    └── test_benchmark_report_schema.py  (to be added/selected after review)
```

## Boundary notes

- The tree intentionally has separate physical, imitation, and Genesis data paths. The 3,548-column and 900-column physical banks are not interchangeable with the 1,216-column Genesis student Hankel.
- The 17 physical CSV destinations normalize the source folder names `foward 50cm`, `back50cm`, `left 30cm`, `right30cm`, `diagonals`, and `random_yawHeading` into stable category names. The source filenames themselves are preserved.
- The raw 120-episode dataset is shown as optional because it is a roughly 249 MB aggregate payload. The canonical processed dataset is still required for exact BC reproduction.
- The expert checkpoint and ONNX artifacts are needed to reproduce dataset collection, but not to run the final learned student. They may be moved to a separate release asset if the thesis release is runtime-only.
- Genesis URDFs are not placed under `models/` in this proposal because the freeze found them inside the installed Genesis package. Add project-local copies only after licensing and hash verification are resolved.
- The current target `requirements.txt` is only a placeholder. It must eventually distinguish Python-only dependencies from ROS2, Gurobi, Genesis, display, and robot/mocap deployment requirements.
- `tests/` entries are proposals, not claims that those exact tests already exist. The inventory pass did not create or modify tests.
