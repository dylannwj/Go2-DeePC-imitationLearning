# Dependency and path audit

Audit scope: files retained in `final_migration_manifest.csv`, plus the omitted candidates needed to explain review decisions. The source tree was inspected statically; no controller, ROS2 node, Genesis scene, solver, training script, or test was executed. Paths below are relative to `/home/dylan/Desktop/Go2Project` unless stated otherwise.

## High-level dependency graph

```text
physical final-yaw UI
  └─ final online receding controller
       ├─ final MIQP solver subprocess
       ├─ physical Hankel bank (balanced yaw bank at runtime)
       ├─ ROS2 + mocap topics
       └─ Gurobi / gurobipy

expert checkpoint + Genesis/MJLab compatibility code
  └─ 120-episode collector
       └─ canonical 240000-transition dataset + episode split
            └─ linear BC scripts
                 └─ final 48→256→256→12 student checkpoint

Genesis benchmark harness
  └─ frozen_controller adapter
       ├─ candidate QP with C2 + vx constraints
       ├─ recovered DeePC controller + Genesis Hankel
       ├─ direct student runtime + final student checkpoint
       └─ Genesis deployment scene / URDF / CPU runtime
```

## Runtime dependency matrix

| Chain | Local imports and data | External/runtime requirements | Migration risk |
|---|---|---|---|
| Maintained DeePC library | `src/deepc_go2/{geometry,hankel,qp,receding,local_controller,miqp}.py` | Python, NumPy, and the solver stack used by the selected caller | The target package is named `src/deepc`; current imports use `deepc_go2`. Preserve a package alias or rewrite every import and test. |
| Physical MIQP solver | `deadzone_miqp_deepc_preview_solver_3INPUT_DOG2.py`; Hankel passed by CLI | NumPy, `gurobipy`, Gurobi license/runtime | The source accepts a Hankel path and output paths, but the online wrapper invokes it as a subprocess. Keep the solver-side alphabet and result schema unchanged. |
| Physical online controller | `miqp_online_receding_deepc_controller_FINAL_3INPUT_YAW.py` | ROS2 `rclpy`, `geometry_msgs`, `std_msgs`, mocap pose stream, subprocess | It hardcodes the `deadzoneQuantized` root, uses `/mocap/go2_pose`, and expects the center report and balanced Hankel under that root. Convert to config/CLI paths. |
| Physical UI | `deepc_ui_node_FINAL_YAW_UI.py` | ROS2, Tkinter, display, mocap | Final-yaw target interaction is coupled to ROS topics and subprocess status. Do not merge with the Genesis UI. |
| Final DeePC | `target_suite_recovery/controllers/final_receding_deepc.py` plus `src/deepc_go2` | NumPy, package-local geometry/Hankel/QP | It computes paths from `Path(__file__).resolve().parents[3]` and inserts `Go2Project/src` into `sys.path`; this breaks after flattening. |
| Champion safety adapter | `final_champion_genesis_ui/frozen_controller.py`, candidate QP, candidate/base configs, constraint audit | NumPy, CVXPY, OSQP/Clarabel as configured | It dynamically loads a source file and compares hashes. The exact C2 source line and no-post-solver-rewrite behavior are release invariants. |
| Genesis direct student | `direct_student_runtime.py`, `linear_bc_common.py`, final `.pt` | PyTorch, NumPy, Genesis | It expects `student_bc/linear_output_revision` as a sibling and uses a `/tmp/final_deepc_student` cache. |
| Genesis deployment scene | `scripts/evaluate_command_bc_genesis.py` and `final_deepc_student/live_ui/app/genesis_window.py` | Genesis 1.3.1, Python 3.12.13, Torch 2.10.0+cpu, NumPy 2.4.4, CVXPY 1.8.2, OSQP 1.1.1, Quadrants, Genesis assets | The evaluator imports `scripts.collect_command_teacher_rollouts` and `scripts.train_command_bc`, even though the final product uses only selected scene/model helpers. Those imports must be isolated or carried into the release. |
| Genesis live UI | `final_deepc_ui.py`, `ui_state.py`, `ui_runtime_bridge.py`, `run_live_product.py` | Tkinter/display plus the Genesis runtime | The launcher inserts multiple source roots into `sys.path`, including `src`, `live_ui`, recovery controllers, and safety-refinement roots. A package/entry-point migration is required. |
| Expert dataset collector | `expert_imitation_dataset/scripts/collect_expert_dataset.py`, `dataset_common.py`, compatibility gate, visual-transfer validator, `src/deepc_go2/genesis_walking/contracts.py` | Genesis/MJLab, Torch/NumPy, installed robot assets | Collector and validator assume `expert_policy_validation` and the `deepc_go2` namespace are present. The external expert asset package must be pinned or documented. |
| Student BC training | `student_bc/linear_output_revision/scripts/*.py`, canonical processed dataset, split files, linear common module | Python, NumPy, PyTorch | Scripts use sibling paths, mutate output directories during finalization, and `evaluate_genesis_linear.py` imports root `scripts` and compatibility modules. |
| Runtime environment | Freeze manifest and recovery launcher | Genesis 1.3.1, Python 3.12.13, CPU Torch, OSQP/CVXPY, Quadrants | The recovery venv is not a portable release dependency. Recreate it from a pinned requirements/environment manifest and re-hash the runtime assets. |

## Local module and data references

### Physical branch

- The final MIQP source imports `numpy` and `gurobipy`; its Hankel loader expects a `.npz` with `Up`, `Uf`, `Yp`, `Yf` and related metadata.
- The final online node imports ROS2 message types and launches the solver with `subprocess.run`. It reads the balanced yaw/back Hankel and center-correction report, and writes solver JSON/CSV outputs.
- The physical UI separately imports Tkinter and ROS2 and subscribes to `/mocap/go2_pose`. No UDP relay module is imported by the exact selected final-yaw source. The hardware relay helpers can therefore remain excluded unless the deployment procedure explicitly requires them.
- The physical MIQP source uses the four-valued `vx` alphabet, three-valued `vy` alphabet, and five-valued `yaw_rate` alphabet recorded in `inventory_summary.md`. A generic continuous `src/deepc_go2` QP must not silently replace this discrete solver.

### Imitation-learning branch

- The collector resolves its project root two levels above the script and imports `deepc_go2.genesis_walking.contracts`.
- It imports `configure_caches` from `expert_policy_validation.mjlab_expert_compatibility_gate.scripts.run_expert_compatibility_gate` and `make_teacher_observation` from `expert_policy_validation.mjlab_genesis_visual_transfer.scripts.run_validation`.
- It records expert artifacts from `expert_policy_validation/mjlab_genesis_visual_transfer/{artifacts,configs}` and Genesis URDF locations from the installed `genesis` package.
- The validator also expects compatibility reports under `expert_policy_validation/mjlab_expert_compatibility_gate/reports` and the same student-contract module.
- `linear_bc_common.py` is the central local dependency for training, preflight, offline evaluation, reload determinism, and finalization. It hardcodes the canonical dataset relative to the Go2Project root and verifies dataset SHA-256 and row counts.
- The final student input is 45 state features plus 3 command features. Its output is already a physical 12-D joint-position target. The Genesis evaluator must not apply the expert action scale a second time.

### Genesis champion branch

- `final_champion_benchmark.py` dynamically resolves the candidate QP, recovered DeePC source, frozen adapter, evaluator, UI scene, and linear student source from their current Go2Project locations.
- `frozen_controller.py` dynamically loads the candidate QP and recovered DeePC source and checks the hashes of the candidate QP, base configuration, candidate configuration, candidate decision, Genesis Hankel, and student checkpoint.
- The candidate QP implements `future_input[1::3] + future_input[2::3] <= float(limit)` and separate `vx` bounds inside the optimization problem. The freeze explicitly requires `post_solver_command_rewriting=false`.
- The Genesis scene uses `urdf/go2/urdf/go2.urdf` and `urdf/plane/plane.urdf` relative to installed Genesis assets. The project does not contain authoritative project-local copies of those URDFs.

## Post-review dependency closure

The first inventory contained 107 proposed items. The final review retained 84 of those items and added 22 concrete closure items. The added items are not historical conveniences:

- `final_deepc_student/live_ui/app/ui_canvas.py` is imported directly by `final_deepc_ui.py`.
- The three package initializers for the live UI, recovered controllers, and champion adapter preserve the package boundaries used by the current import paths.
- `scripts/collect_command_teacher_rollouts.py` and `scripts/train_command_bc.py` are imported at module load by `scripts/evaluate_command_bc_genesis.py`; they are retained as a temporary helper closure and should be narrowed during migration.
- `train_indices.npz`, `val_indices.npz`, `test_indices.npz`, and the three `*_episode_ids.txt` files are read directly by `linear_bc_common.py` and `preflight_linear.py`.
- The Go2 and plane URDF entries plus the seven Go2 DAE meshes and plane OBJ are external Genesis 1.3.1 assets, not project-local source files.

The broad current `src/deepc_go2/__init__.py` is not copied verbatim. It imports `safe_command_set`, `signed_deepc`, and `u6_deepc`, which are outside the selected final chain. The final manifest therefore marks that one file `REGENERATE`: the migration must create a minimal init exporting only retained geometry, Hankel, and QP symbols. This removes the only package-initializer false dependency without pulling historical safe-set families into the release.

### Closure result

| Branch | Local dependencies checked | Result |
|---|---|---|
| Physical | Final MIQP solver, online ROS2 node, final-yaw UI, Hankel banks, center report, collection/build recipes | PASS for selected local files; ROS2, Gurobi, mocap, and the missing final run manifest/metrics remain external or evidence conditions. |
| Imitation learning | Collector, dataset helpers/validator/split builder, expert validation and compatibility helpers, BC scripts, final checkpoint, six split files | PASS; all local imports are represented in the final manifest after adding the split artifacts. |
| Genesis | Benchmark, live product, UI modules including `ui_canvas`, evaluator, root helper closure, frozen adapter, candidate QP, recovered DeePC, student runtime, configs, Hankel, checkpoint | PASS; package/path rewrites are required, but no local dependency is left unrepresented. |

`DEPENDENCY_BLOCKERS=0` means no missing local file remains after these additions. It does not mean the external ROS2, Gurobi, Genesis, PyTorch, display, or mocap environment is already installed.

## Absolute paths and generated path references

### Count used in the terminal handoff

`HARDCODED_PATHS_FOUND=3` means three literal absolute-path sites in the selected runtime Python sources:

1. `evidence_miqp_3input_final_yaw_2026_07_06_180804/source_files/miqp_online_receding_deepc_controller_FINAL_3INPUT_YAW.py:38` — `/home/dylan/Desktop/Go2Project/deadzoneQuantized`.
2. `final_deepc_student/live_ui/app/genesis_window.py:55` — `/tmp/final_deepc_student` cache root.
3. `final_deepc_student/target_suite_recovery/controllers/direct_student_runtime.py:270` — `/tmp/final_deepc_student` cache root.

This count intentionally excludes generated JSON/Markdown provenance. The final Genesis JSON reports and UI integration manifest contain many absolute `/home/dylan/Desktop/Go2Project/...` paths, including per-run log files, installed URDFs, and the recovery Python executable. Normalize these paths before placing the reports in a portable release. The source code also constructs absolute paths indirectly from `Path(__file__)`, `sys.path`, and sibling-directory assumptions; those are migration risks even when no literal `/home/dylan` string appears.

## Asset, version, and licensing assumptions

- Genesis freeze: `1.3.1`.
- Python: `3.12.13`.
- NumPy: `2.4.4`; PyTorch: `2.10.0+cpu`; OSQP: `1.1.1`; CVXPY: `1.8.2`; Quadrants `(1, 2, 0)`.
- Physics timestep: `0.005 s`; control timestep: `0.02 s`; four physics steps per policy action; collision enabled; viewer disabled for the benchmark.
- The Go2 and plane URDF SHA-256 values are recorded in the Genesis freeze, but the files are in an installed site-package/virtual environment. The Go2 URDF references `dae/base.dae`, `dae/calf.dae`, `dae/calf_mirror.dae`, `dae/foot.dae`, `dae/hip.dae`, `dae/thigh.dae`, and `dae/thigh_mirror.dae`; the plane URDF references `plane100.obj`. These meshes are also external Genesis assets. The reviewed plan depends on Genesis-provided assets or a separately documented asset download rather than blindly vendoring the directory.
- Physical execution requires ROS2 and a mocap publisher. The exact physical solver requires a functioning Gurobi installation/license. These are deployment requirements, not Python-only `requirements.txt` entries.

## Required migration checks

1. Decide whether the target package remains importable as `deepc_go2` or becomes `deepc`; update all local imports, contract links, dynamic loaders, and tests consistently.
2. Replace physical hardcoded roots with an explicit config/CLI path and verify the balanced Hankel, center report, solver executable, and output directory.
3. Collapse the Genesis sibling-tree imports into a single package or an explicit `PYTHONPATH`-free entry point.
4. Preserve the two physical Hankels and the separate Genesis Hankel under distinct destination paths.
5. Recreate the final runtime environment from pinned dependencies; verify Genesis URDF hashes and the student/Hankel hashes.
6. Run static import checks, physical solver schema checks, student checkpoint reload checks, and the Genesis benchmark only after the release files are assembled.
7. Keep final evidence immutable; do not regenerate or overwrite the selected 17 physical logs or the 43-case Genesis report during packaging.
