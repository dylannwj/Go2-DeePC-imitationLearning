# Thesis release migration report

## Result

The approved manifest migration is complete in the clean target repository.
The original source tree was read-only throughout this task.

- Approved manifest entries: 129
- `COPY` entries: 93; 68 remain byte-identical and 25 received only the
  portability edits listed below
- `REGENERATE` entries: 1 (`src/deepc/__init__.py`)
- Migrated manifest files: 94
- Manifest category totals: CORE 34, EVIDENCE 22, REPRODUCIBILITY 38
- Source files modified: 0
- Target Git repository initialized: no

## Portability diff summary

The following migrated files were modified. Every change is limited to a
moved-file import, repository-relative path, or required local cache/output
location; controller equations, DeePC parameters, safety logic, Genesis
behavior, and BC network behavior were not changed.

| Migrated file | Exact reason |
|---|---|
| `src/deepc/genesis_recovered_deepc.py` | Resolve the repository root after relocation and import retained `deepc` modules from the clean `src` tree. |
| `src/physical/online_receding_controller_final.py` | Replace the old absolute physical root with clean-repository paths for the solver, balanced Hankel, center report, and local runtime outputs. |
| `src/learned_execution/contracts.py` | Remove the import of the omitted historical mapping package and retain its frozen canonical joint-name tuple locally. |
| `src/learned_execution/direct_student_runtime.py` | Resolve the checkpoint and moved `learned_execution`/DeePC imports from the clean repository; use a local temporary cache directory. |
| `src/learned_execution/candidate_qp.py` | Replace moved sibling-tree paths and old DeePC/UI imports with clean-repository paths and package names. |
| `src/learned_execution/frozen_champion_controller.py` | Replace moved artifact/source paths with clean-repository paths and restore the target-relative validation-decision path constant without altering audit logic. |
| `src/learned_execution/genesis_window.py` | Resolve moved scene imports and use a local temporary cache directory. |
| `src/learned_execution/linear_bc_common.py` | Point the relocated BC helpers to the clean repository, external-release dataset location, and retained split files. |
| `src/learned_execution/genesis_deployment_scene.py` | Import the retained imitation helpers from their new package paths and relocate default output/normalization/checkpoint paths. |
| `scripts/genesis/run_live_product.py` | Resolve the clean repository and `src` roots, update moved runtime imports, and use clean config/checkpoint/Hankel/output paths. |
| `scripts/genesis/validate_direct_student_interface.py` | Resolve the clean repository and import the relocated direct student runtime. |
| `scripts/genesis/final_champion_benchmark.py` | Resolve clean artifact paths, moved runtime imports, and a local temporary cache; benchmark equations and target protocol are unchanged. |
| `scripts/imitation_learning/dataset_common.py` | Point dataset storage at `data/imitation`. |
| `scripts/imitation_learning/build_episode_splits.py` | Import dataset helpers from their relocated package. |
| `scripts/imitation_learning/collect_expert_dataset.py` | Update relocated contract/runtime imports and point expert, external, compatibility, and report paths at the clean layout. |
| `scripts/imitation_learning/validate_expert_dataset.py` | Update relocated contract/runtime imports, clean compatibility paths, and external expert checkpoint path. |
| `scripts/imitation_learning/run_expert_validation.py` | Correct the moved script root/workspace and use clean configs, external assets, and relocated helper imports. |
| `scripts/imitation_learning/run_expert_compatibility_gate.py` | Correct the moved script root/output workspace and use clean configs, external assets, and relocated helper imports. |
| `scripts/imitation_learning/collect_command_teacher_rollouts.py` | Correct the moved script root, clean dataset/checkpoint/config defaults, and generated report location. |
| `scripts/imitation_learning/train_command_bc.py` | Correct the moved script root, clean dataset/checkpoint/normalization/output defaults, and teacher-helper import. |
| `scripts/imitation_learning/train_linear_bc.py` | Import the relocated linear BC common module through `src` and add clean-root import setup. |
| `scripts/imitation_learning/preflight_linear.py` | Import the relocated linear BC common module, add clean-root import setup, and correct the revision-root default. |
| `scripts/imitation_learning/evaluate_student_offline.py` | Import the relocated linear BC common module and add clean-root import setup. |
| `scripts/imitation_learning/reload_determinism.py` | Import the relocated linear BC common module and add clean-root import setup. |
| `scripts/imitation_learning/finalize_linear.py` | Import the relocated linear BC common module, add clean-root import setup, and point the final checkpoint at `models/`. |

`src/deepc/__init__.py` was regenerated as required by the manifest so the
clean package exports only the retained geometry, Hankel, and QP symbols.
The temporary `src/learned_execution/release_paths.py` experiment was removed;
cache paths are now simple local expressions in the two Genesis scene/runtime
files and benchmark harness.

## Verification

- Python syntax parse: `41/41` files passed.
- Selected local imports: passed for DeePC, physical-independent modules,
  Genesis wrappers, benchmark, BC, and imitation scripts.
- Final student checkpoint load: passed; SHA-256 is
  `6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc` and
  architecture is `[48, 256, 256, 12]`.
- Physical raw CSVs: `17`.
- Physical Hankel banks: `2`; the balanced bank has `900` columns and the
  backward bank has `3548` columns.
- Genesis final benchmark: `43` cases, `24` reached, `3` solver failures,
  and `16` hard-safety-event episodes.
- Installed Genesis asset files were located and their Go2/plane URDF hashes
  matched the frozen values.
- Full Genesis import/runtime smoke is pending the frozen Python 3.12.13
  environment; the available `/usr/bin/python3` is 3.10.12 with older
  NumPy/Torch/CVXPY versions.
- Runtime Python scan for the legacy host prefixes: zero occurrences in
  `src/` and `scripts/`.
- `GO2PROJECT_FILES_MODIFIED = 0`.

The full clean-repository scan still finds 111 provenance/documentation
occurrences. They are not runtime Python literals. Exact locations are:

- `configs/genesis/candidate_config.json:99` (1)
- `configs/genesis/frozen_runtime_contract.json:250,252,260,288,335` (5)
- `configs/imitation/closed_loop_result.json:9,21,49,56,63,70,77,84,91,98,105` (11)
- `configs/imitation/expert_artifact_manifest.json:5,14,23,32,41,50,157,159,160,227,233` (11)
- `configs/imitation/final_student_manifest.json:21,74,78,84,111,232` (6)
- `configs/imitation/selected_checkpoint.json:4,28,52,78` (4)
- `docs/repo_inventory/dependency_audit.md:3,99,103` (3)
- `docs/repo_inventory/inventory_summary.md:4,5,110` (3)
- `docs/repo_inventory/migration_report.md:6,7,72` (3)
- `results/genesis/final_champion_benchmark.csv:2–44` (43 per-run log paths)
- `results/genesis/final_genesis_champion_freeze.json:85,89,93,97,101,105,109,113,151,158,161,248,252,256,260,264,268,272,276,280,284,288,292` (23)
- `results/genesis/final_linear_student_bc_report.md:12` (1)

The JSON/CSV/Markdown paths are retained provenance from the approved
evidence and are intentionally not rewritten. No runtime source file contains
a legacy host path.

## Intentionally external or pending inputs

- `data/imitation/expert_imitation_dataset_v1.npz` remains a GitHub Release
  asset because it is 246,636,478 bytes; its required SHA-256 is recorded in
  `data/imitation/README.md`.
- `models/expert_model_500.pt` remains separately distributed and license
  dependent.
- Genesis Go2/plane URDFs and meshes are supplied by Genesis 1.3.1 and are
  not vendored.
- `data/physical/PENDING_FILES.md` records that the canonical physical
  `run_manifest.csv` and `run_metrics.csv` are pending transfer from the Mac
  workspace; neither was reconstructed.
- The optional `frozen_artifact_audit()` still names the target-relative
  `results/genesis/audits/final_validation_decision.json`, which was not an
  approved manifest entry and is therefore absent. The selected benchmark and
  live runtime do not call that optional audit function.

## Handoff

The clean repository is ready for manual Git review. Do not initialize Git or
claim the pending physical manifest/metrics until the Mac transfer is made.
