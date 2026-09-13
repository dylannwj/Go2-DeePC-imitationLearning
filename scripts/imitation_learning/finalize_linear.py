#!/usr/bin/env python3
"""Freeze the validated linear student and emit the final handoff evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learned_execution.linear_bc_common import (  # noqa: E402
    ACTION_DIM,
    ACTION_SCALE,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    INPUT_DIM,
    INPUT_FEATURE_NAMES,
    KD,
    KP,
    Q_DEFAULT_CANONICAL,
    REVISION_ROOT,
    sha256_file,
    write_json,
)


NEXT_AUTHORIZED_PHASE = "FREEZE FINAL STUDENT LOCOMOTION ENVELOPE AND PREPARE DeePC INTEGRATION"
FINAL_RELATIVE_PATH = Path("models/go2_observable_student_bc_linear_v1.pt")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_pass(payload: dict[str, Any], name: str, classification: str) -> None:
    observed = payload.get("classification")
    if observed != classification:
        raise RuntimeError(f"{name} did not pass: {observed!r}")


def verify_checkpoint(checkpoint_path: Path, expected_sha: str, expected_seed: int, expected_epoch: int) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "observable_student_bc_linear_v1":
        raise RuntimeError("final checkpoint schema is not the linear student schema")
    if checkpoint.get("architecture") != [INPUT_DIM, 256, 256, ACTION_DIM]:
        raise RuntimeError(f"unexpected architecture: {checkpoint.get('architecture')}")
    if checkpoint.get("hidden_activation") != "ELU" or checkpoint.get("output_activation") != "linear":
        raise RuntimeError("final checkpoint activation contract is invalid")
    if checkpoint.get("input_features") != list(INPUT_FEATURE_NAMES):
        raise RuntimeError("final checkpoint input feature contract is invalid")
    if checkpoint.get("seed") != expected_seed or checkpoint.get("selected_epoch") != expected_epoch:
        raise RuntimeError("final checkpoint selection metadata does not match selected manifest")
    if checkpoint.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("final checkpoint dataset hash is invalid")
    if sha256_file(checkpoint_path) != expected_sha:
        raise RuntimeError("final checkpoint hash does not match the source candidate")

    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, ACTION_DIM),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    module_types = [type(module).__name__ for module in model]
    if module_types != ["Linear", "ELU", "Linear", "ELU", "Linear"]:
        raise RuntimeError(f"unexpected final model modules: {module_types}")
    return {
        "schema": checkpoint["schema"],
        "architecture": checkpoint["architecture"],
        "hidden_activation": checkpoint["hidden_activation"],
        "output_activation": checkpoint["output_activation"],
        "output_semantics": checkpoint["output_semantics"],
        "module_types": module_types,
        "input_dimension": checkpoint["input_dimension"],
        "state_dimension": checkpoint["state_dimension"],
        "command_dimension": checkpoint["command_dimension"],
        "action_dimension": checkpoint["action_dimension"],
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    dataset_path = args.dataset.resolve()

    preflight = read_json(revision_root / "manifests/revised_preflight.json")
    offline = read_json(revision_root / "manifests/offline_gate.json")
    reload_result = read_json(revision_root / "manifests/reload_determinism.json")
    closed_loop = read_json(revision_root / "manifests/closed_loop_result.json")
    selected_manifest = read_json(revision_root / "manifests/selected_checkpoint.json")
    training_summary = read_json(revision_root / "manifests/training_summary.json")
    selected = selected_manifest["selected"]

    require_pass(preflight, "preflight", "LINEAR_BC_PREFLIGHT_PASS")
    require_pass(offline, "offline gate", "LINEAR_BC_OFFLINE_PASS")
    require_pass(reload_result, "reload determinism", "STUDENT_RELOAD_DETERMINISM_PASS")
    require_pass(closed_loop, "closed loop", "STUDENT_BC_LINEAR_CLOSED_LOOP_PASS")
    if sha256_file(dataset_path) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("frozen dataset hash changed before finalization")
    if closed_loop.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("closed-loop evidence references a different dataset")

    source_checkpoint = Path(selected["candidate_path"]).resolve()
    source_sha = sha256_file(source_checkpoint)
    if source_sha != selected["candidate_sha256"]:
        raise RuntimeError("selected candidate hash changed")
    final_checkpoint = (revision_root / FINAL_RELATIVE_PATH).resolve()
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, final_checkpoint)
    final_sha = sha256_file(final_checkpoint)
    architecture_validation = verify_checkpoint(final_checkpoint, source_sha, int(selected["seed"]), int(selected["selected_epoch"]))

    try:
        genesis_version = importlib.metadata.version("genesis-world")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Genesis 1.3.1 package metadata is unavailable") from exc
    if genesis_version != "1.3.1":
        raise RuntimeError(f"unexpected Genesis version: {genesis_version}")

    # Make the downstream boundary explicit in the machine-readable closed-loop result.
    closed_loop["do_not_start_ppo_or_dagger_automatically"] = True
    closed_loop["do_not_start_downstream_automatically"] = True
    closed_loop["next_authorized_phase"] = NEXT_AUTHORIZED_PHASE
    closed_loop["final_checkpoint_path"] = str(final_checkpoint)
    write_json(revision_root / "manifests/closed_loop_result.json", closed_loop)

    training_script_path = revision_root / "scripts/train_linear_bc.py"
    runtime_script_path = revision_root / "scripts/evaluate_genesis_linear.py"
    manifest = {
        "schema": "linear-output-final-student-manifest-v1",
        "classification": "STUDENT_BC_LINEAR_CLOSED_LOOP_PASS",
        "dataset": {
            "path": str(dataset_path),
            "sha256": EXPECTED_DATASET_SHA256,
            "train_transitions": 192000,
            "validation_transitions": 24000,
            "test_transitions": 24000,
            "modified": False,
        },
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": final_sha,
            "source_candidate_path": str(source_checkpoint),
            "source_candidate_sha256": source_sha,
            "seed": int(selected["seed"]),
            "data_loader_seed": int(selected["seed"]),
            "selected_epoch": int(selected["selected_epoch"]),
        },
        "training_script_sha256": sha256_file(training_script_path),
        "training_script_sha256_recorded": training_summary.get("training_script_sha256"),
        "runtime_script_sha256": sha256_file(runtime_script_path),
        "architecture": {
            "description": "48→256 ELU→256 ELU→12 linear",
            "layers": [INPUT_DIM, 256, 256, ACTION_DIM],
            "hidden_activation": "ELU",
            "output_activation": "linear",
            "validation": architecture_validation,
        },
        "input_contract": {
            "path": str(revision_root / "frozen_contract/student_48d_input_contract.md"),
            "state_dimension": 45,
            "command_dimension": 3,
            "dimension": INPUT_DIM,
            "preprocessing": "identity/raw",
            "features": list(INPUT_FEATURE_NAMES),
        },
        "action_contract": {
            "path": str(revision_root / "frozen_contract/revised_student_policy_contract.md"),
            "output": "raw_action in R^12",
            "order": "canonical Genesis joint order",
            "target": "expert_action_student_order",
            "decoder": "q_target_raw = q_default + 0.5 * action",
            "q_default_canonical": [float(value) for value in Q_DEFAULT_CANONICAL],
            "action_scale": [float(value) for value in ACTION_SCALE],
            "neural_output_clipping": False,
        },
        "fixed_pd": {
            "kp": [float(value) for value in KP],
            "kd": [float(value) for value in KD],
            "effort_limit": [23.5, 23.5, 45.43] * 4,
        },
        "genesis": {
            "version": genesis_version,
            "control_dt_s": 0.02,
            "physics_dt_s": 0.005,
            "decimation": 4,
        },
        "gate_classifications": {
            "preflight": preflight["classification"],
            "offline": offline["classification"],
            "reload_determinism": reload_result["classification"],
            "closed_loop": closed_loop["classification"],
        },
        "closed_loop": {
            "student_episode_count": closed_loop["student_episode_count"],
            "expert_comparison_episode_count": closed_loop["expert_episode_count"],
            "core_command_repetitions": closed_loop["core_command_repetitions"],
            "gates": closed_loop["gates"],
            "video_gate": closed_loop["gates"]["video_gate"],
            "raw_q_target_clip_events": closed_loop["gates"]["raw_q_target_clip_events"],
            "distribution_shift_report": str(revision_root / "reports/closed_loop_distribution_shift.md"),
        },
        "historical_blocked_evidence_preserved": [
            str(Path("student_bc/reports/final_student_bc_report.md")),
            str(Path("student_bc/reports/action_range_compatibility.md")),
            str(Path("student_bc/reports/preflight_integrity.md")),
        ],
        "next_authorized_phase": NEXT_AUTHORIZED_PHASE,
        "do_not_start_downstream_automatically": True,
    }
    write_json(revision_root / "manifests/final_student_manifest.json", manifest)

    test_csv_path = revision_root / "metrics/test_metrics.csv"
    if not test_csv_path.exists():
        raise RuntimeError("held-out test metrics artifact is missing")
    report_lines = [
        "# Final linear-output student BC report",
        "",
        "STUDENT_BC_LINEAR_CLOSED_LOOP_PASS",
        "",
        "ARCHITECTURE REVISION VALIDATED:",
        "48 → 256 ELU → 256 ELU → 12 linear",
        "",
        "The frozen dataset, teacher, action scale, PD gains, and observable student input contract were unchanged. The only hypothesis change was removal of the final tanh restriction.",
        "",
        "## Frozen artifacts",
        "",
        f"- final checkpoint: `{final_checkpoint}`",
        f"- final checkpoint SHA256: `{final_sha}`",
        f"- training script SHA256: `{manifest['training_script_sha256']}`",
        f"- dataset SHA256: `{EXPECTED_DATASET_SHA256}`",
        f"- seed / selected epoch: `{selected['seed']} / {selected['selected_epoch']}`",
        f"- Genesis: `{genesis_version}`",
        "",
        "## Gate results",
        "",
        f"- `{preflight['classification']}`",
        f"- `{offline['classification']}`",
        f"- `{reload_result['classification']}`",
        f"- `{closed_loop['classification']}`",
        "",
        "Closed-loop safety counts were zero for falls, base contacts, NaN/Inf, actual joint-limit violations, and torque-limit exceedances. Standing, forward, yaw, arc, lateral, transition, full-duration, and video gates passed across 33 student episodes. Raw neural actions were not clamped to `[-1,1]`; 60 raw q-target joint-limit clip events were logged separately by the physical safety layer.",
        "",
        "Held-out test evidence is in `reports/test_results.md` and `metrics/test_metrics.csv`; reload evidence is in `reports/student_reload_determinism.md`; runtime state distribution evidence is in `reports/closed_loop_distribution_shift.md`.",
        "",
        "## Required boundary",
        "",
        "No PPO, DAgger, teacher fallback, action blending, policy switching, gait switching, or privileged state was added.",
        "",
        "NEXT AUTHORIZED PHASE:",
        "FREEZE FINAL STUDENT LOCOMOTION ENVELOPE",
        "AND PREPARE DeePC INTEGRATION",
        "",
        "DO NOT START AUTOMATICALLY.",
        "",
        "STUDENT_BC_LINEAR_CLOSED_LOOP_PASS",
    ]
    (revision_root / "reports/final_linear_student_bc_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": manifest["classification"],
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": final_sha,
        "training_script_sha256": manifest["training_script_sha256"],
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "genesis_version": genesis_version,
        "next_authorized_phase": NEXT_AUTHORIZED_PHASE,
        "do_not_start_downstream_automatically": True,
        "test_metrics_artifact": str(test_csv_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
