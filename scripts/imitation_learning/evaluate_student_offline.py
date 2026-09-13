#!/usr/bin/env python3
"""Select a linear BC checkpoint on validation and perform one final test pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learned_execution.linear_bc_common import (  # noqa: E402
    ACTION_DIM,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    JOINT_NAMES,
    REVISION_ROOT,
    action_distribution,
    behavior_metrics,
    load_dataset,
    load_split_indices,
    make_model,
    regression_metrics,
    sha256_file,
    transition_masks,
    write_csv,
    write_json,
)


def predict(checkpoint_path: Path, inputs: np.ndarray) -> np.ndarray:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(torch)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(inputs), 8192):
            rows.append(model(torch.from_numpy(inputs[start : start + 8192])).numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def scalar_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {key: value for key, value in regression_metrics(prediction, target).items() if key != "per_joint"}


def global_range(actions: np.ndarray) -> dict[str, Any]:
    value = np.asarray(actions, dtype=np.float64)
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "p1": float(np.percentile(value, 1.0)),
        "p99": float(np.percentile(value, 99.0)),
        "fraction_outside_minus1_plus1": float(np.mean((value < -1.0) | (value > 1.0))),
    }


def best_candidate(summary: dict[str, Any]) -> dict[str, Any]:
    candidates = sorted(summary["results"], key=lambda row: float(row["validation"]["mse"]))
    return candidates[0]


def candidate_validation_comparison(
    data: dict[str, np.ndarray],
    results: list[dict[str, Any]],
    val_rows: np.ndarray,
    val_transition: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        prediction = predict(Path(result["candidate_path"]), data["student_input"][val_rows].astype(np.float32))
        target = data["expert_action_student_order"][val_rows]
        metrics = regression_metrics(prediction, target)
        behavior = behavior_metrics(prediction, target, data["desired_command"][val_rows], val_transition)
        expert_range = global_range(target)
        student_range = global_range(prediction)
        rows.append(
            {
                "seed": int(result["seed"]),
                "selected_epoch": int(result["selected_epoch"]),
                "candidate_path": result["candidate_path"],
                "candidate_sha256": result["candidate_sha256"],
                "validation_mse": metrics["mse"],
                "validation_mae": metrics["mae"],
                "validation_mean_action_l2": metrics["mean_action_l2"],
                "validation_median_action_l2": metrics["median_action_l2"],
                "validation_p95_action_l2": metrics["p95_action_l2"],
                "validation_transition_mse": behavior["transitions"]["mse"],
                "validation_steady_state_mse": behavior["steady_state"]["mse"],
                "validation_worst_named_behavior_mse": max(
                    float(behavior[name]["mse"])
                    for name in ("standing", "forward", "backward", "lateral_left", "lateral_right", "yaw_left", "yaw_right", "forward_yaw", "mixed")
                ),
                "expert_min": expert_range["min"],
                "expert_max": expert_range["max"],
                "expert_p1": expert_range["p1"],
                "expert_p99": expert_range["p99"],
                "expert_fraction_outside": expert_range["fraction_outside_minus1_plus1"],
                "student_min": student_range["min"],
                "student_max": student_range["max"],
                "student_p1": student_range["p1"],
                "student_p99": student_range["p99"],
                "student_fraction_outside": student_range["fraction_outside_minus1_plus1"],
            }
        )
    return rows


def markdown_behavior(metrics: dict[str, dict[str, Any]], title: str) -> str:
    names = ("standing", "forward", "backward", "lateral_left", "lateral_right", "yaw_left", "yaw_right", "forward_yaw", "mixed", "steady_state", "transitions")
    lines = [
        f"# {title}",
        "",
        "Metrics are computed on raw canonical Genesis-order action targets.",
        "",
        "| Behavior | Samples | MSE | MAE | Mean L2 | Median L2 | p95 L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = metrics[name]
        lines.append(
            f"| `{name}` | {row['samples']} | {row['mse']:.9g} | {row['mae']:.9g} | "
            f"{row['mean_action_l2']:.9g} | {row['median_action_l2']:.9g} | {row['p95_action_l2']:.9g} |"
        )
    return "\n".join(lines)


def markdown_selection(rows: list[dict[str, Any]], selected: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint selection",
        "",
        "Selection was made using validation data only. The test split was not loaded while comparing candidate seeds.",
        "",
        "| Seed | Epoch | Validation MSE | Validation MAE | Mean L2 | Transition MSE | Steady MSE | Candidate SHA256 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        marker = " **SELECTED**" if int(row["seed"]) == int(selected["seed"]) else ""
        lines.append(
            f"| {row['seed']}{marker} | {row['selected_epoch']} | {row['validation_mse']:.9g} | "
            f"{row['validation_mae']:.9g} | {row['validation_mean_action_l2']:.9g} | "
            f"{row['validation_transition_mse']:.9g} | {row['validation_steady_state_mse']:.9g} | `{row['candidate_sha256']}` |"
        )
    lines += [
        "",
        f"Selected seed: `{selected['seed']}`",
        f"Selected epoch: `{selected['selected_epoch']}`",
        f"Selected candidate: `{selected['candidate_path']}`",
        f"Selected checkpoint SHA256: `{selected['candidate_sha256']}`",
        "",
        "The candidate is frozen before the one-time held-out test evaluation.",
    ]
    return "\n".join(lines)


def write_test_csv(path: Path, overall: dict[str, Any], per_joint: list[dict[str, Any]], behaviors: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for metric_name in ("mse", "mae", "mean_action_l2", "median_action_l2", "p95_action_l2", "p99_action_l2", "max_action_l2"):
        rows.append({"row_type": "overall", "behavior": "all", "joint": "all", "metric": metric_name, "value": overall[metric_name], "samples": overall["samples"]})
    for row in per_joint:
        for metric_name in ("mse", "mae", "pearson"):
            rows.append({"row_type": "per_joint", "behavior": "all", "joint": row["joint"], "metric": metric_name, "value": row[metric_name], "samples": overall["samples"]})
    for behavior, metrics in behaviors.items():
        for metric_name in ("mse", "mae", "mean_action_l2", "median_action_l2", "p95_action_l2", "p99_action_l2", "max_action_l2"):
            rows.append({"row_type": "behavior", "behavior": behavior, "joint": "all", "metric": metric_name, "value": metrics[metric_name], "samples": metrics["samples"]})
    write_csv(path, rows)


def offline_gate(
    *,
    training_summary: dict[str, Any],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    val_behavior: dict[str, dict[str, Any]],
    test_behavior: dict[str, dict[str, Any]],
    prediction: np.ndarray,
    target: np.ndarray,
    transition: np.ndarray,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    all_finite = bool(np.isfinite(prediction).all()) and all(
        result["training_safety"]["finite"] and result["training_safety"]["nan_inf_batches"] == 0
        for result in training_summary["results"]
    )
    check("finite outputs and training numerics", all_finite, "all selected/test predictions finite; all seeds had zero NaN/Inf batches")
    prediction_std = np.std(prediction, axis=0)
    no_collapse = bool(np.all(prediction_std > 1.0e-3))
    check("no collapsed output dimensions", no_collapse, f"minimum test prediction std={float(np.min(prediction_std)):.9g}")

    val_mse = float(val_metrics["mse"])
    test_mse = float(test_metrics["mse"])
    train_mse = float(train_metrics["mse"])
    alignment_ratio = test_mse / max(val_mse, 1.0e-12)
    train_val_ratio = val_mse / max(train_mse, 1.0e-12)
    aligned = bool(alignment_ratio <= 1.75 and train_val_ratio <= 8.0)
    check("train/validation/test errors reasonably aligned", aligned, f"train={train_mse:.9g}; val={val_mse:.9g}; test={test_mse:.9g}; test/val={alignment_ratio:.4g}")

    pearsons = np.asarray([row["pearson"] for row in test_metrics["per_joint"]], dtype=float)
    target_variance = np.var(target, axis=0)
    normalized_mse = np.asarray([row["mse"] for row in test_metrics["per_joint"]], dtype=float) / np.maximum(target_variance, 1.0e-12)
    strong = bool(np.nanmean(pearsons) >= 0.90 and np.nanmin(pearsons) >= 0.70 and np.nanmax(normalized_mse) <= 0.20)
    check("strong expert/student action agreement", strong, f"mean_pearson={float(np.nanmean(pearsons)):.6f}; min_pearson={float(np.nanmin(pearsons)):.6f}; max_normalized_mse={float(np.nanmax(normalized_mse)):.6f}")

    named = ("standing", "forward", "backward", "lateral_left", "lateral_right", "yaw_left", "yaw_right", "forward_yaw", "mixed")
    behavior_mses = {name: float(test_behavior[name]["mse"]) for name in named if test_behavior[name]["samples"]}
    worst_behavior_ratio = max(behavior_mses.values()) / max(test_mse, 1.0e-12) if behavior_mses else float("inf")
    no_catastrophic_class = bool(behavior_mses and worst_behavior_ratio <= 4.0)
    check("no catastrophic command-class failure", no_catastrophic_class, f"worst_behavior_mse_ratio={worst_behavior_ratio:.6f}; behavior_mse={behavior_mses}")

    transition_ratio = float(test_behavior["transitions"]["mse"]) / max(float(test_behavior["steady_state"]["mse"]), 1.0e-12)
    transitions_ok = bool(test_behavior["transitions"]["samples"] > 0 and transition_ratio <= 2.5)
    check("transition performance acceptable", transitions_ok, f"transition_mse={test_behavior['transitions']['mse']:.9g}; steady_state_mse={test_behavior['steady_state']['mse']:.9g}; ratio={transition_ratio:.6f}")

    expert_range = global_range(target)
    student_range = global_range(prediction)
    output_fidelity = bool(student_range["max"] > 1.0 and student_range["min"] < -1.0 and student_range["fraction_outside_minus1_plus1"] > 0.01)
    check("output range is not artificially tanh-bounded", output_fidelity, f"expert=[{expert_range['min']:.6g},{expert_range['max']:.6g}]; student=[{student_range['min']:.6g},{student_range['max']:.6g}]; student_fraction_outside={student_range['fraction_outside_minus1_plus1']:.6f}")

    return {
        "schema": "linear-output-bc-offline-gate-v1",
        "checks": checks,
        "target_variance_per_joint": target_variance.tolist(),
        "prediction_std_per_joint": prediction_std.tolist(),
        "classification": "LINEAR_BC_OFFLINE_PASS" if all(item["passed"] for item in checks) else "LINEAR_BC_OFFLINE_BLOCKED",
    }


def plot_offline(revision_root: Path, behavior: dict[str, dict[str, Any]], expert: np.ndarray, student: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = revision_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = ("standing", "forward", "backward", "lateral_left", "lateral_right", "yaw_left", "yaw_right", "forward_yaw", "mixed")
    values = [behavior[name]["mse"] for name in names]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(np.arange(len(names)), values, color="#4472c4")
    axis.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    axis.set_ylabel("MSE")
    axis.set_title("Held-out test action MSE by behavior")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "test_mse_by_behavior.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(expert.reshape(-1), bins=80, alpha=0.65, label="expert", density=True)
    axes[0].hist(student.reshape(-1), bins=80, alpha=0.65, label="student", density=True)
    axes[0].set_title("Global raw action distribution")
    axes[0].legend()
    axes[1].scatter(expert.reshape(-1)[::20], student.reshape(-1)[::20], s=2, alpha=0.15)
    low = float(min(expert.min(), student.min()))
    high = float(max(expert.max(), student.max()))
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set_title("Expert vs student actions")
    axes[1].set_xlabel("expert")
    axes[1].set_ylabel("student")
    figure.tight_layout()
    figure.savefig(plot_dir / "test_expert_student_action_distribution.png", dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--revision-root", type=Path, default=REVISION_ROOT)
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    dataset_path = args.dataset.resolve()
    data = load_dataset(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise SystemExit(f"frozen dataset hash mismatch: {dataset_sha}")
    summary = json.loads((revision_root / "manifests/training_summary.json").read_text(encoding="utf-8"))
    val_rows = load_split_indices("val")
    val_transition_all, _ = transition_masks(data["episode_id"], data["step_index"], data["desired_command"])
    val_inputs = data["student_input"][val_rows].astype(np.float32)
    val_target = data["expert_action_student_order"][val_rows].astype(np.float32)
    val_transition = val_transition_all[val_rows]

    # Candidate selection is deliberately complete before test rows are loaded.
    comparison = candidate_validation_comparison(data, summary["results"], val_rows, val_transition)
    selected = min(comparison, key=lambda row: float(row["validation_mse"]))
    write_json(
        revision_root / "manifests/selected_checkpoint.json",
        {
            "selection_basis": "validation_mse_only",
            "selected": selected,
            "candidates": comparison,
            "dataset_sha256": dataset_sha,
        },
    )
    (revision_root / "reports/checkpoint_selection.md").write_text(markdown_selection(comparison, selected) + "\n", encoding="utf-8")
    selected_path = Path(selected["candidate_path"])
    selected_val_prediction = predict(selected_path, val_inputs)
    selected_val_metrics = regression_metrics(selected_val_prediction, val_target)
    selected_val_behavior = behavior_metrics(selected_val_prediction, val_target, data["desired_command"][val_rows], val_transition)
    (revision_root / "reports/validation_by_behavior.md").write_text(markdown_behavior(selected_val_behavior, "Validation metrics by behavior") + "\n", encoding="utf-8")
    write_json(revision_root / "metrics/selected_validation_metrics.json", {"overall": selected_val_metrics, "by_behavior": selected_val_behavior, "expert_distribution": action_distribution(val_target), "student_distribution": action_distribution(selected_val_prediction)})

    # Held-out test is accessed only after selection has been written and frozen.
    test_rows = load_split_indices("test")
    test_transition = val_transition_all[test_rows]
    test_inputs = data["student_input"][test_rows].astype(np.float32)
    test_target = data["expert_action_student_order"][test_rows].astype(np.float32)
    test_prediction = predict(selected_path, test_inputs)
    test_metrics = regression_metrics(test_prediction, test_target)
    test_behavior = behavior_metrics(test_prediction, test_target, data["desired_command"][test_rows], test_transition)
    train_rows = load_split_indices("train")
    train_prediction = predict(selected_path, data["student_input"][train_rows].astype(np.float32))
    train_metrics = regression_metrics(train_prediction, data["expert_action_student_order"][train_rows].astype(np.float32))
    write_test_csv(revision_root / "metrics/test_metrics.csv", test_metrics, test_metrics["per_joint"], test_behavior)

    gate = offline_gate(
        training_summary=summary,
        train_metrics=train_metrics,
        val_metrics=selected_val_metrics,
        test_metrics=test_metrics,
        val_behavior=selected_val_behavior,
        test_behavior=test_behavior,
        prediction=test_prediction,
        target=test_target,
        transition=test_transition,
    )
    write_json(revision_root / "manifests/offline_gate.json", gate)
    plot_offline(revision_root, test_behavior, test_target, test_prediction)

    expert_range = global_range(test_target)
    student_range = global_range(test_prediction)
    lines = [
        "# Held-out test results",
        "",
        f"Checkpoint selected before test access: `{selected_path}`",
        f"Checkpoint SHA256: `{selected['candidate_sha256']}`",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in ("mse", "mae", "mean_action_l2", "median_action_l2", "p95_action_l2", "p99_action_l2", "max_action_l2"):
        lines.append(f"| `{name}` | {test_metrics[name]:.9g} |")
    lines += [
        "",
        "## Per-joint metrics",
        "",
        "| Joint | MSE | MAE | Pearson |",
        "|---|---:|---:|---:|",
    ]
    for row in test_metrics["per_joint"]:
        pearson = "nan" if not np.isfinite(row["pearson"]) else f"{row['pearson']:.9g}"
        lines.append(f"| `{row['joint']}` | {row['mse']:.9g} | {row['mae']:.9g} | {pearson} |")
    lines += [
        "",
        "## Behavior metrics",
        "",
        markdown_behavior(test_behavior, "Test behavior table").split("\n", 4)[-1],
        "",
        "## Output-range fidelity",
        "",
        f"- expert min/max: `{expert_range['min']:.9g}` / `{expert_range['max']:.9g}`",
        f"- student min/max: `{student_range['min']:.9g}` / `{student_range['max']:.9g}`",
        f"- expert p1/p99: `{expert_range['p1']:.9g}` / `{expert_range['p99']:.9g}`",
        f"- student p1/p99: `{student_range['p1']:.9g}` / `{student_range['p99']:.9g}`",
        f"- expert fraction outside `[-1,1]`: `{expert_range['fraction_outside_minus1_plus1']:.9f}`",
        f"- student fraction outside `[-1,1]`: `{student_range['fraction_outside_minus1_plus1']:.9f}`",
        "",
        "## Offline gate",
        "",
        f"`{gate['classification']}`",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for item in gate["checks"]:
        detail = item["detail"].replace("|", "\\|")
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    (revision_root / "reports/test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_seed": selected["seed"], "selected_epoch": selected["selected_epoch"], "classification": gate["classification"], "test": scalar_metrics(test_prediction, test_target)}, indent=2, sort_keys=True))
    return 0 if gate["classification"] == "LINEAR_BC_OFFLINE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
