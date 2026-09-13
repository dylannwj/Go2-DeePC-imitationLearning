#!/usr/bin/env python3
"""Reverify the frozen dataset for the linear-output BC revision."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    DATASET_SPLIT_ROOT,
    EXPECTED_COUNTS,
    EXPECTED_DATASET_SHA256,
    INPUT_DIM,
    INPUT_FEATURE_NAMES,
    JOINT_NAMES,
    Q_DEFAULT_CANONICAL,
    STATE_DIM,
    STATE_FEATURE_NAMES,
    action_distribution,
    json_safe,
    load_dataset,
    load_split_indices,
    raw_student_targets,
    sha256_file,
    write_json,
)


REQUIRED_SHAPES = {
    "student_state": (240_000, STATE_DIM),
    "desired_command": (240_000, 3),
    "student_input": (240_000, INPUT_DIM),
    "expert_action": (240_000, ACTION_DIM),
    "expert_action_student_order": (240_000, ACTION_DIM),
}


def _episode_ids(path: Path) -> set[int]:
    values = np.loadtxt(path, dtype=str, ndmin=1)
    result = set()
    for value in values.tolist():
        result.add(int(str(value).rsplit("_", 1)[-1]))
    return result


def run(dataset_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    observed_hash = sha256_file(dataset_path)
    check("canonical dataset SHA256", observed_hash == EXPECTED_DATASET_SHA256, f"observed={observed_hash}; expected={EXPECTED_DATASET_SHA256}")
    data = load_dataset(dataset_path)
    missing = sorted(set(REQUIRED_SHAPES) - set(data))
    check("required arrays present", not missing, f"missing={missing or 'none'}")

    for name, expected_shape in REQUIRED_SHAPES.items():
        actual_shape = tuple(data[name].shape) if name in data else None
        check(f"{name} shape", actual_shape == expected_shape, f"observed={actual_shape}; expected={expected_shape}")
    for name in ("student_state", "desired_command", "student_input", "expert_action", "expert_action_student_order", "q_target_student_order"):
        if name in data:
            finite = bool(np.isfinite(data[name]).all())
            check(f"{name} finite", finite, f"finite={finite}; dtype={data[name].dtype}")

    input_error = float(np.max(np.abs(data["student_input"] - np.concatenate((data["student_state"], data["desired_command"]), axis=1))))
    check("student input is exact state + command", input_error <= 1.0e-6, f"max_abs_error={input_error:.3e}")
    check("input feature contract", len(INPUT_FEATURE_NAMES) == INPUT_DIM and len(STATE_FEATURE_NAMES) == STATE_DIM, f"state_features={len(STATE_FEATURE_NAMES)}; input_features={len(INPUT_FEATURE_NAMES)}")

    split_rows: dict[str, np.ndarray] = {}
    split_episodes: dict[str, set[int]] = {}
    split_details: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        ids_path = DATASET_SPLIT_ROOT / f"{split}_episode_ids.txt"
        rows = load_split_indices(split)
        ids = _episode_ids(ids_path)
        row_ids = {int(value) for value in np.unique(data["episode_id"][rows])}
        split_rows[split] = rows
        split_episodes[split] = ids
        split_details[split] = {
            "episodes": len(ids),
            "rows": int(len(rows)),
            "unique_rows": int(len(np.unique(rows))),
            "manifest_episode_ids": sorted(ids),
            "row_episode_ids": sorted(row_ids),
            "row_episode_ids_match_manifest": row_ids == ids,
        }
        check(f"{split} sample count", len(rows) == EXPECTED_COUNTS[split], f"observed={len(rows)}; expected={EXPECTED_COUNTS[split]}")
        check(f"{split} row indices unique", len(rows) == len(np.unique(rows)), f"rows={len(rows)}; unique={len(np.unique(rows))}")
        check(f"{split} rows match episode manifest", row_ids == ids, f"observed={sorted(row_ids)}; expected={sorted(ids)}")

    pairwise = {
        f"{left}∩{right}": len(split_episodes[left] & split_episodes[right])
        for index, left in enumerate(("train", "val", "test"))
        for right in ("train", "val", "test")[index + 1 :]
    }
    all_rows = np.concatenate([split_rows[name] for name in ("train", "val", "test")])
    disjoint = all(value == 0 for value in pairwise.values())
    covers = len(all_rows) == 240_000 and len(np.unique(all_rows)) == 240_000
    check("episode IDs disjoint", disjoint, json.dumps(pairwise, sort_keys=True))
    check("split rows cover dataset exactly once", covers, f"total={len(all_rows)}; unique={len(np.unique(all_rows))}")

    target = raw_student_targets(data)
    reconstructed_q = Q_DEFAULT_CANONICAL[None, :] + 0.5 * target
    q_error = float(np.max(np.abs(reconstructed_q - data["q_target_student_order"])))
    check("q_target decoder reconstruction", q_error <= 1.0e-6, f"max_abs_error={q_error:.3e}")
    check("accepted dataset safety flags remain zero", all(not np.any(data[name]) for name in ("fall_flag", "base_contact", "actual_joint_limit_violation", "torque_limit_violation", "nan_inf_flag")), "all accepted-row safety flags are false")

    raw = np.asarray(data["expert_action"], dtype=np.float64)
    canonical = np.asarray(data["expert_action_student_order"], dtype=np.float64)
    global_stats = {
        "teacher_order_min": float(raw.min()),
        "teacher_order_max": float(raw.max()),
        "canonical_min": float(canonical.min()),
        "canonical_max": float(canonical.max()),
        "fraction_outside_minus1_plus1": float(np.mean((raw < -1.0) | (raw > 1.0))),
        "count_outside_minus1_plus1": int(np.count_nonzero((raw < -1.0) | (raw > 1.0))),
        "count_below_minus1": int(np.count_nonzero(raw < -1.0)),
        "count_above_plus1": int(np.count_nonzero(raw > 1.0)),
    }
    return {
        "schema": "linear-output-bc-preflight-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": observed_hash,
            "expected_sha256": EXPECTED_DATASET_SHA256,
            "arrays": {name: list(value.shape) for name, value in data.items()},
        },
        "checks": checks,
        "split_details": split_details,
        "split_pairwise_episode_intersections": pairwise,
        "global_action_distribution": global_stats,
        "integrity_pass": bool(all(item["passed"] for item in checks)),
        "classification": "LINEAR_BC_PREFLIGHT_PASS" if all(item["passed"] for item in checks) else "LINEAR_BC_PREFLIGHT_BLOCKED",
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ("Index", "Joint", "Min", "Max", "Mean", "Std", "p0.1", "p1", "p50", "p99", "p99.9", "Outside [-1,1]")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row["index"]),
                    f"`{row['joint']}`",
                    *(f"{row[key]:.9g}" for key in ("min", "max", "mean", "std", "p0.1", "p1", "p50", "p99", "p99.9")),
                    f"{100.0 * row['fraction_outside_minus1_plus1']:.6f}%",
                ]
            ) + " |"
        )
    return "\n".join(lines)


def write_reports(result: dict[str, Any], data: dict[str, np.ndarray], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    checks = result["checks"]
    lines = [
        "# Revised linear-output BC preflight",
        "",
        f"`{result['classification']}`",
        "",
        f"Dataset: `{result['dataset']['path']}`",
        f"SHA256: `{result['dataset']['sha256']}`",
        f"Expected SHA256: `{result['dataset']['expected_sha256']}`",
        "",
        "The frozen NPZ was opened read-only. No source dataset, labels, episode, or split file was modified.",
        "",
        "## Integrity checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## Frozen learning dimensions",
        "",
        "- student input: `[240000,48]`",
        "- expert action: `[240000,12]`",
        "- train / val / test: `192000 / 24000 / 24000`",
        "- episode split: `96 / 12 / 12`, disjoint",
        "",
        "## Action-range interpretation",
        "",
        "The raw action range is deliberately not an acceptance blocker in this revision. It is characterized in `raw_action_distribution.md`; labels remain unmodified and the linear output is allowed to predict outside `[-1,1]`.",
        "",
        "LINEAR_BC_PREFLIGHT_PASS" if result["integrity_pass"] else "LINEAR_BC_PREFLIGHT_BLOCKED",
    ]
    (report_dir / "revised_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    teacher_stats = action_distribution(data["expert_action"], ("FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"))
    canonical_stats = action_distribution(data["expert_action_student_order"], JOINT_NAMES)
    global_stats = result["global_action_distribution"]
    distribution_lines = [
        "# Raw expert action distribution",
        "",
        "## Classification",
        "",
        "This is characterization, not a blocker. The frozen expert labels are reported exactly as stored; no clipping, rescaling, row removal, or split change occurred.",
        "",
        f"- teacher-order minimum: `{global_stats['teacher_order_min']:.9f}`",
        f"- teacher-order maximum: `{global_stats['teacher_order_max']:.9f}`",
        f"- labels below `-1`: `{global_stats['count_below_minus1']}`",
        f"- labels above `+1`: `{global_stats['count_above_plus1']}`",
        f"- labels outside `[-1,1]`: `{global_stats['count_outside_minus1_plus1']}` (`{100.0 * global_stats['fraction_outside_minus1_plus1']:.7f}%`)",
        "",
        "## Canonical Genesis order",
        "",
        markdown_table(canonical_stats),
        "",
        "## Teacher/policy order provenance",
        "",
        markdown_table(teacher_stats),
        "",
        "The training target is the same raw action after the frozen named permutation into canonical Genesis order (`expert_action_student_order`).",
    ]
    (report_dir / "raw_action_distribution.md").write_text("\n".join(distribution_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--revision-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    result = run(dataset_path)
    data = load_dataset(dataset_path)
    revision_root = args.revision_root.resolve()
    write_reports(result, data, revision_root / "reports")
    write_json(revision_root / "manifests/revised_preflight.json", result)
    print(json.dumps(json_safe({"classification": result["classification"], **result["global_action_distribution"]}), indent=2, sort_keys=True))
    return 0 if result["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
