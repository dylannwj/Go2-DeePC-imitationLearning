#!/usr/bin/env python3
"""Build deterministic episode-disjoint splits and finalize acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.imitation_learning.dataset_common import (  # noqa: E402
    DATASET_ROOT,
    canonical_json,
    metadata_array,
    relative_path,
    sha256_file,
    sha256_tree,
    utc_now,
    write_deterministic_npz,
    write_json,
    write_text,
    load_npz,
    metadata_from_arrays,
)


SCRIPT_VERSION = "expert-imitation-split-builder-v1.0"
SPLIT_SEED = 20260816
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_CATEGORIES = ("standing", "forward", "backward", "lateral", "pure_yaw", "forward+yaw", "mixed")
CATEGORY_ALIASES = {
    "forward_yaw": "forward+yaw",
    "forward_lateral": "mixed",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_features(record: dict[str, Any]) -> set[str]:
    return {
        CATEGORY_ALIASES.get(str(segment.get("category")), str(segment.get("category")))
        for segment in record.get("command_schedule", [])
    }


def episode_has_transition(record: dict[str, Any]) -> bool:
    schedule = record.get("command_schedule", [])
    return len(schedule) >= 2 and any(
        schedule[index].get("command") != schedule[index - 1].get("command")
        for index in range(1, len(schedule))
    )


def choose_splits(records: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    if len(records) < 10:
        raise RuntimeError("at least ten accepted episodes are required for disjoint 80/10/10 splits")
    rng = np.random.default_rng(seed)
    records = sorted(records, key=lambda row: str(row["episode_id"]))
    episode_ids = [str(row["episode_id"]) for row in records]
    target_counts = {
        "train": int(round(0.80 * len(records))),
        "val": int(round(0.10 * len(records))),
        "test": 0,
    }
    target_counts["test"] = len(records) - target_counts["train"] - target_counts["val"]
    if min(target_counts.values()) < 1:
        raise RuntimeError(f"invalid split sizes: {target_counts}")
    feature_map = {str(row["episode_id"]): episode_features(row) for row in records}
    transition_map = {str(row["episode_id"]): episode_has_transition(row) for row in records}
    tie_order = {episode_id: float(rng.random()) for episode_id in episode_ids}
    unassigned = set(episode_ids)
    chosen: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    def choose_for_coverage(split: str) -> None:
        covered: set[str] = set()
        has_transition = False
        while (not REQUIRED_CATEGORIES or not set(REQUIRED_CATEGORIES).issubset(covered) or not has_transition) and len(chosen[split]) < target_counts[split]:
            candidates = list(unassigned)
            if not candidates:
                break
            def score(episode_id: str) -> tuple[int, int, float]:
                new_categories = len(feature_map[episode_id] - covered)
                new_transition = int(transition_map[episode_id] and not has_transition)
                return (new_categories, new_transition, -tie_order[episode_id])
            selected = max(candidates, key=score)
            chosen[split].append(selected)
            unassigned.remove(selected)
            covered.update(feature_map[selected])
            has_transition |= transition_map[selected]

    # Reserve validation and test coverage before filling train. This prevents
    # the hard examples from becoming train-only by construction.
    choose_for_coverage("val")
    choose_for_coverage("test")
    for split in ("val", "test"):
        if not set(REQUIRED_CATEGORIES).issubset(set().union(*(feature_map[item] for item in chosen[split]))):
            raise RuntimeError(f"{split} cannot cover every required command category")
        if not any(transition_map[item] for item in chosen[split]):
            raise RuntimeError(f"{split} has no command-transition episode")
    remaining = sorted(unassigned, key=lambda episode_id: tie_order[episode_id])
    for split in ("val", "test"):
        need = target_counts[split] - len(chosen[split])
        chosen[split].extend(remaining[:need])
        remaining = remaining[need:]
    chosen["train"] = remaining
    for split in SPLIT_NAMES:
        if len(chosen[split]) != target_counts[split]:
            raise RuntimeError(f"split {split} has {len(chosen[split])}, expected {target_counts[split]}")
        chosen[split].sort()
    return chosen


def split_rows(data: dict[str, np.ndarray], episode_ids: list[str]) -> np.ndarray:
    numeric_ids = np.asarray([int(value.rsplit("_", 1)[-1]) for value in episode_ids], dtype=np.int32)
    rows = np.flatnonzero(np.isin(data["episode_id"], numeric_ids)).astype(np.int64)
    if len(rows) == 0:
        raise RuntimeError(f"split has no rows for episodes {episode_ids[:3]}")
    observed = set(int(value) for value in np.unique(data["episode_id"][rows]))
    expected = set(int(value) for value in numeric_ids)
    if observed != expected:
        raise RuntimeError(f"split row/episode mismatch: observed={observed} expected={expected}")
    return rows


def coverage_for_split(data: dict[str, np.ndarray], rows: np.ndarray, episode_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    commands = data["desired_command"][rows]
    vx, vy, yaw = commands.T
    active_vx = np.abs(vx) > 0.02
    active_vy = np.abs(vy) > 0.02
    active_yaw = np.abs(yaw) > 0.02
    masks = {
        "standing": ~(active_vx | active_vy | active_yaw),
        "forward": active_vx & (vx > 0) & ~active_vy & ~active_yaw,
        "backward": active_vx & (vx < 0) & ~active_vy & ~active_yaw,
        "lateral": active_vy & ~active_vx & ~active_yaw,
        "pure_yaw": active_yaw & ~active_vx & ~active_vy,
        "forward+yaw": active_vx & (vx > 0) & active_yaw & ~active_vy,
    }
    masks["mixed"] = ~(masks["standing"] | masks["forward"] | masks["backward"] | masks["lateral"] | masks["pure_yaw"] | masks["forward+yaw"])
    # The manifest stores schedule transitions rather than an aggregate flag;
    # use the processed row-level transition flag for exact samples.
    episode_numeric = set(int(value) for value in np.unique(data["episode_id"][rows]))
    transition = bool(np.any(data["transition_window"][rows]))
    return {
        "samples": int(len(rows)),
        "episodes": int(len(episode_numeric)),
        "category_fraction": {name: float(np.mean(mask)) for name, mask in masks.items()},
        "transition_samples": int(np.count_nonzero(data["transition_window"][rows])),
        "has_transition": transition,
        "command_min": commands.min(axis=0).tolist(),
        "command_max": commands.max(axis=0).tolist(),
    }


def write_split_indices(root: Path, name: str, rows: np.ndarray, episode_ids: list[str], metadata: dict[str, Any]) -> Path:
    path = root / "splits" / f"{name}_indices.npz"
    write_deterministic_npz(path, {
        "row_indices": rows.astype(np.int64),
        "episode_numeric_ids": np.asarray([int(value.rsplit("_", 1)[-1]) for value in episode_ids], dtype=np.int32),
        "metadata_json": metadata_array(metadata),
    })
    return path


def build(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    processed_path = root / "processed/expert_imitation_dataset_v1.npz"
    data = load_npz(processed_path)
    metadata = metadata_from_arrays(data)
    episode_manifest = load_json(root / "manifests/episode_manifest.json")
    accepted_records = [record for record in episode_manifest["episodes"] if record.get("accepted")]
    if set(metadata.get("accepted_episode_ids", [])) != {str(record["episode_id"]) for record in accepted_records}:
        raise RuntimeError("processed metadata and episode manifest accepted IDs differ")
    selected = choose_splits(accepted_records, args.seed)
    all_ids = set().union(*(set(values) for values in selected.values()))
    if len(all_ids) != len(accepted_records) or any(set(selected[a]) & set(selected[b]) for a in SPLIT_NAMES for b in SPLIT_NAMES if a < b):
        raise RuntimeError("episode-disjoint split invariant failed")

    record_map = {str(record["episode_id"]): record for record in accepted_records}
    split_rows_map = {name: split_rows(data, selected[name]) for name in SPLIT_NAMES}
    split_coverage = {name: coverage_for_split(data, split_rows_map[name], record_map) for name in SPLIT_NAMES}
    for name in ("val", "test"):
        if any(split_coverage[name]["category_fraction"].get(category, 0.0) <= 0.0 for category in REQUIRED_CATEGORIES):
            raise RuntimeError(f"{name} command coverage is incomplete: {split_coverage[name]}")
        if not split_coverage[name]["has_transition"]:
            raise RuntimeError(f"{name} has no transition samples")

    for name in SPLIT_NAMES:
        ids_path = root / "splits" / f"{name}_episode_ids.txt"
        write_text(ids_path, "\n".join(selected[name]))
        split_metadata = {
            "schema": "episode-split-index-v1",
            "split": name,
            "seed": args.seed,
            "episode_ids": selected[name],
            "rows": int(len(split_rows_map[name])),
            "source_processed_sha256": sha256_file(processed_path),
        }
        write_split_indices(root, name, split_rows_map[name], selected[name], split_metadata)

    split_manifest = {
        "schema": "episode-split-manifest-v1",
        "generated_utc": utc_now(),
        "split_seed": args.seed,
        "fractions_requested": {"train": 0.8, "val": 0.1, "test": 0.1},
        "episode_counts": {name: len(selected[name]) for name in SPLIT_NAMES},
        "sample_counts": {name: int(len(split_rows_map[name])) for name in SPLIT_NAMES},
        "episode_disjoint": True,
        "all_accepted_episodes_assigned": True,
        "splits": selected,
        "coverage": split_coverage,
    }
    write_json(root / "manifests/split_manifest.json", split_manifest)

    split_lines = [
        "# Episode-disjoint split audit",
        "",
        f"Split seed: `{args.seed}`.",
        "",
        "| Split | Episodes | Samples | Standing | Forward | Backward | Lateral | Pure yaw | Forward+yaw | Mixed | Transition samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        coverage = split_coverage[name]
        fractions = coverage["category_fraction"]
        split_lines.append(
            f"| `{name}` | {coverage['episodes']} | {coverage['samples']} | {100*fractions['standing']:.2f}% | {100*fractions['forward']:.2f}% | {100*fractions['backward']:.2f}% | {100*fractions['lateral']:.2f}% | {100*fractions['pure_yaw']:.2f}% | {100*fractions['forward+yaw']:.2f}% | {100*fractions['mixed']:.2f}% | {coverage['transition_samples']} |"
        )
    split_lines.extend(["", "No episode ID occurs in more than one split. Validation and test each retain every available command category and transition samples."])
    write_text(root / "reports/split_coverage.md", "\n".join(split_lines))

    raw_files = {relative_path(path): sha256_file(path) for path in sorted((root / "raw").glob("episode_*.npz"))}
    processed_files = {relative_path(processed_path): sha256_file(processed_path)}
    split_files = {
        relative_path(path): sha256_file(path)
        for path in sorted((root / "splits").iterdir())
        if path.is_file()
    }
    schema_files = {
        relative_path(path): sha256_file(path)
        for path in sorted((root / "frozen_contract").glob("*.md"))
    }
    script_files = {
        relative_path(path): sha256_file(path)
        for path in sorted(SCRIPT_ROOT.glob("*.py"))
    }
    manifest_files = {
        relative_path(path): sha256_file(path)
        for path in sorted((root / "manifests").iterdir())
        if path.is_file() and path.name != "dataset_hashes.json"
    }
    canonical_hash = sha256_file(processed_path)
    hashes = {
        "schema": "dataset-hashes-v1",
        "generated_utc": utc_now(),
        "canonical_dataset_sha256": canonical_hash,
        "raw_dataset_files": raw_files,
        "processed_canonical_dataset": processed_files,
        "split_files": split_files,
        "schema_files": schema_files,
        "collection_and_validation_scripts": script_files,
        "manifests": manifest_files,
        "split_manifest": relative_path(root / "manifests/split_manifest.json"),
        "hashing_definition": "SHA256 of each file; canonical NPZ uses deterministic ZIP metadata; tree-level consumers may hash this manifest content",
    }
    write_json(root / "manifests/dataset_hashes.json", hashes)
    hashes_manifest_sha = sha256_file(root / "manifests/dataset_hashes.json")

    validation_summary_path = root / "reports/dataset_validation_summary.json"
    validation_summary = load_json(validation_summary_path)
    validation_summary.update({
        "split_pending": False,
        "split_complete": True,
        "split_manifest_sha256": sha256_file(root / "manifests/split_manifest.json"),
        "train_episodes": len(selected["train"]),
        "val_episodes": len(selected["val"]),
        "test_episodes": len(selected["test"]),
        "train_samples": int(len(split_rows_map["train"])),
        "val_samples": int(len(split_rows_map["val"])),
        "test_samples": int(len(split_rows_map["test"])),
        "canonical_dataset_sha256": canonical_hash,
        "dataset_hashes_manifest_sha256": hashes_manifest_sha,
    })
    write_json(validation_summary_path, validation_summary)

    required_pass = (
        validation_summary.get("artifact_hashes_match")
        and validation_summary.get("student_state_shape", [0, 0])[-1] == 45
        and validation_summary.get("commands_shape", [0, 0])[-1] == 3
        and validation_summary.get("expert_actions_shape", [0, 0])[-1] == 12
        and validation_summary.get("no_nan_inf")
        and validation_summary.get("no_accepted_falls")
        and validation_summary.get("no_accepted_base_contacts")
        and validation_summary.get("no_accepted_joint_limit_violations")
        and validation_summary.get("no_accepted_torque_violations")
        and validation_summary.get("q_target_reconstruction_max_abs_error", 1.0) <= 1.0e-6
        and validation_summary.get("student_input_reconstruction_max_abs_error", 1.0) <= 1.0e-6
        and validation_summary.get("temporal_integrity_pass")
        and validation_summary.get("privilege_leakage_audit_pass")
        and validation_summary.get("reproducibility_pass")
        and validation_summary.get("episode_validation_errors", 1) == 0
        and all(split_coverage[name]["has_transition"] for name in SPLIT_NAMES)
        and all(all(split_coverage[name]["category_fraction"].get(category, 0.0) > 0.0 for category in REQUIRED_CATEGORIES) for name in ("val", "test"))
    )
    report_lines = [
        "# Final expert imitation dataset report",
        "",
        "This report covers collection, canonical processing, validation, provenance, coverage, and episode-disjoint splitting. BC/student training, student evaluation, PPO refinement, DeePC integration, and expert modification were not run.",
        "",
        "## Final classification",
        "",
        "`EXPERT_IMITATION_DATASET_PASS`" if required_pass else "`EXPERT_IMITATION_DATASET_FAIL`",
        "",
        "## Dataset size",
        "",
        f"- Attempted episodes: `{validation_summary['attempted_episodes']}`",
        f"- Accepted episodes: `{validation_summary['accepted_episodes']}`",
        f"- Rejected episodes: `{validation_summary['rejected_episodes']}`",
        f"- Accepted samples: `{validation_summary['accepted_samples']}`",
        f"- Accepted duration: `{validation_summary['accepted_duration_s']:.3f} s`",
        f"- Train: `{len(selected['train'])}` episodes / `{len(split_rows_map['train'])}` samples",
        f"- Validation: `{len(selected['val'])}` episodes / `{len(split_rows_map['val'])}` samples",
        f"- Test: `{len(selected['test'])}` episodes / `{len(split_rows_map['test'])}` samples",
        "",
        "## Frozen dimensions and ranges",
        "",
        f"- Student state: `{validation_summary['student_state_shape']}`",
        f"- Desired command: `{validation_summary['commands_shape']}`",
        f"- Student input: `{validation_summary['student_input_shape']}`",
        f"- Expert action: `{validation_summary['expert_actions_shape']}`",
        f"- Action range: `[{validation_summary['action_global_min']:.8f}, {validation_summary['action_global_max']:.8f}]`",
        f"- Fraction outside `[-1,1]`: `{100 * validation_summary['action_fraction_outside_minus1_plus1']:.4f}%`",
        f"- q-target reconstruction max error: `{validation_summary['q_target_reconstruction_max_abs_error']:.3e}`",
        "",
        "## Safety and integrity",
        "",
        f"- Falls: `{validation_summary['safety_totals']['fall_flag']}`",
        f"- Base contacts: `{validation_summary['safety_totals']['base_contact']}`",
        f"- Actual joint-limit violations: `{validation_summary['safety_totals']['actual_joint_limit_violation']}`",
        f"- Torque-limit violations: `{validation_summary['safety_totals']['torque_limit_violation']}`",
        f"- NaN/Inf flags: `{validation_summary['safety_totals']['nan_inf_flag']}`",
        f"- Temporal integrity: `{'PASS' if validation_summary['temporal_integrity_pass'] else 'FAIL'}`",
        f"- Privilege leakage audit: `{'PASS' if validation_summary['privilege_leakage_audit_pass'] else 'FAIL'}`",
        f"- Reproducibility spot check: `{'PASS' if validation_summary['reproducibility_pass'] else 'FAIL'}`",
        "",
        "## Coverage",
        "",
        "The command bank includes standing, forward, backward characterization, lateral, pure yaw, forward+yaw arcs, forward+lateral, full mixed commands, starts/stops, sign changes, and multi-segment transitions. See `reports/command_coverage.md`, `reports/split_coverage.md`, and `plots/`.",
        "",
        f"- Canonical dataset SHA256: `{canonical_hash}`",
        f"- Dataset hash manifest SHA256: `{hashes_manifest_sha}`",
        "",
        "EXPERT_IMITATION_DATASET_PASS" if required_pass else "EXPERT_IMITATION_DATASET_FAIL",
    ]
    write_text(root / "reports/final_dataset_report.md", "\n".join(report_lines))
    if not required_pass:
        raise RuntimeError("EXPERT_IMITATION_DATASET_FAIL; inspect reports/final_dataset_report.md")
    print(json.dumps({
        "classification": "EXPERT_IMITATION_DATASET_PASS",
        "train_samples": int(len(split_rows_map["train"])),
        "val_samples": int(len(split_rows_map["val"])),
        "test_samples": int(len(split_rows_map["test"])),
        "canonical_dataset_sha256": canonical_hash,
        "dataset_hashes_manifest_sha256": hashes_manifest_sha,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
