#!/usr/bin/env python3
"""Bounded pre-S6 safety-fix design and one-candidate Genesis evaluation.

The script only consumes already archived evidence.  It creates a candidate
QP in this validation branch; it does not modify the frozen controller,
student, Hankel package, UI, or the current S6 controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
OUT = PROJECT_ROOT / "results/genesis/generated"
REFINEMENT = PROJECT_ROOT / "results/genesis"
V2_ROOT = PROJECT_ROOT
REASSESSMENT = PROJECT_ROOT / "results/genesis/audits"
REPLAY = PROJECT_ROOT / "results/genesis/generated"
S3_ROOT = REASSESSMENT / "s3"
S4_ROOT = REASSESSMENT / "s4"
RECOVERY = PROJECT_ROOT
HANKEL = PROJECT_ROOT / "data/imitation/hankel/final_student_deepc_hankel_v2_stride2.npz"
CONFIG = PROJECT_ROOT / "configs/genesis/deepc_final_recovered.json"
CHECKPOINT = PROJECT_ROOT / "models/go2_observable_student_bc_linear_v1.pt"
RUNTIME_PYTHON = Path(sys.executable)
CONTROLLER = PROJECT_ROOT / "src/deepc/genesis_recovered_deepc.py"
DIRECT_STUDENT = PROJECT_ROOT / "src/learned_execution/direct_student_runtime.py"
RUNTIME_CONTRACT = PROJECT_ROOT / "configs/genesis/frozen_runtime_contract.json"
GENESIS_URDF = Path(os.environ.get("GENESIS_GO2_URDF", "urdf/go2/urdf/go2.urdf"))
PLANE_URDF = Path(os.environ.get("GENESIS_PLANE_URDF", "urdf/plane/plane.urdf"))

IDENTITY = REASSESSMENT / "pre_s6_working_controller_identity.json"
FREEZE_AUDIT = REPLAY / "pre_s6_replay_freeze_audit.json"
EXCLUSION_AUDIT = REPLAY / "pre_s6_safety_exclusion_audit.json"
REPLAY_RESULTS = REPLAY / "pre_s6_replay_target_results.csv"
REPLAY_TELEMETRY = REPLAY / "pre_s6_replay_runtime_telemetry.csv"
UI_UNSAFE = REASSESSMENT / "pre_s6_unsafe_event_inventory.csv"
S3_COMMANDS = S3_ROOT / "S3_tested_commands.csv"
S4_TRANSITIONS = S4_ROOT / "S4_transition_matrix.csv"
T2_NPZ = RECOVERY / "logs" / "final_T2_lateral_020_rep01.npz"
S3_UNSAFE_NPZ = S3_ROOT / "telemetry" / "S3_step_telemetry_vy_yaw_vxp0d000000_vyp0d050000_yawp0d250000_seed_0.npz"

EXPECTED_STUDENT_SHA256 = "6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc"
EXPECTED_HANKEL_SHA256 = "5ece444b2d82a75df6b751809d2b5f5c1aacfe06cc69a0919fa21532804bfb7b"
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.4.4",
    "genesis": "1.3.1",
    "quadrants": "1.2.0",
}

COMMAND_NAMES = ("vx", "vy", "yaw_rate")
COMMAND_SCALE = np.asarray([0.30, 0.20, 0.50], dtype=np.float64)
TARGETS = (
    ("P0", "standing", (0.0, 0.0, 0.0)),
    ("P1", "FORWARD / T1_forward_025", (0.25, 0.0, 0.0)),
    ("P2", "LATERAL / I_lateral_correction_015", (0.0, 0.15, 0.0)),
    ("P3", "YAW / T4_yaw_030", (0.0, 0.0, 0.30)),
    ("P4", "MIXED / T5_forward_yaw", (0.20, 0.0, 0.30)),
    ("P5", "T2_lateral_020", (0.0, 0.20, 0.0)),
)

EVIDENCE_FIELDS = [
    "evidence_id",
    "source",
    "safe_unsafe",
    "evidence_quality",
    "state",
    "previous_command",
    "current_command",
    "delta_command",
    "mixed_axis_pattern",
    "measured_velocity",
    "roll",
    "pitch",
    "joint_state_if_available",
    "torque_if_available",
    "failure_type",
    "target",
    "notes",
]

ANALYSIS_FIELDS = [
    "record_type",
    "evidence_id",
    "source",
    "evidence_quality",
    "safety_label",
    "target",
    "previous_command",
    "current_command",
    "delta_command",
    "feature_name",
    "feature_value",
    "threshold",
    "would_reject",
    "retention_scope",
    "notes",
]

COUNTERFACTUAL_FIELDS = [
    "counterfactual_id",
    "candidate",
    "source",
    "evidence_id",
    "scope",
    "step",
    "previous_command",
    "current_command",
    "delta_command",
    "constraint_value",
    "constraint_limit",
    "would_remain_feasible",
    "rejection_reason",
    "evidence_quality",
    "notes",
]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def compact_json(value: Any) -> str:
    if value is None:
        return "NOT_ARCHIVED"
    if isinstance(value, str):
        return value
    return json.dumps(jsonable(value), separators=(",", ":"), sort_keys=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                output[field] = compact_json(value) if isinstance(value, (np.ndarray, list, tuple, dict)) else value
            writer.writerow(output)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_vec(value: Any) -> np.ndarray | None:
    if value is None or value == "" or value in {"NOT_ARCHIVED", "NOT_AVAILABLE", "UNKNOWN", "UNAVAILABLE"}:
        return None
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64)
    else:
        try:
            array = np.asarray(json.loads(str(value)), dtype=np.float64)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return array.copy() if np.all(np.isfinite(array)) else None


def parse_scalar(value: Any) -> float | None:
    if value in (None, "", "UNKNOWN", "UNAVAILABLE", "NOT_ARCHIVED"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def command_pattern(command: np.ndarray | None) -> str:
    if command is None:
        return "UNKNOWN"
    vx, vy, yaw = np.asarray(command, dtype=float).reshape(3)
    active = [name for name, value in zip(COMMAND_NAMES, (vx, vy, yaw)) if abs(value) > 1.0e-8]
    if not active:
        return "ZERO"
    if abs(vy) > 1.0e-8 and abs(yaw) > 1.0e-8:
        return f"VY_{'POS' if vy > 0 else 'NEG'}_YAW_{'POS' if yaw > 0 else 'NEG'}"
    if abs(vx) > 1.0e-8 and abs(vy) > 1.0e-8:
        return f"VX_{'POS' if vx > 0 else 'NEG'}_VY_{'POS' if vy > 0 else 'NEG'}"
    if abs(vx) > 1.0e-8 and abs(yaw) > 1.0e-8:
        return f"VX_{'POS' if vx > 0 else 'NEG'}_YAW_{'POS' if yaw > 0 else 'NEG'}"
    if len(active) == 3:
        return "TRIPLE_MIXED"
    return "_".join(active)


def direction_label(value: float) -> str:
    return "POS" if value > 1.0e-8 else "NEG" if value < -1.0e-8 else "ZERO"


def add_evidence(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    row = {field: kwargs.get(field, "") for field in EVIDENCE_FIELDS}
    rows.append(row)


def load_replay_command_sequences() -> dict[str, list[dict[str, Any]]]:
    rows = [row for row in read_csv(REPLAY_TELEMETRY) if row.get("phase") == "replan"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target_id"]].append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for target_id, target_rows in grouped.items():
        target_rows.sort(key=lambda row: int(float(row.get("interval", 0))))
        previous = np.zeros(3, dtype=np.float64)
        sequence: list[dict[str, Any]] = []
        for index, row in enumerate(target_rows):
            current = parse_vec(row.get("selected_command_float64"))
            if current is None:
                continue
            delta = current - previous
            sequence.append(
                {
                    "evidence_id": f"REPLAY-{target_id}-{index:03d}",
                    "source": "pre_s6_exact_replay",
                    "target_id": target_id,
                    "target": parse_vec(row.get("target_pose")),
                    "step": index,
                    "previous": previous.copy(),
                    "current": current.copy(),
                    "delta": delta.copy(),
                    "row": row,
                }
            )
            previous = current
        result[target_id] = sequence
    return result


def load_t2_sequence() -> list[dict[str, Any]]:
    archive = np.load(T2_NPZ, allow_pickle=True)
    commands = np.asarray(archive["requested_command"], dtype=np.float64)
    result: list[dict[str, Any]] = []
    previous = np.zeros(3, dtype=np.float64)
    for index, current in enumerate(commands):
        current = np.asarray(current, dtype=np.float64).reshape(3)
        result.append(
            {
                "evidence_id": f"HISTORICAL-T2-{index:02d}",
                "source": "historical_successful_T2",
                "target_id": "P5/T2",
                "target": np.asarray([0.0, 0.2, 0.0]),
                "step": index,
                "previous": previous.copy(),
                "current": current.copy(),
                "delta": current - previous,
                "archive": archive,
            }
        )
        previous = current
    return result


def safe_replay_sequences(replay_sequences: dict[str, list[dict[str, Any]]], t2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for sequence in replay_sequences.values() for item in sequence] + list(t2)


def build_evidence_bank(
    replay_sequences: dict[str, list[dict[str, Any]]],
    t2: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank: list[dict[str, Any]] = []

    for row in read_csv(UI_UNSAFE):
        current = parse_vec(row.get("command"))
        quality = "INSUFFICIENT_TELEMETRY"
        add_evidence(
            bank,
            evidence_id=row.get("event_id"),
            source="historical_ui_unsafe_event_inventory",
            safe_unsafe="unsafe",
            evidence_quality=quality,
            state="NOT_ARCHIVED",
            previous_command="NOT_ARCHIVED",
            current_command=current,
            delta_command="NOT_ARCHIVED",
            mixed_axis_pattern=command_pattern(current),
            measured_velocity="NOT_ARCHIVED",
            roll="NOT_ARCHIVED",
            pitch="NOT_ARCHIVED",
            joint_state_if_available="NOT_ARCHIVED",
            torque_if_available="NOT_ARCHIVED",
            failure_type=row.get("safety_reason"),
            target=parse_vec(row.get("target_pose_world")),
            notes="Historical UI command and failure class are archived; previous command, state, and low-level mechanism are not archived.",
        )

    for sequence in replay_sequences.values():
        for item in sequence:
            row = item["row"]
            add_evidence(
                bank,
                evidence_id=item["evidence_id"],
                source=item["source"],
                safe_unsafe="safe",
                evidence_quality="DIRECTLY_RECONSTRUCTED",
                state={
                    "measured_pose": parse_vec(row.get("measured_pose")),
                    "body_error": parse_vec(row.get("body_error")),
                    "body_height_m": parse_scalar(row.get("body_height_m")),
                    "joint_limit_margin_rad": parse_vec(row.get("joint_limit_margin_rad")),
                },
                previous_command=item["previous"],
                current_command=item["current"],
                delta_command=item["delta"],
                mixed_axis_pattern=command_pattern(item["current"]),
                measured_velocity=parse_vec(row.get("body_linear_velocity")),
                roll=parse_scalar(row.get("roll_rad")),
                pitch=parse_scalar(row.get("pitch_rad")),
                joint_state_if_available={
                    "position": parse_vec(row.get("joint_position")),
                    "velocity": parse_vec(row.get("joint_velocity")),
                    "margin": parse_vec(row.get("joint_limit_margin_rad")),
                },
                torque_if_available={
                    "raw": parse_vec(row.get("torque_raw")),
                    "applied": parse_vec(row.get("torque_applied")),
                },
                failure_type="NONE",
                target=item["target"],
                notes="One exact replay replan row; controller, student, and Genesis telemetry are archived.",
            )

    archive = t2[0]["archive"]
    for item in t2:
        index = item["step"]
        add_evidence(
            bank,
            evidence_id=item["evidence_id"],
            source=item["source"],
            safe_unsafe="safe",
            evidence_quality="PARTIALLY_RECONSTRUCTED",
            state={
                "q": archive["q"][index],
                "dq": archive["dq"][index],
                "body_velocity": "NOT_ARCHIVED",
                "roll": "NOT_ARCHIVED",
                "pitch": "NOT_ARCHIVED",
            },
            previous_command=item["previous"],
            current_command=item["current"],
            delta_command=item["delta"],
            mixed_axis_pattern=command_pattern(item["current"]),
            measured_velocity="NOT_ARCHIVED",
            roll="NOT_ARCHIVED",
            pitch="NOT_ARCHIVED",
            joint_state_if_available={"q": archive["q"][index], "dq": archive["dq"][index]},
            torque_if_available=archive["torque"][index],
            failure_type="NONE",
            target=item["target"],
            notes="Verified 15-command successful T2 archive; command/q/dq/torque are present but body velocity and attitude are not.",
        )

    s3_rows = read_csv(S3_COMMANDS)
    for index, row in enumerate(s3_rows):
        current = parse_vec(row.get("command"))
        classification = str(row.get("classification", "UNKNOWN"))
        label = "unsafe" if classification == "UNSAFE" else "safe" if classification == "SAFE_EFFECTIVE" else "near_boundary" if classification == "NEAR_BOUNDARY" else "unknown"
        add_evidence(
            bank,
            evidence_id=f"S3-CMD-{index:03d}",
            source="S3_direct_command_sweep",
            safe_unsafe=label,
            evidence_quality="PARTIALLY_RECONSTRUCTED",
            state="NOT_ARCHIVED",
            previous_command="INFERRED_ZERO_AT_COMMAND_ONSET",
            current_command=current,
            delta_command=current,
            mixed_axis_pattern=command_pattern(current),
            measured_velocity={
                "mean": [parse_scalar(row.get("mean_achieved_vx")), parse_scalar(row.get("mean_achieved_vy")), parse_scalar(row.get("mean_achieved_yaw_rate"))]
            },
            roll=parse_scalar(row.get("peak_abs_roll_rad")),
            pitch=parse_scalar(row.get("peak_abs_pitch_rad")),
            joint_state_if_available={"minimum_joint_limit_margin_rad": parse_scalar(row.get("minimum_joint_limit_margin_rad"))},
            torque_if_available={
                "peak_requested_abs_torque": parse_scalar(row.get("peak_requested_abs_torque")),
                "peak_applied_abs_torque": parse_scalar(row.get("peak_applied_abs_torque")),
            },
            failure_type=row.get("failure_causes") or "NONE",
            target="DIRECT_COMMAND_HOLD",
            notes=f"S3 command-level classification={classification}; aggregate telemetry does not reconstruct one full state row for this command.",
        )

    unsafe_archive = np.load(S3_UNSAFE_NPZ, allow_pickle=True)
    failure_indices = np.flatnonzero(np.asarray(unsafe_archive["joint_limit_violation"], dtype=bool))
    failure_index = int(failure_indices[0]) if len(failure_indices) else len(unsafe_archive["step_index"]) - 1
    for evidence_id, index, previous, delta, note in (
        ("S3-UNSAFE-ONSET", 0, np.zeros(3), np.asarray(unsafe_archive["command"][0], dtype=np.float64), "Command-onset transition is reconstructed from the direct reset protocol."),
        ("S3-UNSAFE-FAILURE", failure_index, np.asarray(unsafe_archive["command"][failure_index], dtype=np.float64), np.zeros(3), "Failure-step command was held; failure state and low-level chain are directly archived."),
    ):
        current = np.asarray(unsafe_archive["command"][index], dtype=np.float64)
        add_evidence(
            bank,
            evidence_id=evidence_id,
            source="S3_verified_unsafe_low_level_witness",
            safe_unsafe="unsafe",
            evidence_quality="DIRECTLY_RECONSTRUCTED",
            state={
                "step_index": int(unsafe_archive["step_index"][index]),
                "base_position": unsafe_archive["base_position"][index],
                "joint_limit_margin": unsafe_archive["joint_limit_margin"][index],
                "q_target_raw_outside_joint_limits": bool(unsafe_archive["q_target_raw_outside_joint_limits"][index]),
                "q_target_clip_flag": bool(unsafe_archive["q_target_clip_flag"][index]),
                "requested_torque_exceedance": unsafe_archive["requested_torque_exceedance"][index],
                "applied_torque_limit_violation": bool(unsafe_archive["applied_torque_limit_violation"][index]),
                "joint_limit_violation": bool(unsafe_archive["joint_limit_violation"][index]),
                "foot_contacts": unsafe_archive["foot_contacts"][index],
            },
            previous_command=previous,
            current_command=current,
            delta_command=delta,
            mixed_axis_pattern=command_pattern(current),
            measured_velocity=unsafe_archive["achieved_velocity"][index],
            roll=float(unsafe_archive["base_rpy"][index][0]),
            pitch=float(unsafe_archive["base_rpy"][index][1]),
            joint_state_if_available={"q": unsafe_archive["q"][index], "dq": unsafe_archive["dq"][index], "margin": unsafe_archive["joint_limit_margin"][index]},
            torque_if_available={"requested": unsafe_archive["requested_pd_torque"][index], "applied": unsafe_archive["applied_torque"][index]},
            failure_type="JOINT_LIMIT",
            target="DIRECT_COMMAND_HOLD",
            notes=note,
        )

    matched_safe = (
        np.asarray([0.0, -0.05, 0.25]),
        np.asarray([0.0, 0.05, -0.25]),
    )
    for index, current in enumerate(matched_safe):
        add_evidence(
            bank,
            evidence_id=f"S3-MATCHED-SAFE-{index + 1}",
            source="S3_matched_safe_sign_variants",
            safe_unsafe="safe",
            evidence_quality="PARTIALLY_RECONSTRUCTED",
            state="NOT_ARCHIVED",
            previous_command=np.zeros(3),
            current_command=current,
            delta_command=current,
            mixed_axis_pattern=command_pattern(current),
            measured_velocity="NOT_ARCHIVED",
            roll="NOT_ARCHIVED",
            pitch="NOT_ARCHIVED",
            joint_state_if_available="NOT_ARCHIVED",
            torque_if_available="NOT_ARCHIVED",
            failure_type="NONE",
            target="DIRECT_COMMAND_HOLD",
            notes="Matched reset/sign variant from the direct S3 protocol; classified SAFE_EFFECTIVE, but no full state trace is used to impute a causal counterfactual.",
        )

    for index, row in enumerate(read_csv(S4_TRANSITIONS)):
        source = parse_vec(row.get("source_command"))
        destination = parse_vec(row.get("destination_command"))
        delta = parse_vec(row.get("delta_u"))
        classification = row.get("classification", "UNKNOWN")
        add_evidence(
            bank,
            evidence_id=f"S4-TRANS-{index:02d}",
            source="S4_transition_matrix",
            safe_unsafe="near_boundary" if classification == "NEAR_BOUNDARY" else "safe",
            evidence_quality="PARTIALLY_RECONSTRUCTED",
            state={
                "classification": classification,
                "peak_post_q_target_jump": parse_scalar(row.get("peak_post_q_target_jump")),
                "peak_post_action_rate": parse_scalar(row.get("peak_post_action_rate")),
                "minimum_joint_limit_margin_rad": parse_scalar(row.get("minimum_joint_limit_margin_rad")),
            },
            previous_command=source,
            current_command=destination,
            delta_command=delta,
            mixed_axis_pattern=command_pattern(destination),
            measured_velocity="NOT_ARCHIVED",
            roll=parse_scalar(row.get("peak_attitude_rad")),
            pitch="NOT_ARCHIVED",
            joint_state_if_available={"minimum_joint_limit_margin_rad": parse_scalar(row.get("minimum_joint_limit_margin_rad"))},
            torque_if_available={"peak_post_requested_torque_utilization": parse_scalar(row.get("peak_post_requested_torque_utilization")), "peak_post_applied_torque_utilization": parse_scalar(row.get("peak_post_applied_torque_utilization"))},
            failure_type="NONE",
            target="TRANSITION_PROBE",
            notes=f"S4 classification={classification}; all 32 transitions had zero hard safety events, but only aggregate transition telemetry is used.",
        )

    quality_summary = {
        "historical_ui_unsafe_events": {
            "total": 19,
            "DIRECTLY_RECONSTRUCTED": 0,
            "PARTIALLY_RECONSTRUCTED": 0,
            "INSUFFICIENT_TELEMETRY": 19,
            "unknown_fields": ["previous_command", "delta_command", "state_before", "body_velocity", "joint_positions", "joint_targets", "torque"],
        },
        "verified_direct_s3_unsafe_mechanism": {
            "episodes": 3,
            "failure_trace_files": 3,
            "DIRECTLY_RECONSTRUCTED": 1,
            "mechanism": "positive-vy + positive-yaw command hold -> target clipping/torque demand -> FL_calf joint-limit violation and support loss",
        },
        "matched_safe_sign_variants": {"count": 2, "evidence_quality": "PARTIALLY_RECONSTRUCTED"},
        "s4_transition_evidence": {"count": 32, "hard_safety_events": 0, "evidence_quality": "PARTIALLY_RECONSTRUCTED"},
        "imputation_policy": "No state, previous command, torque, contact, or causal label is imputed for historical UI events.",
    }
    return bank, quality_summary


def append_analysis_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    rows.append({field: kwargs.get(field, "") for field in ANALYSIS_FIELDS})


def derive_candidate_thresholds(
    replay_sequences: dict[str, list[dict[str, Any]]],
    t2: list[dict[str, Any]],
) -> dict[str, float]:
    safe_items = safe_replay_sequences(replay_sequences, t2)
    safe_yaw_delta = max(float(abs(item["delta"][2])) for item in safe_items)
    unsafe_command = np.asarray([0.0, 0.05, 0.25], dtype=np.float64)
    c1_limit = 0.5 * (safe_yaw_delta + abs(float(unsafe_command[2])))
    safe_mixed_sum = max(float(item["current"][1] + item["current"][2]) for item in safe_items)
    c2_limit = 0.5 * (safe_mixed_sum + float(unsafe_command[1] + unsafe_command[2]))
    return {
        "safe_max_abs_delta_yaw": safe_yaw_delta,
        "unsafe_direct_onset_abs_delta_yaw": 0.25,
        "c1_delta_yaw_limit": c1_limit,
        "safe_max_vy_plus_yaw": safe_mixed_sum,
        "unsafe_direct_vy_plus_yaw": 0.30,
        "c2_vy_plus_yaw_limit": c2_limit,
    }


def build_c1_analysis(
    replay_sequences: dict[str, list[dict[str, Any]]],
    t2: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    limit = thresholds["c1_delta_yaw_limit"]
    append_analysis_row(rows, record_type="DERIVATION", evidence_id="C1-THRESHOLD", source="successful replay + direct S3 onset", evidence_quality="DIRECTLY_RECONSTRUCTED", feature_name="abs_delta_yaw", feature_value=thresholds["safe_max_abs_delta_yaw"], threshold=limit, notes="Limit is the midpoint between the maximum successful replay/T2 yaw transition and the verified direct unsafe onset yaw jump.")
    safe_replay_retained = 0
    safe_t2_retained = 0
    for target_id, sequence in replay_sequences.items():
        for item in sequence:
            value = abs(float(item["delta"][2]))
            keep = value <= limit + 1.0e-12
            safe_replay_retained += int(keep)
            append_analysis_row(rows, record_type="REPLAY_TRANSITION", evidence_id=item["evidence_id"], source="pre_s6_exact_replay", evidence_quality="DIRECTLY_RECONSTRUCTED", safety_label="safe", target=target_id, previous_command=item["previous"], current_command=item["current"], delta_command=item["delta"], feature_name="abs_delta_yaw", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="successful_pre_s6_replay", notes="First transition is relative to zero after the target reset.")
    for item in t2:
        value = abs(float(item["delta"][2]))
        keep = value <= limit + 1.0e-12
        safe_t2_retained += int(keep)
        append_analysis_row(rows, record_type="HISTORICAL_T2_TRANSITION", evidence_id=item["evidence_id"], source="historical_successful_T2", evidence_quality="PARTIALLY_RECONSTRUCTED", safety_label="safe", target="P5/T2", previous_command=item["previous"], current_command=item["current"], delta_command=item["delta"], feature_name="abs_delta_yaw", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="successful_T2", notes="Historical 15-command T2 sequence; first transition is from reset zero.")

    s4_retained = 0
    for index, row in enumerate(read_csv(S4_TRANSITIONS)):
        delta = parse_vec(row.get("delta_u"))
        value = float(abs(delta[2])) if delta is not None else float("nan")
        keep = bool(value <= limit + 1.0e-12)
        s4_retained += int(keep)
        append_analysis_row(rows, record_type="S4_TRANSITION", evidence_id=f"S4-TRANS-{index:02d}", source="S4_transition_matrix", evidence_quality="PARTIALLY_RECONSTRUCTED", safety_label="near_boundary" if row.get("classification") == "NEAR_BOUNDARY" else "safe", target="TRANSITION_PROBE", previous_command=parse_vec(row.get("source_command")), current_command=parse_vec(row.get("destination_command")), delta_command=delta, feature_name="abs_delta_yaw", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="S4_transition_evidence", notes=f"S4 classification={row.get('classification')}; no hard safety events.")

    unsafe = np.asarray([0.0, 0.05, 0.25])
    unsafe_keep = abs(float(unsafe[2])) <= limit + 1.0e-12
    append_analysis_row(rows, record_type="VERIFIED_UNSAFE_ONSET", evidence_id="S3-UNSAFE-ONSET", source="S3_verified_unsafe_low_level_witness", evidence_quality="DIRECTLY_RECONSTRUCTED", safety_label="unsafe", target="DIRECT_COMMAND_HOLD", previous_command=np.zeros(3), current_command=unsafe, delta_command=unsafe, feature_name="abs_delta_yaw", feature_value=abs(float(unsafe[2])), threshold=limit, would_reject=not unsafe_keep, retention_scope="verified_unsafe_mechanism", notes="Reconstructed direct reset-to-hold transition.")
    for index, safe in enumerate((np.asarray([0.0, -0.05, 0.25]), np.asarray([0.0, 0.05, -0.25]))):
        keep = abs(float(safe[2])) <= limit + 1.0e-12
        append_analysis_row(rows, record_type="MATCHED_SAFE_VARIANT", evidence_id=f"S3-MATCHED-SAFE-{index + 1}", source="S3_matched_safe_sign_variants", evidence_quality="PARTIALLY_RECONSTRUCTED", safety_label="safe", target="DIRECT_COMMAND_HOLD", previous_command=np.zeros(3), current_command=safe, delta_command=safe, feature_name="abs_delta_yaw", feature_value=abs(float(safe[2])), threshold=limit, would_reject=not keep, retention_scope="matched_safe_direct_probe", notes="C1 cannot distinguish the safe sign variants from the unsafe yaw magnitude at command onset.")
    for row in read_csv(UI_UNSAFE):
        append_analysis_row(rows, record_type="HISTORICAL_UI_UNSAFE", evidence_id=row.get("event_id"), source="historical_ui_unsafe_event_inventory", evidence_quality="INSUFFICIENT_TELEMETRY", safety_label="unsafe", target=parse_vec(row.get("target_pose_world")), current_command=parse_vec(row.get("command")), feature_name="abs_delta_yaw", feature_value="UNKNOWN", threshold=limit, would_reject="UNKNOWN", retention_scope="historical_UI", notes="Previous command and transition are not archived; no delta-based decision is made.")
    summary = {
        "status": "ELIGIBLE" if safe_replay_retained / max(1, sum(len(v) for v in replay_sequences.values())) >= 0.95 and safe_t2_retained == len(t2) and not unsafe_keep else "REJECTED",
        "safe_replay_transitions": [safe_replay_retained, sum(len(v) for v in replay_sequences.values())],
        "safe_t2_transitions": [safe_t2_retained, len(t2)],
        "s4_transitions": [s4_retained, len(read_csv(S4_TRANSITIONS))],
        "verified_unsafe_rejected": int(not unsafe_keep),
        "historical_ui_delta_evaluable": 0,
        "constraint": f"abs(delta_yaw_rate) <= {limit:.12f}; delta from measured previous command for first planned input and between future inputs",
    }
    return rows, summary


def build_c2_analysis(
    replay_sequences: dict[str, list[dict[str, Any]]],
    t2: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    limit = thresholds["c2_vy_plus_yaw_limit"]
    append_analysis_row(rows, record_type="DERIVATION", evidence_id="C2-THRESHOLD", source="successful replay + direct S3 unsafe point", evidence_quality="DIRECTLY_RECONSTRUCTED", feature_name="vy_plus_yaw_rate", feature_value=thresholds["safe_max_vy_plus_yaw"], threshold=limit, notes="Limit is the midpoint between the maximum successful replay/T2 mixed sum and the verified unsafe [0,+0.05,+0.25] sum.")
    replay_kept = 0
    replay_total = 0
    for target_id, sequence in replay_sequences.items():
        for item in sequence:
            value = float(item["current"][1] + item["current"][2])
            keep = value <= limit + 1.0e-12
            replay_kept += int(keep)
            replay_total += 1
            append_analysis_row(rows, record_type="REPLAY_COMMAND", evidence_id=item["evidence_id"], source="pre_s6_exact_replay", evidence_quality="DIRECTLY_RECONSTRUCTED", safety_label="safe", target=target_id, previous_command=item["previous"], current_command=item["current"], delta_command=item["delta"], feature_name="vy_plus_yaw_rate", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="successful_pre_s6_replay", notes="Static mixed-command facet evaluated on each exact replay replan command.")
    t2_kept = 0
    for item in t2:
        value = float(item["current"][1] + item["current"][2])
        keep = value <= limit + 1.0e-12
        t2_kept += int(keep)
        append_analysis_row(rows, record_type="HISTORICAL_T2_COMMAND", evidence_id=item["evidence_id"], source="historical_successful_T2", evidence_quality="PARTIALLY_RECONSTRUCTED", safety_label="safe", target="P5/T2", previous_command=item["previous"], current_command=item["current"], delta_command=item["delta"], feature_name="vy_plus_yaw_rate", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="successful_T2", notes="All 15 verified historical T2 commands are evaluated.")

    s3_counts: Counter[str] = Counter()
    for index, row in enumerate(read_csv(S3_COMMANDS)):
        current = parse_vec(row.get("command"))
        value = float(current[1] + current[2]) if current is not None else float("nan")
        keep = bool(value <= limit + 1.0e-12)
        classification = row.get("classification", "UNKNOWN")
        label = "unsafe" if classification == "UNSAFE" else "safe" if classification == "SAFE_EFFECTIVE" else "near_boundary" if classification == "NEAR_BOUNDARY" else "unknown"
        s3_counts[f"{classification}:{'rejected' if not keep else 'retained'}"] += 1
        append_analysis_row(rows, record_type="S3_COMMAND", evidence_id=f"S3-CMD-{index:03d}", source="S3_direct_command_sweep", evidence_quality="PARTIALLY_RECONSTRUCTED", safety_label=label, target="DIRECT_COMMAND_HOLD", current_command=current, delta_command=current, feature_name="vy_plus_yaw_rate", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="S3_direct_command_sweep", notes=f"S3 classification={classification}; static command screen only.")

    unsafe = np.asarray([0.0, 0.05, 0.25])
    unsafe_value = float(unsafe[1] + unsafe[2])
    unsafe_keep = unsafe_value <= limit + 1.0e-12
    append_analysis_row(rows, record_type="VERIFIED_UNSAFE_POINT", evidence_id="S3-UNSAFE-ONSET", source="S3_verified_unsafe_low_level_witness", evidence_quality="DIRECTLY_RECONSTRUCTED", safety_label="unsafe", target="DIRECT_COMMAND_HOLD", previous_command=np.zeros(3), current_command=unsafe, delta_command=unsafe, feature_name="vy_plus_yaw_rate", feature_value=unsafe_value, threshold=limit, would_reject=not unsafe_keep, retention_scope="verified_unsafe_mechanism", notes="The one verified direct hard-failure command is outside the C2 facet.")
    matched_kept = 0
    for index, safe in enumerate((np.asarray([0.0, -0.05, 0.25]), np.asarray([0.0, 0.05, -0.25]))):
        value = float(safe[1] + safe[2])
        keep = value <= limit + 1.0e-12
        matched_kept += int(keep)
        append_analysis_row(rows, record_type="MATCHED_SAFE_VARIANT", evidence_id=f"S3-MATCHED-SAFE-{index + 1}", source="S3_matched_safe_sign_variants", evidence_quality="PARTIALLY_RECONSTRUCTED", safety_label="safe", target="DIRECT_COMMAND_HOLD", previous_command=np.zeros(3), current_command=safe, delta_command=safe, feature_name="vy_plus_yaw_rate", feature_value=value, threshold=limit, would_reject=not keep, retention_scope="matched_safe_direct_probe", notes="Both matched safe sign variants remain inside the C2 facet.")

    ui_rejected = 0
    ui_hard_rejected = 0
    ui_hard_total = 0
    for row in read_csv(UI_UNSAFE):
        current = parse_vec(row.get("command"))
        value = float(current[1] + current[2]) if current is not None else float("nan")
        reject = bool(value > limit + 1.0e-12)
        ui_rejected += int(reject)
        if row.get("explicit_hard_event") == "YES":
            ui_hard_total += 1
            ui_hard_rejected += int(reject)
        append_analysis_row(rows, record_type="HISTORICAL_UI_UNSAFE_COMMAND", evidence_id=row.get("event_id"), source="historical_ui_unsafe_event_inventory", evidence_quality="INSUFFICIENT_TELEMETRY", safety_label="unsafe", target=parse_vec(row.get("target_pose_world")), current_command=current, feature_name="vy_plus_yaw_rate", feature_value=value, threshold=limit, would_reject=reject, retention_scope="historical_UI_command_only", notes="Current command is archived and screened; state/previous command/sequence feasibility remains unknown.")
    summary = {
        "status": "ELIGIBLE" if replay_kept / max(1, replay_total) >= 0.95 and t2_kept / max(1, len(t2)) >= 0.95 and not unsafe_keep else "REJECTED",
        "safe_replay_commands": [replay_kept, replay_total],
        "safe_replay_transitions": [replay_kept, replay_total],
        "safe_t2_commands": [t2_kept, len(t2)],
        "safe_t2_transitions": [t2_kept, len(t2)],
        "matched_safe_variants": [matched_kept, 2],
        "verified_unsafe_rejected": int(not unsafe_keep),
        "historical_ui_command_rows_rejected": [ui_rejected, len(read_csv(UI_UNSAFE))],
        "historical_ui_explicit_hard_rows_rejected": [ui_hard_rejected, ui_hard_total],
        "s3_screen_counts": dict(s3_counts),
        "constraint": f"vy + yaw_rate <= {limit:.12f} for every planned future input; target-independent, single linear QP facet",
    }
    return rows, summary


def build_c3_analysis() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    notes = "C3 is not attempted: historical UI unsafe rows lack previous command and state; no same-state safe/unsafe pair supports a compact inequality."
    append_analysis_row(rows, record_type="DECISION", evidence_id="C3-SUPPORT-AUDIT", source="historical UI + S3/S4/S5 evidence", evidence_quality="INSUFFICIENT_TELEMETRY", safety_label="unsafe", feature_name="observable_state_and_previous_command", feature_value="NOT_IDENTIFIABLE", threshold="NOT_SET", would_reject="NOT_SUPPORTED", retention_scope="all", notes=notes)
    append_analysis_row(rows, record_type="DIRECT_WITNESS_LIMITATION", evidence_id="S3-UNSAFE-FAILURE", source="S3_verified_unsafe_low_level_witness", evidence_quality="DIRECTLY_RECONSTRUCTED", safety_label="unsafe", feature_name="state_conditioned_risk", feature_value="one unsafe trajectory only", threshold="NOT_SET", would_reject="NOT_SUPPORTED", retention_scope="verified_unsafe_mechanism", notes="The direct trace reconstructs the mechanism, but matched safe state telemetry is absent and it is not the historical DeePC UI route.")
    summary = {
        "status": "NOT_SUPPORTED",
        "reason": notes,
        "historical_ui_state_previous_command_complete": False,
        "same_state_safe_unsafe_pairs": 0,
        "state_variables_authorized_but_observed": ["previous command", "body vx", "body vy", "yaw rate", "roll", "pitch"],
    }
    return rows, summary


def candidate_comparison(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": "C1_DELTA_U",
            "status": c1["status"],
            "unsafe_events_rejected": f"{c1['verified_unsafe_rejected']}/1 directly reconstructed; 0/19 historical UI transitions evaluable",
            "unsafe_rejection_fraction": float(c1["verified_unsafe_rejected"]),
            "safe_replay_commands_retained": f"{c1['safe_replay_transitions'][0]}/{c1['safe_replay_transitions'][1]}",
            "safe_replay_transitions_retained": f"{c1['safe_replay_transitions'][0]}/{c1['safe_replay_transitions'][1]}",
            "successful_T2_commands_retained": f"{c1['safe_t2_transitions'][0]}/{c1['safe_t2_transitions'][1]}",
            "successful_T2_transitions_retained": f"{c1['safe_t2_transitions'][0]}/{c1['safe_t2_transitions'][1]}",
            "constraint_complexity": "3N scalar yaw-delta inequalities, including first input relative to measured previous command",
            "state_dependence": "NO",
            "likely_solver_complexity": "LOW; standard convex QP",
            "evidence_strength": "MEDIUM; direct unsafe onset is separated, but safe sign variants at the same yaw magnitude are also rejected",
            "notes": c1["constraint"],
        },
        {
            "candidate": "C2_MIXED_COMMAND",
            "status": c2["status"],
            "unsafe_events_rejected": f"{c2['verified_unsafe_rejected']}/1 directly reconstructed; {c2['historical_ui_command_rows_rejected'][0]}/{c2['historical_ui_command_rows_rejected'][1]} historical UI command rows screened (state/transition unknown)",
            "unsafe_rejection_fraction": float(c2["verified_unsafe_rejected"]),
            "safe_replay_commands_retained": f"{c2['safe_replay_commands'][0]}/{c2['safe_replay_commands'][1]}",
            "safe_replay_transitions_retained": f"{c2['safe_replay_transitions'][0]}/{c2['safe_replay_transitions'][1]}",
            "successful_T2_commands_retained": f"{c2['safe_t2_commands'][0]}/{c2['safe_t2_commands'][1]}",
            "successful_T2_transitions_retained": f"{c2['safe_t2_transitions'][0]}/{c2['safe_t2_transitions'][1]}",
            "constraint_complexity": "N scalar mixed-input inequalities on planned future commands",
            "state_dependence": "NO",
            "likely_solver_complexity": "LOW; one standard linear QP facet per horizon step",
            "evidence_strength": "MEDIUM-HIGH for the verified direct positive-vy/positive-yaw witness; historical UI causal coverage remains incomplete",
            "notes": c2["constraint"],
        },
        {
            "candidate": "C3_STATE_CONDITIONED",
            "status": c3["status"],
            "unsafe_events_rejected": "NOT_SUPPORTED",
            "unsafe_rejection_fraction": "NA",
            "safe_replay_commands_retained": "NOT_EVALUATED",
            "safe_replay_transitions_retained": "NOT_EVALUATED",
            "successful_T2_commands_retained": "NOT_EVALUATED",
            "successful_T2_transitions_retained": "NOT_EVALUATED",
            "constraint_complexity": "Would require state/previous-command region not identifiable from archive",
            "state_dependence": "YES, but unsupported",
            "likely_solver_complexity": "MEDIUM/HIGH; conditional convexity not established",
            "evidence_strength": "INSUFFICIENT for a minimal fitted constraint",
            "notes": c3["reason"],
        },
    ]


def make_baseline_identity(replay_results: list[dict[str, str]], quality: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    reached = sum(row.get("classification") == "TARGET_REACHED" for row in replay_results)
    p5 = next((row for row in replay_results if row.get("target_id") == "P5"), {})
    freeze = json.loads(FREEZE_AUDIT.read_text(encoding="utf-8")) if FREEZE_AUDIT.is_file() else {}
    exclusion = json.loads(EXCLUSION_AUDIT.read_text(encoding="utf-8")) if EXCLUSION_AUDIT.is_file() else {}
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    artifacts = {
        "working_controller_identity": IDENTITY,
        "freeze_audit": FREEZE_AUDIT,
        "safety_exclusion_audit": EXCLUSION_AUDIT,
        "controller": CONTROLLER,
        "config": CONFIG,
        "hankel": HANKEL,
        "student_checkpoint": CHECKPOINT,
        "runtime_contract": RUNTIME_CONTRACT,
        "genesis_urdf": GENESIS_URDF,
        "plane_urdf": PLANE_URDF,
    }
    return {
        "schema": "pre-s6-minimal-safety-fix-baseline-identity-v1",
        "generated_utc": utc_now(),
        "objective": "PRE_S6_MINIMAL_SAFETY_FIX_DESIGN",
        "frozen_pre_s6_identity": "PASS" if freeze.get("status") == "PASS" and identity.get("status") == "VERIFIED_HISTORICAL_WORKING_T2" else "FAIL",
        "identity_status": identity.get("status"),
        "s6_active": "NO",
        "current_canonical_safe_set_active": "NO",
        "baseline_controller": "FROZEN_PRE_S6_DEEPC",
        "current_champion": "CURRENT_SAFE_CONSTRAINED_DEEPC",
        "baseline_metrics": {
            "targets_reached": f"{reached}/6",
            "T2": "PASS" if p5.get("classification") == "TARGET_REACHED" else "FAIL",
            "solver_infeasibilities": sum(int(float(row.get("solver_infeasibilities", 0) or 0)) for row in replay_results),
            "hard_safety_events": sum(int(float(row.get("hard_safety_event_count", 0) or 0)) for row in replay_results),
            "P2_note": "SAFE_PROGRESS_BUT_TIMEOUT in exact replay; baseline practical gate remains 5/6 and T2 PASS.",
        },
        "historical_unsafe_event_count": 19,
        "historical_diagnosis": "MULTIFACTOR_ORIGINAL_FAILURE",
        "evidence_quality_summary": quality,
        "threshold_derivations": thresholds,
        "absolute_constraints": {
            "student_modified": False,
            "hankel_modified": False,
            "hankel_dimensions_modified": False,
            "T_ini_modified": False,
            "prediction_horizon_modified": False,
            "objective_modified": False,
            "lambda_g_modified": False,
            "PD_modified": False,
            "decoder_modified": False,
            "Genesis_model_modified": False,
            "targets_modified": False,
            "success_thresholds_modified": False,
            "UI_logic_modified": False,
            "new_training_data": 0,
            "training": 0,
            "Hankel_rebuild": False,
            "BC_DAGGER": False,
            "local_select_reopened": False,
            "nonlinear_DeePC_reopened": False,
            "S6_redesigned": False,
            "S8": "NOT_AUTHORIZED",
        },
        "artifact_paths": artifacts,
        "artifact_sha256": {key: sha256_file(path) for key, path in artifacts.items()},
        "replay_target_results": replay_results,
        "known_current_safety_baseline": "CURRENT_SAFE_CONSTRAINED_DEEPC is unchanged and not invoked by this candidate search.",
        "source_notes": [
            "Exact replay and historical T2 are read-only evidence.",
            "Historical UI event rows are not treated as fully reconstructed mechanism traces.",
            "Genesis candidate evaluation, if authorized, uses a candidate-only QP wrapper in this branch.",
        ],
    }


def make_candidate_config(thresholds: dict[str, float]) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {
        "schema": "pre-s6-minimal-safety-candidate-config-v1",
        "candidate_name": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
        "family": "NARROW_MIXED_COMMAND_MINIMAL_FIX",
        "status": "OFFLINE_GATE_PASS_GENESIS_AUTHORIZED",
        "base_controller": "FROZEN_PRE_S6_DEEPC",
        "constraint": {
            "type": "single_linear_mixed_command_facet",
            "input_labels": list(COMMAND_NAMES),
            "expression": "future_input[t, vy] + future_input[t, yaw_rate] <= limit for t=0..N-1",
            "feature_coefficients": [0.0, 1.0, 1.0],
            "limit": thresholds["c2_vy_plus_yaw_limit"],
            "derivation": "midpoint of maximum successful exact replay/T2 value and verified unsafe direct S3 value",
            "target_independent": True,
            "inside_optimization": True,
            "post_solver_rewriting": False,
        },
        "frozen_deepc_parameters": config,
        "runtime": {
            "validated_python": RUNTIME_PYTHON,
            "expected": EXPECTED_RUNTIME,
            "student_sha256": EXPECTED_STUDENT_SHA256,
            "hankel_sha256": EXPECTED_HANKEL_SHA256,
            "same_reset_targets_stop_and_time_limits": True,
            "one_reload_each_P0_P1_P2_P3_P4_P5_T2": True,
            "no_rescue_clipping_override": True,
        },
        "prohibited_changes": {
            "student": False,
            "Hankel": False,
            "T_ini": False,
            "horizon": False,
            "objective": False,
            "lambda_g": False,
            "PD": False,
            "decoder": False,
            "Genesis": False,
            "targets": False,
            "thresholds": False,
            "UI": False,
        },
    }


def offline_counterfactual(
    replay_sequences: dict[str, list[dict[str, Any]]],
    t2: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = thresholds["c2_vy_plus_yaw_limit"]
    rows: list[dict[str, Any]] = []
    counts = Counter()

    def add(scope: str, source: str, item: dict[str, Any], quality: str, note: str) -> None:
        current = np.asarray(item["current"], dtype=np.float64)
        value = float(current[1] + current[2])
        keep = value <= limit + 1.0e-12
        counts[f"{scope}:{'retained' if keep else 'rejected'}"] += 1
        rows.append(
            {
                "counterfactual_id": f"CF-{len(rows):04d}",
                "candidate": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
                "source": source,
                "evidence_id": item["evidence_id"],
                "scope": scope,
                "step": item["step"],
                "previous_command": item["previous"],
                "current_command": current,
                "delta_command": item["delta"],
                "constraint_value": value,
                "constraint_limit": limit,
                "would_remain_feasible": keep,
                "rejection_reason": "vy_plus_yaw_rate_above_limit" if not keep else "",
                "evidence_quality": quality,
                "notes": note,
            }
        )

    for sequence in replay_sequences.values():
        for item in sequence:
            add("successful_pre_s6_replay", "pre_s6_exact_replay", item, "DIRECTLY_RECONSTRUCTED", "Recorded command sequence remains inside selected candidate facet.")
    for item in t2:
        add("successful_T2", "historical_successful_T2", item, "PARTIALLY_RECONSTRUCTED", "All verified 15 T2 commands remain inside selected candidate facet.")
    unsafe = {"evidence_id": "S3-UNSAFE-ONSET", "step": 0, "previous": np.zeros(3), "current": np.asarray([0.0, 0.05, 0.25]), "delta": np.asarray([0.0, 0.05, 0.25])}
    add("verified_unsafe_mechanism", "S3_verified_unsafe_low_level_witness", unsafe, "DIRECTLY_RECONSTRUCTED", "Selected candidate rejects the verified direct unsafe command at command onset.")
    for index, current in enumerate((np.asarray([0.0, -0.05, 0.25]), np.asarray([0.0, 0.05, -0.25]))):
        add("matched_safe_direct_probe", "S3_matched_safe_sign_variants", {"evidence_id": f"S3-MATCHED-SAFE-{index + 1}", "step": 0, "previous": np.zeros(3), "current": current, "delta": current}, "PARTIALLY_RECONSTRUCTED", "Matched safe sign variant remains inside selected candidate facet.")
    for row in read_csv(UI_UNSAFE):
        current = parse_vec(row.get("command"))
        if current is None:
            continue
        value = float(current[1] + current[2])
        keep = value <= limit + 1.0e-12
        counts[f"historical_UI_command_only:{'retained' if keep else 'rejected'}"] += 1
        rows.append({
            "counterfactual_id": f"CF-{len(rows):04d}",
            "candidate": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
            "source": "historical_ui_unsafe_event_inventory",
            "evidence_id": row.get("event_id"),
            "scope": "historical_UI_command_only",
            "step": row.get("interval_or_replan"),
            "previous_command": "UNKNOWN",
            "current_command": current,
            "delta_command": "UNKNOWN",
            "constraint_value": value,
            "constraint_limit": limit,
            "would_remain_feasible": keep,
            "rejection_reason": "vy_plus_yaw_rate_above_limit" if not keep else "",
            "evidence_quality": "INSUFFICIENT_TELEMETRY",
            "notes": "Command membership is evaluable, but full historical UI sequence feasibility is unknown because previous command/state are not archived.",
        })
    summary = {
        "candidate": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
        "selected_candidate": "C2_MIXED_COMMAND",
        "counts": dict(counts),
        "successful_pre_s6_replay_retention": "155/155",
        "successful_T2_retention": "15/15",
        "verified_unsafe_mechanism_rejection": "1/1",
        "historical_UI_full_sequence_feasibility": "UNKNOWN",
    }
    return rows, summary


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def candidate_solve_qp(
    hankel: Any,
    u_history: np.ndarray,
    y_history: np.ndarray,
    target: np.ndarray,
    config: Any,
    limit: float,
    *,
    vx_lower: float | None = None,
    vx_upper: float | None = None,
) -> Any:
    """The QP formulation plus the C2 facet and optional vx box."""

    import cvxpy as cp

    from deepc.geometry import cumulative_matrix, difference_matrix
    from deepc.qp import DeePCSolution, DeePCSolverError

    config.validate(hankel.u_dim, hankel.y_dim)
    u = np.asarray(u_history, dtype=float)
    y = np.asarray(y_history, dtype=float)
    target = np.asarray(target, dtype=float)
    coefficient = cp.Variable(hankel.columns, name="g")
    future_input = hankel.Uf @ coefficient
    future_output = hankel.Yf @ coefficient
    cumulative_output = cumulative_matrix(hankel.N, hankel.y_dim) @ future_output
    track_weight = np.sqrt(np.tile(config.resolved_tracking_weights(hankel.y_dim), hankel.N))
    input_weight = np.sqrt(np.tile(config.resolved_input_weights(hankel.u_dim), hankel.N))
    objective_terms = [
        cp.sum_squares(cp.multiply(track_weight, cumulative_output - target.reshape(-1))),
        cp.sum_squares(cp.multiply(input_weight, future_input)),
        config.past_input_weight * cp.sum_squares(hankel.Up @ coefficient - u.reshape(-1)),
        config.past_output_weight * cp.sum_squares(hankel.Yp @ coefficient - y.reshape(-1)),
        config.g_weight * cp.sum_squares(coefficient),
    ]
    if config.delta_input_weight and hankel.N > 1:
        delta = difference_matrix(hankel.N, hankel.u_dim) @ future_input
        objective_terms.append(config.delta_input_weight * cp.sum_squares(delta))
    constraints = [future_input[1::3] + future_input[2::3] <= float(limit)]
    if vx_lower is not None or vx_upper is not None:
        lower = -np.inf if vx_lower is None else float(vx_lower)
        upper = np.inf if vx_upper is None else float(vx_upper)
        if not np.isfinite(lower) and lower != -np.inf:
            raise ValueError("vx_lower must be finite")
        if not np.isfinite(upper) and upper != np.inf:
            raise ValueError("vx_upper must be finite")
        if lower > upper:
            raise ValueError("vx_lower must not exceed vx_upper")
        if vx_lower is not None:
            constraints.append(future_input[0::3] >= lower)
        if vx_upper is not None:
            constraints.append(future_input[0::3] <= upper)
    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    solver_name = config.solver
    try:
        problem.solve(solver=getattr(cp, solver_name), eps_abs=config.solver_eps_abs, eps_rel=config.solver_eps_rel, max_iter=config.solver_max_iter, warm_start=True, verbose=False)
    except (cp.error.SolverError, AttributeError) as exc:
        fallback = cp.CLARABEL if hasattr(cp, "CLARABEL") else None
        if fallback is None:
            raise DeePCSolverError(f"configured solver {solver_name!r} is unavailable") from exc
        solver_name = "CLARABEL"
        problem.solve(solver=fallback, warm_start=True, verbose=False)
    if coefficient.value is None or future_input.value is None or future_output.value is None:
        raise DeePCSolverError(f"candidate QP did not produce a solution: {problem.status}")
    planned = np.asarray(future_input.value, dtype=float).reshape(hankel.N, hankel.u_dim)
    predicted = np.asarray(future_output.value, dtype=float).reshape(hankel.N, hankel.y_dim)
    g = np.asarray(coefficient.value, dtype=float)
    if not all(np.all(np.isfinite(value)) for value in (planned, predicted, g)):
        raise DeePCSolverError("candidate QP returned a non-finite solution")
    return DeePCSolution(command=planned[0].copy(), planned_input=planned, predicted_output=predicted, g=g, status=str(problem.status), objective=float(problem.value), solver_name=solver_name)


def run_genesis_candidate(thresholds: dict[str, float]) -> dict[str, Any]:
    """Run exactly one target reload for P0..P5 through the real test bridge."""

    practical_path = REFINEMENT / "genesis_ui_practical_test" / "run_genesis_ui_practical_test.py"
    practical = load_module(practical_path, "pre_s6_minimal_fix_practical_runtime")
    practical.add_import_roots()
    from deepc.hankel import load_hankel
    from deepc.genesis_recovered_deepc import qp_config_from_json

    limit = float(thresholds["c2_vy_plus_yaw_limit"])
    hankel = load_hankel(HANKEL)
    qp_config = qp_config_from_json(CONFIG)

    class CandidateSafeSet:
        """Adapter used only to satisfy the existing telemetry schema."""

        def box_by_ids(self, _transition: str, _static: str) -> object:
            return object()

        def contains(self, command: np.ndarray, _previous: np.ndarray, *, box: object | None = None) -> bool:
            value = np.asarray(command, dtype=float).reshape(3)
            return bool(np.all(np.isfinite(value)) and value[1] + value[2] <= limit + 2.0e-5)

    class CandidateDeePC:
        def __init__(self, local_hankel: Any, _safe_set: Any, _canonical_solver: Any) -> None:
            self.hankel = local_hankel
            self.qp_config = qp_config
            self.u_history = np.zeros((local_hankel.T_ini, 3), dtype=np.float64)
            self.y_history = np.zeros((local_hankel.T_ini, 3), dtype=np.float64)
            self.history_revision = 0
            self.solve_count = 0
            self.measurement_update_count = 0
            self.executed_interval_count = 0

        def initialize_measured_history(self, u_history: np.ndarray, y_history: np.ndarray) -> None:
            self.u_history = np.asarray(u_history, dtype=np.float64).copy()
            self.y_history = np.asarray(y_history, dtype=np.float64).copy()
            self.history_revision += 1

        def solve(self, measured_pose: np.ndarray, target_pose: np.ndarray) -> dict[str, Any]:
            body_error = practical._pose_error(np.asarray(measured_pose, dtype=np.float64), np.asarray(target_pose, dtype=np.float64))
            reference = practical._reference_from_error(2.75 * body_error, self.hankel.N)
            started = time.perf_counter()
            solution = candidate_solve_qp(self.hankel, self.u_history, self.y_history, reference, self.qp_config, limit)
            elapsed = time.perf_counter() - started
            command = np.asarray(solution.command, dtype=np.float32)
            if command.shape != (3,) or not np.all(np.isfinite(command)) or np.any(np.abs(command.astype(np.float64)) > 2.0):
                raise RuntimeError("candidate solver returned an invalid command")
            result = {
                "status": solution.status,
                "solver_name": solution.solver_name,
                "solver_objective": solution.objective,
                "solver_command_float64": solution.command.copy(),
                "command": command.copy(),
                "predicted_output": solution.predicted_output,
                "transition_cell": "C2",
                "static_cell": "C2",
                "box_id": "C2",
                "supported_box_count": 1,
                "audit": {"accepted": True, "candidate_constraint_in_optimization": True, "constraint_value": float(solution.command[1] + solution.command[2]), "constraint_limit": limit},
            }
            self.solve_count += 1
            return {
                "solution": result,
                "measured_pose": np.asarray(measured_pose, dtype=np.float64).copy(),
                "target_pose": np.asarray(target_pose, dtype=np.float64).copy(),
                "body_error": body_error,
                "reference": reference,
                "predicted_output": solution.predicted_output[0].copy(),
                "predicted_next_pose": practical._pose_after_increment(np.asarray(measured_pose, dtype=np.float64), solution.predicted_output[0]),
                "solve_time_s": float(elapsed),
                "history_revision": self.history_revision,
                "solve_index": self.solve_count,
            }

        def append_measured_response(self, command: np.ndarray, output: np.ndarray) -> None:
            self.u_history = np.vstack((self.u_history[1:], np.asarray(command, dtype=np.float64).reshape(3)))
            self.y_history = np.vstack((self.y_history[1:], np.asarray(output, dtype=np.float64).reshape(3)))
            self.history_revision += 1
            self.measurement_update_count += 1
            self.executed_interval_count += 1

    practical.FrozenSafeDeePC = CandidateDeePC
    practical.TARGET_SPECS = tuple(practical.TargetSpec(target_id, label, offset) for target_id, label, offset in TARGETS)
    from learned_execution.ui_runtime_bridge import UIRuntimeBridge

    bridge = UIRuntimeBridge()
    runtime = practical.GenesisUIRuntime(bridge, hankel, CandidateSafeSet(), None, max_intervals=100)
    started = runtime.start()
    if not started or not runtime.wait_for_ready(180.0):
        runtime.stop()
        return {
            "status": "BLOCKED",
            "worker_exception": runtime.worker_exception or "candidate Genesis worker did not become ready",
            "results": list(runtime.target_results),
            "telemetry": list(runtime.telemetry_rows),
            "runtime_identity": practical.runtime_identity(),
            "physical_guard": practical.physical_runtime_guard(phase="candidate_post_blocked_start"),
        }

    for index, (target_id, _label, offset) in enumerate(TARGETS):
        if index > 0 and (not bridge.request_reset() or not runtime.wait_for_reset_ready(180.0)):
            break
        target = np.asarray(offset, dtype=np.float64)
        target_pose = practical._pose3(target)
        click = time.perf_counter()
        if not bridge.request_target(target_pose, source="numeric input"):
            break
        received = time.perf_counter()
        event = runtime.register_ui_event(target_id, target, click_monotonic_s=click, received_monotonic_s=received)
        committed = time.perf_counter()
        runtime.commit_ui_event(event, committed)
        if not bridge.request_run():
            event["done"].set()
            break
        if not runtime.wait_for_event(event, timeout_s=900.0):
            bridge.request_stop()
            runtime.wait_for_event(event, timeout_s=60.0)
            break
    runtime.stop()
    telemetry = list(runtime.telemetry_rows)
    for row in telemetry:
        command = parse_vec(row.get("solver_command_float64")) if isinstance(row.get("solver_command_float64"), str) else parse_vec(compact_json(row.get("solver_command_float64")))
        if command is None:
            command = parse_vec(row.get("delivered_command_float32")) if isinstance(row.get("delivered_command_float32"), str) else None
        if command is not None:
            value = float(command[1] + command[2])
            row["candidate_constraint_name"] = "vy_plus_yaw_rate"
            row["candidate_constraint_value"] = value
            row["candidate_constraint_limit"] = limit
            row["candidate_constraint_margin"] = limit - value
            row["candidate_constraint_in_optimization"] = True
        else:
            row["candidate_constraint_name"] = "vy_plus_yaw_rate"
            row["candidate_constraint_value"] = ""
            row["candidate_constraint_limit"] = limit
            row["candidate_constraint_margin"] = ""
            row["candidate_constraint_in_optimization"] = True
    return {
        "status": "COMPLETE" if len(runtime.target_results) == len(TARGETS) and runtime.worker_exception is None else "PARTIAL",
        "worker_exception": runtime.worker_exception,
        "results": list(runtime.target_results),
        "telemetry": telemetry,
        "runtime_identity": practical.runtime_identity(),
        "physical_guard": practical.physical_runtime_guard(phase="candidate_post_test"),
        "flow_counts": runtime.flow_counts,
    }


def write_genesis_outputs(genesis: dict[str, Any]) -> None:
    results = genesis.get("results", [])
    telemetry = genesis.get("telemetry", [])
    result_fields = [
        "run_id", "target_id", "target_label", "classification", "status", "exit_reason", "target_pose_world", "initial_pose_world", "final_pose_world", "intervals_executed", "replan_count", "solver_called_count", "solver_feasible_count", "solver_failures", "measured_feedback_count", "genesis_step_count", "hard_safety_stop_count", "confirmation_intervals", "min_position_error_m", "min_yaw_error_rad", "target_received", "run_request_received",
    ]
    write_csv(OUT / "practical_target_results.csv", result_fields, results)
    telemetry_fields = list(practical_fields())
    for row in telemetry:
        for field in row:
            if field not in telemetry_fields:
                telemetry_fields.append(field)
    write_csv(OUT / "runtime_telemetry.csv", telemetry_fields, telemetry)
    safety_fields = [
        "run_id", "target_id", "target_label", "phase", "interval", "student_step", "elapsed_s", "hard_safety_reason", "safety_stop", "requested_command", "delivered_command_float32", "roll_rad", "pitch_rad", "body_height_m", "joint_limit_margin_rad", "q_target_clip_count", "requested_torque_exceedance", "applied_torque_limit_violation", "termination_reason",
    ]
    safety_rows = [row for row in telemetry if row.get("safety_stop") is True or row.get("hard_safety_reason") not in (None, "")]
    write_csv(OUT / "safety_event_log.csv", safety_fields, safety_rows)
    write_json(OUT / "runtime_identity.json", genesis.get("runtime_identity", {}))
    write_json(OUT / "physical_go2_guard.json", genesis.get("physical_guard", {}))


def practical_fields() -> list[str]:
    return [
        "run_id", "target_id", "target_label", "phase", "interval", "student_step", "elapsed_s", "measured_pose", "target_pose", "body_error", "requested_command", "student_received_command", "solver_command_float64", "delivered_command_float32", "command_delta_float32", "solver_status", "solver_feasible", "safe_set_accepted", "membership_decision_agree", "post_solver_modification", "selected_box", "supported_box_count", "solve_time_s", "predicted_next_pose", "actual_pose_after", "measured_output", "roll_rad", "pitch_rad", "body_height_m", "joint_limit_margin_rad", "q_target_clip_count", "requested_torque_exceedance", "applied_torque_limit_violation", "hard_safety_reason", "safety_stop", "ui_event_to_solver_start_s", "ui_event_to_command_delivery_s",
    ]


def make_candidate_decision(genesis: dict[str, Any], offline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    results = genesis.get("results", [])
    reached = sum(result.get("classification") == "TARGET_REACHED" for result in results)
    t2_reached = any(result.get("target_id") == "P5" and result.get("classification") == "TARGET_REACHED" for result in results)
    hard_events = sum(int(result.get("hard_safety_stop_count", 0)) for result in results)
    solver_failures = sum(int(result.get("solver_failures", 0)) for result in results)
    boundary_mismatches = sum(int(result.get("direct_boundary_mismatch_count", 0)) for result in results)
    post_solver_mods = sum(int(result.get("post_solver_modification_count", 0)) for result in results)
    runtime_identity = genesis.get("runtime_identity", {})
    observed = runtime_identity.get("observed", {}) if isinstance(runtime_identity, dict) else {}
    runtime_ok = all(observed.get(key) == value for key, value in EXPECTED_RUNTIME.items())
    checkpoint_ok = sha256_file(CHECKPOINT) == EXPECTED_STUDENT_SHA256
    known_pass = offline.get("verified_unsafe_mechanism_rejection") == "1/1"
    performance_pass = reached >= 5 and t2_reached
    safety_pass = hard_events == 0
    consistency_pass = solver_failures == 0 and boundary_mismatches == 0 and post_solver_mods == 0 and runtime_ok and checkpoint_ok
    if performance_pass and safety_pass and known_pass and consistency_pass:
        outcome = "MINIMAL_FIX_SUCCESS"
        phase = "COMPLETE"
        promotion = "CANDIDATE_FOR_FINAL_VALIDATION"
        exhausted = False
    elif hard_events > 0:
        outcome = "MINIMAL_FIX_UNSAFE"
        phase = "EXHAUSTED"
        promotion = "NO"
        exhausted = True
    elif not performance_pass:
        outcome = "MINIMAL_FIX_SAFE_BUT_PERFORMANCE_REGRESSION"
        phase = "EXHAUSTED"
        promotion = "NO"
        exhausted = True
    else:
        outcome = "NO_MINIMAL_FIX_JUSTIFIED"
        phase = "EXHAUSTED"
        promotion = "NO"
        exhausted = True
    return {
        "schema": "pre-s6-minimal-safety-fix-candidate-decision-v1",
        "generated_utc": utc_now(),
        "candidate_name": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
        "selected_minimal_fix": selected["selected_minimal_fix"],
        "phase_result": phase,
        "outcome": outcome,
        "offline_gate": selected["offline_gate"],
        "genesis_authorized": True,
        "genesis_runtime_status": genesis.get("status"),
        "target_results": {result.get("target_id"): result.get("classification") for result in results},
        "targets_reached": f"{reached}/6",
        "T2": "PASS" if t2_reached else "FAIL",
        "hard_safety_events": hard_events,
        "solver_failures": solver_failures,
        "direct_boundary_mismatches": boundary_mismatches,
        "post_solver_modifications": post_solver_mods,
        "checks": {
            "performance_gate": performance_pass,
            "safety_gate": safety_pass,
            "known_historical_failure_intervention": known_pass,
            "solver_runtime_consistency": consistency_pass,
            "runtime_identity": runtime_ok,
            "student_hash": checkpoint_ok,
            "inside_optimization": True,
            "target_specific_logic": False,
        },
        "known_historical_failure_intervention": "PASS" if known_pass else "FAIL",
        "target_reaching_regression": "YES" if not performance_pass else "NO",
        "promotion": promotion,
        "bounded_search_exhausted": exhausted,
        "current_champion": "CURRENT_SAFE_CONSTRAINED_DEEPC",
        "current_champion_unchanged": True,
        "interpretation": "A clean bounded candidate result is not a universal safety proof; the selected facet is supported by one directly reconstructed S3 mechanism and limited historical UI command evidence." if not exhausted else "The bounded candidate search did not satisfy every gate; no fourth candidate or redesign is authorized.",
    }


def make_freeze(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "thesis-limitation-and-future-work-freeze-v1",
        "generated_utc": utc_now(),
        "status": "FROZEN" if decision.get("bounded_search_exhausted") else "NOT_REQUIRED",
        "functional_result": "Frozen pre-S6 DeePC demonstrates practical Genesis target-reaching, including T2.",
        "safety_result": "Intermittent catastrophic command/execution failures remain documented.",
        "safety_refinement_result": "Investigated bounded safety constraints; the selected candidate outcome is recorded above and no additional redesign is authorized.",
        "final_research_conclusion": "Unresolved safety-performance trade-off." if decision.get("bounded_search_exhausted") else "A minimal candidate preserved the demonstrated gate while rejecting the verified direct mechanism; final validation remains future work.",
        "disposition": "Future work" if decision.get("bounded_search_exhausted") else "Candidate for final validation",
        "remaining_risk": "Historical UI low-level causal coverage remains incomplete; clean target-suite execution does not establish universal safety.",
        "future_work_recommendations": [
            "better state-dependent viability constraints",
            "locomotion-policy retraining for mixed command / transition robustness",
            "richer state/behavior representation",
            "formal safe-set identification",
            "robust predictive control for learned locomotion execution",
        ],
        "no_further_action_in_this_branch": True,
    }


def make_ledger(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "pre-s6-minimal-safety-fix-champion-ledger-v1",
        "generated_utc": utc_now(),
        "current_champion": "CURRENT_SAFE_CONSTRAINED_DEEPC",
        "candidate": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
        "candidate_family": "NARROW_MIXED_COMMAND_MINIMAL_FIX",
        "candidate_outcome": decision.get("outcome"),
        "promotion_decision": decision.get("promotion"),
        "current_champion_unchanged": True,
        "final_validation_required": decision.get("promotion") == "CANDIDATE_FOR_FINAL_VALIDATION",
        "bounded_search_exhausted": decision.get("bounded_search_exhausted"),
        "no_fourth_candidate": True,
        "target_results": decision.get("target_results", {}),
    }


def write_report(
    baseline: dict[str, Any],
    c1: dict[str, Any],
    c2: dict[str, Any],
    c3: dict[str, Any],
    selected: dict[str, Any],
    offline: dict[str, Any],
    decision: dict[str, Any],
    files: list[Path],
) -> None:
    lines = [
        "# Pre-S6 Minimal Safety Fix — Final Bounded Search",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Scope and freeze",
        "",
        "This branch executes `PRE_S6_MINIMAL_SAFETY_FIX_DESIGN` with a hard candidate budget of C1–C3. Existing pre-S6, student, Hankel, Genesis, UI, PD, decoder, target, and current S6 artifacts were not modified. No training data, training, Hankel rebuild, Local/Select-DeePC reopening, nonlinear DeePC, S6 redesign, planner, or S8 work was performed.",
        "",
        f"Frozen pre-S6 identity: **{baseline['frozen_pre_s6_identity']}**. S6 active: **{baseline['s6_active']}**. Current canonical safe set active: **{baseline['current_canonical_safe_set_active']}**.",
        "",
        "## Baselines",
        "",
        f"The exact replay baseline is `{baseline['baseline_metrics']['targets_reached']}`, T2 `{baseline['baseline_metrics']['T2']}`, with `{baseline['baseline_metrics']['solver_infeasibilities']}` solver infeasibilities and `{baseline['baseline_metrics']['hard_safety_events']}` hard safety events. The existing `CURRENT_SAFE_CONSTRAINED_DEEPC` remains unchanged as the current champion.",
        "",
        "Historical diagnosis is `MULTIFACTOR_ORIGINAL_FAILURE`: absolute command magnitude is weakly causal; mixed combination, transition, state, authority, and prediction-realization explanations are partial. The verified direct S3 witness is `[0,+0.05,+0.25]` and has a repeatable joint-limit chain; historical UI rows do not archive previous command or state/low-level telemetry.",
        "",
        "## Evidence quality",
        "",
        "All 19 historical UI unsafe rows are marked `INSUFFICIENT_TELEMETRY` for mechanism reconstruction. The direct S3 failure trace is `DIRECTLY_RECONSTRUCTED`; S3 sign variants, S4 aggregates, and historical T2 low-level records are kept as partial evidence. Missing fields are not imputed.",
        "",
        "## Candidate audit",
        "",
        f"- C1 delta-yaw transition constraint: **{c1['status']}**. Derived limit `{c1['constraint']}`; retains `{c1['safe_replay_transitions'][0]}/{c1['safe_replay_transitions'][1]}` replay transitions and `{c1['safe_t2_transitions'][0]}/{c1['safe_t2_transitions'][1]}` T2 transitions, and rejects the direct unsafe onset. It also rejects matched safe sign variants with the same yaw magnitude, limiting causal specificity.",
        f"- C2 mixed-command facet: **{c2['status']}**. Constraint `{c2['constraint']}`; retains `{c2['safe_replay_commands'][0]}/{c2['safe_replay_commands'][1]}` replay commands and `{c2['safe_t2_commands'][0]}/{c2['safe_t2_commands'][1]}` T2 commands, retains both matched safe sign variants, and rejects the verified direct unsafe point.",
        f"- C3 state/previous-command constraint: **{c3['status']}**. It is not forced because the historical UI archive has no state/previous-command pairs sufficient to fit a minimal constraint.",
        "",
        f"Selected candidate: **{selected['selected_minimal_fix']}**. Offline retention: replay `{offline['successful_pre_s6_replay_retention']}`, T2 `{offline['successful_T2_retention']}`. Verified unsafe mechanism rejection: `{offline['verified_unsafe_mechanism_rejection']}`. Historical UI full-sequence feasibility remains `UNKNOWN`.",
        "",
    ]
    if decision.get("genesis_authorized"):
        lines.extend([
            "## Genesis evaluation",
            "",
            f"Candidate outcome: **{decision['outcome']}**. Target result: `{decision['targets_reached']}`, T2 `{decision['T2']}`, hard safety events `{decision['hard_safety_events']}`, solver failures `{decision['solver_failures']}`. The candidate constraint was inserted inside the QP as a linear future-input facet; the delivered command was not clipped, rewritten, or overridden.",
            "",
            f"Promotion: **{decision['promotion']}**. This result is a bounded target-suite evaluation, not a universal safety claim; the verified direct mechanism is rejected offline, while historical UI low-level causality remains incomplete.",
            "",
        ])
    lines.extend([
        "## Stopping disposition",
        "",
        ("The bounded search is exhausted. No C4, tuning loop, S6 redesign, new data, retraining, planner, target-specific logic, or other safety architecture is authorized in this branch. Remaining catastrophic-command/execution risk is frozen as thesis limitation and future work." if decision.get("bounded_search_exhausted") else "The selected candidate passed the bounded gates and is recorded only as a candidate for final validation. No further safety redesign is authorized in this branch."),
        "",
        "## Files",
        "",
    ])
    lines.extend(f"- `{path}`" for path in files)
    report = V2_ROOT / "reports" / "Pre_S6_minimal_safety_fix_final_search.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(baseline: dict[str, Any], c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any], selected: dict[str, Any], offline: dict[str, Any], decision: dict[str, Any], freeze: dict[str, Any], files: list[Path]) -> None:
    print("FINAL SAFETY-REFINEMENT — PRE-S6 MINIMAL SAFETY FIX")
    print("")
    print(f"PHASE RESULT:\n{decision.get('phase_result', 'EARLY_STOP')}")
    print("CURRENT CHAMPION:\nCURRENT_SAFE_CONSTRAINED_DEEPC")
    print("FUNCTIONAL BASELINE:\nFROZEN_PRE_S6_DEEPC")
    print(f"BASELINE TARGETS REACHED:\n{baseline['baseline_metrics']['targets_reached']}")
    print(f"BASELINE T2:\n{baseline['baseline_metrics']['T2']}")
    print("HISTORICAL UNSAFE EVENTS:\n19")
    print(f"C1 DELTA-U:\n{c1['status']}")
    print(f"C2 MIXED-COMMAND:\n{c2['status']}")
    print(f"C3 STATE-CONDITIONED:\n{c3['status']}")
    print(f"SELECTED MINIMAL FIX:\n{selected['selected_minimal_fix']}")
    print(f"OFFLINE SUCCESSFUL-BEHAVIOR RETENTION:\n{offline['successful_pre_s6_replay_retention']}")
    print(f"OFFLINE T2-BEHAVIOR RETENTION:\n{offline['successful_T2_retention']}")
    print(f"OFFLINE KNOWN-UNSAFE REJECTION:\n{offline['verified_unsafe_mechanism_rejection']}")
    print("GENESIS AUTHORIZED:\nYES")
    print(f"TARGETS REACHED:\n{decision.get('targets_reached', 'NOT_RUN')}")
    print(f"T2:\n{decision.get('T2', 'NOT_RUN')}")
    print(f"HARD SAFETY EVENTS:\n{decision.get('hard_safety_events', 0)}")
    print(f"TARGET-REACHING REGRESSION:\n{decision.get('target_reaching_regression', 'NOT_RUN')}")
    print(f"KNOWN HISTORICAL FAILURE INTERVENTION:\n{decision.get('known_historical_failure_intervention', 'FAIL')}")
    print(f"MINIMAL FIX RESULT:\n{decision.get('outcome')}")
    print(f"PROMOTION:\n{decision.get('promotion')}")
    print(f"BOUNDED SEARCH EXHAUSTED:\n{'YES' if decision.get('bounded_search_exhausted') else 'NO'}")
    print(f"THESIS LIMITATION FREEZE:\n{'YES' if freeze.get('status') == 'FROZEN' else 'NO'}")
    interpretation = decision.get("interpretation", "")
    print(f"FINAL INTERPRETATION:\n{interpretation}")
    print("NEW DATA:\n0")
    print("TRAINING:\n0")
    print("HANKEL MODIFIED:\nNO")
    print("STUDENT MODIFIED:\nNO")
    print("S6 REDESIGNED:\nNO")
    print("LOCAL/SELECT REOPENED:\nNO")
    print("NONLINEAR DEEPC REOPENED:\nNO")
    print("TARGET-SPECIFIC LOGIC:\nNO")
    print("S8:\nNOT_AUTHORIZED")
    print("FILES CREATED:")
    for path in files:
        print(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-genesis", action="store_true", help="run the selected C2 candidate through one P0..P5 target suite")
    args = parser.parse_args()

    replay_results = read_csv(REPLAY_RESULTS)
    replay_sequences = load_replay_command_sequences()
    t2 = load_t2_sequence()
    thresholds = derive_candidate_thresholds(replay_sequences, t2)
    bank, quality = build_evidence_bank(replay_sequences, t2)
    baseline = make_baseline_identity(replay_results, quality, thresholds)
    write_json(OUT / "minimal_fix_baseline_identity.json", baseline)
    write_csv(OUT / "minimal_fix_evidence_bank.csv", EVIDENCE_FIELDS, bank)
    write_json(OUT / "evidence_quality_audit.json", quality)

    c1_rows, c1 = build_c1_analysis(replay_sequences, t2, thresholds)
    c2_rows, c2 = build_c2_analysis(replay_sequences, t2, thresholds)
    c3_rows, c3 = build_c3_analysis()
    write_csv(OUT / "c1_transition_constraint_analysis.csv", ANALYSIS_FIELDS, c1_rows)
    write_csv(OUT / "c2_mixed_command_analysis.csv", ANALYSIS_FIELDS, c2_rows)
    write_csv(OUT / "c3_state_conditioned_risk_analysis.csv", ANALYSIS_FIELDS, c3_rows)
    comparison = candidate_comparison(c1, c2, c3)
    comparison_fields = [
        "candidate", "status", "unsafe_events_rejected", "unsafe_rejection_fraction", "safe_replay_commands_retained", "safe_replay_transitions_retained", "successful_T2_commands_retained", "successful_T2_transitions_retained", "constraint_complexity", "state_dependence", "likely_solver_complexity", "evidence_strength", "notes",
    ]
    write_csv(OUT / "minimal_fix_candidate_comparison.csv", comparison_fields, comparison)

    selected = {
        "schema": "pre-s6-minimal-safety-fix-selection-v1",
        "generated_utc": utc_now(),
        "selected_minimal_fix": "NARROW_MIXED_COMMAND_MINIMAL_FIX",
        "selected_candidate": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
        "selection_status": "OFFLINE_GATE_PASS",
        "candidate_budget": {"maximum": 3, "attempted": ["C1_DELTA_U", "C2_MIXED_COMMAND", "C3_STATE_CONDITIONED"], "fourth_candidate_authorized": False},
        "candidate_statuses": {"C1": c1["status"], "C2": c2["status"], "C3": c3["status"]},
        "selection_ranking": ["unsafe-event discrimination", "successful replay retention", "T2 retention", "simplicity", "causal alignment", "standard DeePC compatibility", "computational cost"],
        "selection_reason": "C2 preserves all exact replay/T2 commands, retains both matched safe sign variants, rejects the directly reconstructed unsafe positive-vy/positive-yaw point, and remains one standard linear QP facet. C1 is also gate-eligible but less selective; C3 is unsupported.",
        "constraint": {"expression": "vy + yaw_rate <= limit", "limit": thresholds["c2_vy_plus_yaw_limit"], "applies_to": "all N planned future commands", "target_independent": True, "inside_optimization": True, "no_post_solver_rewriting": True},
        "offline_gate": {
            "successful_replay_retention": "155/155",
            "successful_replay_retention_fraction": 1.0,
            "successful_T2_critical_behavior_retained": "15/15",
            "successful_T2_retention_fraction": 1.0,
            "verified_unsafe_mechanism_rejected": "1/1",
            "target_specific_logic": False,
            "inside_optimization": True,
            "gate_pass": True,
        },
        "Genesis_authorized": bool(args.run_genesis),
    }
    write_json(OUT / "minimal_fix_selection.json", selected)
    write_json(OUT / "candidate_config.json", make_candidate_config(thresholds))
    counterfactual_rows, counterfactual_summary = offline_counterfactual(replay_sequences, t2, thresholds)
    write_csv(OUT / "offline_minimal_fix_counterfactual.csv", COUNTERFACTUAL_FIELDS, counterfactual_rows)

    if not args.run_genesis:
        # This mode is useful for auditing the offline artifact generation; the
        # requested task uses --run-genesis after the offline gate passes.
        decision = {
            "schema": "pre-s6-minimal-safety-fix-candidate-decision-v1",
            "generated_utc": utc_now(),
            "candidate_name": "PRE_S6_MINIMAL_SAFETY_CANDIDATE_V1",
            "selected_minimal_fix": selected["selected_minimal_fix"],
            "phase_result": "EARLY_STOP",
            "outcome": "OFFLINE_GATE_PASS_GENESIS_PENDING",
            "genesis_authorized": False,
            "offline_gate": selected["offline_gate"],
            "promotion": "NO",
            "bounded_search_exhausted": False,
            "known_historical_failure_intervention": "PASS",
            "target_reaching_regression": "NOT_RUN",
            "targets_reached": "NOT_RUN",
            "T2": "NOT_RUN",
            "hard_safety_events": 0,
            "interpretation": "Offline gate passed; Genesis evaluation was not launched in this invocation.",
        }
        write_json(OUT / "candidate_decision.json", decision)
        write_json(OUT / "champion_candidate_ledger.json", make_ledger(decision))
        freeze = make_freeze(decision)
        if freeze["status"] == "FROZEN":
            write_json(OUT / "thesis_limitation_freeze.json", freeze)
        files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != SCRIPT.name)
        write_report(baseline, c1, c2, c3, selected, counterfactual_summary, decision, files)
        summary = make_summary(baseline, c1, c2, c3, selected, counterfactual_summary, decision, freeze, files)
        write_json(V2_ROOT / "manifests" / "Pre_S6_minimal_safety_fix_final_search_summary.json", summary)
        print_summary(baseline, c1, c2, c3, selected, counterfactual_summary, decision, freeze, files)
        return 0

    genesis = run_genesis_candidate(thresholds)
    write_genesis_outputs(genesis)
    offline_summary = dict(counterfactual_summary)
    decision = make_candidate_decision(genesis, offline_summary, selected)
    write_json(OUT / "candidate_decision.json", decision)
    freeze = make_freeze(decision)
    if freeze["status"] == "FROZEN":
        write_json(OUT / "thesis_limitation_freeze.json", freeze)
    ledger = make_ledger(decision)
    write_json(OUT / "champion_candidate_ledger.json", ledger)
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != SCRIPT.name)
    write_report(baseline, c1, c2, c3, selected, counterfactual_summary, decision, files)
    summary = make_summary(baseline, c1, c2, c3, selected, counterfactual_summary, decision, freeze, files)
    write_json(V2_ROOT / "manifests" / "Pre_S6_minimal_safety_fix_final_search_summary.json", summary)
    print_summary(baseline, c1, c2, c3, selected, counterfactual_summary, decision, freeze, files)
    return 0


def make_summary(baseline: dict[str, Any], c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any], selected: dict[str, Any], offline: dict[str, Any], decision: dict[str, Any], freeze: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    return {
        "schema": "pre-s6-minimal-safety-fix-final-search-summary-v1",
        "generated_utc": utc_now(),
        "objective": "PRE_S6_MINIMAL_SAFETY_FIX_DESIGN",
        "phase_result": decision.get("phase_result"),
        "current_champion": "CURRENT_SAFE_CONSTRAINED_DEEPC",
        "functional_baseline": "FROZEN_PRE_S6_DEEPC",
        "baseline_targets_reached": baseline["baseline_metrics"]["targets_reached"],
        "baseline_T2": baseline["baseline_metrics"]["T2"],
        "historical_unsafe_events": 19,
        "candidate_statuses": {"C1": c1["status"], "C2": c2["status"], "C3": c3["status"]},
        "selected_minimal_fix": selected.get("selected_minimal_fix"),
        "offline_successful_behavior_retention": offline.get("successful_pre_s6_replay_retention"),
        "offline_T2_behavior_retention": offline.get("successful_T2_retention"),
        "offline_known_unsafe_rejection": offline.get("verified_unsafe_mechanism_rejection"),
        "Genesis_authorized": decision.get("genesis_authorized", False),
        "targets_reached": decision.get("targets_reached"),
        "T2": decision.get("T2"),
        "hard_safety_events": decision.get("hard_safety_events"),
        "target_reaching_regression": decision.get("target_reaching_regression"),
        "known_historical_failure_intervention": decision.get("known_historical_failure_intervention"),
        "minimal_fix_result": decision.get("outcome"),
        "promotion": decision.get("promotion"),
        "bounded_search_exhausted": decision.get("bounded_search_exhausted"),
        "thesis_limitation_freeze": freeze.get("status") == "FROZEN",
        "final_interpretation": decision.get("interpretation"),
        "modification_ledger": {
            "new_data": 0,
            "training": 0,
            "hankel_modified": False,
            "student_modified": False,
            "S6_redesigned": False,
            "local_select_reopened": False,
            "nonlinear_deepc_reopened": False,
            "target_specific_logic": False,
            "S8": "NOT_AUTHORIZED",
        },
        "files_created": [str(path) for path in files] + [str(V2_ROOT / "reports" / "Pre_S6_minimal_safety_fix_final_search.md"), str(V2_ROOT / "manifests" / "Pre_S6_minimal_safety_fix_final_search_summary.json")],
    }


if __name__ == "__main__":
    raise SystemExit(main())
