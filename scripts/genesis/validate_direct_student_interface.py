#!/usr/bin/env python3
"""Validate direct DeePC command delivery to the frozen student in Genesis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from learned_execution.direct_student_runtime import (  # noqa: E402
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    DirectStudentBatch,
    FrozenObservableStudent,
    make_genesis_scene,
    pose_from_snapshot,
    sha256_file,
)

COMMANDS = [
    ("zero", (0.00, 0.00, 0.00)),
    ("forward_020", (0.20, 0.00, 0.00)),
    ("forward_040", (0.40, 0.00, 0.00)),
    ("lateral_left_005", (0.00, +0.05, 0.00)),
    ("lateral_left_010", (0.00, +0.10, 0.00)),
    ("lateral_left_015", (0.00, +0.15, 0.00)),
    ("lateral_right_005", (0.00, -0.05, 0.00)),
    ("lateral_right_010", (0.00, -0.10, 0.00)),
    ("lateral_right_015", (0.00, -0.15, 0.00)),
    ("yaw_left_020", (0.00, 0.00, +0.20)),
    ("yaw_left_040", (0.00, 0.00, +0.40)),
    ("yaw_right_010", (0.00, 0.00, -0.10)),
    ("mixed_forward_left_yaw", (0.20, +0.10, +0.20)),
    ("mixed_forward_right_yaw", (0.20, -0.10, -0.10)),
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(repetitions: int, duration_s: float) -> dict[str, Any]:
    steps = int(round(duration_s / 0.02))
    if not np.isclose(steps * 0.02, duration_s):
        raise ValueError("duration must be an integer multiple of 0.02")
    commands = np.asarray([command for _name, command in COMMANDS], dtype=np.float32)
    scene = make_genesis_scene(len(commands))
    student = FrozenObservableStudent()
    runtime = DirectStudentBatch(scene, student)
    rows: list[dict[str, Any]] = []
    equality_count = 0
    comparison_count = 0
    safety_totals = {name: 0 for name in ("fall", "base_contact", "nan_inf", "joint_violation", "torque_violation")}
    completed = 0
    failure: str | None = None

    for repetition in range(repetitions):
        scene.reset()
        runtime.reset()
        pose = pose_from_snapshot(runtime.observe())
        for step_index in range(steps):
            result = runtime.step(commands)
            current = pose_from_snapshot(result.after)
            yaw_delta = np.arctan2(np.sin(current[:, 2] - pose[:, 2]), np.cos(current[:, 2] - pose[:, 2]))
            pose[:, 2] += yaw_delta
            pose[:, :2] = current[:, :2]
            equal = np.all(result.requested_command == result.student_received_command, axis=1)
            equality_count += int(np.count_nonzero(equal))
            comparison_count += len(commands)
            for index, (name, command) in enumerate(COMMANDS):
                safety = {
                    "fall": bool(result.safety["fall"][index]),
                    "base_contact": bool(result.safety["base_contact"][index]),
                    "nan_inf": bool(result.safety["nan_inf"][index]),
                    "joint_violation": bool(result.safety["actual_joint_limit_violation"][index]),
                    "torque_violation": bool(result.safety["torque_limit_exceedance"][index]),
                }
                for key, value in safety.items():
                    safety_totals[key] += int(value)
                rows.append(
                    {
                        "repetition": repetition,
                        "step": step_index,
                        "condition": name,
                        "requested_command": list(command),
                        "student_received_command": result.student_received_command[index].tolist(),
                        "command_equal": bool(equal[index]),
                        "x": float(pose[index, 0]),
                        "y": float(pose[index, 1]),
                        "yaw": float(pose[index, 2]),
                        "body_vx": float(result.after["body_linear_velocity"][index, 0]),
                        "body_vy": float(result.after["body_linear_velocity"][index, 1]),
                        "body_yaw_rate": float(result.after["body_angular_velocity"][index, 2]),
                        **safety,
                    }
                )
            failed_names = [name for name, value in safety_totals.items() if value]
            if failed_names:
                failure = "SAFETY_FAILURE: " + ", ".join(failed_names)
                break
        if failure:
            break
        completed += 1

    np.savez_compressed(
        WORKSPACE / "logs/direct_student_interface_validation.npz",
        requested_command=np.asarray([row["requested_command"] for row in rows], dtype=np.float32),
        student_received_command=np.asarray([row["student_received_command"] for row in rows], dtype=np.float32),
        command_equal=np.asarray([row["command_equal"] for row in rows], dtype=bool),
        x=np.asarray([row["x"] for row in rows], dtype=np.float64),
        y=np.asarray([row["y"] for row in rows], dtype=np.float64),
        yaw=np.asarray([row["yaw"] for row in rows], dtype=np.float64),
    )
    path = WORKSPACE / "logs/direct_student_interface_validation.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    passed = (
        completed == repetitions
        and equality_count == comparison_count
        and not any(safety_totals.values())
    )
    result = {
        "schema": "direct-student-interface-validation-v1",
        "classification": "DIRECT_STUDENT_INTERFACE_PASS" if passed else "DIRECT_STUDENT_INTERFACE_BLOCKED",
        "repetitions_requested": repetitions,
        "repetitions_completed": completed,
        "duration_s": duration_s,
        "steps_per_condition": steps,
        "condition_count": len(COMMANDS),
        "command_comparisons": comparison_count,
        "command_equal_count": equality_count,
        "command_mismatch_count": comparison_count - equality_count,
        "safety_totals": safety_totals,
        "failure": failure,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "runtime_path": "raw DeePC command -> measured 45-D state + exact command -> frozen student -> physical PD/Genesis",
        "legacy_projection_present": False,
    }
    write_json(WORKSPACE / "reports/direct_student_interface_validation.json", result)
    lines = [
        "# Direct frozen-student interface validation",
        "",
        f"`{result['classification']}`",
        "",
        "The tested path is direct: `requested_command == student_received_command` before the command is appended to the 45-D observable state.",
        "No command guard, projection, positive-`vy` deadzone, clipping, smoothing, mode switch, expert, or fallback is present.",
        "",
        f"- conditions: `{len(COMMANDS)}`",
        f"- repetitions: `{completed}/{repetitions}`",
        f"- duration per condition: `{duration_s:.2f} s`",
        f"- command comparisons: `{comparison_count}`",
        f"- exact equal comparisons: `{equality_count}`",
        f"- mismatch count: `{comparison_count - equality_count}`",
        f"- safety totals: `{safety_totals}`",
        f"- checkpoint SHA256: `{result['checkpoint_sha256']}`",
        "",
        "The small positive-`vy` conditions are intentionally retained as direct commands; their measured response is evidence about the final plant, not a reason to restore the historical projection.",
        "",
        "| Condition | Command |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{list(command)}` |" for name, command in COMMANDS)
    (WORKSPACE / "reports/direct_student_interface_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args(argv)
    result = run(args.repetitions, args.duration)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] == "DIRECT_STUDENT_INTERFACE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
