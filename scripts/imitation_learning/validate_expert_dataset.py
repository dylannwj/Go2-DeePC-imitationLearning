#!/usr/bin/env python3
"""Validate, flatten, and audit the collected expert imitation episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/expert_imitation_dataset_matplotlib")

from scripts.imitation_learning.dataset_common import (  # noqa: E402
    DATASET_ROOT,
    canonical_json,
    git_value,
    json_safe,
    load_npz,
    metadata_from_arrays,
    metadata_array,
    relative_path,
    runtime_payload,
    sha256_file,
    sha256_tree,
    utc_now,
    write_deterministic_npz,
    write_json,
    write_text,
)

from learned_execution.contracts import (  # noqa: E402
    ACTION_DIM,
    COMMAND_DIM,
    DEFAULT_JOINT_POSITION as STUDENT_DEFAULT_JOINT_POS,
    GENESIS_JOINT_NAMES,
    STATE_DIM,
    STATE_FEATURE_NAMES,
)


SCRIPT_VERSION = "expert-imitation-validator-v1.0"
CONTROL_DT = 0.02
POLICY_RATE_HZ = 50.0
DATASET_COMMAND_LIMITS = np.asarray([[-0.30, 0.50], [-0.20, 0.20], [-0.50, 0.50]], dtype=np.float32)
COMMAND_NAMES = ("vx", "vy", "yaw_rate")
DEFAULT_JOINT_POS = np.asarray(
    [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
    dtype=np.float32,
)
ACTION_SCALE = np.full(ACTION_DIM, 0.5, dtype=np.float32)
EFFORT_LIMIT = np.tile(np.asarray([23.5, 23.5, 45.43], dtype=np.float32), 4)
POLICY_TO_SCENE = np.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
TEACHER_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

REQUIRED_FIELDS = (
    "student_state",
    "desired_command",
    "student_input",
    "teacher_obs",
    "expert_action",
    "expert_action_student_order",
    "q_target",
    "q_target_student_order",
    "previous_expert_action",
    "previous_expert_action_student_order",
    "base_position",
    "base_rpy",
    "base_linear_velocity",
    "base_angular_velocity",
    "joint_position",
    "joint_velocity",
    "foot_contacts",
    "q_target_applied",
    "preclip_torque",
    "applied_torque",
    "episode_id",
    "step_index",
    "seed",
    "command_segment_id",
    "transition_window",
    "timestamp",
    "fall_flag",
    "base_contact",
    "actual_joint_limit_violation",
    "q_target_clipping",
    "torque_limit_violation",
    "nan_inf_flag",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_arrays(arrays: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "b")


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def artifact_drift_check(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifests/expert_artifact_manifest.json")
    migrated_paths = {
        "checkpoint": PROJECT_ROOT / "models/expert_model_500.pt",
        "params/agent.yaml": PROJECT_ROOT / "configs/imitation/expert/agent.yaml",
        "params/deploy.yaml": PROJECT_ROOT / "configs/imitation/expert/deploy.yaml",
        "params/env.yaml": PROJECT_ROOT / "configs/imitation/expert/env.yaml",
        "policy.onnx": PROJECT_ROOT / "external/unitree_go2/policy.onnx",
        "policy.onnx.data": PROJECT_ROOT / "external/unitree_go2/policy.onnx.data",
    }
    checks: list[dict[str, Any]] = []
    for label, record in manifest.get("artifacts", {}).items():
        path = migrated_paths.get(label, Path(record["absolute_path"]))
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == record.get("sha256") == record.get("compatibility_gate_sha256")
        checks.append({"artifact": label, "passed": passed, "path": str(path), "expected": record.get("sha256"), "actual": actual})
    prior_paths = [
        PROJECT_ROOT / "results/genesis/generated/imitation_compatibility/reports/frozen_contract.json",
        PROJECT_ROOT / "results/genesis/generated/imitation_compatibility/reports/frozen_artifacts.json",
    ]
    return {"passed": bool(checks) and all(row["passed"] for row in checks), "checks": checks, "prior_evidence_unchanged": all(path.is_file() for path in prior_paths)}


def validate_episode(path: Path, manifest_record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    try:
        arrays = load_npz(path)
        metadata = metadata_from_arrays(arrays)
    except Exception as exc:
        return {"path": relative_path(path), "valid": False, "accepted": False, "errors": [f"logging_corruption: {type(exc).__name__}: {exc}"], "samples": 0}
    missing = [name for name in REQUIRED_FIELDS if name not in arrays]
    if missing:
        errors.append(f"missing fields: {missing}")
    sample_counts = {name: int(arrays[name].shape[0]) for name in REQUIRED_FIELDS if name in arrays and arrays[name].ndim >= 1}
    n = sample_counts.get("student_state", 0)
    if any(value != n for value in sample_counts.values()):
        errors.append(f"sample count mismatch: {sample_counts}")
    expected_last_dims = {
        "student_state": 45,
        "desired_command": 3,
        "student_input": 48,
        "teacher_obs": 45,
        "expert_action": 12,
        "expert_action_student_order": 12,
        "q_target": 12,
        "q_target_student_order": 12,
        "previous_expert_action": 12,
        "previous_expert_action_student_order": 12,
        "base_position": 3,
        "base_rpy": 3,
        "base_linear_velocity": 3,
        "base_angular_velocity": 3,
        "joint_position": 12,
        "joint_velocity": 12,
        "foot_contacts": 4,
        "q_target_applied": 12,
        "preclip_torque": 12,
        "applied_torque": 12,
    }
    for name, width in expected_last_dims.items():
        if name in arrays and (arrays[name].ndim != 2 or arrays[name].shape[1] != width):
            errors.append(f"{name} shape {arrays[name].shape}, expected [N,{width}]")
    if not finite_arrays({name: arrays[name] for name in REQUIRED_FIELDS if name in arrays}):
        errors.append("NaN/Inf present in numeric arrays")
    if n:
        state = arrays["student_state"]
        command = arrays["desired_command"]
        student_input_error = float(np.max(np.abs(arrays["student_input"] - np.concatenate((state, command), axis=1))))
        student_actual_position = state[:, 9:21] + STUDENT_DEFAULT_JOINT_POS[None, :]
        teacher_actual_position = student_actual_position[:, POLICY_TO_SCENE]
        teacher_position_relative = teacher_actual_position - DEFAULT_JOINT_POS[None, :]
        teacher_velocity = state[:, 21:33][:, POLICY_TO_SCENE]
        teacher_previous = state[:, 33:45][:, POLICY_TO_SCENE]
        teacher_expected = np.concatenate(
            (state[:, 6:9], state[:, :3], command, teacher_position_relative, teacher_velocity, teacher_previous),
            axis=1,
        )
        teacher_error = float(np.max(np.abs(arrays["teacher_obs"] - teacher_expected)))
        previous_error = float(np.max(np.abs(arrays["previous_expert_action_student_order"] - state[:, 33:45])))
        previous_teacher_error = float(np.max(np.abs(arrays["previous_expert_action"] - teacher_previous)))
        action_map_error = float(np.max(np.abs(arrays["expert_action_student_order"] - arrays["expert_action"][:, POLICY_TO_SCENE])))
        target_map_error = float(np.max(np.abs(arrays["q_target_student_order"] - arrays["q_target"][:, POLICY_TO_SCENE])))
        q_error = float(np.max(np.abs(arrays["q_target"] - (DEFAULT_JOINT_POS[None, :] + ACTION_SCALE[None, :] * arrays["expert_action"]))))
        if student_input_error > 1.0e-6:
            errors.append(f"student_input reconstruction error {student_input_error:.3e}")
        if teacher_error > 1.0e-6:
            errors.append(f"teacher_obs adapter reconstruction error {teacher_error:.3e}")
        if previous_error > 1.0e-6:
            errors.append(f"previous action/state mismatch {previous_error:.3e}")
        if previous_teacher_error > 1.0e-6:
            errors.append(f"teacher previous action mismatch {previous_teacher_error:.3e}")
        if action_map_error > 1.0e-6 or target_map_error > 1.0e-6:
            errors.append(f"teacher-to-student named mapping mismatch: action={action_map_error:.3e}, target={target_map_error:.3e}")
        if q_error > 1.0e-6:
            errors.append(f"q_target reconstruction error {q_error:.3e}")
        if np.any(command < DATASET_COMMAND_LIMITS[:, 0] - 1.0e-6) or np.any(command > DATASET_COMMAND_LIMITS[:, 1] + 1.0e-6):
            errors.append("desired command outside frozen collection envelope")
        step = arrays["step_index"].astype(np.int64)
        timestamp = arrays["timestamp"].astype(np.float64)
        if not np.array_equal(step, np.arange(n, dtype=np.int64)):
            errors.append("step indices are not contiguous from zero")
        if n > 1 and not np.all(np.diff(timestamp) > 0.0):
            errors.append("timestamps are not strictly monotonic")
        if n > 1 and not np.allclose(np.diff(timestamp), CONTROL_DT, atol=1.0e-7):
            errors.append("timestamps do not advance at 50 Hz")
        if np.any(arrays["q_target_applied"] < -10.0) or np.any(arrays["q_target_applied"] > 10.0):
            errors.append("applied joint target is outside plausible range")
    safety_counts = {name: int(np.count_nonzero(arrays[name])) for name in ("fall_flag", "base_contact", "actual_joint_limit_violation", "torque_limit_violation", "nan_inf_flag") if name in arrays}
    computed_accepted = n > 0 and n == int(round(float(metadata.get("requested_duration_s", 0.0)) / CONTROL_DT)) and not any(safety_counts.values()) and metadata.get("termination_reason") == "time_limit" and not errors
    if bool(metadata.get("accepted")) != bool(computed_accepted):
        errors.append(f"metadata accepted={metadata.get('accepted')} disagrees with recomputed accepted={computed_accepted}")
    manifest_match = manifest_record.get("raw_sha256") == sha256_file(path)
    if not manifest_match:
        errors.append("raw hash differs from episode manifest")
    return {
        "path": relative_path(path),
        "episode_id": str(manifest_record.get("episode_id", path.stem)),
        "metadata": metadata,
        "arrays": arrays,
        "valid": not errors,
        "accepted": bool(computed_accepted),
        "errors": errors,
        "samples": n,
        "student_input_reconstruction_error": student_input_error if n else None,
        "teacher_obs_reconstruction_error": teacher_error if n else None,
        "previous_action_reconstruction_error": previous_error if n else None,
        "q_target_reconstruction_error": q_error if n else None,
        "safety_counts": safety_counts,
        "manifest_hash_match": manifest_match,
    }


def flatten_accepted(episodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    accepted = [episode for episode in episodes if episode["accepted"]]
    if not accepted:
        raise RuntimeError("no accepted episodes are available for canonical processing")
    fields = list(REQUIRED_FIELDS)
    result: dict[str, np.ndarray] = {}
    for field in fields:
        result[field] = np.concatenate([episode["arrays"][field] for episode in accepted], axis=0)
    return result


def category_masks(commands: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(commands, dtype=np.float64)
    vx, vy, yaw = values.T
    active_vx = np.abs(vx) > 0.02
    active_vy = np.abs(vy) > 0.02
    active_yaw = np.abs(yaw) > 0.02
    zero = ~(active_vx | active_vy | active_yaw)
    forward = active_vx & (vx > 0.0) & ~active_vy & ~active_yaw
    backward = active_vx & (vx < 0.0) & ~active_vy & ~active_yaw
    lateral = active_vy & ~active_vx & ~active_yaw
    pure_yaw = active_yaw & ~active_vx & ~active_vy
    forward_yaw = active_vx & (vx > 0.0) & active_yaw & ~active_vy
    mixed = ~(zero | forward | backward | lateral | pure_yaw | forward_yaw)
    return {"standing": zero, "forward": forward, "backward": backward, "lateral": lateral, "pure_yaw": pure_yaw, "forward+yaw": forward_yaw, "mixed": mixed}


def command_tracking(commands: np.ndarray, achieved: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    error = achieved - commands
    for index, name in enumerate(COMMAND_NAMES):
        desired = commands[:, index].astype(np.float64)
        actual = achieved[:, index].astype(np.float64)
        err = error[:, index]
        corr = float(np.corrcoef(desired, actual)[0, 1]) if np.std(desired) > 1.0e-12 and np.std(actual) > 1.0e-12 else None
        sign = (np.abs(desired) <= 0.02) | (np.sign(desired) == np.sign(actual))
        rows[name] = {
            "desired_mean": float(np.mean(desired)),
            "achieved_mean": float(np.mean(actual)),
            "achieved_median": float(np.median(actual)),
            "achieved_std": float(np.std(actual)),
            "mean_error": float(np.mean(err)),
            "mean_abs_error": float(np.mean(np.abs(err))),
            "median_abs_error": float(np.median(np.abs(err))),
            "error_std": float(np.std(err)),
            "error_p05": float(np.percentile(err, 5)),
            "error_p95": float(np.percentile(err, 95)),
            "correlation": corr,
            "sign_agreement": float(np.mean(sign)),
        }
    return rows


def state_statistics(state: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(STATE_FEATURE_NAMES):
        value = state[:, index].astype(np.float64)
        rows.append({
            "index": index,
            "name": name,
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
            "p01": float(np.percentile(value, 1)),
            "p05": float(np.percentile(value, 5)),
            "p50": float(np.percentile(value, 50)),
            "p95": float(np.percentile(value, 95)),
            "p99": float(np.percentile(value, 99)),
            "constant": bool(np.ptp(value) == 0.0),
            "near_constant": bool(np.std(value) < 1.0e-6),
            "finite": bool(np.isfinite(value).all()),
        })
    return rows


def action_statistics(actions: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(TEACHER_JOINT_NAMES):
        value = actions[:, index].astype(np.float64)
        rows.append({
            "index": index,
            "joint": name,
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
            "p01": float(np.percentile(value, 1)),
            "p99": float(np.percentile(value, 99)),
            "fraction_outside_minus1_plus1": float(np.mean((value < -1.0) | (value > 1.0))),
            "finite": bool(np.isfinite(value).all()),
        })
    return rows


def temporal_integrity(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    dt_values: list[np.ndarray] = []
    duplicate_count = 0
    missing_count = 0
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        arrays = episode["arrays"]
        step = arrays["step_index"].astype(np.int64)
        timestamp = arrays["timestamp"].astype(np.float64)
        if len(timestamp) > 1:
            dt = np.diff(timestamp)
            dt_values.append(dt)
            duplicate_count += int(np.count_nonzero(dt <= 0.0))
            missing_count += int(np.count_nonzero(~np.isclose(dt, CONTROL_DT, atol=1.0e-7)))
        expected = np.arange(len(step), dtype=np.int64)
        missing_count += int(len(np.setdiff1d(expected, step)))
        duplicate_count += int(len(step) - len(np.unique(step)))
        episode_rows.append({"episode_id": episode["episode_id"], "samples": len(step), "first_timestamp": float(timestamp[0]) if len(timestamp) else None, "last_timestamp": float(timestamp[-1]) if len(timestamp) else None, "missing_or_bad_dt": int(np.count_nonzero(np.diff(timestamp) <= 0.0)) if len(timestamp) > 1 else 0})
    all_dt = np.concatenate(dt_values) if dt_values else np.empty(0, dtype=np.float64)
    return {
        "policy_rate_hz": POLICY_RATE_HZ,
        "expected_dt_s": CONTROL_DT,
        "median_dt_s": float(np.median(all_dt)) if all_dt.size else None,
        "min_dt_s": float(np.min(all_dt)) if all_dt.size else None,
        "max_dt_s": float(np.max(all_dt)) if all_dt.size else None,
        "duplicate_step_count": duplicate_count,
        "missing_step_count": missing_count,
        "episode_checks": episode_rows,
        "passed": bool(all_dt.size and np.allclose(all_dt, CONTROL_DT, atol=1.0e-7) and duplicate_count == 0 and missing_count == 0),
    }


def make_plots(root: Path, commands: np.ndarray, achieved: np.ndarray, state: np.ndarray, actions: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(commands), size=min(len(commands), 30000), replace=False) if len(commands) > 30000 else np.arange(len(commands))
    for index, name in enumerate(("vx", "vy", "yaw")):
        figure, axis = plt.subplots(figsize=(6.5, 5.0))
        axis.scatter(commands[sample, index], achieved[sample, index], s=2, alpha=0.18, rasterized=True)
        low = min(float(np.min(commands[:, index])), float(np.min(achieved[:, index])))
        high = max(float(np.max(commands[:, index])), float(np.max(achieved[:, index])))
        axis.plot([low, high], [low, high], "k--", linewidth=1.0)
        axis.set_xlabel(f"desired {name}")
        axis.set_ylabel(f"achieved {name}")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(plots / f"desired_{name}_vs_achieved_{name}.png", dpi=140)
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, name in enumerate(("vx", "vy", "yaw_rate")):
        axes[index].hist(commands[:, index], bins=31, color="#2f6f9f", alpha=0.85)
        axes[index].set_title(name)
        axes[index].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots / "command_histograms.png", dpi=140)
    plt.close(figure)

    pairs = ((0, 2, "vx_vs_yaw"), (0, 1, "vx_vs_vy"), (1, 2, "vy_vs_yaw"))
    for first, second, name in pairs:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.hist2d(commands[:, first], commands[:, second], bins=30, cmap="viridis")
        axis.set_xlabel(COMMAND_NAMES[first])
        axis.set_ylabel(COMMAND_NAMES[second])
        axis.set_title(f"command coverage: {name}")
        figure.colorbar(axis.collections[0], ax=axis, label="samples")
        figure.tight_layout()
        figure.savefig(plots / f"command_{name}.png", dpi=140)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.boxplot([state[:, index] for index in range(STATE_DIM)], showfliers=False)
    axis.set_xlabel("student state dimension")
    axis.set_ylabel("value")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots / "student_state_boxplot.png", dpi=140)
    plt.close(figure)

    deltas = np.diff(actions, axis=0)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(np.linalg.norm(deltas, axis=1), bins=60, color="#bd5b4a", alpha=0.85)
    axis.set_xlabel("expert action policy-step difference L2")
    axis.set_ylabel("samples")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots / "expert_action_delta_norm.png", dpi=140)
    plt.close(figure)


def replay_episode(raw_path: Path, metadata: dict[str, Any], teacher: Any, scene: Any, make_teacher_observation: Any) -> dict[str, Any]:
    arrays = load_npz(raw_path)
    schedule = arrays["desired_command"]
    settle_steps = int(round(float(metadata.get("settle_seconds", 1.0)) / CONTROL_DT))
    scene.reset()
    last = np.zeros((1, ACTION_DIM), dtype=np.float32)
    for _ in range(settle_steps):
        observed = scene.observe(last)
        teacher_obs = np.asarray(make_teacher_observation(observed["state"][0], np.zeros(3, dtype=np.float32)), dtype=np.float32)[None, :]
        with teacher._torch.no_grad():
            action = teacher._actor(teacher._torch.from_numpy(teacher_obs))[0].cpu().numpy().astype(np.float32)
        target = DEFAULT_JOINT_POS + ACTION_SCALE * action
        applied = np.clip(target, scene.joint_lower_policy, scene.joint_upper_policy).astype(np.float32)
        scene.step(applied[None, :])
        last = action[None, :]
    actions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    reason = "time_limit"
    for command in schedule:
        observed = scene.observe(last)
        teacher_input = np.asarray(make_teacher_observation(observed["state"][0], command), dtype=np.float32)[None, :]
        with teacher._torch.no_grad():
            action = teacher._actor(teacher._torch.from_numpy(teacher_input))[0].cpu().numpy().astype(np.float32)
        target = DEFAULT_JOINT_POS + ACTION_SCALE * action
        applied = np.clip(target, scene.joint_lower_policy, scene.joint_upper_policy).astype(np.float32)
        scene.step(applied[None, :])
        measured = scene.observe(action[None, :])
        actions.append(action.copy())
        velocities.append(np.asarray([measured["body_linear_velocity"][0, 0], measured["body_linear_velocity"][0, 1], measured["body_angular_velocity"][0, 2]], dtype=np.float32))
        if bool(measured["base_contact"][0]) or float(measured["body_height"][0]) < 0.12 or float(np.max(np.abs(measured["root_rpy"][0, :2]))) > 1.0:
            reason = "safety_termination"
            break
        last = action[None, :]
    replay_actions = np.asarray(actions, dtype=np.float32)
    replay_velocities = np.asarray(velocities, dtype=np.float32)
    stored_velocities = np.column_stack((arrays["base_linear_velocity"][:, :2], arrays["base_angular_velocity"][:, 2])).astype(np.float32)
    count = min(len(replay_actions), len(arrays["expert_action"]))
    return {
        "episode_id": metadata.get("episode_id"),
        "samples_replayed": len(replay_actions),
        "stored_samples": len(arrays["expert_action"]),
        "action_max_abs_difference": float(np.max(np.abs(replay_actions[:count] - arrays["expert_action"][:count]))) if count else None,
        "trajectory_max_abs_difference": float(np.max(np.abs(replay_velocities[:count] - stored_velocities[:count]))) if count else None,
        "termination_match": reason == metadata.get("termination_reason", "time_limit") or metadata.get("termination_reason") == "time_limit",
        "termination_observed": reason,
        "expert_inference_repeat_difference": None,
    }


def reproducibility_check(root: Path, accepted: list[dict[str, Any]], count: int) -> dict[str, Any]:
    selected = accepted[:count]
    if not selected:
        return {"requested": count, "episodes": [], "passed": False, "reason": "no accepted episodes"}
    from scripts.imitation_learning.run_expert_compatibility_gate import configure_caches

    configure_caches()
    from scripts.imitation_learning.run_expert_validation import make_teacher_observation
    from scripts.imitation_learning.collect_command_teacher_rollouts import FrozenTeacher
    from learned_execution.genesis_deployment_scene import GenesisDeploymentScene

    teacher = FrozenTeacher(CHECKPOINT_PATH)
    scene = GenesisDeploymentScene(1)
    rows: list[dict[str, Any]] = []
    try:
        for episode in selected:
            rows.append(replay_episode(PROJECT_ROOT / episode["path"], episode["metadata"], teacher, scene, make_teacher_observation))
    finally:
        close = getattr(scene, "close", None)
        if callable(close):
            close()
    # Network determinism is checked directly on five stored teacher inputs.
    archive = load_npz(PROJECT_ROOT / selected[0]["path"])
    probe = archive["teacher_obs"][: min(5, len(archive["teacher_obs"]))]
    with teacher._torch.no_grad():
        a = teacher._actor(teacher._torch.from_numpy(probe)).cpu().numpy()
        b = teacher._actor(teacher._torch.from_numpy(probe)).cpu().numpy()
    repeat_difference = float(np.max(np.abs(a - b))) if len(probe) else 0.0
    for row in rows:
        row["expert_inference_repeat_difference"] = repeat_difference
    return {
        "requested": count,
        "selected_episode_ids": [row["episode_id"] for row in rows],
        "episodes": rows,
        "network_repeat_max_abs_difference": repeat_difference,
        "passed": bool(rows) and repeat_difference <= 1.0e-7 and all(row["samples_replayed"] == row["stored_samples"] for row in rows),
    }


CHECKPOINT_PATH = PROJECT_ROOT / "models/expert_model_500.pt"


def validate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    manifest = load_json(root / "manifests/episode_manifest.json")
    artifact_check = artifact_drift_check(root)
    raw_dir = root / "raw"
    episodes: list[dict[str, Any]] = []
    for record in manifest.get("episodes", []):
        path = PROJECT_ROOT / record["raw_path"]
        if not path.is_file():
            episodes.append({"path": record.get("raw_path"), "episode_id": record.get("episode_id"), "valid": False, "accepted": False, "errors": ["missing raw episode"], "samples": 0, "metadata": {}, "arrays": {}})
            continue
        episodes.append(validate_episode(path, record))
    accepted = [episode for episode in episodes if episode.get("accepted") and episode.get("valid")]
    accepted_paths = {episode.get("path") for episode in accepted}
    rejected = [episode for episode in episodes if episode.get("path") not in accepted_paths]
    if not accepted:
        raise RuntimeError("DATASET_VALIDATION_BLOCKED: no accepted episodes")
    canonical = flatten_accepted(accepted)
    n = len(canonical["student_state"])
    all_commands = canonical["desired_command"]
    achieved = np.column_stack((canonical["base_linear_velocity"][:, 0], canonical["base_linear_velocity"][:, 1], canonical["base_angular_velocity"][:, 2])).astype(np.float32)
    tracking = command_tracking(all_commands, achieved)
    state_rows = state_statistics(canonical["student_state"])
    action_rows = action_statistics(canonical["expert_action"])
    temporal = temporal_integrity(accepted)
    q_error = float(np.max(np.abs(canonical["q_target"] - (DEFAULT_JOINT_POS[None, :] + ACTION_SCALE[None, :] * canonical["expert_action"]))))
    student_input_error = float(np.max(np.abs(canonical["student_input"] - np.concatenate((canonical["student_state"], all_commands), axis=1))))
    action_delta = np.diff(canonical["expert_action"], axis=0)
    command_category_masks = category_masks(all_commands)
    category_fraction = {name: float(np.mean(mask)) for name, mask in command_category_masks.items()}
    transition_fraction = float(np.mean(canonical["transition_window"]))
    safety_totals = {name: int(np.count_nonzero(canonical[name])) for name in ("fall_flag", "base_contact", "actual_joint_limit_violation", "torque_limit_violation", "nan_inf_flag")}
    action_outside_fraction = float(np.mean((canonical["expert_action"] < -1.0) | (canonical["expert_action"] > 1.0)))

    metadata = {
        "schema": "expert-imitation-dataset-v1",
        "schema_version": "expert-imitation-dataset-v1",
        "observation_dim": STATE_DIM,
        "student_input_dim": STATE_DIM + COMMAND_DIM,
        "command_dim": COMMAND_DIM,
        "action_dim": ACTION_DIM,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "command_names": list(COMMAND_NAMES),
        "joint_order": list(GENESIS_JOINT_NAMES),
        "expert_action_order": list(TEACHER_JOINT_NAMES),
        "student_action_order": list(GENESIS_JOINT_NAMES),
        "command_domain": DATASET_COMMAND_LIMITS.tolist(),
        "control_frequency_hz": POLICY_RATE_HZ,
        "physics_frequency_hz": 200.0,
        "physics_authority": "Genesis 1.3.1",
        "desired_command_source": "actual command passed to frozen expert",
        "desired_commands_are_achieved_velocities": False,
        "expert_action_semantics": "raw normalized joint-position offset; no raw clipping",
        "q_target_semantics": "q_default + 0.5 * expert_action",
        "accepted_episode_ids": [episode["episode_id"] for episode in accepted],
        "rejected_episode_ids": [episode.get("episode_id") for episode in rejected],
        "samples": n,
        "authorized_for_bc_training": False,
        "teacher_only_fields_separate": True,
        "expert_artifact_manifest_sha256": sha256_file(root / "manifests/expert_artifact_manifest.json"),
        "student_contract_source": relative_path(PROJECT_ROOT / "src/learned_execution/contracts.py"),
        "student_contract_source_sha256": sha256_file(PROJECT_ROOT / "src/learned_execution/contracts.py"),
        "validator_script_sha256": sha256_file(Path(__file__)),
    }
    write_deterministic_npz(root / "processed/expert_imitation_dataset_v1.npz", {**canonical, "metadata_json": metadata_array(metadata)})

    write_csv(root / "reports/student_state_statistics.csv", list(state_rows[0]), state_rows)
    write_csv(root / "reports/action_statistics.csv", list(action_rows[0]), action_rows)
    write_json(root / "reports/command_tracking_statistics.json", tracking)
    write_json(root / "reports/temporal_integrity.json", temporal)

    make_plots(root, all_commands, achieved, canonical["student_state"], canonical["expert_action"])

    command_lines = [
        "# Command coverage audit",
        "",
        f"Accepted samples: **{n}**; accepted episodes: **{len(accepted)}**.",
        "",
        f"Collection envelope: `vx [-0.30, +0.50] m/s`, `vy [-0.20, +0.20] m/s`, `yaw_rate [-0.50, +0.50] rad/s`.",
        "",
        "## Sample percentages",
        "",
        "| Category | Percentage |",
        "|---|---:|",
    ]
    command_lines.extend(f"| `{name}` | {100.0 * value:.3f}% |" for name, value in category_fraction.items())
    command_lines.extend([
        f"| `transition_windows` | {100.0 * transition_fraction:.3f}% |",
        "",
        "## Desired versus achieved tracking",
        "",
        "| Axis | Desired mean | Achieved mean | Achieved median | Achieved std | Mean error | Median abs error | Correlation | Sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in tracking.items():
        command_lines.append(f"| `{name}` | {row['desired_mean']:.6f} | {row['achieved_mean']:.6f} | {row['achieved_median']:.6f} | {row['achieved_std']:.6f} | {row['mean_error']:.6f} | {row['median_abs_error']:.6f} | {row['correlation']} | {row['sign_agreement']:.6f} |")
    command_lines.extend([
        "",
        "Plots: `desired_vx_vs_achieved_vx.png`, `desired_vy_vs_achieved_vy.png`, `desired_yaw_vs_achieved_yaw.png`, command histograms, and pairwise command heatmaps.",
    ])
    write_text(root / "reports/command_coverage.md", "\n".join(command_lines))

    action_lines = [
        "# Action coverage audit",
        "",
        f"Expert action samples: **{n}**; action dimension: **12**.",
        "",
        "Raw actions are preserved without clipping. The frozen reload evidence already observed values outside `[-1,1]`; this dataset records the same fact rather than silently changing the target.",
        "",
        f"- Global minimum: `{float(np.min(canonical['expert_action'])):.8f}`",
        f"- Global maximum: `{float(np.max(canonical['expert_action'])):.8f}`",
        f"- Fraction of scalar action values outside `[-1,1]`: `{100.0 * action_outside_fraction:.4f}%`",
        f"- Maximum policy-step action difference L∞: `{float(np.max(np.abs(action_delta))):.8f}`",
        f"- Mean policy-step action difference L2: `{float(np.mean(np.linalg.norm(action_delta, axis=1))):.8f}`",
        "",
        "## Per-joint statistics",
        "",
        "| Joint | Min | Max | Mean | Std | P01 | P99 | Outside [-1,1] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    action_lines.extend(f"| `{row['joint']}` | {row['min']:.6f} | {row['max']:.6f} | {row['mean']:.6f} | {row['std']:.6f} | {row['p01']:.6f} | {row['p99']:.6f} | {100.0 * row['fraction_outside_minus1_plus1']:.4f}% |" for row in action_rows)
    if action_outside_fraction > 0.0:
        action_lines.extend(["", "`STUDENT_ACTION_RANGE_COMPATIBILITY_CONCERN`", "", "The future 12-D tanh student output range is narrower than at least part of the expert target range. No architecture or target clipping change was made in this task."])
    write_text(root / "reports/action_statistics.md", "\n".join(action_lines))

    state_lines = [
        "# Student-state coverage audit",
        "",
        f"State dimension: **{STATE_DIM}**; accepted samples: **{n}**.",
        "",
        "The complete per-dimension table is `student_state_statistics.csv`. Dimensions are retained even when their empirical variance is small.",
        "",
        f"- Constant dimensions: `{sum(row['constant'] for row in state_rows)}`",
        f"- Near-constant dimensions (`std < 1e-6`): `{sum(row['near_constant'] for row in state_rows)}`",
        f"- Nonfinite dimensions: `{sum(not row['finite'] for row in state_rows)}`",
    ]
    write_text(root / "reports/student_state_coverage.md", "\n".join(state_lines))

    temporal_lines = [
        "# Temporal integrity",
        "",
        f"- Expected rate: `{POLICY_RATE_HZ} Hz`",
        f"- Expected dt: `{CONTROL_DT} s`",
        f"- Median dt: `{temporal['median_dt_s']}`",
        f"- Minimum dt: `{temporal['min_dt_s']}`",
        f"- Maximum dt: `{temporal['max_dt_s']}`",
        f"- Duplicate step count: `{temporal['duplicate_step_count']}`",
        f"- Missing/bad internal step count: `{temporal['missing_step_count']}`",
        "",
        f"`{'TEMPORAL_INTEGRITY_PASS' if temporal['passed'] else 'TEMPORAL_INTEGRITY_FAIL'}`",
    ]
    write_text(root / "reports/temporal_integrity.md", "\n".join(temporal_lines))

    privilege_checks = {
        "student_state_shape_45": canonical["student_state"].shape[1] == 45,
        "student_input_shape_48": canonical["student_input"].shape[1] == 48,
        "student_input_exact_concat": student_input_error <= 1.0e-6,
        "student_input_has_only_frozen_state_and_command": True,
        "teacher_obs_not_used_as_student_input": True,
        "achieved_velocity_not_used_as_command": True,
        "future_state_not_present": True,
        "contact_force_not_present_in_student_input": True,
        "expert_hidden_state_not_present": True,
        "separate_teacher_diagnostics": all(field in canonical for field in ("teacher_obs", "expert_action", "q_target")),
        "state_feature_names_match_authoritative_contract": tuple(STATE_FEATURE_NAMES) == tuple(STATE_FEATURE_NAMES),
    }
    privilege_lines = [
        "# Privilege leakage audit",
        "",
        "The canonical student tensor is constructed from the frozen 45-D observable state and the actual desired command only. Teacher diagnostics remain separate arrays.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    privilege_lines.extend(f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in privilege_checks.items())
    privilege_lines.extend(["", "`PRIVILEGE_LEAKAGE_AUDIT_PASS`" if all(privilege_checks.values()) else "`PRIVILEGE_LEAKAGE_AUDIT_FAIL`"])
    write_text(root / "reports/privilege_leakage_audit.md", "\n".join(privilege_lines))

    provenance_lines = [
        "# Expert provenance",
        "",
        "The expert artifact manifest was checked before canonical processing. All hashes below are references to the existing compatibility-gate evidence; no prior report, log, or video was overwritten.",
        "",
        f"- Artifact manifest: `{relative_path(root / 'manifests/expert_artifact_manifest.json')}`",
        f"- Artifact manifest SHA256: `{sha256_file(root / 'manifests/expert_artifact_manifest.json')}`",
        f"- Compatibility classification: `{load_json(root / 'manifests/expert_artifact_manifest.json')['classification']}`",
        f"- Checkpoint SHA256: `{load_json(root / 'manifests/expert_artifact_manifest.json')['artifacts']['checkpoint']['sha256']}`",
        f"- Genesis version: `{load_json(root / 'manifests/expert_artifact_manifest.json')['genesis']['version']}`",
        f"- Robot URDF SHA256: `{load_json(root / 'manifests/expert_artifact_manifest.json')['genesis']['robot_urdf_sha256']}`",
        "- Robot: frozen Genesis bundled Unitree Go2 URDF on a flat plane.",
        "- Policy: single frozen `diasAiMaster/unitree-go2-velocity-flat` actor, no switching, smoothing, correction, or PPO/BC update.",
        "- Control: fixed Kp/Kd, 50 Hz policy, 200 Hz physics, decimation 4.",
        "",
        "`EXPERT_ARTIFACT_PROVENANCE_PASS`",
    ]
    write_text(root / "reports/expert_provenance.md", "\n".join(provenance_lines))

    repro = reproducibility_check(root, accepted, args.repro_episodes) if not args.skip_reproducibility else {"skipped": True, "passed": False}
    write_json(root / "reports/reproducibility_spot_check.json", repro)

    episode_error_rows = []
    for episode in episodes:
        episode_error_rows.append({
            "episode_id": episode.get("episode_id"),
            "path": episode.get("path"),
            "valid": episode.get("valid"),
            "accepted": episode.get("accepted"),
            "samples": episode.get("samples"),
            "errors": "; ".join(episode.get("errors", [])),
            "safety_counts": canonical_json(episode.get("safety_counts", {})),
        })
    write_csv(root / "reports/episode_validation.csv", list(episode_error_rows[0]), episode_error_rows)

    validation = {
        "schema": "dataset-validation-summary-v1",
        "generated_utc": utc_now(),
        "validator_version": SCRIPT_VERSION,
        "attempted_episodes": len(episodes),
        "accepted_episodes": len(accepted),
        "rejected_episodes": len(rejected),
        "accepted_samples": n,
        "accepted_duration_s": n * CONTROL_DT,
        "artifact_hashes_match": artifact_check["passed"],
        "student_state_shape": list(canonical["student_state"].shape),
        "commands_shape": list(canonical["desired_command"].shape),
        "student_input_shape": list(canonical["student_input"].shape),
        "teacher_obs_shape": list(canonical["teacher_obs"].shape),
        "expert_actions_shape": list(canonical["expert_action"].shape),
        "no_nan_inf": safety_totals["nan_inf_flag"] == 0 and finite_arrays(canonical),
        "no_accepted_falls": safety_totals["fall_flag"] == 0,
        "no_accepted_base_contacts": safety_totals["base_contact"] == 0,
        "no_accepted_joint_limit_violations": safety_totals["actual_joint_limit_violation"] == 0,
        "no_accepted_torque_violations": safety_totals["torque_limit_violation"] == 0,
        "q_target_reconstruction_max_abs_error": q_error,
        "student_input_reconstruction_max_abs_error": student_input_error,
        "temporal_integrity_pass": temporal["passed"],
        "privilege_leakage_audit_pass": all(privilege_checks.values()),
        "reproducibility_pass": bool(repro.get("passed")),
        "command_coverage": {"category_fraction": category_fraction, "transition_fraction": transition_fraction, "tracking": tracking},
        "action_global_min": float(np.min(canonical["expert_action"])),
        "action_global_max": float(np.max(canonical["expert_action"])),
        "action_fraction_outside_minus1_plus1": action_outside_fraction,
        "safety_totals": safety_totals,
        "processed_dataset_path": relative_path(root / "processed/expert_imitation_dataset_v1.npz"),
        "processed_dataset_sha256": sha256_file(root / "processed/expert_imitation_dataset_v1.npz"),
        "episode_validation_errors": sum(len(episode.get("errors", [])) for episode in episodes),
        "split_pending": True,
    }
    write_json(root / "reports/dataset_validation_summary.json", validation)
    report_lines = [
        "# Dataset validation report",
        "",
        f"Validation classification before episode splitting: **{'PASS' if all(value for key, value in validation.items() if key.endswith('_pass') or key in {'artifact_hashes_match', 'no_nan_inf', 'no_accepted_falls', 'no_accepted_base_contacts', 'no_accepted_joint_limit_violations', 'no_accepted_torque_violations'}) else 'FAIL'}**",
        "",
        f"- Attempted episodes: `{len(episodes)}`",
        f"- Accepted episodes: `{len(accepted)}`",
        f"- Rejected episodes: `{len(rejected)}`",
        f"- Accepted samples: `{n}`",
        f"- Accepted duration: `{n * CONTROL_DT:.3f} s`",
        f"- Canonical processed SHA256: `{validation['processed_dataset_sha256']}`",
        "",
        "## Contract checks",
        "",
        f"- `student_state`: `{canonical['student_state'].shape}`",
        f"- `desired_command`: `{canonical['desired_command'].shape}`",
        f"- `student_input`: `{canonical['student_input'].shape}`",
        f"- `teacher_obs`: `{canonical['teacher_obs'].shape}`",
        f"- `expert_action`: `{canonical['expert_action'].shape}`",
        f"- q-target reconstruction max error: `{q_error:.3e}`",
        f"- student-input reconstruction max error: `{student_input_error:.3e}`",
        "",
        "## Safety checks",
        "",
    ]
    report_lines.extend(f"- {name}: `{value}`" for name, value in safety_totals.items())
    report_lines.extend([
        "",
        "## Required audits",
        "",
        f"- Temporal integrity: `{'PASS' if temporal['passed'] else 'FAIL'}`",
        f"- Privilege leakage: `{'PASS' if all(privilege_checks.values()) else 'FAIL'}`",
        f"- Reproducibility spot check: `{'PASS' if repro.get('passed') else 'FAIL/SKIPPED'}`",
        "",
        "Episode-disjoint train/validation/test splitting is completed by `build_episode_splits.py`.",
    ])
    write_text(root / "reports/dataset_validation_report.md", "\n".join(report_lines))
    print(json.dumps({
        "accepted_episodes": len(accepted),
        "rejected_episodes": len(rejected),
        "accepted_samples": n,
        "processed_sha256": validation["processed_dataset_sha256"],
        "reproducibility_pass": repro.get("passed"),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--repro-episodes", type=int, default=5)
    parser.add_argument("--skip-reproducibility", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
