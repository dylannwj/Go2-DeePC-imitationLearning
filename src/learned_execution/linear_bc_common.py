"""Shared constants and read-only helpers for the linear-output BC revision."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = REPO_ROOT
DATASET_PATH = REPO_ROOT / "data/imitation/expert_imitation_dataset_v1.npz"
DATASET_SPLIT_ROOT = REPO_ROOT / "data/imitation/splits"
EXPECTED_DATASET_SHA256 = "7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99"
EXPECTED_COUNTS = {"train": 192_000, "val": 24_000, "test": 24_000}

STATE_DIM = 45
COMMAND_DIM = 3
INPUT_DIM = 48
ACTION_DIM = 12

JOINT_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
COMMAND_NAMES = ("vx", "vy", "yaw_rate")
STATE_FEATURE_NAMES = (
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "base_linear_velocity_x",
    "base_linear_velocity_y",
    "base_linear_velocity_z",
    "base_angular_velocity_x",
    "base_angular_velocity_y",
    "base_angular_velocity_z",
    *(f"joint_pos_rel_{name}" for name in JOINT_NAMES),
    *(f"joint_vel_rel_{name}" for name in JOINT_NAMES),
    *(f"previous_action_{name}" for name in JOINT_NAMES),
)
INPUT_FEATURE_NAMES = STATE_FEATURE_NAMES + tuple(f"command_{name}" for name in COMMAND_NAMES)

# Decoder pose and named map from the validated expert execution environment.
# This is separate from the frozen student relative-position reference.
POLICY_TO_SCENE = np.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
Q_DEFAULT_POLICY = np.asarray(
    [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
    dtype=np.float32,
)
Q_DEFAULT_CANONICAL = Q_DEFAULT_POLICY[POLICY_TO_SCENE]
STUDENT_JOINT_REFERENCE = np.asarray(
    [
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        1.0,
        -1.5,
        0.0,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)
ACTION_SCALE = np.full(ACTION_DIM, 0.5, dtype=np.float32)
KP = np.tile(np.asarray([20.0, 20.0, 40.0], dtype=np.float32), 4)
KD = np.tile(np.asarray([1.0, 1.0, 2.0], dtype=np.float32), 4)
EFFORT_LIMIT = np.tile(np.asarray([23.5, 23.5, 45.43], dtype=np.float32), 4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(values[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            converted = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(json_safe(value), separators=(",", ":"))
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                converted[field] = value
            writer.writerow(converted)


def load_dataset(path: Path = DATASET_PATH) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_split_indices(name: str) -> np.ndarray:
    with np.load(DATASET_SPLIT_ROOT / f"{name}_indices.npz", allow_pickle=False) as archive:
        return np.asarray(archive["row_indices"], dtype=np.int64)


def raw_student_targets(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return untouched raw actions in the canonical Genesis joint order."""

    target = np.asarray(data["expert_action_student_order"], dtype=np.float32)
    if target.shape[-1] != ACTION_DIM:
        raise ValueError(f"unexpected target shape {target.shape}")
    return target


def make_model(torch: Any) -> Any:
    """Construct exactly 48 -> 256 ELU -> 256 ELU -> 12 linear."""

    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, ACTION_DIM),
    )


def command_masks(commands: np.ndarray) -> dict[str, np.ndarray]:
    """Return mutually readable command-class masks for offline reports."""

    value = np.asarray(commands, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != COMMAND_DIM:
        raise ValueError(f"commands must have shape [N,3], got {value.shape}")
    active = np.abs(value) > 1.0e-7
    vx, vy, yaw = value.T
    zero_vy = ~active[:, 1]
    zero_yaw = ~active[:, 2]
    masks = {
        "standing": ~np.any(active, axis=1),
        "forward": (vx > 0.0) & zero_vy & zero_yaw,
        "backward": (vx < 0.0) & zero_vy & zero_yaw,
        "lateral_left": (~active[:, 0]) & (vy > 0.0) & zero_yaw,
        "lateral_right": (~active[:, 0]) & (vy < 0.0) & zero_yaw,
        "yaw_left": (~active[:, 0]) & (~active[:, 1]) & (yaw > 0.0),
        "yaw_right": (~active[:, 0]) & (~active[:, 1]) & (yaw < 0.0),
        "forward_yaw": active[:, 0] & zero_vy & active[:, 2],
    }
    assigned = np.zeros(len(value), dtype=bool)
    for mask in masks.values():
        assigned |= mask
    masks["mixed"] = ~assigned
    masks["lateral"] = masks["lateral_left"] | masks["lateral_right"]
    masks["yaw"] = masks["yaw_left"] | masks["yaw_right"]
    masks["arcs"] = masks["forward_yaw"]
    return masks


def transition_masks(
    episode_ids: np.ndarray,
    step_indices: np.ndarray,
    commands: np.ndarray,
    *,
    half_window_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark samples within +/- 0.5 s of a command boundary."""

    episode_ids = np.asarray(episode_ids)
    step_indices = np.asarray(step_indices)
    commands = np.asarray(commands)
    transition = np.zeros(len(commands), dtype=bool)
    boundary = np.zeros(len(commands), dtype=bool)
    for episode in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode)
        if len(rows) < 2:
            continue
        local_change = np.any(np.abs(np.diff(commands[rows], axis=0)) > 1.0e-7, axis=1)
        for local_index in np.flatnonzero(local_change) + 1:
            center = int(rows[local_index])
            boundary[center] = True
            distance = np.abs(step_indices[rows] - step_indices[center])
            transition[rows[distance <= half_window_steps]] = True
    return transition, boundary


def action_distribution(actions: np.ndarray, names: tuple[str, ...] = JOINT_NAMES) -> list[dict[str, Any]]:
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(names):
        raise ValueError(f"actions must have shape [N,{len(names)}], got {value.shape}")
    rows = []
    for index, name in enumerate(names):
        column = value[:, index]
        rows.append(
            {
                "index": index,
                "joint": name,
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "p0.1": float(np.percentile(column, 0.1)),
                "p1": float(np.percentile(column, 1.0)),
                "p50": float(np.percentile(column, 50.0)),
                "p99": float(np.percentile(column, 99.0)),
                "p99.9": float(np.percentile(column, 99.9)),
                "fraction_outside_minus1_plus1": float(np.mean((column < -1.0) | (column > 1.0))),
            }
        )
    return rows


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != ACTION_DIM:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(target),):
        raise ValueError(f"mask shape mismatch: {mask.shape}")
    if not np.any(mask):
        return {"samples": 0, "mse": None, "mae": None, "mean_action_l2": None, "median_action_l2": None, "p95_action_l2": None, "p99_action_l2": None, "max_action_l2": None, "per_joint": []}
    error = prediction[mask] - target[mask]
    l2 = np.linalg.norm(error, axis=1)
    per_joint = []
    for index, name in enumerate(JOINT_NAMES):
        per_joint.append(
            {
                "index": index,
                "joint": name,
                "mse": float(np.mean(error[:, index] ** 2)),
                "mae": float(np.mean(np.abs(error[:, index]))),
                "pearson": _pearson(prediction[mask, index], target[mask, index]),
            }
        )
    return {
        "samples": int(np.count_nonzero(mask)),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "mean_action_l2": float(np.mean(l2)),
        "median_action_l2": float(np.median(l2)),
        "p95_action_l2": float(np.percentile(l2, 95.0)),
        "p99_action_l2": float(np.percentile(l2, 99.0)),
        "max_action_l2": float(np.max(l2)),
        "per_joint": per_joint,
    }


def behavior_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    commands: np.ndarray,
    transition: np.ndarray,
) -> dict[str, dict[str, Any]]:
    masks = command_masks(commands)
    result = {name: regression_metrics(prediction, target, mask) for name, mask in masks.items()}
    result["transitions"] = regression_metrics(prediction, target, transition)
    result["steady_state"] = regression_metrics(prediction, target, ~transition)
    return result


def flatten_per_joint(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for joint in metrics.get("per_joint", []):
        rows.append(dict(joint))
    return rows
