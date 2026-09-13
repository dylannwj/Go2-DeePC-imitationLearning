#!/usr/bin/env python3
"""Train and evaluate a command-conditioned Go2 behaviour-cloning policy.

This script intentionally has no PPO runner, environment randomization, or
teacher-update path.  The teacher and the command-conditioned dataset are
read-only inputs.  The learned policy receives the 42-D state observation and
the 3-D desired command in the same layout as the recovered teacher:

    [state[0:6], command[0:3], state[6:42]] -> 12-D joint-position target

The closed-loop evaluation uses the same deployment-equivalent MuJoCo scene
and PD/action mapping as ``collect_command_teacher_rollouts.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "command-bc-v1.0"
OBS_DIM = 42
COMMAND_DIM = 3
ACTION_DIM = 12
INPUT_DIM = OBS_DIM + COMMAND_DIM
COMMAND_AXES = ("vx", "vy", "yaw")
SENSITIVITY_THRESHOLD = 1.0e-3
SENSITIVITY_FRACTION_THRESHOLD = 0.95
DEFAULT_SEED = 20260804
ARCHITECTURE = (INPUT_DIM, 512, 256, 128, ACTION_DIM)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def make_policy_input(observations: np.ndarray, commands: np.ndarray) -> np.ndarray:
    """Insert commands into the recovered teacher's 45-D observation layout."""

    observations = np.asarray(observations, dtype=np.float32)
    commands = np.asarray(commands, dtype=np.float32)
    if observations.shape[:-1] != commands.shape[:-1]:
        raise ValueError("observations and commands must have matching leading dimensions")
    if observations.shape[-1] != OBS_DIM or commands.shape[-1] != COMMAND_DIM:
        raise ValueError("unexpected observation or command dimension")
    return np.concatenate((observations[..., :6], commands, observations[..., 6:]), axis=-1)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validation_checks(data: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = ("observations", "commands", "actions")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    missing = [key for key in required if key not in data]
    add("required arrays present", not missing, "missing=" + (", ".join(missing) or "none"))
    if missing:
        return checks, {"valid": False, "checks": checks}

    observations = data["observations"]
    commands = data["commands"]
    actions = data["actions"]
    if observations.ndim == 3:
        episodes, steps = observations.shape[:2]
    else:
        episodes, steps = -1, -1

    expected = {
        "observations": [episodes, steps, OBS_DIM],
        "commands": [episodes, steps, COMMAND_DIM],
        "actions": [episodes, steps, ACTION_DIM],
    }
    shapes_ok = True
    for key, shape in expected.items():
        actual = list(data[key].shape)
        passed = actual == shape
        shapes_ok &= passed
        add(f"{key} shape", passed, f"actual={actual}; expected={shape}")
    add("required shapes", shapes_ok, "obs[N,T,42], commands[N,T,3], actions[N,T,12]")

    for key in ("observations", "commands", "actions"):
        array = data[key]
        passed = np.issubdtype(array.dtype, np.number) and bool(np.isfinite(array).all())
        add(f"{key} finite", passed, f"dtype={array.dtype}; finite={passed}")

    layout_ok = False
    layout_detail = "policy_observations not present"
    if "policy_observations" in data and shapes_ok:
        expected_policy = make_policy_input(observations, commands)
        difference = float(np.max(np.abs(expected_policy - data["policy_observations"])))
        layout_ok = difference <= 1.0e-6
        layout_detail = f"max_abs_difference={difference:.3e}"
    add("teacher-compatible command layout", layout_ok, layout_detail)

    constant_commands = bool(np.all(commands == commands[:, :1])) if shapes_ok else False
    add("command constant within episode", constant_commands, str(constant_commands))

    episode_id_ok = True
    if "episode_ids" in data and shapes_ok:
        expected_ids = np.arange(episodes, dtype=data["episode_ids"].dtype)[:, None]
        episode_id_ok = bool(np.all(data["episode_ids"] == expected_ids))
    add("episode ids contiguous", episode_id_ok, str(episode_id_ok))

    timestamp_ok = True
    if "timestamps" in data and data["timestamps"].ndim == 2:
        timestamp_ok = bool(np.all(np.diff(data["timestamps"], axis=1) > 0.0))
    add("timestamps strictly increasing", timestamp_ok, str(timestamp_ok))

    finite_extras = True
    extra_details: list[str] = []
    for key in ("policy_observations", "raw_actions", "velocities", "next_velocities", "timestamps"):
        if key not in data:
            continue
        array = data[key]
        current = np.issubdtype(array.dtype, np.number) and bool(np.isfinite(array).all())
        finite_extras &= current
        extra_details.append(f"{key}={current}")
    add("numeric auxiliary arrays finite", finite_extras, "; ".join(extra_details) or "none")

    valid = bool(all(check["passed"] for check in checks))
    details = {
        "valid": valid,
        "episodes": int(episodes),
        "steps_per_episode": int(steps),
        "samples": int(episodes * steps) if episodes >= 0 else 0,
        "shapes": {key: list(value.shape) for key, value in data.items()},
        "checks": checks,
        "command_min": commands.min(axis=(0, 1)).tolist() if shapes_ok else None,
        "command_max": commands.max(axis=(0, 1)).tolist() if shapes_ok else None,
        "action_min": actions.min(axis=(0, 1)).tolist() if shapes_ok else None,
        "action_max": actions.max(axis=(0, 1)).tolist() if shapes_ok else None,
    }
    return checks, details


def write_dataset_validation_report(
    path: Path,
    dataset_path: Path,
    dataset_hash: str,
    details: dict[str, Any],
    phase: str,
) -> None:
    lines = [
        "# Command-Conditioned BC Dataset Validation",
        "",
        f"Validation phase: **{phase}**  ",
        f"Dataset: `{dataset_path}`  ",
        f"SHA-256: `{dataset_hash}`  ",
        "",
        "## Schema",
        "",
        f"- Episodes: `{details.get('episodes', 'unknown')}`",
        f"- Steps per episode: `{details.get('steps_per_episode', 'unknown')}`",
        f"- Samples: `{details.get('samples', 'unknown')}`",
        "- Required input: `observations + commands`",
        "- Required output: `actions` (12-D joint-position target)",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in details.get("checks", []):
        result = "PASS" if check["passed"] else "FAIL"
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {result} | {detail} |")
    lines.extend(
        [
            "",
            f"Overall validation: **{'VALID' if details.get('valid') else 'INVALID'}**",
            "",
            "The report is generated before the BC optimizer is constructed. The source NPZ is read-only; this training pipeline never writes to the dataset path.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def deterministic_split(episodes: int, seed: int) -> dict[str, list[int]]:
    if episodes < 10:
        raise ValueError("at least 10 episodes are required for an 80/10/10 split")
    generator = np.random.default_rng(seed)
    order = generator.permutation(episodes).tolist()
    train_count = int(0.8 * episodes)
    validation_count = int(0.1 * episodes)
    return {
        "train": [int(value) for value in order[:train_count]],
        "validation": [int(value) for value in order[train_count : train_count + validation_count]],
        "test": [int(value) for value in order[train_count + validation_count :]],
    }


def compute_stats(array: np.ndarray, episodes: list[int], name: str) -> dict[str, Any]:
    values = np.asarray(array[episodes], dtype=np.float32).reshape(-1, array.shape[-1])
    mean = values.mean(axis=0).astype(np.float32)
    raw_std = values.std(axis=0).astype(np.float32)
    std = np.maximum(raw_std, 1.0e-6).astype(np.float32)
    return {
        "schema": "go2_command_bc_normalization_v1",
        "name": name,
        "mean": mean,
        "std": std,
        "raw_std": raw_std,
        "count": int(values.shape[0]),
        "source_episode_ids": [int(value) for value in episodes],
        "epsilon": 1.0e-6,
    }


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def normalized_inputs(
    observations: np.ndarray,
    commands: np.ndarray,
    obs_stats: dict[str, Any],
    command_stats: dict[str, Any],
) -> np.ndarray:
    state_normalized = (
        np.asarray(observations, dtype=np.float32) - obs_stats["mean"]
    ) / obs_stats["std"]
    command_normalized = (
        np.asarray(commands, dtype=np.float32) - command_stats["mean"]
    ) / command_stats["std"]
    return make_policy_input(state_normalized, command_normalized).astype(np.float32)


def flatten_split(
    data: dict[str, np.ndarray],
    episodes: list[int],
    obs_stats: dict[str, Any],
    command_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations = data["observations"][episodes]
    commands = data["commands"][episodes]
    actions = data["actions"][episodes]
    inputs = normalized_inputs(observations, commands, obs_stats, command_stats)
    sample_episodes = np.repeat(np.asarray(episodes, dtype=np.int32), observations.shape[1])
    sample_steps = np.tile(np.arange(observations.shape[1], dtype=np.int32), len(episodes))
    return (
        inputs.reshape(-1, INPUT_DIM),
        actions.reshape(-1, ACTION_DIM).astype(np.float32),
        sample_episodes,
        sample_steps,
    )


def correlation_and_slope(desired: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    desired = np.asarray(desired, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if desired.size < 2 or np.std(desired) <= 1.0e-12:
        return math.nan, math.nan
    correlation = float(np.corrcoef(desired, actual)[0, 1])
    slope = float(np.cov(desired, actual, bias=True)[0, 1] / np.var(desired))
    return correlation, slope


def tracking_statistics(commands: np.ndarray, velocities: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    commands = np.asarray(commands, dtype=np.float64).reshape(-1, COMMAND_DIM)
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, COMMAND_DIM)
    for index, axis in enumerate(COMMAND_AXES):
        desired = commands[:, index]
        actual = velocities[:, index]
        error = actual - desired
        correlation, slope = correlation_and_slope(desired, actual)
        result[axis] = {
            "desired_mean": float(np.mean(desired)),
            "actual_mean": float(np.mean(actual)),
            "mean_error": float(np.mean(error)),
            "mean_abs_error": float(np.mean(np.abs(error))),
            "std_error": float(np.std(error)),
            "correlation": correlation,
            "slope": slope,
        }
    return result


def import_torch() -> Any:
    import torch

    return torch


def make_model(torch: Any) -> Any:
    class CommandBCPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(INPUT_DIM, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, ACTION_DIM),
            )

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    return CommandBCPolicy()


def evaluate_model(
    model: Any,
    torch: Any,
    inputs: Any,
    actions: Any,
    action_mean: Any,
    action_std: Any,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    sum_squared = 0.0
    sum_absolute = 0.0
    sum_action_l2 = 0.0
    max_action_l2 = 0.0
    sample_count = int(inputs.shape[0])
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            prediction = model(inputs[start:end]) * action_std + action_mean
            difference = prediction - actions[start:end]
            difference_cpu = difference.detach().cpu()
            sum_squared += float(torch.sum(difference_cpu * difference_cpu).item())
            sum_absolute += float(torch.sum(torch.abs(difference_cpu)).item())
            row_l2 = torch.linalg.vector_norm(difference_cpu, dim=1)
            sum_action_l2 += float(torch.sum(row_l2).item())
            max_action_l2 = max(max_action_l2, float(torch.max(row_l2).item()))
    element_count = sample_count * ACTION_DIM
    mse = sum_squared / element_count
    return {
        "loss": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(sum_absolute / element_count),
        "mean_action_l2": float(sum_action_l2 / sample_count),
        "max_action_l2": float(max_action_l2),
        "samples": sample_count,
    }


def train_model(
    model: Any,
    torch: Any,
    train_inputs: np.ndarray,
    train_actions: np.ndarray,
    validation_inputs: np.ndarray,
    validation_actions: np.ndarray,
    action_stats: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Any, list[dict[str, float]], int, int, float]:
    train_inputs_t = torch.from_numpy(train_inputs)
    train_actions_t = torch.from_numpy(train_actions)
    validation_inputs_t = torch.from_numpy(validation_inputs)
    validation_actions_t = torch.from_numpy(validation_actions)
    action_mean = torch.from_numpy(action_stats["mean"])
    action_std = torch.from_numpy(action_stats["std"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(config["scheduler_patience"])),
        min_lr=1.0e-6,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    early_stopping_patience = int(config["early_stopping_patience"])
    best_validation_loss = math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    optimizer_steps = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_inputs_t.shape[0], generator=generator)
        batch_loss_sum = 0.0
        for start in range(0, train_inputs_t.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(train_inputs_t[indices]) * action_std + action_mean
            loss = torch.mean((prediction - train_actions_t[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            batch_loss_sum += float(loss.detach().item()) * int(indices.shape[0])

        train_metrics = evaluate_model(
            model,
            torch,
            train_inputs_t,
            train_actions_t,
            action_mean,
            action_std,
            batch_size,
        )
        validation_metrics = evaluate_model(
            model,
            torch,
            validation_inputs_t,
            validation_actions_t,
            action_mean,
            action_std,
            batch_size,
        )
        scheduler.step(validation_metrics["loss"])
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batch_train_loss": float(batch_loss_sum / train_inputs_t.shape[0]),
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "validation_rmse": validation_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "validation_mae": validation_metrics["mae"],
            "train_mean_action_l2": train_metrics["mean_action_l2"],
            "validation_mean_action_l2": validation_metrics["mean_action_l2"],
        }
        history.append(row)
        if validation_metrics["loss"] < best_validation_loss - float(config["min_delta"]):
            best_validation_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:3d}/{epochs}: train_loss={train_metrics['loss']:.6e} "
                f"val_loss={validation_metrics['loss']:.6e} "
                f"val_l2={validation_metrics['mean_action_l2']:.6e}"
            )
        if epochs_without_improvement >= early_stopping_patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best model state")
    model.load_state_dict(best_state)
    return model, history, best_epoch, optimizer_steps, float(best_validation_loss)


def predict_actions(
    model: Any,
    torch: Any,
    observations: np.ndarray,
    commands: np.ndarray,
    obs_stats: dict[str, Any],
    command_stats: dict[str, Any],
    action_stats: dict[str, Any],
    batch_size: int = 8192,
) -> np.ndarray:
    inputs = normalized_inputs(observations, commands, obs_stats, command_stats).reshape(-1, INPUT_DIM)
    action_mean = torch.from_numpy(action_stats["mean"])
    action_std = torch.from_numpy(action_stats["std"])
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        inputs_t = torch.from_numpy(inputs)
        for start in range(0, inputs.shape[0], batch_size):
            output = model(inputs_t[start : start + batch_size]) * action_std + action_mean
            predictions.append(output.cpu().numpy().astype(np.float32))
    return np.concatenate(predictions, axis=0).reshape(observations.shape[:-1] + (ACTION_DIM,))


def command_sensitivity(
    model: Any,
    torch: Any,
    data: dict[str, np.ndarray],
    test_episodes: list[int],
    obs_stats: dict[str, Any],
    command_stats: dict[str, Any],
    action_stats: dict[str, Any],
    output_path: Path,
    plot_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations = data["observations"][test_episodes].reshape(-1, OBS_DIM)
    episode_ids = np.repeat(np.asarray(test_episodes, dtype=np.int32), data["observations"].shape[1])
    step_ids = np.tile(np.arange(data["observations"].shape[1], dtype=np.int32), len(test_episodes))
    zero = np.zeros(COMMAND_DIM, dtype=np.float32)
    probes = [
        ("vx_A_to_B", "A", zero, "B", np.asarray([0.1, 0.0, 0.0], dtype=np.float32), "vx"),
        ("vx_B_to_C", "B", np.asarray([0.1, 0.0, 0.0], dtype=np.float32), "C", np.asarray([0.2, 0.0, 0.0], dtype=np.float32), "vx"),
        ("vx_A_to_C", "A", zero, "C", np.asarray([0.2, 0.0, 0.0], dtype=np.float32), "vx"),
        ("vy_zero_to_positive", "A", zero, "D", np.asarray([0.0, 0.1, 0.0], dtype=np.float32), "vy"),
        ("yaw_zero_to_positive", "A", zero, "E", np.asarray([0.0, 0.0, 0.3], dtype=np.float32), "yaw"),
    ]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    plot_values: list[tuple[str, str, float]] = []
    for probe_id, name_a, command_a, name_b, command_b, axis in probes:
        commands_a = np.broadcast_to(command_a, (observations.shape[0], COMMAND_DIM)).copy()
        commands_b = np.broadcast_to(command_b, (observations.shape[0], COMMAND_DIM)).copy()
        prediction_a = predict_actions(
            model, torch, observations, commands_a, obs_stats, command_stats, action_stats
        )
        prediction_b = predict_actions(
            model, torch, observations, commands_b, obs_stats, command_stats, action_stats
        )
        differences = np.linalg.norm(prediction_a - prediction_b, axis=1)
        fraction = float(np.mean(differences > SENSITIVITY_THRESHOLD))
        entry = {
            "probe": probe_id,
            "axis": axis,
            "command_a_name": name_a,
            "command_a": command_a.tolist(),
            "command_b_name": name_b,
            "command_b": command_b.tolist(),
            "states": int(differences.size),
            "mean_action_l2": float(np.mean(differences)),
            "median_action_l2": float(np.median(differences)),
            "max_action_l2": float(np.max(differences)),
            "fraction_l2_gt_threshold": fraction,
            "threshold": SENSITIVITY_THRESHOLD,
            "pass": bool(
                np.mean(differences) > SENSITIVITY_THRESHOLD
                and fraction >= SENSITIVITY_FRACTION_THRESHOLD
            ),
        }
        summary[probe_id] = entry
        plot_values.append((probe_id, axis, float(np.mean(differences))))
        for index, difference in enumerate(differences.tolist()):
            rows.append(
                {
                    "state_episode": int(episode_ids[index]),
                    "state_step": int(step_ids[index]),
                    "probe": probe_id,
                    "axis": axis,
                    "command_a_name": name_a,
                    "command_a_vx": float(command_a[0]),
                    "command_a_vy": float(command_a[1]),
                    "command_a_yaw": float(command_a[2]),
                    "command_b_name": name_b,
                    "command_b_vx": float(command_b[0]),
                    "command_b_vy": float(command_b[1]),
                    "command_b_yaw": float(command_b[2]),
                    "action_l2": float(difference),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [value[0] for value in plot_values]
    values = [value[2] for value in plot_values]
    colors = ["#4472c4" if value[1] == "vx" else "#70ad47" if value[1] == "vy" else "#ed7d31" for value in plot_values]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(labels, values, color=colors)
    axis.axhline(SENSITIVITY_THRESHOLD, color="black", linestyle="--", linewidth=1, label="threshold")
    axis.set_ylabel("mean action difference (L2)")
    axis.set_title("Command sensitivity on identical held-out states")
    axis.tick_params(axis="x", rotation=28)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return summary, rows


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def run_closed_loop_evaluation(
    model: Any,
    torch: Any,
    scene_path: Path,
    teacher_checkpoint: Path,
    obs_stats: dict[str, Any],
    command_stats: dict[str, Any],
    action_stats: dict[str, Any],
    rollout_steps: int,
    torch_threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.imitation_learning.collect_command_teacher_rollouts import (
        ACTION_SCALE,
        DEFAULT_JOINT_POS,
        MuJoCoDeploymentScene,
        FrozenTeacher,
        command_catalog,
        command_velocity_from_base_velocity,
    )

    teacher = FrozenTeacher(teacher_checkpoint, torch_threads=torch_threads)
    rollout_data: dict[str, dict[int, dict[str, Any]]] = {"teacher": {}, "bc": {}}
    commands_by_id: dict[int, np.ndarray] = {}

    for policy_name in ("teacher", "bc"):
        environment = MuJoCoDeploymentScene(scene_path)
        for catalog_entry in command_catalog():
            command_id = int(catalog_entry["command_id"])
            command = np.asarray(catalog_entry["command"], dtype=np.float32)
            commands_by_id[command_id] = command
            last_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
            velocities: list[np.ndarray] = []
            trajectory: list[list[float]] = []
            environment.reset()
            for step in range(rollout_steps):
                state, policy_observation, _, _, _, _ = environment.observation(
                    command, last_raw_action
                )
                if policy_name == "teacher":
                    raw_action = teacher(policy_observation)
                    target_action = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
                else:
                    target_action = predict_actions(
                        model,
                        torch,
                        state[None, :],
                        command[None, :],
                        obs_stats,
                        command_stats,
                        action_stats,
                        batch_size=1,
                    )[0]
                    raw_action = (target_action - DEFAULT_JOINT_POS) / ACTION_SCALE
                if not np.isfinite(target_action).all() or not np.isfinite(raw_action).all():
                    raise FloatingPointError(f"non-finite {policy_name} action at command {command_id}, step {step}")
                environment.step(target_action)
                velocity = command_velocity_from_base_velocity(environment.base_velocity())
                qpos = environment.data.qpos.copy()
                if not np.isfinite(velocity).all() or not np.isfinite(qpos).all():
                    raise FloatingPointError(f"non-finite {policy_name} state at command {command_id}, step {step}")
                velocities.append(velocity)
                trajectory.append(
                    [
                        step * 0.02,
                        float(qpos[0]),
                        float(qpos[1]),
                        _yaw_from_quaternion(qpos[3:7]),
                    ]
                )
                last_raw_action = np.asarray(raw_action, dtype=np.float32)
            rollout_data[policy_name][command_id] = {
                "velocities": np.asarray(velocities, dtype=np.float32),
                "trajectory": np.asarray(trajectory, dtype=np.float64),
            }

    tracking_rows: list[dict[str, Any]] = []
    tracking_summary: dict[str, Any] = {}
    for policy_name in ("teacher", "bc"):
        all_commands = np.concatenate(
            [np.broadcast_to(commands_by_id[command_id], (rollout_steps, COMMAND_DIM)) for command_id in sorted(commands_by_id)],
            axis=0,
        )
        all_velocities = np.concatenate(
            [rollout_data[policy_name][command_id]["velocities"] for command_id in sorted(commands_by_id)],
            axis=0,
        )
        tracking_summary[policy_name] = tracking_statistics(all_commands, all_velocities)
        for command_id in sorted(commands_by_id):
            command = commands_by_id[command_id]
            values = rollout_data[policy_name][command_id]["velocities"]
            catalog_entry = next(item for item in command_catalog() if int(item["command_id"]) == command_id)
            for index, axis in enumerate(COMMAND_AXES):
                desired = float(command[index])
                actual = values[:, index]
                error = actual - desired
                tracking_rows.append(
                    {
                        "policy": policy_name,
                        "command_id": command_id,
                        "command_name": catalog_entry["name"],
                        "axis": axis,
                        "desired": desired,
                        "achieved_mean": float(np.mean(actual)),
                        "achieved_std": float(np.std(actual)),
                        "error_mean": float(np.mean(error)),
                        "absolute_error_mean": float(np.mean(np.abs(error))),
                        "error_std": float(np.std(error)),
                        "samples": int(actual.size),
                    }
                )

    return {
        "tracking_rows": tracking_rows,
        "tracking_summary": tracking_summary,
        "rollout_data": rollout_data,
        "commands": commands_by_id,
        "rollout_steps": rollout_steps,
    }, tracking_summary


def write_tracking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_training_plots(plot_dir: Path, history: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="train", color="#4472c4")
    axis.set_xlabel("epoch")
    axis.set_ylabel("physical action MSE")
    axis.set_title("Training loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "training_loss.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(epochs, [row["validation_loss"] for row in history], label="validation", color="#c00000")
    axis.set_xlabel("epoch")
    axis.set_ylabel("physical action MSE")
    axis.set_title("Validation loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "validation_loss.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(epochs, [row["validation_rmse"] for row in history], label="validation RMSE", color="#70ad47")
    axis.plot(epochs, [row["validation_mean_action_l2"] for row in history], label="validation mean action L2", color="#ed7d31")
    axis.set_xlabel("epoch")
    axis.set_ylabel("action error (rad)")
    axis.set_title("Action prediction error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "action_prediction_error.png", dpi=160)
    plt.close(figure)


def make_tracking_plot(plot_dir: Path, rollout: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    commands = rollout["commands"]
    data = rollout["rollout_data"]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for index, axis_name in enumerate(COMMAND_AXES):
        desired = np.asarray([commands[key][index] for key in sorted(commands)], dtype=np.float64)
        axis = axes[index]
        for policy_name, color, marker in (("teacher", "#4472c4", "o"), ("bc", "#c00000", "s")):
            achieved = np.asarray(
                [np.mean(data[policy_name][key]["velocities"][:, index]) for key in sorted(commands)],
                dtype=np.float64,
            )
            axis.plot(desired, achieved, marker=marker, color=color, linewidth=1.2, label=policy_name)
        low = float(min(np.min(desired), np.min([axis.get_ylim()[0], axis.get_ylim()[1]])))
        high = float(max(np.max(desired), np.max([axis.get_ylim()[0], axis.get_ylim()[1]])))
        axis.plot([low, high], [low, high], "k--", linewidth=1, label="identity" if index == 0 else None)
        axis.set_xlabel(f"desired {axis_name}")
        axis.set_ylabel(f"achieved {axis_name}")
        axis.set_title(axis_name)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Command versus achieved velocity")
    figure.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "command_vs_achieved_velocity.png", dpi=160)
    plt.close(figure)


def make_trajectory_plot(plot_dir: Path, rollout: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    commands = rollout["commands"]
    selected = [
        key for key in sorted(commands)
        if np.allclose(commands[key], [0.0, 0.0, 0.0])
        or np.allclose(commands[key], [0.1, 0.0, 0.0])
        or np.allclose(commands[key], [0.2, 0.0, 0.0])
    ]
    figure, axes = plt.subplots(1, len(selected), figsize=(5.0 * len(selected), 4.2), squeeze=False)
    axes_flat = axes[0]
    for axis, command_id in zip(axes_flat, selected):
        command = commands[command_id]
        for policy_name, color in (("teacher", "#4472c4"), ("bc", "#c00000")):
            trajectory = rollout["rollout_data"][policy_name][command_id]["trajectory"]
            axis.plot(trajectory[:, 1], trajectory[:, 2], color=color, label=policy_name)
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.set_title(f"command [{command[0]:.1f}, {command[1]:.1f}, {command[2]:.1f}]")
        axis.grid(alpha=0.25)
    axes_flat[0].legend()
    figure.suptitle("Teacher versus BC trajectory comparison")
    figure.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "teacher_vs_bc_trajectory.png", dpi=160)
    plt.close(figure)


def write_training_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_report(
    path: Path,
    dataset_path: Path,
    dataset_hash_before: str,
    dataset_hash_after: str,
    validation: dict[str, Any],
    split: dict[str, list[int]],
    config: dict[str, Any],
    training: dict[str, Any],
    sensitivity: dict[str, Any],
    tracking: dict[str, Any],
    decision: str,
    next_step: str,
    artifacts: dict[str, str],
) -> None:
    policy = training["policy"]
    lines = [
        "# Command-Conditioned Behaviour Cloning Training Report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Result",
        "",
        f"Decision: **{decision}**  ",
        f"Next step: **{next_step}**",
        "",
        "This is behaviour cloning only. No PPO runner or PPO optimizer was used.",
        "",
        "## Dataset validation",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Samples: `{validation['samples']}` ({validation['episodes']} episodes × {validation['steps_per_episode']} steps)",
        f"- Validation report: `{artifacts['dataset_validation']}`",
        f"- Hash before training: `{dataset_hash_before}`",
        f"- Hash after evaluation: `{dataset_hash_after}`",
        f"- Dataset unchanged: **{'PASS' if dataset_hash_before == dataset_hash_after else 'FAIL'}**",
        "",
        "## Policy and objective",
        "",
        f"- Architecture: `{policy['architecture']}`",
        f"- Input dimension: `{policy['input_dimension']}` (42-D observation + 3-D command)",
        f"- Command layout: `{policy['input_layout']}`",
        f"- Output dimension: `{policy['output_dimension']}` (joint-position target)",
        f"- Loss: `{policy['loss']}`",
        f"- Optimizer: `{policy['optimizer']}`",
        "",
        "## Deterministic episode split",
        "",
        f"- Seed: `{config['seed']}`",
        f"- Train: `{len(split['train'])}` episodes / `{len(split['train']) * validation['steps_per_episode']}` samples",
        f"- Validation: `{len(split['validation'])}` episodes / `{len(split['validation']) * validation['steps_per_episode']}` samples",
        f"- Test: `{len(split['test'])}` episodes / `{len(split['test']) * validation['steps_per_episode']}` samples",
        "- Split unit: episode; individual timesteps were not split across partitions.",
        f"- Train episode ids: `{split['train']}`",
        f"- Validation episode ids: `{split['validation']}`",
        f"- Test episode ids: `{split['test']}`",
        "",
        "## Training",
        "",
        f"- Best epoch: `{training['best_epoch']}`",
        f"- Optimizer steps: `{training['optimizer_steps']}`",
        f"- Final validation loss (physical action MSE): `{training['final_validation_loss']:.8e}`",
        f"- Validation action RMSE: `{training['validation_metrics']['rmse']:.8e}` rad",
        f"- Validation mean action error: `{training['validation_metrics']['mean_action_l2']:.8e}` rad L2",
        f"- Validation action MAE: `{training['validation_metrics']['mae']:.8e}` rad",
        f"- Constant-train-mean validation baseline: `{training['validation_baseline_loss']:.8e}`",
        f"- Baseline improvement: `{training['baseline_improvement']:.3%}`",
        "",
        "## Command sensitivity",
        "",
        "Counterfactual commands were evaluated on identical held-out state observations.",
        "",
        "| Probe | Axis | Mean action L2 | Median | Fraction above threshold | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for probe_id, entry in sensitivity.items():
        lines.append(
            f"| {probe_id} | {entry['axis']} | {entry['mean_action_l2']:.6e} | "
            f"{entry['median_action_l2']:.6e} | {entry['fraction_l2_gt_threshold']:.3f} | "
            f"{'PASS' if entry['pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Closed-loop teacher comparison", ""])
    lines.append("Achieved body-frame velocities were measured in the same MuJoCo deployment-equivalent scene for the frozen teacher and BC policy.")
    lines.extend(
        [
            "",
            "| Policy | Axis | Desired mean | Achieved mean | Mean absolute tracking error | Correlation |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for policy_name in ("teacher", "bc"):
        for axis in COMMAND_AXES:
            entry = tracking[policy_name][axis]
            corr = "n/a" if not math.isfinite(entry["correlation"]) else f"{entry['correlation']:.4f}"
            lines.append(
                f"| {policy_name} | {axis} | {entry['desired_mean']:.5f} | {entry['actual_mean']:.5f} | "
                f"{entry['mean_abs_error']:.5f} | {corr} |"
            )
    lines.extend(
        [
            "",
            "## Constraint counters",
            "",
            "- PPO updates: **0**",
            "- Teacher changes: **0**",
            "- Dataset writes: **0**",
            "- Original Kine2Go data writes: **0**",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, artifact in artifacts.items():
        lines.append(f"- `{name}`: `{artifact}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/imitation/expert_imitation_dataset_v1.npz"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("models/expert_model_500.pt"))
    parser.add_argument("--scene", type=Path, default=Path("external/unitree_go2/scene_go2.xml"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="allow replacement of existing BC-only artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    dataset_path = (repo_root / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    teacher_checkpoint = (repo_root / args.teacher_checkpoint).resolve() if not args.teacher_checkpoint.is_absolute() else args.teacher_checkpoint.resolve()
    scene_path = (repo_root / args.scene).resolve() if not args.scene.is_absolute() else args.scene.resolve()
    report_dir = repo_root / "results/genesis/generated/imitation"
    plot_dir = report_dir / "bc_training_plots"
    checkpoint_path = repo_root / "models/go2_command_bc_v1.pt"
    obs_stats_path = repo_root / "data/imitation/normalization/obs_stats.pkl"
    action_stats_path = repo_root / "data/imitation/normalization/action_stats.pkl"
    command_stats_path = repo_root / "data/imitation/normalization/command_stats.pkl"
    validation_report_path = report_dir / "bc_dataset_validation.md"
    training_report_path = report_dir / "bc_training_report.md"
    summary_path = report_dir / "bc_training_summary.json"
    sensitivity_path = report_dir / "bc_command_sensitivity.csv"
    tracking_path = report_dir / "bc_tracking_evaluation.csv"
    provenance_path = report_dir / "bc_training_provenance.json"
    history_path = report_dir / "bc_training_history.csv"

    for path in (dataset_path, teacher_checkpoint, scene_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.force and checkpoint_path.exists():
        raise FileExistsError(f"BC checkpoint already exists; pass --force to replace it: {checkpoint_path}")

    dataset_hash_before = sha256_file(dataset_path)
    teacher_hash_before = sha256_file(teacher_checkpoint)
    data = load_dataset(dataset_path)
    _, validation = validation_checks(data)
    write_dataset_validation_report(
        validation_report_path,
        dataset_path,
        dataset_hash_before,
        validation,
        "pre-training",
    )
    if not validation["valid"]:
        raise ValueError(f"dataset validation failed; see {validation_report_path}")

    split = deterministic_split(validation["episodes"], args.seed)
    obs_stats = compute_stats(data["observations"], split["train"], "observation")
    command_stats = compute_stats(data["commands"], split["train"], "command")
    action_stats = compute_stats(data["actions"], split["train"], "action")
    save_pickle(obs_stats_path, obs_stats)
    save_pickle(action_stats_path, action_stats)
    save_pickle(command_stats_path, command_stats)

    torch = import_torch()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass

    train_inputs, train_actions, _, _ = flatten_split(data, split["train"], obs_stats, command_stats)
    validation_inputs, validation_actions, _, _ = flatten_split(
        data, split["validation"], obs_stats, command_stats
    )
    test_inputs, test_actions, _, _ = flatten_split(data, split["test"], obs_stats, command_stats)
    config = {
        "schema": "go2_command_bc_v1_config",
        "script_version": SCRIPT_VERSION,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "scheduler_patience": int(args.scheduler_patience),
        "early_stopping_patience": int(args.early_stopping_patience),
        "min_delta": float(args.min_delta),
        "torch_threads": int(args.torch_threads),
        "input_layout": "[state[0:6], command[0:3], state[6:42]]",
        "loss": "mean((predicted_joint_position_target - teacher_joint_position_target)^2) in physical action units",
        "optimizer": "AdamW",
        "device": "cpu",
    }
    model = make_model(torch)
    action_mean_np = action_stats["mean"]
    baseline_loss = float(np.mean((validation_actions - action_mean_np) ** 2))
    model, history, best_epoch, optimizer_steps, best_validation_loss = train_model(
        model,
        torch,
        train_inputs,
        train_actions,
        validation_inputs,
        validation_actions,
        action_stats,
        config,
    )
    make_training_plots(plot_dir, history)
    write_training_history(history_path, history)

    action_mean_t = torch.from_numpy(action_stats["mean"])
    action_std_t = torch.from_numpy(action_stats["std"])
    train_metrics = evaluate_model(
        model,
        torch,
        torch.from_numpy(train_inputs),
        torch.from_numpy(train_actions),
        action_mean_t,
        action_std_t,
        args.batch_size,
    )
    validation_metrics = evaluate_model(
        model,
        torch,
        torch.from_numpy(validation_inputs),
        torch.from_numpy(validation_actions),
        action_mean_t,
        action_std_t,
        args.batch_size,
    )
    test_metrics = evaluate_model(
        model,
        torch,
        torch.from_numpy(test_inputs),
        torch.from_numpy(test_actions),
        action_mean_t,
        action_std_t,
        args.batch_size,
    )

    checkpoint = {
        "schema": "go2_command_bc_v1",
        "script_version": SCRIPT_VERSION,
        "model_state_dict": model.state_dict(),
        "architecture": list(ARCHITECTURE),
        "input_dimension": INPUT_DIM,
        "observation_dimension": OBS_DIM,
        "command_dimension": COMMAND_DIM,
        "action_dimension": ACTION_DIM,
        "input_layout": config["input_layout"],
        "output_semantics": "12-D joint position target in physical units",
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "normalization": {
            "observation": {"mean": obs_stats["mean"], "std": obs_stats["std"]},
            "command": {"mean": command_stats["mean"], "std": command_stats["std"]},
            "action": {"mean": action_stats["mean"], "std": action_stats["std"]},
        },
        "config": config,
        "dataset_sha256": dataset_hash_before,
        "split": split,
        "best_epoch": best_epoch,
        "metrics": {"train": train_metrics, "validation": validation_metrics, "test": test_metrics},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    sensitivity_summary, _ = command_sensitivity(
        model,
        torch,
        data,
        split["test"],
        obs_stats,
        command_stats,
        action_stats,
        sensitivity_path,
        plot_dir / "command_sensitivity.png",
    )
    rollout, tracking_summary = run_closed_loop_evaluation(
        model,
        torch,
        scene_path,
        teacher_checkpoint,
        obs_stats,
        command_stats,
        action_stats,
        int(args.rollout_steps),
        int(args.torch_threads),
    )
    write_tracking_csv(tracking_path, rollout["tracking_rows"])
    make_tracking_plot(plot_dir, rollout)
    make_trajectory_plot(plot_dir, rollout)

    dataset_hash_after = sha256_file(dataset_path)
    teacher_hash_after = sha256_file(teacher_checkpoint)
    sensitivity_pass = all(bool(entry["pass"]) for entry in sensitivity_summary.values())
    tracking_finite = all(
        math.isfinite(float(value))
        for policy_values in tracking_summary.values()
        for axis_values in policy_values.values()
        for key, value in axis_values.items()
        if key not in {"correlation", "slope"} or math.isfinite(float(value))
    )
    baseline_improvement = 1.0 - validation_metrics["loss"] / baseline_loss if baseline_loss > 0.0 else 0.0
    offline_pass = bool(
        np.isfinite(validation_metrics["loss"])
        and baseline_improvement > 0.90
        and validation_metrics["mean_action_l2"] < 0.10
    )
    decision_valid = bool(
        validation["valid"]
        and dataset_hash_before == dataset_hash_after
        and teacher_hash_before == teacher_hash_after
        and offline_pass
        and sensitivity_pass
        and tracking_finite
    )
    decision = "BC POLICY VALID" if decision_valid else "BC POLICY REQUIRES DEBUGGING"
    next_step = "PPO REFINEMENT" if decision_valid else "FIX BC PIPELINE"

    artifacts = {
        "checkpoint": str(checkpoint_path),
        "obs_stats": str(obs_stats_path),
        "action_stats": str(action_stats_path),
        "command_stats": str(command_stats_path),
        "dataset_validation": str(validation_report_path),
        "training_history": str(history_path),
        "training_report": str(training_report_path),
        "summary": str(summary_path),
        "provenance": str(provenance_path),
        "command_sensitivity": str(sensitivity_path),
        "tracking_evaluation": str(tracking_path),
        "plots": str(plot_dir),
    }
    training_summary = {
        "policy": {
            "architecture": "45 -> 512 -> ELU -> 256 -> ELU -> 128 -> ELU -> 12",
            "input_dimension": INPUT_DIM,
            "command_dimension": COMMAND_DIM,
            "output_dimension": ACTION_DIM,
            "input_layout": config["input_layout"],
            "loss": config["loss"],
            "optimizer": config["optimizer"],
        },
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(optimizer_steps),
        "final_validation_loss": float(best_validation_loss),
        "validation_metrics": validation_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "validation_baseline_loss": baseline_loss,
        "baseline_improvement": baseline_improvement,
        "offline_pass": offline_pass,
    }
    build_report(
        training_report_path,
        dataset_path,
        dataset_hash_before,
        dataset_hash_after,
        validation,
        split,
        config,
        training_summary,
        sensitivity_summary,
        tracking_summary,
        decision,
        next_step,
        artifacts,
    )

    provenance = {
        "schema": "go2_command_bc_v1_provenance",
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "git_commit": git_commit(repo_root),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": {
            "path": str(dataset_path),
            "sha256_before": dataset_hash_before,
            "sha256_after": dataset_hash_after,
            "writes": 0,
        },
        "teacher": {
            "checkpoint_path": str(teacher_checkpoint),
            "sha256_before": teacher_hash_before,
            "sha256_after": teacher_hash_after,
            "changes": 0,
            "frozen": True,
            "evaluation_only": True,
        },
        "source_scene": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "config": config,
        "split": split,
        "counters": {
            "ppo_updates": 0,
            "teacher_changes": 0,
            "dataset_writes": 0,
            "original_kine2go_data_writes": 0,
            "bc_optimizer_steps": int(optimizer_steps),
        },
        "artifacts": artifacts,
        "decision": decision,
        "next_step": next_step,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, default=json_default) + "\n", encoding="utf-8")
    summary = {
        "schema": "go2_command_bc_v1_summary",
        "generated_utc": utc_now(),
        "dataset": {
            "name": "go2_command_teacher_v1",
            "path": str(dataset_path),
            "episodes": validation["episodes"],
            "steps_per_episode": validation["steps_per_episode"],
            "samples": validation["samples"],
            "sha256": dataset_hash_after,
            "validation": validation["valid"],
        },
        "policy": training_summary["policy"],
        "split": split,
        "config": config,
        "training": training_summary,
        "command_sensitivity": sensitivity_summary,
        "closed_loop_tracking": tracking_summary,
        "closed_loop_rollout_steps": int(args.rollout_steps),
        "counters": provenance["counters"],
        "decision": decision,
        "next_step": next_step,
        "artifacts": artifacts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")

    write_dataset_validation_report(
        validation_report_path,
        dataset_path,
        dataset_hash_after,
        validation,
        "pre-training validation; post-training hash rechecked in bc_training_report.md",
    )

    print("\nCOMMAND-CONDITIONED BC TRAINING COMPLETE")
    print("\nDataset:")
    print("go2_command_teacher_v1")
    print(f"\nSamples:\n{validation['samples']}")
    print("\nPolicy:")
    print(f"architecture: {training_summary['policy']['architecture']}")
    print(f"\nFinal validation loss:\n{validation_metrics['loss']:.8e}")
    print(f"\nAction error:\nmean_action_l2={validation_metrics['mean_action_l2']:.8e}, rmse={validation_metrics['rmse']:.8e}")
    print(f"\nCommand sensitivity:\n{'PASS' if sensitivity_pass else 'FAIL'}")
    for axis, probe in (("vx", "vx_A_to_C"), ("vy", "vy_zero_to_positive"), ("yaw", "yaw_zero_to_positive")):
        print(f"\n{axis} command response:\n{'PASS' if sensitivity_summary[probe]['pass'] else 'FAIL'}")
    print(f"\nDecision:\n{decision}")
    print(f"\nNext step:\n{next_step}")
    print("\nCounters: PPO updates=0, teacher changes=0, dataset writes=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
