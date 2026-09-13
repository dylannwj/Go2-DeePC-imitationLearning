#!/usr/bin/env python3
"""Collect a command-conditioned dataset from the recovered Go2 PPO teacher.

This collector is deliberately standalone.  It reconstructs the frozen actor
from the checkpoint and runs it in the bundled Unitree MuJoCo deployment scene
because the optional mjlab Python packages are not installed on the host.
It never creates a training runner, optimizer, or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "command-teacher-rollout-v1.1"
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = 4
OBSERVATION_DIM = 42
POLICY_OBSERVATION_DIM = 45
ACTION_DIM = 12
COMMAND_DIM = 3

# The policy order is FL, FR, RL, RR.  The bundled deployment/MuJoCo scene
# order is FR, FL, RR, RL.  This permutation is the recovered deploy config's
# joint_ids_map and is an involution, so it maps in either direction.
POLICY_TO_SCENE = np.array(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64
)
DEFAULT_JOINT_POS = np.array(
    [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
    dtype=np.float32,
)
ACTION_SCALE = np.full(ACTION_DIM, 0.5, dtype=np.float32)
KP = np.tile(np.array([20.0, 20.0, 40.0], dtype=np.float32), 4)
KD = np.tile(np.array([1.0, 1.0, 2.0], dtype=np.float32), 4)
EFFORT_LIMIT = np.tile(np.array([23.5, 23.5, 45.43], dtype=np.float32), 4)
SCENE_LEGS = ("FR", "FL", "RR", "RL")
POLICY_LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("hip", "thigh", "calf")
AXES = ("vx", "vy", "yaw")

STATE_FEATURE_NAMES = (
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    *(f"joint_pos_rel_{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS),
    *(f"joint_vel_rel_{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS),
    *(f"last_action_{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS),
)
POLICY_FEATURE_NAMES = (
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "command_vx",
    "command_vy",
    "command_yaw_rate",
    *(f"joint_pos_rel_{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS),
    *(f"joint_vel_rel_{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS),
    *(f"last_action_{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def command_catalog() -> list[dict[str, Any]]:
    """Return required commands plus larger mixed commands for coverage."""

    commands: list[tuple[str, tuple[float, float, float]]] = [
        ("standing", (0.0, 0.0, 0.0)),
        ("forward_005", (0.05, 0.0, 0.0)),
        ("forward_010", (0.10, 0.0, 0.0)),
        ("forward_015", (0.15, 0.0, 0.0)),
        ("forward_020", (0.20, 0.0, 0.0)),
        ("forward_030", (0.30, 0.0, 0.0)),
        ("backward_005", (-0.05, 0.0, 0.0)),
        ("backward_010", (-0.10, 0.0, 0.0)),
        ("lateral_left_010", (0.0, 0.10, 0.0)),
        ("lateral_left_005", (0.0, 0.05, 0.0)),
        ("lateral_right_005", (0.0, -0.05, 0.0)),
        ("lateral_right_010", (0.0, -0.10, 0.0)),
        ("turn_left_030", (0.0, 0.0, 0.30)),
        ("turn_left_050", (0.0, 0.0, 0.50)),
        ("turn_right_030", (0.0, 0.0, -0.30)),
        ("turn_right_050", (0.0, 0.0, -0.50)),
        ("lateral_left_020", (0.0, 0.20, 0.0)),
        ("lateral_right_020", (0.0, -0.20, 0.0)),
        ("lateral_left_030", (0.0, 0.30, 0.0)),
        ("lateral_right_030", (0.0, -0.30, 0.0)),
        ("forward_left", (0.20, 0.20, 0.0)),
        ("forward_right", (0.20, -0.20, 0.0)),
        ("forward_turn_left", (0.20, 0.0, 0.30)),
        ("forward_turn_right", (0.20, 0.0, -0.30)),
        ("backward_turn_left", (-0.10, 0.0, 0.30)),
        ("backward_turn_right", (-0.10, 0.0, -0.30)),
        ("lateral_left_turn_left", (0.0, 0.20, 0.30)),
        ("lateral_right_turn_right", (0.0, -0.20, -0.30)),
    ]
    return [
        {"command_id": index, "name": name, "command": list(values)}
        for index, (name, values) in enumerate(commands)
    ]


def make_schedule(episodes: int) -> tuple[np.ndarray, np.ndarray]:
    catalog = command_catalog()
    ids = np.arange(episodes, dtype=np.int32) % len(catalog)
    commands = np.asarray([catalog[int(index)]["command"] for index in ids], dtype=np.float32)
    return ids, commands


def inverse_rotate_vector(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the base frame."""

    import mujoco

    result = np.zeros(3, dtype=np.float64)
    conjugate = np.array(
        [
            quaternion_wxyz[0],
            -quaternion_wxyz[1],
            -quaternion_wxyz[2],
            -quaternion_wxyz[3],
        ],
        dtype=np.float64,
    )
    mujoco.mju_rotVecQuat(result, np.asarray(vector, dtype=np.float64), conjugate)
    return result.astype(np.float32)


def command_velocity_from_base_velocity(base_velocity: np.ndarray) -> np.ndarray:
    """Extract body-frame [vx, vy, yaw_rate] from [lin xyz, ang xyz]."""

    base_velocity = np.asarray(base_velocity, dtype=np.float32)
    return np.asarray(
        [base_velocity[0], base_velocity[1], base_velocity[5]], dtype=np.float32
    )


class FrozenTeacher:
    """RSL-RL-compatible deterministic actor reconstructed from a checkpoint."""

    def __init__(self, checkpoint_path: Path, torch_threads: int = 1):
        import torch

        torch.set_num_threads(max(1, torch_threads))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model_state_dict")
        state = checkpoint["model_state_dict"]
        required = [
            "actor.0.weight",
            "actor.0.bias",
            "actor.2.weight",
            "actor.2.bias",
            "actor.4.weight",
            "actor.4.bias",
            "actor.6.weight",
            "actor.6.bias",
            "actor_obs_normalizer._mean",
            "actor_obs_normalizer._std",
        ]
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"Checkpoint actor is missing keys: {missing}")

        class Actor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(POLICY_OBSERVATION_DIM, 512),
                    torch.nn.ELU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, ACTION_DIM),
                )
                network_state = {
                    f"{index}.{suffix}": state[f"actor.{index}.{suffix}"]
                    for index in (0, 2, 4, 6)
                    for suffix in ("weight", "bias")
                }
                self.network.load_state_dict(network_state, strict=True)
                self.register_buffer("mean", state["actor_obs_normalizer._mean"])
                self.register_buffer("std", state["actor_obs_normalizer._std"])

            def forward(self, observation: Any) -> Any:
                normalized = (observation - self.mean) / (self.std + 0.01)
                return self.network(normalized)

        self._torch = torch
        self._actor = Actor().eval()
        self.iteration = int(checkpoint.get("iter", -1))
        self.observation_dim = int(state["actor.0.weight"].shape[1])
        self.action_dim = int(state["actor.6.weight"].shape[0])

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            action = self._actor(tensor)[0].cpu().numpy()
        return np.asarray(action, dtype=np.float32)


class MuJoCoDeploymentScene:
    """Single-env flat Go2 scene with deployment-equivalent PD control."""

    def __init__(self, scene_path: Path, root_height: float = 0.32):
        import mujoco

        if not np.array_equal(POLICY_TO_SCENE[POLICY_TO_SCENE], np.arange(ACTION_DIM)):
            raise ValueError("Recovered joint map is expected to be an involution")
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.model.opt.timestep = PHYSICS_DT
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.model.opt.impratio = 1.0
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 10
        self.model.opt.tolerance = 1.0e-8
        self.model.opt.ls_iterations = 20
        self.model.opt.ls_tolerance = 0.01
        self.model.opt.gravity[:] = (0.0, 0.0, -9.81)
        self.data = mujoco.MjData(self.model)
        self.root_height = float(root_height)

        def sensor_address(name: str) -> int:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0:
                raise ValueError(f"MuJoCo scene is missing sensor {name!r}")
            return int(self.model.sensor_adr[sensor_id])

        def joint_qpos_address(name: str) -> int:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo scene is missing joint {name!r}")
            return int(self.model.jnt_qposadr[joint_id])

        self.scene_pos_sensor = np.asarray(
            [
                sensor_address(f"{leg}_{joint}_pos")
                for leg in SCENE_LEGS
                for joint in JOINTS
            ],
            dtype=np.int64,
        )
        self.scene_vel_sensor = np.asarray(
            [
                sensor_address(f"{leg}_{joint}_vel")
                for leg in SCENE_LEGS
                for joint in JOINTS
            ],
            dtype=np.int64,
        )
        self.scene_qpos = np.asarray(
            [
                joint_qpos_address(f"{leg}_{joint}_joint")
                for leg in SCENE_LEGS
                for joint in JOINTS
            ],
            dtype=np.int64,
        )
        self.imu_gyro = sensor_address("imu_gyro")
        self.imu_quat = sensor_address("imu_quat")
        self.floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.foot_geoms = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in SCENE_LEGS],
            dtype=np.int64,
        )
        if self.floor_geom < 0 or np.any(self.foot_geoms < 0):
            raise ValueError("MuJoCo scene is missing named floor or foot geoms")

    def reset(self) -> np.ndarray:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = (0.0, 0.0, self.root_height, 1.0, 0.0, 0.0, 0.0)
        scene_default = DEFAULT_JOINT_POS[POLICY_TO_SCENE]
        self.data.qpos[self.scene_qpos] = scene_default
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return np.zeros(ACTION_DIM, dtype=np.float32)

    def observation(
        self, command: np.ndarray, last_action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        command = np.asarray(command, dtype=np.float32)
        last_action = np.asarray(last_action, dtype=np.float32)
        scene_pos = self.data.sensordata[self.scene_pos_sensor].astype(np.float32)
        scene_vel = self.data.sensordata[self.scene_vel_sensor].astype(np.float32)
        policy_pos = scene_pos[POLICY_TO_SCENE]
        policy_vel = scene_vel[POLICY_TO_SCENE]
        quaternion = self.data.sensordata[self.imu_quat : self.imu_quat + 4]
        projected_gravity = inverse_rotate_vector(quaternion, np.array([0.0, 0.0, -1.0]))
        base_ang_vel = self.data.sensordata[self.imu_gyro : self.imu_gyro + 3].astype(
            np.float32
        )
        state = np.concatenate(
            [base_ang_vel, projected_gravity, policy_pos - DEFAULT_JOINT_POS, policy_vel, last_action]
        ).astype(np.float32)
        policy_observation = np.concatenate([state[:6], command, state[6:]]).astype(np.float32)
        base_pose = self.data.qpos[:7].astype(np.float32).copy()
        base_velocity = self.base_velocity()
        contacts = self.contacts()
        return state, policy_observation, base_pose, base_velocity, contacts, policy_pos

    def base_velocity(self) -> np.ndarray:
        quaternion = self.data.qpos[3:7]
        linear_body = inverse_rotate_vector(quaternion, self.data.qvel[:3])
        angular_body = self.data.sensordata[self.imu_gyro : self.imu_gyro + 3].astype(
            np.float32
        )
        return np.concatenate([linear_body, angular_body]).astype(np.float32)

    def contacts(self) -> np.ndarray:
        contact = np.zeros(len(SCENE_LEGS), dtype=bool)
        foot_to_index = {int(geom): index for index, geom in enumerate(self.foot_geoms)}
        for index in range(self.data.ncon):
            pair = (int(self.data.contact[index].geom1), int(self.data.contact[index].geom2))
            if self.floor_geom not in pair:
                continue
            other = pair[0] if pair[1] == self.floor_geom else pair[1]
            if other in foot_to_index:
                contact[foot_to_index[other]] = True
        return contact

    def step(self, target_policy: np.ndarray) -> None:
        target_policy = np.asarray(target_policy, dtype=np.float32)
        for _ in range(DECIMATION):
            scene_pos = self.data.sensordata[self.scene_pos_sensor].astype(np.float32)
            scene_vel = self.data.sensordata[self.scene_vel_sensor].astype(np.float32)
            policy_pos = scene_pos[POLICY_TO_SCENE]
            policy_vel = scene_vel[POLICY_TO_SCENE]
            torque_policy = np.clip(
                KP * (target_policy - policy_pos) - KD * policy_vel,
                -EFFORT_LIMIT,
                EFFORT_LIMIT,
            )
            self.data.ctrl[:] = torque_policy[POLICY_TO_SCENE]
            self.mujoco.mj_step(self.model, self.data)


def correlation_and_slope(desired: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    desired = np.asarray(desired, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    finite = np.isfinite(desired) & np.isfinite(actual)
    desired = desired[finite]
    actual = actual[finite]
    if desired.size < 2 or np.std(desired) <= 1.0e-12:
        return math.nan, math.nan
    correlation = float(np.corrcoef(desired, actual)[0, 1])
    slope = float(np.cov(desired, actual, bias=True)[0, 1] / np.var(desired))
    return correlation, slope


def tracking_statistics(commands: np.ndarray, velocities: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    flat_commands = commands.reshape(-1, COMMAND_DIM)
    flat_velocities = velocities.reshape(-1, COMMAND_DIM)
    for axis_index, axis_name in enumerate(AXES):
        desired = flat_commands[:, axis_index]
        actual = flat_velocities[:, axis_index]
        error = actual - desired
        correlation, slope = correlation_and_slope(desired, actual)
        result[axis_name] = {
            "desired_mean": float(np.mean(desired)),
            "actual_mean": float(np.mean(actual)),
            "mean_error": float(np.mean(error)),
            "mean_abs_error": float(np.mean(np.abs(error))),
            "std_error": float(np.std(error)),
            "correlation": correlation,
            "slope": slope,
        }
    return result


def command_rows(
    command_ids: np.ndarray,
    episode_commands: np.ndarray,
    velocities: np.ndarray,
    timestamps: np.ndarray,
) -> list[dict[str, Any]]:
    catalog = command_catalog()
    rows: list[dict[str, Any]] = []
    for command_id in sorted(np.unique(command_ids).tolist()):
        episode_indices = np.flatnonzero(command_ids == command_id)
        desired = episode_commands[episode_indices[0]]
        actual = velocities[episode_indices]
        duration = float(timestamps[episode_indices, -1].max() + CONTROL_DT)
        name = catalog[int(command_id)]["name"]
        for axis_index, axis_name in enumerate(AXES):
            values = actual[:, :, axis_index].reshape(-1)
            errors = values - desired[axis_index]
            rows.append(
                {
                    "command_id": int(command_id),
                    "command_name": name,
                    "axis": axis_name,
                    "desired_vx": float(desired[0]),
                    "desired_vy": float(desired[1]),
                    "desired_yaw_rate": float(desired[2]),
                    "episodes": int(len(episode_indices)),
                    "samples": int(values.size),
                    "duration_s": duration,
                    "actual_mean": float(np.mean(values)),
                    "actual_std": float(np.std(values)),
                    "error_mean": float(np.mean(errors)),
                    "absolute_error_mean": float(np.mean(np.abs(errors))),
                    "error_std": float(np.std(errors)),
                }
            )
    return rows


def write_statistics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rollout statistics rows were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    plot_dir: Path,
    commands: np.ndarray,
    velocities: np.ndarray,
    timestamps: np.ndarray,
    actions: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    flat_commands = commands.reshape(-1, COMMAND_DIM)
    flat_velocities = velocities.reshape(-1, COMMAND_DIM)
    labels = ("vx", "vy", "yaw_rate")
    for index, label in enumerate(labels):
        figure, axis = plt.subplots(figsize=(6.5, 5.0))
        desired = flat_commands[:, index]
        actual = flat_velocities[:, index]
        axis.scatter(desired, actual, s=3, alpha=0.12, rasterized=True)
        low = float(min(np.min(desired), np.min(actual)))
        high = float(max(np.max(desired), np.max(actual)))
        axis.plot([low, high], [low, high], "k--", linewidth=1, label="identity")
        unique = np.unique(desired)
        means = np.array([np.mean(actual[np.isclose(desired, value)]) for value in unique])
        axis.plot(unique, means, "o-", linewidth=1.5, label="per-command mean")
        axis.set_xlabel(f"desired {label} (body frame)")
        axis.set_ylabel(f"achieved {label} (body frame)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / f"desired_vs_achieved_{'yaw' if index == 2 else label}.png", dpi=160)
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.hist(flat_commands[:, index], bins=24, color="#4472c4", edgecolor="white")
        axis.set_title(label)
        axis.set_xlabel("desired command")
        axis.set_ylabel("samples")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "command_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    episode_lengths = np.full(timestamps.shape[0], timestamps.shape[1], dtype=int)
    axis.bar(np.arange(len(episode_lengths)), episode_lengths, color="#70ad47")
    axis.set_xlabel("episode id")
    axis.set_ylabel("samples")
    axis.set_title("Episode lengths")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "episode_lengths.png", dpi=160)
    plt.close(figure)

    action_flat = actions.reshape(-1, ACTION_DIM)
    action_std = np.std(action_flat, axis=0)
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.bar(np.arange(ACTION_DIM), action_std, color="#ed7d31")
    axis.set_xlabel("policy joint index")
    axis.set_ylabel("target position standard deviation (rad)")
    axis.set_title("Action variance")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "action_variance.png", dpi=160)
    plt.close(figure)

    action_min = np.min(action_flat, axis=0)
    action_max = np.max(action_flat, axis=0)
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    joints = np.arange(ACTION_DIM)
    axis.vlines(joints, action_min, action_max, color="#5b9bd5", linewidth=3)
    axis.scatter(joints, action_min, color="#2f5597", label="min")
    axis.scatter(joints, action_max, color="#c00000", label="max")
    axis.set_xlabel("policy joint index")
    axis.set_ylabel("joint-position target (rad)")
    axis.set_title("Joint-position target ranges")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "joint_position_ranges.png", dpi=160)
    plt.close(figure)


def validate_dataset(
    observations: np.ndarray,
    policy_observations: np.ndarray,
    commands: np.ndarray,
    actions: np.ndarray,
    raw_actions: np.ndarray,
    velocities: np.ndarray,
    timestamps: np.ndarray,
    episode_ids: np.ndarray,
    command_ids: np.ndarray,
    contacts: np.ndarray,
    episode_commands: np.ndarray,
    episodes: int,
    episode_seconds: float,
) -> dict[str, Any]:
    expected_shapes = {
        "observations": [episodes, timestamps.shape[1], OBSERVATION_DIM],
        "policy_observations": [episodes, timestamps.shape[1], POLICY_OBSERVATION_DIM],
        "commands": [episodes, timestamps.shape[1], COMMAND_DIM],
        "actions": [episodes, timestamps.shape[1], ACTION_DIM],
        "raw_actions": [episodes, timestamps.shape[1], ACTION_DIM],
        "velocities": [episodes, timestamps.shape[1], COMMAND_DIM],
        "timestamps": [episodes, timestamps.shape[1]],
        "episode_ids": [episodes, timestamps.shape[1]],
        "command_ids": [episodes, timestamps.shape[1]],
        "contacts": [episodes, timestamps.shape[1], 4],
    }
    actual_shapes = {
        "observations": list(observations.shape),
        "policy_observations": list(policy_observations.shape),
        "commands": list(commands.shape),
        "actions": list(actions.shape),
        "raw_actions": list(raw_actions.shape),
        "velocities": list(velocities.shape),
        "timestamps": list(timestamps.shape),
        "episode_ids": list(episode_ids.shape),
        "command_ids": list(command_ids.shape),
        "contacts": list(contacts.shape),
    }
    shape_check = all(actual_shapes[key] == value for key, value in expected_shapes.items())
    finite_check = all(
        np.all(np.isfinite(array))
        for array in (observations, policy_observations, commands, actions, raw_actions, velocities, timestamps)
    )
    timestamp_check = bool(np.all(np.diff(timestamps, axis=1) > 0.0))
    episode_id_check = bool(np.all(episode_ids == np.arange(episodes, dtype=np.int32)[:, None]))
    command_id_check = bool(np.all(command_ids == command_ids[:, :1]))
    duration_check = bool(10.0 <= episode_seconds <= 20.0)

    unique_commands = np.unique(episode_commands, axis=0)
    required_values = {
        "vx": (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, -0.05, -0.10),
        "vy": (0.0, -0.10, -0.05, 0.05, 0.10),
        "yaw": (0.0, -0.50, -0.30, 0.30, 0.50),
    }
    coverage = {
        axis: sorted(
            float(value)
            for value in np.unique(episode_commands[:, index])
            if any(np.isclose(value, required) for required in required_values[axis])
        )
        for index, axis in enumerate(AXES)
    }
    coverage_check = all(
        all(any(np.isclose(value, found) for found in coverage[axis]) for value in values)
        for axis, values in required_values.items()
    )
    action_flat = actions.reshape(-1, ACTION_DIM)
    action_variance = np.var(action_flat, axis=0)
    action_diversity = len(
        {
            hashlib.sha256(np.round(actions[index], 6).tobytes()).hexdigest()
            for index in range(actions.shape[0])
        }
    )
    structure_valid = all(
        (
            shape_check,
            finite_check,
            timestamp_check,
            episode_id_check,
            command_id_check,
            duration_check,
            coverage_check,
            action_diversity >= 2,
            bool(np.all(action_variance > 1.0e-8)),
        )
    )
    tracking = tracking_statistics(commands, velocities)
    tracking_valid = all(
        math.isfinite(tracking[axis]["correlation"])
        and tracking[axis]["correlation"] > 0.20
        and math.isfinite(tracking[axis]["slope"])
        and tracking[axis]["slope"] > 0.0
        for axis in AXES
    )
    return {
        "shape_check": shape_check,
        "expected_shapes": expected_shapes,
        "actual_shapes": actual_shapes,
        "finite_check": finite_check,
        "timestamps_monotonic": timestamp_check,
        "episode_ids_contiguous": episode_id_check,
        "command_constant_per_episode": command_id_check,
        "episode_duration_check": duration_check,
        "command_coverage_check": coverage_check,
        "unique_commands": unique_commands.tolist(),
        "unique_vx_commands": sorted(float(value) for value in np.unique(episode_commands[:, 0])),
        "unique_vy_commands": sorted(float(value) for value in np.unique(episode_commands[:, 1])),
        "unique_yaw_commands": sorted(float(value) for value in np.unique(episode_commands[:, 2])),
        "required_command_coverage": coverage,
        "action_unique_trajectories": action_diversity,
        "action_variance": action_variance.tolist(),
        "joint_position_min": np.min(action_flat, axis=0).tolist(),
        "joint_position_max": np.max(action_flat, axis=0).tolist(),
        "tracking": tracking,
        "structure_valid": structure_valid,
        "tracking_valid": tracking_valid,
        "valid": bool(structure_valid and tracking_valid),
    }


def write_audit_report(
    path: Path,
    summary: dict[str, Any],
    checkpoint_path: Path,
    scene_path: Path,
    dataset_path: Path,
    episode_seconds: float,
) -> None:
    tracking = summary["validation"]["tracking"]
    lines = [
        "# Command-conditioned teacher dataset audit",
        "",
        f"Collection status: **{'VALID' if summary['validation']['valid'] else 'INVALID'}**",
        "",
        "## Provenance",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint SHA-256: `{summary['source_hashes']['checkpoint']}`",
        f"- MuJoCo scene: `{scene_path}`",
        f"- dataset: `{dataset_path}`",
        f"- backend: `{summary['backend']}`",
        f"- episode duration: {episode_seconds:.3f} s",
        f"- control period: {CONTROL_DT:.3f} s",
        "",
        "## Dataset format",
        "",
        "The command-free robot state is stored in `observations`; the exact 45-dimensional",
        "actor input is stored in `policy_observations`; desired commands are stored",
        "independently in `commands`. `actions` are processed 12-joint position targets",
        "and `raw_actions` preserves the deterministic PPO actor output.",
        "",
        "```text",
        "observations       [N,T,42]",
        "policy_observations[N,T,45]",
        "commands           [N,T,3]",
        "actions            [N,T,12]",
        "raw_actions        [N,T,12]",
        "velocities         [N,T,3]  body-frame achieved [vx, vy, yaw_rate]",
        "timestamps         [N,T]",
        "base_pose          [N,T,7]  position xyz + quaternion wxyz",
        "base_velocity      [N,T,6] body linear xyz + angular xyz",
        "contacts           [N,T,4] FR, FL, RR, RL",
        "```",
        "",
        "## Command diversity",
        "",
        f"- unique vx commands: {summary['validation']['unique_vx_commands']}",
        f"- unique vy commands: {summary['validation']['unique_vy_commands']}",
        f"- unique yaw commands: {summary['validation']['unique_yaw_commands']}",
        f"- required coverage check: `{summary['validation']['command_coverage_check']}`",
        "",
        "## Command achievement",
        "",
        "| axis | mean error | absolute error | error std | correlation | slope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for axis in AXES:
        stats = tracking[axis]
        lines.append(
            f"| {axis} | {stats['mean_error']:.6f} | {stats['mean_abs_error']:.6f} "
            f"| {stats['std_error']:.6f} | {stats['correlation']:.6f} | {stats['slope']:.6f} |"
        )
    lines += [
        "",
        "## Quality checks",
        "",
        f"- shape check: `{summary['validation']['shape_check']}`",
        f"- timestamps monotonic: `{summary['validation']['timestamps_monotonic']}`",
        f"- no NaN or inf: `{summary['validation']['finite_check']}`",
        f"- episode IDs contiguous: `{summary['validation']['episode_ids_contiguous']}`",
        f"- commands constant per episode: `{summary['validation']['command_constant_per_episode']}`",
        f"- action unique trajectories: `{summary['validation']['action_unique_trajectories']}`",
        f"- structure valid: `{summary['validation']['structure_valid']}`",
        f"- teacher tracking valid: `{summary['validation']['tracking_valid']}`",
        "",
        "## Required counters",
        "",
        "```text",
        "training steps: 0",
        "optimizer steps: 0",
        "checkpoint changes: 0",
        "original dataset writes: 0",
        "```",
        "",
        "## Recommendation",
        "",
        f"**{summary['recommended_next_step']}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=root, help="Go2Project root (default: inferred from script)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models/expert_model_500.pt",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=root / "external/unitree_go2/scene_go2.xml",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/imitation/expert_imitation_dataset_v1.npz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    scene_path = args.scene.resolve()
    output_path = args.output.resolve()
    report_dir = project_root / "results/genesis/generated/imitation"
    audit_path = report_dir / "command_teacher_dataset_audit.md"
    summary_path = report_dir / "command_teacher_dataset_summary.json"
    statistics_path = report_dir / "command_teacher_rollout_statistics.csv"
    provenance_path = report_dir / "command_teacher_provenance.json"
    plot_dir = report_dir / "command_teacher_dataset_plots"

    if args.episodes < 100:
        raise ValueError("The required collection has a minimum of 100 episodes")
    if not 10.0 <= args.episode_seconds <= 20.0:
        raise ValueError("Each episode must be between 10 and 20 seconds")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
    targets = (output_path, audit_path, summary_path, statistics_path, provenance_path, plot_dir)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite collection outputs; remove only a newly-created failed run "
            f"after review if necessary: {existing}"
        )

    np.random.seed(args.seed)
    episode_steps = int(round(args.episode_seconds / CONTROL_DT))
    actual_episode_seconds = episode_steps * CONTROL_DT
    if not math.isclose(actual_episode_seconds, args.episode_seconds, abs_tol=1.0e-9):
        raise ValueError("episode-seconds must be an exact multiple of the 0.02 s control period")
    episode_command_ids, episode_commands = make_schedule(args.episodes)
    teacher = FrozenTeacher(checkpoint_path, torch_threads=args.torch_threads)
    if teacher.observation_dim != POLICY_OBSERVATION_DIM or teacher.action_dim != ACTION_DIM:
        raise ValueError(
            f"Unexpected actor dimensions: {teacher.observation_dim} -> {teacher.action_dim}"
        )
    env = MuJoCoDeploymentScene(scene_path)

    observations = np.empty((args.episodes, episode_steps, OBSERVATION_DIM), dtype=np.float32)
    policy_observations = np.empty(
        (args.episodes, episode_steps, POLICY_OBSERVATION_DIM), dtype=np.float32
    )
    commands = np.empty((args.episodes, episode_steps, COMMAND_DIM), dtype=np.float32)
    actions = np.empty((args.episodes, episode_steps, ACTION_DIM), dtype=np.float32)
    raw_actions = np.empty((args.episodes, episode_steps, ACTION_DIM), dtype=np.float32)
    velocities = np.empty((args.episodes, episode_steps, COMMAND_DIM), dtype=np.float32)
    next_velocities = np.empty((args.episodes, episode_steps, COMMAND_DIM), dtype=np.float32)
    timestamps = np.empty((args.episodes, episode_steps), dtype=np.float64)
    episode_ids = np.empty((args.episodes, episode_steps), dtype=np.int32)
    command_ids = np.empty((args.episodes, episode_steps), dtype=np.int32)
    base_pose = np.empty((args.episodes, episode_steps, 7), dtype=np.float32)
    base_velocity = np.empty((args.episodes, episode_steps, 6), dtype=np.float32)
    contacts = np.empty((args.episodes, episode_steps, 4), dtype=bool)

    print(
        f"Collecting {args.episodes} episodes × {episode_steps} samples "
        f"({actual_episode_seconds:.2f} s each)"
    )
    for episode in range(args.episodes):
        command = episode_commands[episode]
        last_action = env.reset()
        for step in range(episode_steps):
            state, policy_obs, pose, base_vel, contact, _ = env.observation(command, last_action)
            raw_action = teacher(policy_obs)
            target_action = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
            timestamps[episode, step] = env.data.time
            episode_ids[episode, step] = episode
            command_ids[episode, step] = episode_command_ids[episode]
            observations[episode, step] = state
            policy_observations[episode, step] = policy_obs
            commands[episode, step] = command
            raw_actions[episode, step] = raw_action
            actions[episode, step] = target_action
            velocities[episode, step] = command_velocity_from_base_velocity(base_vel)
            base_pose[episode, step] = pose
            base_velocity[episode, step] = base_vel
            contacts[episode, step] = contact
            env.step(target_action)
            next_velocities[episode, step] = command_velocity_from_base_velocity(
                env.base_velocity()
            )
            last_action = raw_action
        if (episode + 1) % 10 == 0 or episode == 0 or episode + 1 == args.episodes:
            print(f"  episode {episode + 1:3d}/{args.episodes}: command={command.tolist()}")

    validation = validate_dataset(
        observations,
        policy_observations,
        commands,
        actions,
        raw_actions,
        velocities,
        timestamps,
        episode_ids,
        command_ids,
        contacts,
        episode_commands,
        args.episodes,
        actual_episode_seconds,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        policy_observations=policy_observations,
        commands=commands,
        actions=actions,
        raw_actions=raw_actions,
        velocities=velocities,
        next_velocities=next_velocities,
        timestamps=timestamps,
        episode_ids=episode_ids,
        command_ids=command_ids,
        base_pose=base_pose,
        base_velocity=base_velocity,
        contacts=contacts,
        episode_lengths=np.full(args.episodes, episode_steps, dtype=np.int32),
        state_feature_names=np.asarray(STATE_FEATURE_NAMES),
        policy_feature_names=np.asarray(POLICY_FEATURE_NAMES),
        command_catalog_json=np.asarray(json.dumps(command_catalog(), default=json_default)),
    )
    dataset_hash = sha256_file(output_path)
    make_plots(plot_dir, commands, velocities, timestamps, actions)
    rows = command_rows(episode_command_ids, episode_commands, velocities, timestamps)
    write_statistics_csv(statistics_path, rows)

    source_hashes = {
        "checkpoint": sha256_file(checkpoint_path),
        "scene": sha256_file(scene_path),
        "deploy_config": sha256_file(
            project_root / "configs/imitation/expert/deploy.yaml"
        ),
        "collector": sha256_file(Path(__file__).resolve()),
        "dataset": dataset_hash,
    }
    tracking = validation["tracking"]
    recommended = "TRAIN BC" if validation["valid"] else "FIX TEACHER COLLECTION"
    summary: dict[str, Any] = {
        "schema": "go2_command_teacher_v1",
        "script_version": SCRIPT_VERSION,
        "collection_time_utc": utc_now(),
        "backend": "direct_mujoco_deployment_equivalent",
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_iteration": teacher.iteration,
        "scene": str(scene_path),
        "episodes": args.episodes,
        "samples": int(args.episodes * episode_steps),
        "episode_steps": episode_steps,
        "episode_seconds": actual_episode_seconds,
        "control_dt": CONTROL_DT,
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "observation_dim": OBSERVATION_DIM,
        "policy_observation_dim": POLICY_OBSERVATION_DIM,
        "command_dim": COMMAND_DIM,
        "action_dim": ACTION_DIM,
        "command_frame": "body",
        "command_order": ["vx", "vy", "yaw_rate"],
        "policy_joint_order": [f"{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS],
        "dataset": str(output_path),
        "dataset_sha256": dataset_hash,
        "source_hashes": source_hashes,
        "command_catalog": command_catalog(),
        "tracking": tracking,
        "validation": validation,
        "counters": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "checkpoint_changes": 0,
            "original_dataset_writes": 0,
        },
        "recommended_next_step": recommended,
        "artifacts": {
            "interface_audit": str(report_dir / "teacher_rollout_interface_audit.md"),
            "dataset_audit": str(audit_path),
            "summary": str(summary_path),
            "statistics_csv": str(statistics_path),
            "provenance": str(provenance_path),
            "plots": str(plot_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")
    write_audit_report(
        audit_path,
        summary,
        checkpoint_path,
        scene_path,
        output_path,
        actual_episode_seconds,
    )
    provenance = {
        "schema": "go2_command_teacher_v1_provenance",
        "script_version": SCRIPT_VERSION,
        "collection_time_utc": summary["collection_time_utc"],
        "host": {"platform": platform.platform(), "python": sys.version},
        "backend": summary["backend"],
        "source_files": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": source_hashes["checkpoint"]},
            "scene": {"path": str(scene_path), "sha256": source_hashes["scene"]},
            "deploy_config": {
                "path": str(project_root / "configs/imitation/expert/deploy.yaml"),
                "sha256": source_hashes["deploy_config"],
            },
            "collector": {"path": str(Path(__file__).resolve()), "sha256": source_hashes["collector"]},
            "dataset": {"path": str(output_path), "sha256": source_hashes["dataset"]},
        },
        "sampling": {
            "episodes": args.episodes,
            "episode_seconds": actual_episode_seconds,
            "seed": args.seed,
            "command_catalog": command_catalog(),
            "commands_constant_per_episode": True,
        },
        "teacher": {
            "ppo_checkpoint_frozen": True,
            "deterministic_actor_mean": True,
            "observation_normalization_frozen": True,
            "policy_observation_order": list(POLICY_FEATURE_NAMES),
            "state_observation_order": list(STATE_FEATURE_NAMES),
            "joint_map": POLICY_TO_SCENE.tolist(),
            "action_scale": ACTION_SCALE.tolist(),
            "default_joint_pos": DEFAULT_JOINT_POS.tolist(),
            "pd_kp": KP.tolist(),
            "pd_kd": KD.tolist(),
        },
        "counters": summary["counters"],
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, default=json_default) + "\n", encoding="utf-8"
    )

    print("\nCOMMAND TEACHER DATASET COLLECTION COMPLETE")
    print(f"Teacher checkpoint: {checkpoint_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Samples: {args.episodes * episode_steps}")
    print("Command coverage:")
    print(f"vx: {validation['unique_vx_commands']}")
    print(f"vy: {validation['unique_vy_commands']}")
    print(f"yaw: {validation['unique_yaw_commands']}")
    print("Tracking:")
    print(f"vx correlation: {tracking['vx']['correlation']:.6f}")
    print(f"vy correlation: {tracking['vy']['correlation']:.6f}")
    print(f"yaw correlation: {tracking['yaw']['correlation']:.6f}")
    print("Dataset:")
    print("VALID" if validation["valid"] else "INVALID")
    print("\nRecommended next step:")
    print(recommended)
    print("\nCounters: training steps=0, optimizer steps=0, checkpoint changes=0, original dataset writes=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
