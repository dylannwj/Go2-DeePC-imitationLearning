#!/usr/bin/env python3
"""Validate the frozen command-conditioned BC policy in native Genesis.

This evaluator is intentionally evaluation-only.  It loads the existing BC
checkpoint and dataset, reconstructs the training observation/action contract,
and runs the policy through a headless Genesis rigid-body scene.  There is no
optimizer, runner, teacher update, dataset writer, pose replay, or PPO path in
this file.

The recovered teacher collector is the canonical source for the policy state
layout, command insertion, joint permutation, target interpretation, gains,
and simulation timing.  Genesis is used only as the dynamics backend here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.imitation_learning.collect_command_teacher_rollouts import (  # noqa: E402
    ACTION_DIM,
    ACTION_SCALE,
    AXES,
    COMMAND_DIM,
    CONTROL_DT,
    DECIMATION,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT,
    JOINTS,
    KD,
    KP,
    OBSERVATION_DIM,
    PHYSICS_DT,
    POLICY_FEATURE_NAMES,
    POLICY_LEGS,
    POLICY_TO_SCENE,
    SCENE_LEGS,
    STATE_FEATURE_NAMES,
)
from scripts.imitation_learning.train_command_bc import make_model, make_policy_input  # noqa: E402

SCRIPT_VERSION = "command-bc-genesis-evaluator-v1.1-interface-fix"
INPUT_DIM = OBSERVATION_DIM + COMMAND_DIM
GENESIS_URDF_RELATIVE = Path("urdf/go2/urdf/go2.urdf")
GENESIS_FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
GENESIS_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
# The recovered BC/PPO training environment initializes the base at 0.40 m.
# Root height is not part of the 42-D policy observation, but it affects the
# initial contact/dynamics state and therefore the standing gate.
GENESIS_ROOT_HEIGHT = 0.40
GENESIS_BASE_LINK_NAME = "base"
MAX_ROLL_PITCH_RAD = 1.0
MIN_ROOT_HEIGHT_M = 0.12
CONTACT_FORCE_TERMINATION_N = 1.0
JOINT_LIMIT_TOLERANCE_RAD = 0.02
SIGN_DEADBAND_MOTION = 0.005
FULL_DURATION_TOLERANCE_STEPS = 0
DEFAULT_SEED_BASE = 20260804
DEFAULT_EPISODE_SECONDS = 10.0
DEFAULT_STATIC_REPETITIONS = 5
DEFAULT_TRANSITION_EPISODES = 10
VIDEO_FPS = int(round(1.0 / CONTROL_DT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def sign_token(value: float) -> str:
    return f"{value:+.2f}".replace("+", "p").replace("-", "m").replace(".", "d")


def command_label(command: np.ndarray) -> str:
    values = np.asarray(command, dtype=float)
    return f"vx{sign_token(values[0])}_vy{sign_token(values[1])}_yaw{sign_token(values[2])}"


def make_command_bank() -> list[dict[str, Any]]:
    """Return the required bank with out-of-dataset values substituted.

    The dataset component limits are vx=[-0.10, 0.30], vy=[-0.30, 0.30],
    yaw=[-0.50, 0.50].  The substitutions are recorded in the interface and
    rollout reports rather than silently testing unsupported values.
    """

    rows: list[tuple[str, str, tuple[float, float, float]]] = [
        ("standing", "standing", (0.00, 0.00, 0.00)),
        ("forward", "forward_020", (0.20, 0.00, 0.00)),
        ("forward", "forward_030_substituted_for_040", (0.30, 0.00, 0.00)),
        ("backward", "backward_005_substituted_for_020", (-0.05, 0.00, 0.00)),
        ("backward", "backward_010_substituted_for_040", (-0.10, 0.00, 0.00)),
        ("lateral", "lateral_left_010_substituted_for_015", (0.00, 0.10, 0.00)),
        ("lateral", "lateral_right_010_substituted_for_015", (0.00, -0.10, 0.00)),
        ("lateral", "lateral_left_030", (0.00, 0.30, 0.00)),
        ("lateral", "lateral_right_030", (0.00, -0.30, 0.00)),
        ("yaw", "yaw_left_030", (0.00, 0.00, 0.30)),
        ("yaw", "yaw_right_030", (0.00, 0.00, -0.30)),
        ("yaw", "yaw_left_050_substituted_for_060", (0.00, 0.00, 0.50)),
        ("yaw", "yaw_right_050_substituted_for_060", (0.00, 0.00, -0.50)),
        ("diagonal", "diagonal_forward_left", (0.30, 0.20, 0.00)),
        ("diagonal", "diagonal_forward_right", (0.30, -0.20, 0.00)),
        ("diagonal", "diagonal_backward_left_substituted_for_minus_030", (-0.10, 0.20, 0.00)),
        ("diagonal", "diagonal_backward_right_substituted_for_minus_030", (-0.10, -0.20, 0.00)),
        ("translation_plus_yaw", "curve_forward_left", (0.30, 0.00, 0.30)),
        ("translation_plus_yaw", "curve_forward_right", (0.30, 0.00, -0.30)),
        ("translation_plus_yaw", "curve_backward_left_substituted_for_minus_025", (-0.10, 0.00, 0.30)),
        ("translation_plus_yaw", "curve_backward_right_substituted_for_minus_025", (-0.10, 0.00, -0.30)),
        ("translation_plus_yaw", "combined_forward_left", (0.30, 0.20, 0.30)),
        ("translation_plus_yaw", "combined_forward_right", (0.30, -0.20, -0.30)),
    ]
    return [
        {
            "command_id": index,
            "category": category,
            "name": name,
            "command": np.asarray(command, dtype=np.float32),
        }
        for index, (category, name, command) in enumerate(rows)
    ]


def transition_bank() -> list[dict[str, Any]]:
    """Return ten deterministic 2-second-segment transition episodes."""

    zero = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
    forward = np.asarray((0.20, 0.0, 0.0), dtype=np.float32)
    backward = np.asarray((-0.10, 0.0, 0.0), dtype=np.float32)
    left = np.asarray((0.0, 0.20, 0.0), dtype=np.float32)
    right = np.asarray((0.0, -0.20, 0.0), dtype=np.float32)
    yaw_left = np.asarray((0.0, 0.0, 0.30), dtype=np.float32)
    yaw_right = np.asarray((0.0, 0.0, -0.30), dtype=np.float32)
    diagonal = np.asarray((0.30, 0.20, 0.0), dtype=np.float32)
    curve = np.asarray((0.30, 0.0, 0.30), dtype=np.float32)
    sequences = [
        ("stand_forward_stand", (zero, forward, zero)),
        ("stand_forward_stand_repeat", (zero, forward, zero)),
        ("stand_backward_stand", (zero, backward, zero)),
        ("stand_backward_stand_repeat", (zero, backward, zero)),
        ("stand_lateral_left_right_stand", (zero, left, right, zero)),
        ("stand_lateral_left_right_stand_repeat", (zero, left, right, zero)),
        ("stand_yaw_left_right_stand", (zero, yaw_left, yaw_right, zero)),
        ("stand_yaw_left_right_stand_repeat", (zero, yaw_left, yaw_right, zero)),
        ("forward_diagonal_curve_stand", (forward, diagonal, curve, zero)),
        ("forward_backward_stand", (forward, backward, zero)),
    ]
    return [
        {
            "transition_id": index,
            "name": name,
            "segments": [np.asarray(command, dtype=np.float32) for command in commands],
        }
        for index, (name, commands) in enumerate(sequences)
    ]


def get_genesis_urdf() -> Path:
    import genesis as gs

    return Path(gs.__file__).resolve().parent / "assets" / GENESIS_URDF_RELATIVE


def build_interface_audit(
    checkpoint_path: Path,
    dataset_path: Path,
    teacher_checkpoint: Path,
    normalization_paths: dict[str, Path],
) -> dict[str, Any]:
    """Reconstruct and validate the complete BC-to-Genesis contract."""

    import torch

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_is_dict = isinstance(checkpoint, dict)
    check("checkpoint is a dictionary", checkpoint_is_dict, type(checkpoint).__name__)
    state = checkpoint.get("model_state_dict", {}) if checkpoint_is_dict else {}
    required_state = {
        "network.0.weight": (512, INPUT_DIM),
        "network.0.bias": (512,),
        "network.2.weight": (256, 512),
        "network.2.bias": (256,),
        "network.4.weight": (128, 256),
        "network.4.bias": (128,),
        "network.6.weight": (ACTION_DIM, 128),
        "network.6.bias": (ACTION_DIM,),
    }
    state_shapes_ok = True
    state_shape_details: list[str] = []
    for name, expected_shape in required_state.items():
        actual = tuple(state[name].shape) if name in state else None
        passed = actual == expected_shape
        state_shapes_ok &= passed
        state_shape_details.append(f"{name}={actual}, expected={expected_shape}")
    check("checkpoint network tensor shapes", state_shapes_ok, "; ".join(state_shape_details))

    architecture = checkpoint.get("architecture") if checkpoint_is_dict else None
    check(
        "checkpoint architecture",
        architecture == [INPUT_DIM, 512, 256, 128, ACTION_DIM],
        f"actual={architecture}; expected={[INPUT_DIM, 512, 256, 128, ACTION_DIM]}",
    )
    dimensions = {
        "input_dimension": checkpoint.get("input_dimension"),
        "observation_dimension": checkpoint.get("observation_dimension"),
        "command_dimension": checkpoint.get("command_dimension"),
        "action_dimension": checkpoint.get("action_dimension"),
    }
    check(
        "checkpoint dimensions",
        dimensions == {
            "input_dimension": INPUT_DIM,
            "observation_dimension": OBSERVATION_DIM,
            "command_dimension": COMMAND_DIM,
            "action_dimension": ACTION_DIM,
        },
        json.dumps(dimensions, sort_keys=True),
    )

    with np.load(dataset_path, allow_pickle=False) as data:
        data_shapes = {key: list(data[key].shape) for key in data.files}
        observations = np.asarray(data["observations"])
        commands = np.asarray(data["commands"])
        actions = np.asarray(data["actions"])
        policy_observations = np.asarray(data["policy_observations"])
        state_feature_names = tuple(str(item) for item in data["state_feature_names"])
        policy_feature_names = tuple(str(item) for item in data["policy_feature_names"])
        dataset_finite = all(
            np.isfinite(data[key]).all()
            for key in ("observations", "commands", "actions", "policy_observations")
        )
        layout_expected = make_policy_input(observations, commands)
        layout_error = float(np.max(np.abs(layout_expected - policy_observations)))
        command_min = commands.reshape(-1, COMMAND_DIM).min(axis=0)
        command_max = commands.reshape(-1, COMMAND_DIM).max(axis=0)

    check(
        "dataset dimensions",
        observations.ndim == 3
        and observations.shape[-1] == OBSERVATION_DIM
        and commands.shape[-1] == COMMAND_DIM
        and actions.shape[-1] == ACTION_DIM
        and policy_observations.shape[-1] == INPUT_DIM,
        json.dumps(
            {
                "observations": list(observations.shape),
                "commands": list(commands.shape),
                "actions": list(actions.shape),
                "policy_observations": list(policy_observations.shape),
            }
        ),
    )
    check("dataset finite", bool(dataset_finite), "observations, commands, actions, policy_observations")
    check("dataset policy layout", layout_error <= 1.0e-6, f"max_abs_difference={layout_error:.3e}")
    check(
        "dataset commands constant per episode",
        bool(np.all(commands == commands[:, :1])),
        str(bool(np.all(commands == commands[:, :1]))),
    )
    check(
        "state feature ordering",
        state_feature_names == tuple(STATE_FEATURE_NAMES),
        f"count={len(state_feature_names)}; expected_count={len(STATE_FEATURE_NAMES)}",
    )
    check(
        "policy feature ordering",
        policy_feature_names == tuple(POLICY_FEATURE_NAMES),
        f"count={len(policy_feature_names)}; expected_count={len(POLICY_FEATURE_NAMES)}",
    )

    normalization = checkpoint.get("normalization", {}) if checkpoint_is_dict else {}
    normalization_checks: dict[str, Any] = {}
    for name, dimension in (("observation", OBSERVATION_DIM), ("command", COMMAND_DIM), ("action", ACTION_DIM)):
        entry = normalization.get(name, {})
        mean = np.asarray(entry.get("mean", []), dtype=np.float32)
        std = np.asarray(entry.get("std", []), dtype=np.float32)
        passed = mean.shape == (dimension,) and std.shape == (dimension,) and np.all(np.isfinite(std)) and np.all(std > 0)
        normalization_checks[name] = {
            "passed": bool(passed),
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "std_min": float(np.min(std)) if std.size else None,
        }
        check(f"checkpoint {name} normalization", passed, json.dumps(normalization_checks[name]))

        stats_path = normalization_paths.get(name)
        if stats_path is not None and stats_path.is_file():
            with stats_path.open("rb") as stream:
                stats = pickle.load(stream)
            file_match = np.allclose(mean, np.asarray(stats["mean"], dtype=np.float32)) and np.allclose(
                std, np.asarray(stats["std"], dtype=np.float32)
            )
            check(f"checkpoint versus {name} stats file", bool(file_match), str(bool(file_match)))

    policy_to_scene_ok = np.array_equal(np.sort(POLICY_TO_SCENE), np.arange(ACTION_DIM))
    check(
        "policy-to-Genesis joint permutation",
        bool(policy_to_scene_ok),
        f"policy_order={list(POLICY_LEGS)}; Genesis_order={list(SCENE_LEGS)}; map={POLICY_TO_SCENE.tolist()}",
    )
    check(
        "policy output semantics",
        checkpoint.get("output_semantics") == "12-D joint position target in physical units",
        str(checkpoint.get("output_semantics")),
    )

    try:
        genesis_urdf = get_genesis_urdf()
        genesis_available = genesis_urdf.is_file()
        genesis_urdf_hash = sha256_file(genesis_urdf) if genesis_available else None
        import genesis as gs

        genesis_version = getattr(gs, "__version__", "unknown")
        genesis_python = str(Path(gs.__file__).resolve())
    except Exception as exc:  # pragma: no cover - exercised only on missing runtime
        genesis_available = False
        genesis_urdf = Path("unavailable")
        genesis_urdf_hash = None
        genesis_version = "unavailable"
        genesis_python = f"unavailable: {type(exc).__name__}: {exc}"
    check("Genesis runtime and Go2 URDF available", genesis_available, str(genesis_urdf))

    teacher_available = teacher_checkpoint.is_file()
    check("teacher checkpoint available for provenance", teacher_available, str(teacher_checkpoint))

    details = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "schema": checkpoint.get("schema"),
            "architecture": architecture,
            "dimensions": dimensions,
            "state_keys": sorted(state),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "shapes": data_shapes,
            "command_min": command_min.tolist(),
            "command_max": command_max.tolist(),
        },
        "teacher_checkpoint": {
            "path": str(teacher_checkpoint),
            "sha256": sha256_file(teacher_checkpoint) if teacher_available else None,
        },
        "normalization": normalization_checks,
        "observation": {
            "dimension": OBSERVATION_DIM,
            "ordering": list(STATE_FEATURE_NAMES),
            "construction": "[base_ang_vel[3], projected_gravity[3], joint_pos_rel[12], joint_vel_rel[12], last_action[12]]",
            "command_insertion": "[state[0:6], normalized command [vx, vy, yaw_rate], state[6:42]]",
            "normalization": "state and command normalized with frozen checkpoint train statistics; RSL-RL epsilon is not used by this BC model",
        },
        "command": {
            "dimension": COMMAND_DIM,
            "ordering": ["vx", "vy", "yaw_rate"],
            "frame": "body",
            "units": ["m/s", "m/s", "rad/s"],
            "dataset_component_min": command_min.tolist(),
            "dataset_component_max": command_max.tolist(),
        },
        "action": {
            "dimension": ACTION_DIM,
            "meaning": "physical joint-position target in policy order",
            "policy_joint_order": [f"{leg}_{joint}" for leg in POLICY_LEGS for joint in JOINTS],
            "Genesis_joint_order": list(GENESIS_JOINT_NAMES),
            "policy_to_Genesis_index": POLICY_TO_SCENE.tolist(),
            "historical_raw_teacher_interpretation": "teacher raw action -> default_joint_pos + 0.5 * raw action",
            "BC_target_source": "the checkpoint output is already the physical 12-D target; Genesis PD consumes it directly",
            "default_joint_position_policy": DEFAULT_JOINT_POS.tolist(),
            "action_scale_from_teacher_collector": ACTION_SCALE.tolist(),
            "pd_kp_policy": KP.tolist(),
            "pd_kd_policy": KD.tolist(),
            "effort_limit_policy": EFFORT_LIMIT.tolist(),
        },
        "timing": {
            "control_dt_s": CONTROL_DT,
            "control_hz": 1.0 / CONTROL_DT,
            "Genesis_physics_dt_s": PHYSICS_DT,
            "Genesis_physics_hz": 1.0 / PHYSICS_DT,
            "physics_steps_per_policy_action": DECIMATION,
            "Genesis_integrator": "native rigid-body scene.step() with Newton constraint solver",
        },
        "startup": {
            "root_position": [0.0, 0.0, GENESIS_ROOT_HEIGHT],
            "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_position": DEFAULT_JOINT_POS.tolist(),
            "velocities": "zero",
            "last_action": "zero",
            "reset": "Genesis set_pos/set_quat/set_dofs_position followed by native dynamics; no trajectory state is replayed",
        },
        "frames": {
            "command": "body frame, [vx, vy, yaw_rate]",
            "root_position": "Genesis world frame",
            "achieved_velocity": "root linear velocity rotated into the instantaneous body frame; yaw rate is body angular velocity z",
            "quaternion": "Genesis wxyz",
            "positive_yaw": "counter-clockwise about +z",
        },
        "termination": {
            "invalid": "any non-finite action or measured state",
            "root_height_below_m": MIN_ROOT_HEIGHT_M,
            "roll_or_pitch_above_rad": MAX_ROLL_PITCH_RAD,
            "base_contact_force_above_N": CONTACT_FORCE_TERMINATION_N,
            "joint_limit_tolerance_rad": JOINT_LIMIT_TOLERANCE_RAD,
            "time_limit": "requested episode duration",
        },
        "genesis": {
            "version": genesis_version,
            "python": genesis_python,
            "urdf": str(genesis_urdf),
            "urdf_sha256": genesis_urdf_hash,
        },
        "checks": checks,
        "valid": bool(all(item["passed"] for item in checks)),
    }
    return details


def write_interface_audit(path: Path, audit: dict[str, Any]) -> None:
    result = "PASS" if audit["valid"] else "FAIL"
    lines = [
        "# BC Genesis Interface Audit",
        "",
        f"Interface reconstruction: **{result}**",
        "",
        "This audit was generated before closed-loop evaluation. It reconstructs the dimensions and runtime contract from the frozen checkpoint, the read-only training dataset, and the existing collector/training implementation.",
        "",
        "## Checkpoint and dimensions",
        "",
        f"- Checkpoint: `{audit['checkpoint']['path']}`",
        f"- SHA-256: `{audit['checkpoint']['sha256']}`",
        f"- Format/schema: `{audit['checkpoint']['schema']}`",
        f"- Architecture: `{audit['checkpoint']['architecture']}`",
        f"- Dimensions: `{audit['checkpoint']['dimensions']}`",
        "- Hidden activations: `ELU`",
        "- Output: physical 12-joint position target",
        "",
        "## Interface fields",
        "",
        "### Observation",
        "",
        "```text",
        "0:3   base_ang_vel_x, base_ang_vel_y, base_ang_vel_z",
        "3:6   projected_gravity_x, projected_gravity_y, projected_gravity_z",
        "6:18  joint_pos - default_joint_pos, policy leg/joint order",
        "18:30 joint_vel, policy leg/joint order",
        "30:42 previous policy action, policy leg/joint order",
        "```",
        "",
        "The normalized 45-vector is `[normalized_state[0:6], normalized_command[0:3], normalized_state[6:42]]`. State and command means/stds are frozen in the checkpoint and are not recomputed during evaluation.",
        "",
        "### Command",
        "",
        f"- Order: `{audit['command']['ordering']}`",
        f"- Frame: `{audit['command']['frame']}`",
        f"- Units: `{audit['command']['units']}`",
        f"- Dataset component limits: min `{audit['command']['dataset_component_min']}`, max `{audit['command']['dataset_component_max']}`",
        "",
        "### Action and joint mapping",
        "",
        f"- BC output semantics: `{audit['action']['meaning']}`",
        f"- Policy order: `{audit['action']['policy_joint_order']}`",
        f"- Genesis order: `{audit['action']['Genesis_joint_order']}`",
        f"- Policy-to-Genesis index map: `{audit['action']['policy_to_Genesis_index']}`",
        f"- Default policy joint position: `{audit['action']['default_joint_position_policy']}`",
        f"- PD Kp: `{audit['action']['pd_kp_policy']}`",
        f"- PD Kd: `{audit['action']['pd_kd_policy']}`",
        f"- Effort clip: `{audit['action']['effort_limit_policy']}`",
        "",
        "The historical teacher generated the equivalent target as `default_joint_pos + 0.5 * raw_action`; the BC checkpoint was trained against those already-processed physical targets, so the Genesis evaluator applies the BC output directly as the PD target and does not apply the 0.5 scale a second time.",
        "",
        "## Timing, Genesis model, and frames",
        "",
        f"- Control rate: `{audit['timing']['control_hz']:.1f} Hz` (`{audit['timing']['control_dt_s']:.6f} s`)",
        f"- Genesis physics timestep: `{audit['timing']['Genesis_physics_dt_s']:.6f} s` (`{audit['timing']['Genesis_physics_hz']:.1f} Hz`)",
        f"- Native physics steps per policy action: `{audit['timing']['physics_steps_per_policy_action']}`",
        f"- Genesis version: `{audit['genesis']['version']}`",
        f"- Go2 URDF: `{audit['genesis']['urdf']}`",
        f"- Go2 URDF SHA-256: `{audit['genesis']['urdf_sha256']}`",
        f"- Command frame: `{audit['frames']['command']}`",
        f"- Achieved velocity frame: `{audit['frames']['achieved_velocity']}`",
        "",
        "## Reset and termination",
        "",
        f"- Reset root position: `{audit['startup']['root_position']}`",
        f"- Reset root quaternion: `{audit['startup']['root_quaternion_wxyz']}`",
        f"- Reset joint position: `{audit['startup']['joint_position']}`",
        "- Reset velocities and previous action: zero",
        f"- Termination conditions: `{audit['termination']}`",
        "",
        "## Audit checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for item in audit["checks"]:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    lines.extend(
        [
            "",
            "The existing project did not contain a Genesis environment with this recovered 50 Hz / 0.005 s / physical-target contract. The evaluator therefore supplies a thin Genesis dynamics adapter while reusing the canonical feature names, command layout, joint map, PD values, reset values, and timing from the recovered deployment collector. It performs native `scene.step()` calls only; it never overwrites a rollout trajectory after reset.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


class FrozenBCPolicy:
    """Checkpoint-backed deterministic policy with no optimizer state."""

    def __init__(self, checkpoint_path: Path):
        import torch

        self.torch = torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.checkpoint = checkpoint
        self.model = make_model(torch)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.normalization = {
            key: {
                "mean": np.asarray(value["mean"], dtype=np.float32),
                "std": np.asarray(value["std"], dtype=np.float32),
            }
            for key, value in checkpoint["normalization"].items()
        }

    def __call__(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        command = np.asarray(command, dtype=np.float32)
        state_normalized = (
            state - self.normalization["observation"]["mean"]
        ) / self.normalization["observation"]["std"]
        command_normalized = (
            command - self.normalization["command"]["mean"]
        ) / self.normalization["command"]["std"]
        policy_input = make_policy_input(state_normalized, command_normalized).astype(np.float32)
        with self.torch.no_grad():
            normalized_action = self.model(self.torch.from_numpy(policy_input))
        action = (
            normalized_action.detach().cpu().numpy()
            * self.normalization["action"]["std"]
            + self.normalization["action"]["mean"]
        )
        return np.asarray(action, dtype=np.float32)


def physical_target_to_policy_action(target_policy: np.ndarray) -> np.ndarray:
    """Recover the raw policy action used by the BC ``last_action`` feature.

    The BC labels are physical joint-position targets, while the training
    observation stores the pre-scale policy action.  The collector and BC
    closed-loop evaluator use ``default + 0.5 * raw_action``; keep this
    inverse mapping in policy joint order and do not clip it.
    """

    target_policy = np.asarray(target_policy, dtype=np.float32)
    if target_policy.shape[-1] != ACTION_DIM:
        raise ValueError(f"physical target has shape {target_policy.shape}; expected last dimension {ACTION_DIM}")
    return ((target_policy - DEFAULT_JOINT_POS) / ACTION_SCALE).astype(np.float32)


def quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, 1.0e-12)


def quat_inverse_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the body frame for wxyz quaternions."""

    q = quat_normalize(quaternion)
    vector = np.asarray(vector, dtype=np.float64)
    w, x, y, z = [q[..., index] for index in range(4)]
    rotation = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rotation[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[..., 0, 1] = 2.0 * (x * y - w * z)
    rotation[..., 0, 2] = 2.0 * (x * z + w * y)
    rotation[..., 1, 0] = 2.0 * (x * y + w * z)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[..., 1, 2] = 2.0 * (y * z - w * x)
    rotation[..., 2, 0] = 2.0 * (x * z - w * y)
    rotation[..., 2, 1] = 2.0 * (y * z + w * x)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.einsum("...ji,...j->...i", rotation, vector)


def quat_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    q = quat_normalize(quaternion)
    w, x, y, z = [q[..., index] for index in range(4)]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(pitch_argument)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


class GenesisDeploymentScene:
    """Minimal native Genesis scene matching the recovered deployment path."""

    def __init__(self, num_envs: int, *, camera: bool = False, camera_res: tuple[int, int] = (640, 480)):
        import genesis as gs
        import torch

        self.gs = gs
        self.torch = torch
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.cpu, logging_level="warning")
        self.num_envs = int(num_envs)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=PHYSICS_DT, substeps=1),
            rigid_options=gs.options.RigidOptions(
                dt=PHYSICS_DT,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(GENESIS_URDF_RELATIVE),
                pos=(0.0, 0.0, GENESIS_ROOT_HEIGHT),
                quat=(1.0, 0.0, 0.0, 0.0),
                links_to_keep=list(GENESIS_FOOT_NAMES),
            ),
            visualize_contact=False,
        )
        self.camera = None
        if camera:
            self.camera = self.scene.add_camera(
                res=camera_res,
                pos=(1.2, -1.2, 0.8),
                lookat=(0.0, 0.0, 0.18),
                fov=40,
                GUI=False,
            )
        self.scene.build(n_envs=self.num_envs)
        self.motor_dofs = [
            int(self.robot.get_joint(name).dofs_idx_local[0]) for name in GENESIS_JOINT_NAMES
        ]
        if len(set(self.motor_dofs)) != ACTION_DIM:
            raise ValueError(f"Genesis motor DOFs are not unique: {self.motor_dofs}")
        self.foot_link_indices = [
            int(self.robot.get_link(name).idx - self.robot.link_start) for name in GENESIS_FOOT_NAMES
        ]
        self.base_link_index = int(self.robot.get_link(GENESIS_BASE_LINK_NAME).idx - self.robot.link_start)
        lower, upper = self.robot.get_dofs_limit(self.motor_dofs)
        self.joint_lower_scene = numpy_value(lower).reshape(-1)
        self.joint_upper_scene = numpy_value(upper).reshape(-1)
        if self.joint_lower_scene.shape != (ACTION_DIM,) or self.joint_upper_scene.shape != (ACTION_DIM,):
            raise ValueError("Genesis joint limits did not resolve to twelve actuated DOFs")
        self.joint_lower_policy = self.joint_lower_scene[POLICY_TO_SCENE]
        self.joint_upper_policy = self.joint_upper_scene[POLICY_TO_SCENE]
        self.kp_scene = KP[POLICY_TO_SCENE].astype(np.float32)
        self.kd_scene = KD[POLICY_TO_SCENE].astype(np.float32)
        self.effort_scene = EFFORT_LIMIT[POLICY_TO_SCENE].astype(np.float32)
        self.default_scene = DEFAULT_JOINT_POS[POLICY_TO_SCENE].astype(np.float32)

    def reset(self) -> None:
        positions = np.repeat(
            np.asarray([[0.0, 0.0, GENESIS_ROOT_HEIGHT]], dtype=np.float32),
            self.num_envs,
            axis=0,
        )
        quaternions = np.repeat(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            self.num_envs,
            axis=0,
        )
        joint_positions = np.repeat(self.default_scene[None, :], self.num_envs, axis=0)
        self.robot.set_pos(self.torch.from_numpy(positions), zero_velocity=True)
        self.robot.set_quat(self.torch.from_numpy(quaternions), zero_velocity=True)
        self.robot.set_dofs_position(
            position=self.torch.from_numpy(joint_positions),
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        position_scene = numpy_value(self.robot.get_dofs_position(self.motor_dofs)).astype(np.float32)
        velocity_scene = numpy_value(self.robot.get_dofs_velocity(self.motor_dofs)).astype(np.float32)
        position_scene = position_scene.reshape(self.num_envs, ACTION_DIM)
        velocity_scene = velocity_scene.reshape(self.num_envs, ACTION_DIM)
        return position_scene[:, POLICY_TO_SCENE], velocity_scene[:, POLICY_TO_SCENE]

    def observe(self, last_action: np.ndarray) -> dict[str, np.ndarray]:
        last_action = np.asarray(last_action, dtype=np.float32).reshape(self.num_envs, ACTION_DIM)
        position, velocity = self._joint_state()
        root_position = numpy_value(self.robot.get_pos()).astype(np.float32).reshape(self.num_envs, 3)
        root_quaternion = numpy_value(self.robot.get_quat()).astype(np.float32).reshape(self.num_envs, 4)
        world_linear_velocity = numpy_value(self.robot.get_vel()).astype(np.float32).reshape(self.num_envs, 3)
        world_angular_velocity = numpy_value(self.robot.get_ang()).astype(np.float32).reshape(self.num_envs, 3)
        base_linear_velocity = quat_inverse_rotate(root_quaternion, world_linear_velocity).astype(np.float32)
        base_angular_velocity = quat_inverse_rotate(root_quaternion, world_angular_velocity).astype(np.float32)
        projected_gravity = quat_inverse_rotate(
            root_quaternion,
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0]), (self.num_envs, 3)),
        ).astype(np.float32)
        root_rpy = quat_to_rpy(root_quaternion).astype(np.float32)
        state = np.concatenate(
            (
                base_angular_velocity,
                projected_gravity,
                position - DEFAULT_JOINT_POS[None, :],
                velocity,
                last_action,
            ),
            axis=1,
        ).astype(np.float32)
        contact_forces = numpy_value(self.robot.get_links_net_contact_force()).astype(np.float32)
        contact_forces = contact_forces.reshape(self.num_envs, -1, 3)
        foot_contact = (
            np.linalg.norm(contact_forces[:, self.foot_link_indices, :], axis=-1)
            > CONTACT_FORCE_TERMINATION_N
        )
        base_contact = np.linalg.norm(contact_forces[:, self.base_link_index, :], axis=-1) > CONTACT_FORCE_TERMINATION_N
        joint_margin = np.minimum(
            position - self.joint_lower_policy[None, :],
            self.joint_upper_policy[None, :] - position,
        ).astype(np.float32)
        return {
            "state": state,
            "root_position": root_position,
            "root_quaternion": root_quaternion,
            "root_rpy": root_rpy,
            "world_linear_velocity": world_linear_velocity,
            "world_angular_velocity": world_angular_velocity,
            "body_linear_velocity": base_linear_velocity,
            "body_angular_velocity": base_angular_velocity,
            "joint_position": position.astype(np.float32),
            "joint_velocity": velocity.astype(np.float32),
            "foot_contacts": foot_contact.astype(bool),
            "base_contact": base_contact.astype(bool),
            "body_height": root_position[:, 2].copy(),
            "joint_limit_margin": joint_margin,
        }

    def step(self, target_policy: np.ndarray) -> np.ndarray:
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(self.num_envs, ACTION_DIM)
        current_position, current_velocity = self._joint_state()
        torque_policy = KP[None, :] * (target_policy - current_position) - KD[None, :] * current_velocity
        torque_policy = np.clip(torque_policy, -EFFORT_LIMIT[None, :], EFFORT_LIMIT[None, :])
        torque_scene = torque_policy[:, POLICY_TO_SCENE].astype(np.float32)
        self.robot.control_dofs_force(self.torch.from_numpy(torque_scene), self.motor_dofs)
        for _ in range(DECIMATION):
            self.scene.step()
        return torque_policy.astype(np.float32)

    def render_frame(self, env_index: int = 0) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("render requested on a scene without a camera")
        position = numpy_value(self.robot.get_pos()).reshape(self.num_envs, 3)[env_index]
        robot_position = position.astype(np.float32)
        self.camera.set_pose(
            pos=robot_position + np.asarray([-1.1, -1.1, 0.70], dtype=np.float32),
            lookat=robot_position + np.asarray([0.0, 0.0, 0.16], dtype=np.float32),
        )
        rendered = self.camera.render()
        frame = rendered[0] if isinstance(rendered, tuple) else rendered
        frame = np.asarray(frame)
        if frame.dtype.kind == "f":
            frame = np.clip(frame * (255.0 if np.max(frame) <= 1.0 else 1.0), 0.0, 255.0)
        frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame


class EpisodeRecorder:
    """In-memory time-series recorder for one dynamics episode."""

    FIELDS = (
        "timestamp",
        "command",
        "root_position",
        "root_quaternion",
        "root_rpy",
        "world_linear_velocity",
        "world_angular_velocity",
        "body_linear_velocity",
        "body_angular_velocity",
        "policy_action",
        "joint_position",
        "joint_velocity",
        "foot_contacts",
        "body_height",
        "joint_limit_margin",
        "termination_flag",
    )

    def __init__(self) -> None:
        self.values: dict[str, list[np.ndarray | float | int]] = {field: [] for field in self.FIELDS}

    def append(
        self,
        timestamp: float,
        command: np.ndarray,
        snapshot: dict[str, np.ndarray],
        action: np.ndarray,
        termination_flag: bool,
        index: int = 0,
    ) -> None:
        self.values["timestamp"].append(float(timestamp))
        self.values["command"].append(np.asarray(command[index], dtype=np.float32).copy())
        for name in (
            "root_position",
            "root_quaternion",
            "root_rpy",
            "world_linear_velocity",
            "world_angular_velocity",
            "body_linear_velocity",
            "body_angular_velocity",
            "joint_position",
            "joint_velocity",
            "foot_contacts",
            "joint_limit_margin",
        ):
            self.values[name].append(np.asarray(snapshot[name][index]).copy())
        self.values["policy_action"].append(np.asarray(action[index], dtype=np.float32).copy())
        self.values["body_height"].append(float(snapshot["body_height"][index]))
        self.values["termination_flag"].append(int(bool(termination_flag)))

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(values) for name, values in self.values.items()}


def termination_for(
    snapshot: dict[str, np.ndarray],
    action: np.ndarray,
    index: int,
) -> tuple[bool, str, int]:
    values = [
        snapshot[name][index]
        for name in (
            "root_position",
            "root_quaternion",
            "root_rpy",
            "world_linear_velocity",
            "world_angular_velocity",
            "body_linear_velocity",
            "body_angular_velocity",
            "joint_position",
            "joint_velocity",
        )
    ]
    finite = all(np.all(np.isfinite(value)) for value in values) and np.all(np.isfinite(action[index]))
    if not finite:
        return True, "invalid_nan_or_inf", 1
    if bool(snapshot["base_contact"][index]):
        return True, "base_contact", 0
    if float(snapshot["body_height"][index]) < MIN_ROOT_HEIGHT_M:
        return True, "root_height_below_threshold", 0
    roll, pitch = np.abs(snapshot["root_rpy"][index, :2])
    if roll > MAX_ROLL_PITCH_RAD or pitch > MAX_ROLL_PITCH_RAD:
        return True, "roll_or_pitch_threshold", 0
    if np.any(snapshot["joint_limit_margin"][index] < -JOINT_LIMIT_TOLERANCE_RAD):
        return True, "joint_limit_violation", 0
    return False, "", 0


def response_delay(command: np.ndarray, velocity: np.ndarray, timestamps: np.ndarray) -> float | None:
    command = np.asarray(command, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    delays: list[float] = []
    for axis in range(COMMAND_DIM):
        desired = float(command[-1, axis])
        if abs(desired) <= 1.0e-9:
            continue
        threshold = max(SIGN_DEADBAND_MOTION, 0.5 * abs(desired))
        matched = np.where(np.sign(desired) * velocity[:, axis] >= threshold)[0]
        if matched.size:
            delays.append(float(timestamps[int(matched[0])]))
    return float(np.mean(delays)) if delays else None


def metric_for_episode(
    episode_id: int,
    phase: str,
    category: str,
    name: str,
    seed: int,
    requested_duration: float,
    arrays: dict[str, np.ndarray],
    termination_reason: str,
    initial_root: np.ndarray,
    initial_yaw: float,
) -> dict[str, Any]:
    timestamps = np.asarray(arrays["timestamp"], dtype=float)
    command = np.asarray(arrays["command"], dtype=float)
    velocity = np.asarray(arrays["body_linear_velocity"], dtype=float)
    velocity = np.column_stack((velocity[:, 0], velocity[:, 1], arrays["body_angular_velocity"][:, 2]))
    error = velocity - command
    count = len(timestamps)
    steady_start = max(0, count // 2)
    steady_error = error[steady_start:]
    if count:
        final_position = np.asarray(arrays["root_position"][-1], dtype=float)
        final_rpy = np.asarray(arrays["root_rpy"][-1], dtype=float)
        displacement_world = final_position[:2] - np.asarray(initial_root, dtype=float)[:2]
        c, s = math.cos(initial_yaw), math.sin(initial_yaw)
        displacement_body = np.asarray((c * displacement_world[0] + s * displacement_world[1], -s * displacement_world[0] + c * displacement_world[1]))
    else:
        final_position = np.full(3, np.nan)
        final_rpy = np.full(3, np.nan)
        displacement_body = np.full(2, np.nan)
    active_axis_pairs: list[dict[str, Any]] = []
    steady_velocity = velocity[steady_start:] if count else np.empty((0, 3))
    for axis, axis_name in enumerate(AXES):
        desired = float(command[-1, axis]) if count else 0.0
        if abs(desired) <= 1.0e-9:
            continue
        achieved = float(np.mean(steady_velocity[:, axis])) if len(steady_velocity) else float("nan")
        correct = bool(math.isfinite(achieved) and desired * achieved > 0.0 and abs(achieved) >= SIGN_DEADBAND_MOTION)
        active_axis_pairs.append(
            {
                "axis": axis_name,
                "desired": desired,
                "achieved_steady_mean": achieved,
                "correct_sign": correct,
            }
        )
    full_duration = count >= int(round(requested_duration / CONTROL_DT)) - FULL_DURATION_TOLERANCE_STEPS
    termination_flag = bool(np.any(arrays["termination_flag"])) if count else True
    invalid_count = 0
    if termination_reason == "invalid_nan_or_inf":
        invalid_count = 1
    joint_violation_count = int(np.count_nonzero(np.any(arrays["joint_limit_margin"] < -JOINT_LIMIT_TOLERANCE_RAD, axis=1))) if count else 0
    survival_time = requested_duration if not termination_flag else float(timestamps[np.argmax(arrays["termination_flag"])]) if count else 0.0
    minimum_margin = float(np.min(arrays["joint_limit_margin"])) if count else float("nan")
    stop_drift = float(np.linalg.norm(displacement_body)) if category == "standing" else float("nan")
    return {
        "episode_id": int(episode_id),
        "phase": phase,
        "category": category,
        "movement_type": name,
        "seed": int(seed),
        "command_vx": float(command[-1, 0]) if count else 0.0,
        "command_vy": float(command[-1, 1]) if count else 0.0,
        "command_yaw_rate": float(command[-1, 2]) if count else 0.0,
        "requested_duration_s": float(requested_duration),
        "survival_time_s": survival_time,
        "steps": int(count),
        "full_duration": bool(full_duration and not termination_flag),
        "fall_status": bool(termination_reason in {"base_contact", "root_height_below_threshold", "roll_or_pitch_threshold"}),
        "termination_flag": bool(termination_flag),
        "termination_reason": termination_reason or "time_limit",
        "mean_abs_vx_tracking_error": float(np.mean(np.abs(error[:, 0]))) if count else float("nan"),
        "mean_abs_vy_tracking_error": float(np.mean(np.abs(error[:, 1]))) if count else float("nan"),
        "mean_abs_yaw_tracking_error": float(np.mean(np.abs(error[:, 2]))) if count else float("nan"),
        "rmse_vx": float(np.sqrt(np.mean(error[:, 0] ** 2))) if count else float("nan"),
        "rmse_vy": float(np.sqrt(np.mean(error[:, 1] ** 2))) if count else float("nan"),
        "rmse_yaw_rate": float(np.sqrt(np.mean(error[:, 2] ** 2))) if count else float("nan"),
        "steady_state_vx_error": float(np.mean(np.abs(steady_error[:, 0]))) if len(steady_error) else float("nan"),
        "steady_state_vy_error": float(np.mean(np.abs(steady_error[:, 1]))) if len(steady_error) else float("nan"),
        "steady_state_yaw_error": float(np.mean(np.abs(steady_error[:, 2]))) if len(steady_error) else float("nan"),
        "command_response_delay_s": response_delay(command, velocity, timestamps) if count else None,
        "stop_drift_m": stop_drift,
        "final_body_displacement_x_m": float(displacement_body[0]),
        "final_body_displacement_y_m": float(displacement_body[1]),
        "max_abs_roll_rad": float(np.max(np.abs(arrays["root_rpy"][:, 0]))) if count else float("nan"),
        "max_abs_pitch_rad": float(np.max(np.abs(arrays["root_rpy"][:, 1]))) if count else float("nan"),
        "minimum_body_height_m": float(np.min(arrays["body_height"])) if count else float("nan"),
        "maximum_body_height_m": float(np.max(arrays["body_height"])) if count else float("nan"),
        "maximum_action_magnitude": float(np.max(np.abs(arrays["policy_action"]))) if count else float("nan"),
        "mean_action_magnitude": float(np.mean(np.linalg.norm(arrays["policy_action"], axis=1))) if count else float("nan"),
        "maximum_action_jump": float(np.max(np.abs(np.diff(arrays["policy_action"], axis=0)))) if count > 1 else 0.0,
        "minimum_joint_limit_margin_rad": minimum_margin,
        "joint_limit_violation_count": joint_violation_count,
        "invalid_action_or_state_count": int(invalid_count),
        "mean_foot_contact_count": float(np.mean(np.sum(arrays["foot_contacts"], axis=1))) if count else float("nan"),
        "active_axis_pairs": active_axis_pairs,
        "final_root_position_x_m": float(final_position[0]),
        "final_root_position_y_m": float(final_position[1]),
        "final_root_height_m": float(final_position[2]),
        "final_roll_rad": float(final_rpy[0]),
        "final_pitch_rad": float(final_rpy[1]),
        "final_yaw_rad": float(final_rpy[2]),
    }


def run_episode_batch(
    scene: GenesisDeploymentScene,
    policy: FrozenBCPolicy,
    commands_at_step: Callable[[int, int], np.ndarray],
    names: list[str],
    categories: list[str],
    seeds: list[int],
    episode_ids: list[int],
    episode_seconds: float,
    phase: str,
    render: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[np.ndarray]]]:
    if scene.num_envs != len(seeds):
        raise ValueError("scene batch size and seed count differ")
    max_steps = int(round(episode_seconds / CONTROL_DT))
    scene.reset()
    last_action = np.zeros((scene.num_envs, ACTION_DIM), dtype=np.float32)
    initial = scene.observe(last_action)
    initial_root = initial["root_position"].copy()
    initial_yaw = quat_to_rpy(initial["root_quaternion"])[:, 2].copy()
    recorders = [EpisodeRecorder() for _ in seeds]
    active = np.ones(scene.num_envs, dtype=bool)
    reasons = ["" for _ in seeds]
    frames: list[list[np.ndarray]] = [[] for _ in seeds]

    for step in range(max_steps):
        command = np.asarray(
            [commands_at_step(step, index) for index in range(scene.num_envs)],
            dtype=np.float32,
        )
        observation = scene.observe(last_action)
        proposed_action = policy(observation["state"], command)
        if proposed_action.shape != (scene.num_envs, ACTION_DIM):
            raise ValueError(f"policy returned {proposed_action.shape}, expected {(scene.num_envs, ACTION_DIM)}")
        applied_action = proposed_action.copy()
        for index in range(scene.num_envs):
            if not active[index]:
                applied_action[index] = observation["joint_position"][index]
        scene.step(applied_action)
        # Genesis receives the physical target, but BC's next observation
        # expects the corresponding raw/pre-scale policy action.
        next_last_action = physical_target_to_policy_action(applied_action)
        measured = scene.observe(next_last_action)
        for index in range(scene.num_envs):
            if active[index]:
                terminated, reason, _ = termination_for(measured, applied_action, index)
                if terminated:
                    active[index] = False
                    reasons[index] = reason
            termination_flag = bool(not active[index])
            recorders[index].append(
                (step + 1) * CONTROL_DT,
                command,
                measured,
                applied_action,
                termination_flag,
                index,
            )
        if render:
            frames[0].append(scene.render_frame(0))
        last_action = next_last_action
        if not np.any(active):
            break

    metrics: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for index, recorder in enumerate(recorders):
        episode_arrays = recorder.arrays()
        arrays.append(episode_arrays)
        metrics.append(
            metric_for_episode(
                episode_ids[index],
                phase,
                categories[index],
                names[index],
                seeds[index],
                episode_seconds,
                episode_arrays,
                reasons[index],
                initial_root[index],
                float(initial_yaw[index]),
            )
        )
    return metrics, arrays, frames


def make_transition_scheduler(segments: list[np.ndarray], segment_steps: int) -> Callable[[int, int], np.ndarray]:
    def schedule(step: int, _index: int) -> np.ndarray:
        segment = min(step // segment_steps, len(segments) - 1)
        return segments[segment]

    return schedule


def run_static_bank(
    scene: GenesisDeploymentScene,
    policy: FrozenBCPolicy,
    commands: list[dict[str, Any]],
    repetitions: int,
    episode_seconds: float,
    seed_base: int,
    episode_id_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]], int]:
    metrics: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    episode_id = episode_id_start
    for command_spec in commands:
        seeds = [seed_base + episode_id * 100 + repetition for repetition in range(repetitions)]
        command = np.asarray(command_spec["command"], dtype=np.float32)

        def scheduler(_step: int, _index: int, command: np.ndarray = command) -> np.ndarray:
            return command

        batch_metrics, batch_arrays, _ = run_episode_batch(
            scene,
            policy,
            scheduler,
            [command_spec["name"]] * repetitions,
            [command_spec["category"]] * repetitions,
            seeds,
            list(range(episode_id, episode_id + repetitions)),
            episode_seconds,
            "static_command",
        )
        metrics.extend(batch_metrics)
        arrays.extend(batch_arrays)
        episode_id += repetitions
        print(
            f"static {command_spec['name']}: "
            f"survival={sum(row['full_duration'] for row in batch_metrics)}/{len(batch_metrics)}",
            flush=True,
        )
    return metrics, arrays, episode_id


def run_transition_bank(
    scene: GenesisDeploymentScene,
    policy: FrozenBCPolicy,
    transitions: list[dict[str, Any]],
    episode_seconds: float,
    seed_base: int,
    episode_id_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]], int]:
    segment_seconds = 2.0
    segment_steps = int(round(segment_seconds / CONTROL_DT))
    metrics: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    episode_id = episode_id_start
    for transition in transitions:
        segments = transition["segments"]
        scheduler = make_transition_scheduler(segments, segment_steps)
        metrics_batch, arrays_batch, _ = run_episode_batch(
            scene,
            policy,
            scheduler,
            [transition["name"]],
            ["transitions"],
            [seed_base + episode_id],
            [episode_id],
            max(episode_seconds, segment_seconds * len(segments)),
            "transition",
        )
        metrics.extend(metrics_batch)
        arrays.extend(arrays_batch)
        episode_id += 1
        print(
            f"transition {transition['name']}: "
            f"survival={int(metrics_batch[0]['full_duration'])}/1 "
            f"reason={metrics_batch[0]['termination_reason']}",
            flush=True,
        )
    return metrics, arrays, episode_id


def save_timeseries(path: Path, metrics: list[dict[str, Any]], arrays: list[dict[str, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {}
    for row, episode_arrays in zip(metrics, arrays, strict=True):
        prefix = f"episode_{int(row['episode_id']):04d}"
        for name, array in episode_arrays.items():
            payload[f"{prefix}__{name}"] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(json_safe(value), separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def aggregate_command_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metrics:
        groups.setdefault((str(row["phase"]), str(row["movement_type"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (phase, movement_type), rows in groups.items():
        category = str(rows[0]["category"])
        sign_pairs = [pair for row in rows for pair in row["active_axis_pairs"]]
        result.append(
            {
                "phase": phase,
                "category": category,
                "movement_type": movement_type,
                "episodes": len(rows),
                "full_duration_survival_rate": float(np.mean([row["full_duration"] for row in rows])),
                "mean_survival_time_s": float(np.mean([row["survival_time_s"] for row in rows])),
                "mean_abs_vx_tracking_error": float(np.mean([row["mean_abs_vx_tracking_error"] for row in rows])),
                "mean_abs_vy_tracking_error": float(np.mean([row["mean_abs_vy_tracking_error"] for row in rows])),
                "mean_abs_yaw_tracking_error": float(np.mean([row["mean_abs_yaw_tracking_error"] for row in rows])),
                "rmse_vx": float(np.mean([row["rmse_vx"] for row in rows])),
                "rmse_vy": float(np.mean([row["rmse_vy"] for row in rows])),
                "rmse_yaw_rate": float(np.mean([row["rmse_yaw_rate"] for row in rows])),
                "correct_motion_sign_rate": float(np.mean([pair["correct_sign"] for pair in sign_pairs])) if sign_pairs else None,
                "minimum_body_height_m": float(np.min([row["minimum_body_height_m"] for row in rows])),
                "maximum_abs_roll_rad": float(np.max([row["max_abs_roll_rad"] for row in rows])),
                "maximum_abs_pitch_rad": float(np.max([row["max_abs_pitch_rad"] for row in rows])),
                "mean_stop_drift_m": float(np.nanmean([row["stop_drift_m"] for row in rows])) if any(math.isfinite(float(row["stop_drift_m"])) for row in rows) else None,
                "invalid_action_or_state_episodes": int(sum(row["invalid_action_or_state_count"] > 0 for row in rows)),
            }
        )
    result.sort(key=lambda row: (row["phase"], row["category"], row["movement_type"]))
    return result


def category_aggregate(metrics: list[dict[str, Any]], phase: str = "static_command") -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in metrics:
        if row["phase"] == phase:
            groups.setdefault(str(row["category"]), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for category, rows in groups.items():
        pairs = [pair for row in rows for pair in row["active_axis_pairs"]]
        result[category] = {
            "category": category,
            "episodes": len(rows),
            "full_duration_survival_rate": float(np.mean([row["full_duration"] for row in rows])),
            "correct_motion_sign_rate": float(np.mean([pair["correct_sign"] for pair in pairs])) if pairs else None,
            "mean_abs_vx_tracking_error": float(np.mean([row["mean_abs_vx_tracking_error"] for row in rows])),
            "mean_abs_vy_tracking_error": float(np.mean([row["mean_abs_vy_tracking_error"] for row in rows])),
            "mean_abs_yaw_tracking_error": float(np.mean([row["mean_abs_yaw_tracking_error"] for row in rows])),
            "rmse_vx": float(np.mean([row["rmse_vx"] for row in rows])),
            "rmse_vy": float(np.mean([row["rmse_vy"] for row in rows])),
            "rmse_yaw_rate": float(np.mean([row["rmse_yaw_rate"] for row in rows])),
            "mean_stop_drift_m": float(np.nanmean([row["stop_drift_m"] for row in rows])) if any(math.isfinite(float(row["stop_drift_m"])) for row in rows) else None,
            "maximum_action_magnitude": float(np.max([row["maximum_action_magnitude"] for row in rows])),
            "minimum_joint_limit_margin_rad": float(np.min([row["minimum_joint_limit_margin_rad"] for row in rows])),
            "invalid_action_or_state_episodes": int(sum(row["invalid_action_or_state_count"] > 0 for row in rows)),
        }
    return result


def evaluate_gate(metrics: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    static = [row for row in metrics if row["phase"] == "static_command"]
    standing = [row for row in static if row["category"] == "standing"]
    categories = sorted({str(row["category"]) for row in static if row["category"] != "standing"})
    all_sign_pairs = [pair for row in static for pair in row["active_axis_pairs"]]
    sign_rate = float(np.mean([pair["correct_sign"] for pair in all_sign_pairs])) if all_sign_pairs else 0.0
    no_invalid = sum(row["invalid_action_or_state_count"] for row in metrics) == 0
    all_axis_rows: dict[str, list[dict[str, Any]]] = {axis: [] for axis in AXES}
    for row in static:
        for pair in row["active_axis_pairs"]:
            all_axis_rows[pair["axis"]].append(pair)
    axis_response: dict[str, Any] = {}
    for axis in AXES:
        rows = all_axis_rows[axis]
        positive = [row for row in rows if row["desired"] > 0]
        negative = [row for row in rows if row["desired"] < 0]
        axis_response[axis] = {
            "active_pairs": len(rows),
            "sign_rate": float(np.mean([row["correct_sign"] for row in rows])) if rows else 0.0,
            "positive_sign_rate": float(np.mean([row["correct_sign"] for row in positive])) if positive else 0.0,
            "negative_sign_rate": float(np.mean([row["correct_sign"] for row in negative])) if negative else 0.0,
            "positive_tested": bool(positive),
            "negative_tested": bool(negative),
        }
    category_summary = category_aggregate(metrics)
    overall_survival = float(np.mean([row["full_duration"] for row in static])) if static else 0.0
    major_category_survival = {
        category: category_summary.get(category, {}).get("full_duration_survival_rate", 0.0)
        for category in categories
    }
    combined_rows = [row for row in static if row["category"] in {"diagonal", "translation_plus_yaw"}]
    combined_sign_rate = float(
        np.mean([pair["correct_sign"] for row in combined_rows for pair in row["active_axis_pairs"]])
    ) if any(row["active_axis_pairs"] for row in combined_rows) else 0.0
    standing_ok = bool(
        standing
        and all(row["full_duration"] and not row["fall_status"] and row["invalid_action_or_state_count"] == 0 for row in standing)
    )
    axis_sensitivity_ok = bool(
        all(
            value["positive_tested"]
            and value["negative_tested"]
            and value["positive_sign_rate"] >= 0.90
            and value["negative_sign_rate"] >= 0.90
            for value in axis_response.values()
        )
    )
    combined_response_ok = bool(combined_rows and combined_sign_rate >= 0.90)
    stop_rows = [row for row in static if row["category"] == "standing" and math.isfinite(float(row["stop_drift_m"]))]
    stop_ok = bool(stop_rows and max(float(row["stop_drift_m"]) for row in stop_rows) <= 0.50)
    standing_reason_counts: dict[str, int] = {}
    for row in standing:
        reason = str(row["termination_reason"])
        standing_reason_counts[reason] = standing_reason_counts.get(reason, 0) + 1
    recommended_ok = bool(
        overall_survival >= 0.80
        and all(value >= 0.60 for value in major_category_survival.values())
        and sign_rate >= 0.90
        and standing_ok
        and no_invalid
    )
    mandatory_ok = bool(
        audit["valid"]
        and standing_ok
        and no_invalid
        and axis_sensitivity_ok
        and combined_response_ok
        and stop_ok
        and overall_survival > 0.0
    )
    if not audit["valid"]:
        decision = "BC GENESIS GATE FAIL — INTERFACE AUDIT FAILURE"
        next_step = "FIX RUNTIME INTERFACE OR BC INITIALIZATION"
    elif not standing_ok:
        decision = "BC GENESIS GATE FAIL — STANDING FAILURE"
        next_step = "FIX RUNTIME INTERFACE OR BC INITIALIZATION"
    elif not mandatory_ok:
        decision = "BC GENESIS GATE FAIL — DIAGNOSE BEFORE PPO"
        next_step = "FIX RUNTIME INTERFACE OR BC INITIALIZATION"
    elif recommended_ok:
        decision = "BC GENESIS GATE PASS — PROCEED TO PPO"
        next_step = "PPO REFINEMENT"
    else:
        decision = "BC GENESIS GATE PARTIAL — TARGETED DATA OR INTERFACE CORRECTION"
        next_step = "TARGETED DATA OR INTERFACE CORRECTION"
    return {
        "decision": decision,
        "next_step": next_step,
        "overall_full_duration_survival_rate": overall_survival,
        "standing_survival_rate": float(np.mean([row["full_duration"] for row in standing])) if standing else 0.0,
        "major_category_survival_rate": major_category_survival,
        "correct_motion_sign_rate": sign_rate,
        "axis_response": axis_response,
        "combined_response_sign_rate": combined_sign_rate,
        "standing_ok": standing_ok,
        "no_invalid_actions_or_states": no_invalid,
        "axis_sensitivity_ok": axis_sensitivity_ok,
        "combined_response_ok": combined_response_ok,
        "stop_ok": stop_ok,
        "mandatory_conditions_ok": mandatory_ok,
        "recommended_pass_standard_ok": recommended_ok,
        "weakest_category": min(major_category_survival, key=major_category_survival.get) if major_category_survival else None,
        "standing_termination_reasons": standing_reason_counts,
        "standing_mean_survival_time_s": float(np.mean([row["survival_time_s"] for row in standing])) if standing else None,
        "diagnosis": (
            "All neutral Genesis episodes failed after finite startup inference, with the repeated terminal event "
            f"{standing_reason_counts} and no invalid actions/states. The checkpoint/dataset interface audit passed; "
            "the evidence therefore points to a runtime control/model realization mismatch or an incompatible BC "
            "initialization state, not an optimizer or dataset mutation. Movement and transition banks were stopped "
            "per the standing-failure rule."
            if standing and not standing_ok
            else "No standing-failure diagnosis was triggered."
        ),
        "category_summary": category_summary,
    }


def plot_results(plot_dir: Path, metrics: list[dict[str, Any]], arrays: list[dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    static = [row for row in metrics if row["phase"] == "static_command"]
    summary = category_aggregate(metrics)
    categories = list(summary)
    if categories:
        x = np.arange(len(categories))
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.bar(x, [summary[item]["full_duration_survival_rate"] for item in categories], color="#4472c4")
        axis.set_ylim(0.0, 1.05)
        axis.set_xticks(x, categories, rotation=25, ha="right")
        axis.set_ylabel("full-duration survival rate")
        axis.set_title("Genesis survival rate by command category")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_dir / "survival_rate_by_command_category.png", dpi=160)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for index, (field, label) in enumerate(
            (("mean_abs_vx_tracking_error", "vx"), ("mean_abs_vy_tracking_error", "vy"), ("mean_abs_yaw_tracking_error", "yaw rate"))
        ):
            axes[index].bar(x, [summary[item][field] for item in categories], color="#ed7d31")
            axes[index].set_xticks(x, categories, rotation=25, ha="right")
            axes[index].set_ylabel("mean absolute error")
            axes[index].set_title(label)
            axes[index].grid(axis="y", alpha=0.25)
        figure.suptitle("Tracking error by command category")
        figure.tight_layout()
        figure.savefig(plot_dir / "tracking_error_by_command_category.png", dpi=160)
        plt.close(figure)

    # Static commanded-versus-achieved plots use steady-state means per episode.
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis_index, axis_name in enumerate(AXES):
        desired: list[float] = []
        achieved: list[float] = []
        for _static_index, row in enumerate(static):
            episode_arrays = arrays[metrics.index(row)]
            commands = episode_arrays["command"]
            velocity = np.column_stack(
                (episode_arrays["body_linear_velocity"][:, 0], episode_arrays["body_linear_velocity"][:, 1], episode_arrays["body_angular_velocity"][:, 2])
            )
            start = len(commands) // 2
            desired.append(float(commands[-1, axis_index]))
            achieved.append(float(np.mean(velocity[start:, axis_index])))
        axes[axis_index].scatter(desired, achieved, s=18, alpha=0.75, color="#4472c4")
        if desired:
            low = min(min(desired), min(achieved))
            high = max(max(desired), max(achieved))
            axes[axis_index].plot([low, high], [low, high], "k--", linewidth=1)
        axes[axis_index].set_xlabel(f"commanded {axis_name}")
        axes[axis_index].set_ylabel(f"achieved {axis_name}")
        axes[axis_index].grid(alpha=0.25)
    figure.suptitle("Commanded versus achieved body-frame velocity")
    figure.tight_layout()
    figure.savefig(plot_dir / "commanded_vs_achieved_vx_vy_yaw.png", dpi=160)
    plt.close(figure)

    representative_indices = [index for index, row in enumerate(metrics) if row["movement_type"] in {"standing", "forward_020", "backward_010_substituted_for_040", "lateral_left_010_substituted_for_015", "yaw_left_030", "diagonal_forward_left", "curve_forward_left"}]
    if representative_indices:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for index in representative_indices[:7]:
            series = arrays[index]
            axis.plot(series["timestamp"], series["body_height"], label=metrics[index]["movement_type"])
        axis.set_xlabel("time (s)")
        axis.set_ylabel("body height (m)")
        axis.set_title("Body height over time")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plot_dir / "body_height_over_time.png", dpi=160)
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for index in representative_indices[:7]:
            series = arrays[index]
            axes[0].plot(series["timestamp"], series["root_rpy"][:, 0], label=metrics[index]["movement_type"])
            axes[1].plot(series["timestamp"], series["root_rpy"][:, 1], label=metrics[index]["movement_type"])
        axes[0].set_ylabel("roll (rad)")
        axes[1].set_ylabel("pitch (rad)")
        axes[1].set_xlabel("time (s)")
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        axes[0].legend(fontsize=8, ncol=2)
        figure.suptitle("Roll and pitch over time")
        figure.tight_layout()
        figure.savefig(plot_dir / "roll_and_pitch_over_time.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 4.5))
        names = [metrics[index]["movement_type"] for index in representative_indices[:7]]
        values = [metrics[index]["stop_drift_m"] if math.isfinite(float(metrics[index]["stop_drift_m"])) else 0.0 for index in representative_indices[:7]]
        axis.bar(np.arange(len(names)), values, color="#70ad47")
        axis.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
        axis.set_ylabel("stop drift (m)")
        axis.set_title("Stop drift")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_dir / "stop_drift.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 5))
        series = arrays[representative_indices[0]]
        axis.plot(series["timestamp"], series["policy_action"])
        axis.set_xlabel("time (s)")
        axis.set_ylabel("12-D physical target")
        axis.set_title(f"Representative action traces: {metrics[representative_indices[0]]['movement_type']}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_dir / "representative_action_traces.png", dpi=160)
        plt.close(figure)


def write_video(path: Path, frames: list[np.ndarray], fps: int = VIDEO_FPS) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7)


def render_representative_videos(
    policy: FrozenBCPolicy,
    command_specs: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    episode_seconds: float,
    seed_base: int,
    video_dir: Path,
    standing_only: bool = False,
) -> list[dict[str, Any]]:
    selected_names = {
        "standing",
        "forward_020",
        "backward_010_substituted_for_040",
        "lateral_left_010_substituted_for_015",
        "lateral_right_010_substituted_for_015",
        "yaw_left_030",
        "yaw_right_030",
        "diagonal_forward_left",
        "curve_forward_left",
    }
    selected = [spec for spec in command_specs if spec["name"] in selected_names]
    if standing_only:
        selected = [spec for spec in selected if spec["category"] == "standing"]
    video_rows: list[dict[str, Any]] = []
    video_dir.mkdir(parents=True, exist_ok=True)
    video_index = 0
    for spec in selected:
        scene = GenesisDeploymentScene(1, camera=True)
        command = np.asarray(spec["command"], dtype=np.float32)
        metrics, _arrays, frames = run_episode_batch(
            scene,
            policy,
            lambda _step, _index, command=command: command,
            [spec["name"]],
            [spec["category"]],
            [seed_base + 9000 + video_index],
            [100000 + video_index],
            episode_seconds,
            "video_representative",
            render=True,
        )
        row = metrics[0]
        status = "PASS" if row["full_duration"] else "FAIL"
        filename = f"{spec['name']}__{command_label(command)}__seed{row['seed']}__{status}.mp4"
        path = video_dir / filename
        write_video(path, frames[0])
        video_rows.append({"movement_type": spec["name"], "command": command, "seed": row["seed"], "pass": status == "PASS", "path": str(path)})
        print(f"video {path.name}", flush=True)
        video_index += 1

    if standing_only or not transitions:
        return video_rows
    transition = transitions[8] if len(transitions) > 8 else transitions[-1]
    scene = GenesisDeploymentScene(1, camera=True)
    segment_steps = int(round(2.0 / CONTROL_DT))
    metrics, _arrays, frames = run_episode_batch(
        scene,
        policy,
        make_transition_scheduler(transition["segments"], segment_steps),
        [transition["name"]],
        ["transitions"],
        [seed_base + 9999],
        [100000 + video_index],
        max(episode_seconds, 2.0 * len(transition["segments"])),
        "video_transition",
        render=True,
    )
    row = metrics[0]
    status = "PASS" if row["full_duration"] else "FAIL"
    command = transition["segments"][0]
    filename = f"transition__{transition['name']}__{command_label(command)}__seed{row['seed']}__{status}.mp4"
    path = video_dir / filename
    write_video(path, frames[0])
    video_rows.append({"movement_type": transition["name"], "command": command, "seed": row["seed"], "pass": status == "PASS", "path": str(path)})
    print(f"video {path.name}", flush=True)
    return video_rows


def write_rollout_report(
    path: Path,
    audit: dict[str, Any],
    metrics: list[dict[str, Any]],
    command_summary: list[dict[str, Any]],
    gate: dict[str, Any],
    hashes: dict[str, Any],
    artifacts: dict[str, Any],
    video_rows: list[dict[str, Any]],
    elapsed_s: float,
) -> None:
    static = [row for row in metrics if row["phase"] == "static_command"]
    invalid_count = int(sum(row["invalid_action_or_state_count"] for row in metrics))
    lines = [
        "# BC Genesis Closed-Loop Rollout Report",
        "",
        f"Generated: `{utc_now()}`",
        f"Evaluator: `{SCRIPT_VERSION}`",
        f"Wall time: `{elapsed_s:.1f} s`",
        "",
        "## Main recommendation",
        "",
        f"**{gate['decision']}**",
        f"NEXT STEP: **{gate['next_step']}**",
        "",
        "No PPO update or BC optimizer update was run during this validation.",
        "",
        "## Evaluation coverage",
        "",
        f"- Total episodes: `{len(metrics)}`",
        f"- Static command episodes: `{len(static)}`",
        f"- Transition episodes: `{sum(row['phase'] == 'transition' for row in metrics)}`",
        f"- Full-duration survival rate: `{gate['overall_full_duration_survival_rate']:.3%}`",
        f"- Standing survival rate: `{gate['standing_survival_rate']:.3%}`",
        f"- Correct motion sign rate: `{gate['correct_motion_sign_rate']:.3%}`",
        f"- NaN/invalid action or state count: `{invalid_count}`",
        f"- Weakest command category by survival: `{gate['weakest_category']}`",
        f"- Movement bank status: `{'NOT RUN — standing failure' if not gate['standing_ok'] else 'RUN'}`",
        "",
        "## Gate checks",
        "",
        "| Condition | Result | Evidence |",
        "|---|---|---|",
        f"| Interface audit | {'PASS' if audit['valid'] else 'FAIL'} | `{audit['checkpoint']['dimensions']}`; canonical ordering checks |",
        f"| Standing stability | {'PASS' if gate['standing_ok'] else 'FAIL'} | survival `{gate['standing_survival_rate']:.3%}` |",
        f"| No invalid actions/states | {'PASS' if gate['no_invalid_actions_or_states'] else 'FAIL'} | count `{invalid_count}` |",
        f"| All command axes have signed response | {'PASS' if gate['axis_sensitivity_ok'] else 'FAIL'} | `{gate['axis_response']}` |",
        f"| Combined commands combine axes | {'PASS' if gate['combined_response_ok'] else 'FAIL'} | sign rate `{gate['combined_response_sign_rate']:.3%}` |",
        f"| Stop stability | {'PASS' if gate['stop_ok'] else 'FAIL'} | standing stop-drift check |",
        f"| Recommended survival standard | {'PASS' if gate['recommended_pass_standard_ok'] else 'FAIL'} | overall `{gate['overall_full_duration_survival_rate']:.3%}`; per-category `{gate['major_category_survival_rate']}` |",
        "",
        "## Command-category summary",
        "",
        "| Phase | Category | Episodes | Survival | Sign rate | MAE vx | MAE vy | MAE yaw |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_aggregate(metrics).values():
        lines.append(
            f"| static | {row['category']} | {row['episodes']} | {row['full_duration_survival_rate']:.3f} | "
            f"{row['correct_motion_sign_rate'] if row['correct_motion_sign_rate'] is not None else 'n/a'} | "
            f"{row['mean_abs_vx_tracking_error']:.4f} | {row['mean_abs_vy_tracking_error']:.4f} | {row['mean_abs_yaw_tracking_error']:.4f} |"
        )
    if not gate["standing_ok"]:
        lines.extend(
            [
                "",
                "## Standing failure diagnosis",
                "",
                f"- Termination reasons: `{gate['standing_termination_reasons']}`",
                f"- Mean survival time: `{gate['standing_mean_survival_time_s']:.3f} s`",
                f"- Diagnosis: {gate['diagnosis']}",
                "- The evaluator did not run movement or transition episodes after this gate failed.",
            ]
        )
    transition_rows = [row for row in command_summary if row["phase"] == "transition"]
    if transition_rows:
        lines.extend(["", "### Transition episodes", "", "| Sequence | Episodes | Survival | MAE vx | MAE vy | MAE yaw |", "|---|---:|---:|---:|---:|---:|"])
        for row in transition_rows:
            lines.append(
                f"| {row['movement_type']} | {row['episodes']} | {row['full_duration_survival_rate']:.3f} | "
                f"{row['mean_abs_vx_tracking_error']:.4f} | {row['mean_abs_vy_tracking_error']:.4f} | {row['mean_abs_yaw_tracking_error']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interface and provenance",
            "",
            f"- Interface audit: `{artifacts['interface_audit']}`",
            f"- Checkpoint: `{audit['checkpoint']['path']}`",
            f"- Dataset: `{audit['dataset']['path']}`",
            f"- Teacher checkpoint: `{audit['teacher_checkpoint']['path']}`",
            f"- Genesis URDF: `{audit['genesis']['urdf']}`",
            f"- Checkpoint hash unchanged: **{hashes['bc_checkpoint']['unchanged']}**",
            f"- Dataset hash unchanged: **{hashes['dataset']['unchanged']}**",
            f"- Teacher hash unchanged: **{hashes['teacher_checkpoint']['unchanged']}**",
            "",
            "## Required counters",
            "",
            "- PPO optimizer updates: **0**",
            "- BC optimizer updates: **0**",
            "- Teacher checkpoint changes: **0**",
            "- Dataset writes: **0**",
            "- BC checkpoint writes: **0**",
            "",
            "## Videos",
            "",
        ]
    )
    for row in video_rows:
        lines.append(f"- `{row['movement_type']}`: `{row['path']}` ({'PASS' if row['pass'] else 'FAIL'})")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, artifact in artifacts.items():
        lines.append(f"- `{name}`: `{artifact}`")
    lines.extend(["", "The time-series logs contain post-step root state, body-frame achieved velocity, quaternion, joint state, contacts, body height, action, termination flag, and commanded values for every recorded episode.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/expert_model_500.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/imitation/expert_imitation_dataset_v1.npz"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--episode-seconds", type=float, default=DEFAULT_EPISODE_SECONDS)
    parser.add_argument("--static-repetitions", type=int, default=DEFAULT_STATIC_REPETITIONS)
    parser.add_argument("--transition-episodes", type=int, default=DEFAULT_TRANSITION_EPISODES)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--command-name", action="append", default=None, help="restrict static bank during development; omit for full bank")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> int:
    args = parse_args()
    if args.episode_seconds <= 0.0:
        raise ValueError("episode duration must be positive")
    if args.static_repetitions <= 0:
        raise ValueError("static repetitions must be positive")
    checkpoint_path = resolve_path(args.checkpoint)
    dataset_path = resolve_path(args.dataset)
    report_dir = REPO_ROOT / "results/genesis/generated"
    interface_audit_path = report_dir / "bc_genesis_interface_audit.md"
    rollout_report_path = report_dir / "bc_genesis_rollout_report.md"
    summary_path = report_dir / "bc_genesis_rollout_summary.json"
    episode_metrics_path = report_dir / "bc_genesis_episode_metrics.csv"
    command_summary_path = report_dir / "bc_genesis_command_summary.csv"
    plot_dir = report_dir / "bc_genesis_plots"
    video_dir = report_dir / "bc_genesis_videos"
    timeseries_path = report_dir / "bc_genesis_rollout_timeseries.npz"
    normalization_paths = {
        "observation": REPO_ROOT / "data/imitation/normalization/obs_stats.pkl",
        "command": REPO_ROOT / "data/imitation/normalization/command_stats.pkl",
        "action": REPO_ROOT / "data/imitation/normalization/action_stats.pkl",
    }
    if args.teacher_checkpoint is None:
        provenance_path = report_dir / "bc_training_provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            teacher_checkpoint = resolve_path(Path(provenance["teacher"]["checkpoint_path"]))
        else:
            teacher_checkpoint = REPO_ROOT / "models/expert_model_500.pt"
    else:
        teacher_checkpoint = resolve_path(args.teacher_checkpoint)

    for path in (checkpoint_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes_before = {
        "bc_checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "teacher_checkpoint": {"path": str(teacher_checkpoint), "sha256": sha256_file(teacher_checkpoint) if teacher_checkpoint.is_file() else None},
    }
    start_time = time.monotonic()
    audit = build_interface_audit(checkpoint_path, dataset_path, teacher_checkpoint, normalization_paths)
    write_interface_audit(interface_audit_path, audit)
    print(f"interface audit: {'PASS' if audit['valid'] else 'FAIL'}", flush=True)
    if args.audit_only:
        return 0 if audit["valid"] else 2
    if not audit["valid"]:
        hashes_after = {
            key: {**value, "sha256_after": value["sha256"], "unchanged": True}
            for key, value in hashes_before.items()
        }
        gate = evaluate_gate([], audit)
        artifacts = {"interface_audit": str(interface_audit_path), "rollout_report": str(rollout_report_path), "summary": str(summary_path), "episode_metrics": str(episode_metrics_path), "command_summary": str(command_summary_path), "plots": str(plot_dir), "videos": str(video_dir)}
        summary = {"schema": "go2_command_bc_genesis_rollout_v1", "generated_utc": utc_now(), "gate": gate, "audit": audit, "hashes": hashes_after, "counters": {"ppo_optimizer_updates": 0, "bc_optimizer_updates": 0, "teacher_checkpoint_changes": 0, "dataset_writes": 0, "bc_checkpoint_writes": 0}, "artifacts": artifacts}
        write_json(summary_path, summary)
        write_rollout_report(rollout_report_path, audit, [], [], gate, hashes_after, artifacts, [], time.monotonic() - start_time)
        return 2

    policy = FrozenBCPolicy(checkpoint_path)
    command_bank = make_command_bank()
    if args.command_name:
        selected = set(args.command_name)
        command_bank = [row for row in command_bank if row["name"] in selected]
        if not command_bank:
            raise ValueError(f"no command matches --command-name={sorted(selected)}")
    transitions = transition_bank()
    metrics: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    episode_id = 0
    scene = GenesisDeploymentScene(args.static_repetitions)
    standing_spec = [row for row in command_bank if row["category"] == "standing"]
    nonstanding_specs = [row for row in command_bank if row["category"] != "standing"]
    standing_metrics, standing_arrays, episode_id = run_static_bank(
        scene,
        policy,
        standing_spec,
        args.static_repetitions,
        args.episode_seconds,
        args.seed_base,
        episode_id,
    )
    metrics.extend(standing_metrics)
    arrays.extend(standing_arrays)
    standing_ok = bool(
        standing_metrics
        and all(
            row["full_duration"]
            and not row["fall_status"]
            and row["invalid_action_or_state_count"] == 0
            for row in standing_metrics
        )
    )
    if standing_ok and nonstanding_specs:
        static_metrics, static_arrays, episode_id = run_static_bank(
            scene,
            policy,
            nonstanding_specs,
            args.static_repetitions,
            args.episode_seconds,
            args.seed_base,
            episode_id,
        )
        metrics.extend(static_metrics)
        arrays.extend(static_arrays)
    elif not standing_ok:
        print("standing gate failed; movement bank and transitions not run", flush=True)

    if standing_ok and args.transition_episodes:
        if args.transition_episodes != len(transitions):
            transitions = transitions[: args.transition_episodes]
        transition_scene = GenesisDeploymentScene(1)
        transition_metrics, transition_arrays, episode_id = run_transition_bank(
            transition_scene,
            policy,
            transitions,
            args.episode_seconds,
            args.seed_base + 500000,
            episode_id,
        )
        metrics.extend(transition_metrics)
        arrays.extend(transition_arrays)

    write_csv(episode_metrics_path, metrics)
    command_summary = aggregate_command_summary(metrics)
    write_csv(command_summary_path, command_summary)
    save_timeseries(timeseries_path, metrics, arrays)
    plot_results(plot_dir, metrics, arrays)
    gate = evaluate_gate(metrics, audit)

    video_rows: list[dict[str, Any]] = []
    if not args.skip_videos and not args.command_name:
        video_rows = render_representative_videos(
            policy,
            make_command_bank(),
            transition_bank(),
            args.episode_seconds,
            args.seed_base,
            video_dir,
            standing_only=not standing_ok,
        )

    hashes_after: dict[str, Any] = {}
    for key, before in hashes_before.items():
        after = sha256_file(Path(before["path"])) if before["path"] and Path(before["path"]).is_file() else None
        hashes_after[key] = {
            **before,
            "sha256_after": after,
            "unchanged": before["sha256"] == after,
        }
    artifacts: dict[str, Any] = {
        "evaluator": str(Path(__file__).resolve()),
        "interface_audit": str(interface_audit_path),
        "rollout_report": str(rollout_report_path),
        "summary": str(summary_path),
        "episode_metrics": str(episode_metrics_path),
        "command_summary": str(command_summary_path),
        "timeseries": str(timeseries_path),
        "plots": str(plot_dir),
        "videos": str(video_dir),
    }
    summary = {
        "schema": "go2_command_bc_genesis_rollout_v1",
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now(),
        "git_commit": git_commit(),
        "host": {"platform": platform.platform(), "python": sys.version, "python_executable": sys.executable},
        "runtime": {"genesis_version": audit["genesis"]["version"], "genesis_python": audit["genesis"]["python"], "genesis_urdf": audit["genesis"]["urdf"], "genesis_urdf_sha256": audit["genesis"]["urdf_sha256"]},
        "interface": audit,
        "evaluation": {"episode_seconds": args.episode_seconds, "static_repetitions_per_command": args.static_repetitions, "transition_episodes": args.transition_episodes, "seed_base": args.seed_base, "static_commands": [{"name": row["name"], "category": row["category"], "command": row["command"]} for row in make_command_bank()], "command_substitutions": "All tested components are within dataset limits. +0.40 vx -> +0.30; -0.20/-0.40 vx -> -0.05/-0.10; +/-0.60 yaw -> +/-0.50; +/-0.15 lateral -> +/-0.10 or +/-0.20; +/-0.25 vx -> +0.30/-0.10.", "transition_segment_seconds": 2.0},
        "episodes": len(metrics),
        "gate": gate,
        "hashes": hashes_after,
        "counters": {"ppo_optimizer_updates": 0, "bc_optimizer_updates": 0, "teacher_checkpoint_changes": 0, "dataset_writes": 0, "bc_checkpoint_writes": 0},
        "videos": video_rows,
        "artifacts": artifacts,
    }
    write_json(summary_path, summary)
    write_rollout_report(rollout_report_path, audit, metrics, command_summary, gate, hashes_after, artifacts, video_rows, time.monotonic() - start_time)
    print(f"FINAL DECISION: {gate['decision']}", flush=True)
    print(f"episodes: {len(metrics)}; full_duration_survival: {gate['overall_full_duration_survival_rate']:.3%}", flush=True)
    return 0 if gate["decision"].startswith("BC GENESIS GATE PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
