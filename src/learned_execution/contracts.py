"""Frozen interface contract for the Genesis-native walking branch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Genesis' bundled Go2 URDF uses this exact named order.  Keep the small
# frozen mapping local to the release rather than importing the omitted
# historical reference-backend package.
CANONICAL_JOINTS = (
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

STATE_DIM = 45
COMMAND_DIM = 3
ACTION_DIM = 12
POLICY_OBSERVATION_DIM = STATE_DIM + COMMAND_DIM
OBSERVATION_SCHEMA_VERSION = "genesis-unified-backward-observation-v2"
ACTION_SCHEMA_VERSION = "genesis-unified-backward-raw-action-v2"
COMMAND_SCHEMA_VERSION = "genesis-body-command-v2"
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = int(round(CONTROL_DT / PHYSICS_DT))

COMMAND_NAMES = ("vx", "vy", "yaw_rate")
COMMAND_FRAME = "body"
COMMAND_UNITS = ("m/s", "m/s", "rad/s")
COMMAND_LIMITS = np.asarray([[-0.3, 0.3], [-0.2, 0.2], [-0.5, 0.5]], dtype=np.float32)
UNIFIED_BACKWARD_COMMAND_LIMITS = np.asarray(
    [[-0.1108709499, -0.0279288553], [0.0, 0.0], [0.0, 0.0]],
    dtype=np.float32,
)

# Genesis' bundled Go2 URDF uses this exact named order.  No index-based
# order from the historical MuJoCo branch is imported into this contract.
GENESIS_JOINT_NAMES = tuple(CANONICAL_JOINTS)
GENESIS_FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
GENESIS_BASE_LINK_NAME = "base"
GENESIS_URDF = "urdf/go2/urdf/go2.urdf"
GENESIS_URDF_SHA256 = "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4"

# This is the observable student reference pose, in canonical Genesis order.
# It is deliberately not the action decoder pose.  The compatibility name is
# retained for old reset/audit callers; new unified-teacher code must use the
# explicit names below so the two coordinate systems cannot be conflated.
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
RESET_JOINT_POSITION = STUDENT_JOINT_REFERENCE.copy()
DEFAULT_JOINT_POSITION = RESET_JOINT_POSITION.copy()

# Source-of-truth action decoder pose.  This is in canonical Genesis order,
# not the historical policy/MuJoCo order.  Neural output is intentionally
# untouched here; the physical joint-limit layer lives at the environment/PD
# boundary.
Q_DEFAULT_CANONICAL = np.asarray(
    [
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
    ],
    dtype=np.float32,
)

# The bounded teacher/student raw action is a dimensionless 12-D offset.  The
# decoder below does not apply a neural-output clamp or activation.
ACTION_SCALE = np.full(ACTION_DIM, 0.5, dtype=np.float32)
KP = np.tile(np.asarray([20.0, 20.0, 40.0], dtype=np.float32), 4)
KD = np.tile(np.asarray([1.0, 1.0, 2.0], dtype=np.float32), 4)
EFFORT_LIMIT = np.tile(np.asarray([23.5, 23.5, 45.43], dtype=np.float32), 4)

# These are contract metadata, not a second policy or a mode selector.  The
# same actor always receives this state layout and emits this target layout.
IDENTITY_NORMALIZATION = {
    "kind": "identity",
    "observation": {
        "mean": [0.0] * POLICY_OBSERVATION_DIM,
        "std": [1.0] * POLICY_OBSERVATION_DIM,
        "clip_min": None,
        "clip_max": None,
    },
}

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
    *(f"joint_pos_rel_{name.removesuffix('_joint')}" for name in GENESIS_JOINT_NAMES),
    *(f"joint_vel_rel_{name.removesuffix('_joint')}" for name in GENESIS_JOINT_NAMES),
    *(f"previous_raw_action_{name.removesuffix('_joint')}" for name in GENESIS_JOINT_NAMES),
)
POLICY_FEATURE_NAMES = STATE_FEATURE_NAMES + tuple(f"command_{name}" for name in COMMAND_NAMES)

if len(STATE_FEATURE_NAMES) != STATE_DIM:
    raise RuntimeError(f"Genesis walking state contract has {len(STATE_FEATURE_NAMES)} fields")
if len(POLICY_FEATURE_NAMES) != POLICY_OBSERVATION_DIM:
    raise RuntimeError(f"Genesis walking policy contract has {len(POLICY_FEATURE_NAMES)} fields")


def validate_command(
    command: np.ndarray | Sequence[float], *, allow_batch: bool = True
) -> np.ndarray:
    """Validate and return a body-frame ``[vx, vy, yaw_rate]`` command."""

    array = np.asarray(command, dtype=np.float32)
    if array.shape[-1:] != (COMMAND_DIM,):
        raise ValueError(f"command must end in shape (3,), got {array.shape}")
    if not allow_batch and array.ndim != 1:
        raise ValueError(f"command must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("command contains NaN or inf")
    lower = COMMAND_LIMITS[:, 0]
    upper = COMMAND_LIMITS[:, 1]
    if np.any(array < lower) or np.any(array > upper):
        raise ValueError(
            f"command outside Genesis limits {COMMAND_LIMITS.tolist()}: {array.tolist()}"
        )
    return array


def make_policy_input(state: np.ndarray, command: np.ndarray) -> np.ndarray:
    """Append the command to the complete Genesis state observation."""

    state_array = np.asarray(state, dtype=np.float32)
    command_array = validate_command(command)
    if state_array.shape[:-1] != command_array.shape[:-1]:
        raise ValueError(
            f"state and command leading shapes differ: {state_array.shape} vs {command_array.shape}"
        )
    if state_array.shape[-1] != STATE_DIM:
        raise ValueError(f"state must end in ({STATE_DIM},), got {state_array.shape}")
    return np.concatenate((state_array, command_array), axis=-1).astype(np.float32)


def raw_action_to_target(raw_action: np.ndarray) -> np.ndarray:
    """Decode raw policy offsets into an unclipped physical q target."""

    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape[-1] != ACTION_DIM:
        raise ValueError(f"raw action must end in ({ACTION_DIM},), got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw action contains NaN or inf")
    return Q_DEFAULT_CANONICAL + ACTION_SCALE * raw


def action_target_to_raw(target: np.ndarray) -> np.ndarray:
    """Invert the decoder for diagnostics; never use this for previous action."""

    value = np.asarray(target, dtype=np.float32)
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"target must end in ({ACTION_DIM},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("target contains NaN or inf")
    return (value - Q_DEFAULT_CANONICAL) / ACTION_SCALE


def contract_payload() -> dict[str, object]:
    """Return the serializable policy/environment interface contract."""

    return {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "command_schema_version": COMMAND_SCHEMA_VERSION,
        "state_dim": STATE_DIM,
        "policy_observation_dim": POLICY_OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "command_dim": COMMAND_DIM,
        "state_features": list(STATE_FEATURE_NAMES),
        "policy_features": list(POLICY_FEATURE_NAMES),
        "joint_order": list(GENESIS_JOINT_NAMES),
        "command_order": list(COMMAND_NAMES),
        "command_frame": COMMAND_FRAME,
        "command_units": list(COMMAND_UNITS),
        "command_limits": COMMAND_LIMITS.tolist(),
        "unified_backward_command_limits": UNIFIED_BACKWARD_COMMAND_LIMITS.tolist(),
        "student_joint_reference": STUDENT_JOINT_REFERENCE.tolist(),
        "q_default_canonical": Q_DEFAULT_CANONICAL.tolist(),
        "action_scale": ACTION_SCALE.tolist(),
        "action_output_semantics": "raw_action_12; no tanh, clamp, or output normalization in student",
        "decoder_semantics": "q_target_raw = Q_DEFAULT_CANONICAL + 0.50 * raw_action",
        "physical_safety_layer": "joint-limit clip q_target_raw to q_target_applied before PD",
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "effort_limit": EFFORT_LIMIT.tolist(),
        "control_dt_s": CONTROL_DT,
        "physics_dt_s": PHYSICS_DT,
        "decimation": DECIMATION,
        "previous_action_semantics": "previous_raw_action[12]; reset zeros; store raw action after inference",
    }


def identity_normalization_payload() -> dict[str, object]:
    """Return a fresh copy of the identity normalization state."""

    return {
        "kind": IDENTITY_NORMALIZATION["kind"],
        "observation": {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in IDENTITY_NORMALIZATION["observation"].items()
        },
    }


def command_catalog() -> list[dict[str, object]]:
    """Return the required phase-C command bank plus mixed commands."""

    values: list[tuple[str, tuple[float, float, float]]] = [
        ("standing", (0.0, 0.0, 0.0)),
        ("forward_005", (0.05, 0.0, 0.0)),
        ("forward_010", (0.10, 0.0, 0.0)),
        ("forward_020", (0.20, 0.0, 0.0)),
        ("forward_030", (0.30, 0.0, 0.0)),
        ("backward_005", (-0.05, 0.0, 0.0)),
        ("backward_010", (-0.10, 0.0, 0.0)),
        ("lateral_right_010", (0.0, -0.10, 0.0)),
        ("lateral_left_005", (0.0, 0.05, 0.0)),
        ("lateral_right_005", (0.0, -0.05, 0.0)),
        ("lateral_left_010", (0.0, 0.10, 0.0)),
        ("yaw_right_030", (0.0, 0.0, -0.30)),
        ("yaw_left_030", (0.0, 0.0, 0.30)),
        ("yaw_right_050", (0.0, 0.0, -0.50)),
        ("yaw_left_050", (0.0, 0.0, 0.50)),
        ("forward_left", (0.20, 0.10, 0.0)),
        ("forward_right", (0.20, -0.10, 0.0)),
        ("forward_yaw_left", (0.20, 0.0, 0.30)),
        ("forward_yaw_right", (0.20, 0.0, -0.30)),
    ]
    return [
        {"command_id": index, "name": name, "command": list(command)}
        for index, (name, command) in enumerate(values)
    ]
