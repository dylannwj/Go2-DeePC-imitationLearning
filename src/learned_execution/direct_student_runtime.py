"""Direct runtime for the frozen observable student.

This module deliberately has no command guard, projection, dead zone, rate
limiter, mode selector, expert fallback, or navigation rule.  A finite DeePC
command is appended to the measured 45-D observable state exactly as the
frozen checkpoint expects.  Joint-target and torque clipping are physical
Genesis protections and are reported rather than fed back as a navigation
command.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from .linear_bc_common import (  # noqa: E402
    ACTION_DIM,
    ACTION_SCALE,
    EFFORT_LIMIT,
    INPUT_DIM,
    KD,
    KP,
    POLICY_TO_SCENE,
    Q_DEFAULT_CANONICAL,
    STATE_DIM,
    STUDENT_JOINT_REFERENCE,
)

CHECKPOINT_PATH = REPO_ROOT / "models/go2_observable_student_bc_linear_v1.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return array


class FrozenObservableStudent:
    """Read-only loader for exactly ``48 -> 256 ELU -> 256 ELU -> 12``."""

    def __init__(self, checkpoint_path: str | Path = CHECKPOINT_PATH, *, verify_hash: bool = True):
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        observed = sha256_file(self.checkpoint_path)
        if verify_hash and observed != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(f"frozen student hash mismatch: {observed} != {EXPECTED_CHECKPOINT_SHA256}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("student checkpoint is not a mapping")
        expected = {
            "schema": "observable_student_bc_linear_v1",
            "architecture": [48, 256, 256, 12],
            "hidden_activation": "ELU",
            "output_activation": "linear",
            "input_dimension": INPUT_DIM,
            "state_dimension": STATE_DIM,
            "command_dimension": 3,
            "action_dimension": ACTION_DIM,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r}, expected {value!r}")
        model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, ACTION_DIM),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.sha256 = observed
        self.checkpoint = checkpoint

    def __call__(self, student_input: np.ndarray) -> np.ndarray:
        value = np.asarray(student_input, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != INPUT_DIM or not np.all(np.isfinite(value)):
            raise ValueError(f"student input must be finite with shape [batch,{INPUT_DIM}], got {value.shape}")
        with self.torch.no_grad():
            output = self.model(self.torch.from_numpy(value))
        action = output.detach().cpu().numpy().astype(np.float32)
        if action.shape != (len(value), ACTION_DIM) or not np.all(np.isfinite(action)):
            raise FloatingPointError("frozen student produced an invalid raw action")
        return action


def _canonical_scene(value: np.ndarray) -> np.ndarray:
    """Convert the deployment scene's policy view to canonical Genesis order."""

    return np.asarray(value, dtype=np.float32)[:, POLICY_TO_SCENE]


def build_student_state(snapshot: dict[str, np.ndarray], previous_raw_action: np.ndarray) -> np.ndarray:
    """Build the frozen 45-D observable state from current Genesis data."""

    previous = _finite(previous_raw_action, (len(previous_raw_action), ACTION_DIM), "previous_raw_action")
    batch = len(previous)
    gravity = _finite(snapshot["state"][:, 3:6], (batch, 3), "projected_gravity")
    linear = _finite(snapshot["body_linear_velocity"], (batch, 3), "body_linear_velocity")
    angular = _finite(snapshot["body_angular_velocity"], (batch, 3), "body_angular_velocity")
    position = _finite(_canonical_scene(snapshot["joint_position"]), (batch, ACTION_DIM), "joint_position")
    velocity = _finite(_canonical_scene(snapshot["joint_velocity"]), (batch, ACTION_DIM), "joint_velocity")
    state = np.concatenate(
        (
            gravity,
            linear,
            angular,
            position - STUDENT_JOINT_REFERENCE[None, :],
            velocity,
            previous,
        ),
        axis=1,
    ).astype(np.float32)
    if state.shape != (batch, STATE_DIM) or not np.all(np.isfinite(state)):
        raise FloatingPointError(f"frozen observable state contract failed: {state.shape}")
    return state


@dataclass(frozen=True)
class DirectStep:
    """All telemetry from one 50 Hz direct student action."""

    requested_command: np.ndarray
    student_received_command: np.ndarray
    student_state: np.ndarray
    student_input: np.ndarray
    raw_action: np.ndarray
    q_target_raw: np.ndarray
    q_target_applied: np.ndarray
    torque_raw: np.ndarray
    torque_applied: np.ndarray
    before: dict[str, np.ndarray]
    after: dict[str, np.ndarray]
    safety: dict[str, np.ndarray]


def _safety(
    scene: Any,
    snapshot: dict[str, np.ndarray],
    raw_action: np.ndarray,
    q_target_raw: np.ndarray,
    q_target_applied: np.ndarray,
    torque_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    q = _canonical_scene(snapshot["joint_position"])
    dq = _canonical_scene(snapshot["joint_velocity"])
    lower = np.asarray(scene.joint_lower_scene, dtype=np.float32)[None, :]
    upper = np.asarray(scene.joint_upper_scene, dtype=np.float32)[None, :]
    finite = (
        np.isfinite(raw_action).all(axis=1)
        & np.isfinite(q_target_raw).all(axis=1)
        & np.isfinite(q_target_applied).all(axis=1)
        & np.isfinite(torque_raw).all(axis=1)
        & np.isfinite(q).all(axis=1)
        & np.isfinite(dq).all(axis=1)
        & np.isfinite(snapshot["root_position"]).all(axis=1)
        & np.isfinite(snapshot["root_rpy"]).all(axis=1)
    )
    base_contact = np.asarray(snapshot["base_contact"], dtype=bool)
    fall = (
        (snapshot["body_height"] < 0.12)
        | (np.max(np.abs(snapshot["root_rpy"][:, :2]), axis=1) > 1.0)
        | base_contact
    )
    return {
        "fall": fall,
        "base_contact": base_contact,
        "nan_inf": ~finite,
        "actual_joint_limit_violation": np.any((q < lower - 1.0e-6) | (q > upper + 1.0e-6), axis=1),
        "torque_limit_exceedance": np.any(np.abs(torque_raw) > EFFORT_LIMIT[None, :] + 1.0e-6, axis=1),
        "q_target_raw_outside_joint_limits": np.any((q_target_raw < lower) | (q_target_raw > upper), axis=1),
        "q_target_clip_flag": np.any(np.abs(q_target_applied - q_target_raw) > 1.0e-7, axis=1),
    }


class DirectStudentBatch:
    """Vectorized direct student + fixed-PD Genesis adapter."""

    def __init__(self, scene: Any, student: FrozenObservableStudent | None = None):
        self.scene = scene
        self.student = student or FrozenObservableStudent()
        self.num_envs = int(scene.num_envs)
        self.previous_raw_action = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float32)

    def reset(self) -> None:
        self.previous_raw_action.fill(0.0)

    def observe(self) -> dict[str, np.ndarray]:
        return self.scene.observe(self.previous_raw_action[:, POLICY_TO_SCENE])

    def step(self, command: np.ndarray) -> DirectStep:
        requested = np.asarray(command, dtype=np.float32)
        if requested.shape != (self.num_envs, 3) or not np.all(np.isfinite(requested)):
            raise ValueError(f"direct command must be finite with shape {(self.num_envs, 3)}, got {requested.shape}")
        before = self.observe()
        state = build_student_state(before, self.previous_raw_action)
        # This is the architectural boundary: the exact DeePC command is the
        # exact command appended to the frozen state.  There is no alternate
        # command variable and no transformation between these two arrays.
        student_received = requested.copy()
        student_input = np.concatenate((state, student_received), axis=1).astype(np.float32)
        raw_action = self.student(student_input)
        q_target_raw = Q_DEFAULT_CANONICAL[None, :] + ACTION_SCALE[None, :] * raw_action
        q_target_applied = np.clip(
            q_target_raw,
            np.asarray(self.scene.joint_lower_scene, dtype=np.float32)[None, :],
            np.asarray(self.scene.joint_upper_scene, dtype=np.float32)[None, :],
        ).astype(np.float32)
        q = _canonical_scene(before["joint_position"])
        dq = _canonical_scene(before["joint_velocity"])
        torque_raw = KP[None, :] * (q_target_applied - q) - KD[None, :] * dq
        torque_applied = np.clip(torque_raw, -EFFORT_LIMIT[None, :], EFFORT_LIMIT[None, :]).astype(np.float32)
        self.scene.step(q_target_applied[:, POLICY_TO_SCENE])
        self.previous_raw_action = raw_action.copy()
        after = self.observe()
        safety = _safety(self.scene, after, raw_action, q_target_raw, q_target_applied, torque_raw)
        return DirectStep(
            requested_command=requested.copy(),
            student_received_command=student_received.copy(),
            student_state=state.copy(),
            student_input=student_input.copy(),
            raw_action=raw_action.copy(),
            q_target_raw=q_target_raw.copy(),
            q_target_applied=q_target_applied.copy(),
            torque_raw=torque_raw.copy(),
            torque_applied=torque_applied.copy(),
            before=before,
            after=after,
            safety=safety,
        )


def make_genesis_scene(num_envs: int, *, camera: bool = False, camera_res: tuple[int, int] = (640, 480)) -> Any:
    """Construct the repository's native Genesis 1.3.1 Go2 scene."""

    cache_root = Path(tempfile.gettempdir()) / "deepc_thesis_cache"
    (cache_root / "genesis").mkdir(parents=True, exist_ok=True)
    (cache_root / "quadrants").mkdir(parents=True, exist_ok=True)
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(cache_root / "genesis"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    try:
        import quadrants as qd

        if not getattr(qd.init, "_final_deepc_student_patched", False):
            original_init = qd.init

            def patched_init(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("offline_cache_file_path", str(cache_root / "quadrants"))
                return original_init(*args, **kwargs)

            patched_init._final_deepc_student_patched = True  # type: ignore[attr-defined]
            qd.init = patched_init
    except ImportError:
        pass
    from .genesis_deployment_scene import GenesisDeploymentScene

    return GenesisDeploymentScene(num_envs, camera=camera, camera_res=camera_res)


def pose_from_snapshot(snapshot: dict[str, np.ndarray]) -> np.ndarray:
    """Return measured world `[x,y,yaw]` for one or more environments."""

    position = np.asarray(snapshot["root_position"], dtype=np.float64)
    yaw = np.asarray(snapshot["root_rpy"], dtype=np.float64)[:, 2:3]
    return np.concatenate((position[:, :2], yaw), axis=1)


def body_increment(previous_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    """Measured local `[forward,left,yaw]` increment between Genesis poses."""

    previous = np.asarray(previous_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    delta = current[:, :2] - previous[:, :2]
    yaw0 = previous[:, 2]
    dyaw = np.arctan2(np.sin(current[:, 2] - yaw0), np.cos(current[:, 2] - yaw0))
    return np.column_stack(
        (
            np.cos(yaw0) * delta[:, 0] + np.sin(yaw0) * delta[:, 1],
            -np.sin(yaw0) * delta[:, 0] + np.cos(yaw0) * delta[:, 1],
            dyaw,
        )
    ).astype(np.float32)


def direct_interface_audit() -> dict[str, Any]:
    """Return a machine-readable audit of the immutable direct interface."""

    return {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "architecture": [48, 256, 256, 12],
        "input_layout": "measured_observable_state[45] + deePC_command[3]",
        "command_names": ["vx", "vy", "yaw_rate"],
        "direct_boundary": "requested_command == student_received_command",
        "navigation_transformations": [],
        "safety_transformations": ["Genesis joint-target limits", "fixed-PD torque limits"],
        "student_action_decoder": "q_default + 0.5 * raw_action",
        "student_observation_is_measured": True,
        "expert_fallback": False,
        "policy_switching": False,
    }
