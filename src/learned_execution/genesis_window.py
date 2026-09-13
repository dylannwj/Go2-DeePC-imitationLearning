"""Native Genesis 1.3.1 viewer scene used by the live product.

This is a viewer-enabled presentation variant of the already validated
Genesis deployment scene.  It keeps the same URDF, joint order, timing,
reset, observation, and fixed-PD path; it does not add a second controller.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from .genesis_deployment_scene import (  # noqa: E402
    ACTION_DIM,
    CONTACT_FORCE_TERMINATION_N,
    DECIMATION,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT,
    GENESIS_BASE_LINK_NAME,
    GENESIS_FOOT_NAMES,
    GENESIS_JOINT_NAMES,
    GENESIS_ROOT_HEIGHT,
    GENESIS_URDF_RELATIVE,
    KD,
    KP,
    PHYSICS_DT,
    POLICY_TO_SCENE,
    numpy_value,
    quat_inverse_rotate,
    quat_to_rpy,
)


class GenesisViewerScene:
    """Viewer-enabled scene implementing the final runtime's scene contract."""

    def __init__(
        self,
        num_envs: int = 1,
        *,
        viewer_res: tuple[int, int] = (880, 760),
        seed: int = 20260804,
    ) -> None:
        import genesis as gs
        import torch

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

        self.gs = gs
        self.torch = torch
        if not getattr(gs, "_initialized", False):
            gs.init(seed=seed, backend=gs.cpu, logging_level="warning")
        self.num_envs = int(num_envs)
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=viewer_res,
                run_in_thread=True,
                refresh_rate=60,
                realtime_factor=1.0,
                camera_pos=(1.8, -1.8, 1.15),
                camera_lookat=(0.0, 0.0, 0.18),
                camera_fov=42,
                enable_help_text=True,
                enable_default_keybinds=True,
            ),
            sim_options=gs.options.SimOptions(dt=PHYSICS_DT, substeps=1),
            rigid_options=gs.options.RigidOptions(
                dt=PHYSICS_DT,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=True,
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
        self.scene.build(n_envs=self.num_envs)
        self.motor_dofs = [
            int(self.robot.get_joint(name).dofs_idx_local[0]) for name in GENESIS_JOINT_NAMES
        ]
        if len(self.motor_dofs) != ACTION_DIM or len(set(self.motor_dofs)) != ACTION_DIM:
            raise ValueError(f"Genesis motor DOFs are not unique: {self.motor_dofs}")
        self.foot_link_indices = [
            int(self.robot.get_link(name).idx - self.robot.link_start) for name in GENESIS_FOOT_NAMES
        ]
        self.base_link_index = int(self.robot.get_link(GENESIS_BASE_LINK_NAME).idx - self.robot.link_start)
        lower, upper = self.robot.get_dofs_limit(self.motor_dofs)
        self.joint_lower_scene = numpy_value(lower).astype(np.float32).reshape(-1)
        self.joint_upper_scene = numpy_value(upper).astype(np.float32).reshape(-1)
        if self.joint_lower_scene.shape != (ACTION_DIM,) or self.joint_upper_scene.shape != (ACTION_DIM,):
            raise ValueError("Genesis joint limits did not resolve to twelve actuated DOFs")

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
        joints = np.repeat(DEFAULT_JOINT_POS[None, :], self.num_envs, axis=0)
        self.robot.set_pos(self.torch.from_numpy(positions), zero_velocity=True)
        self.robot.set_quat(self.torch.from_numpy(quaternions), zero_velocity=True)
        self.robot.set_dofs_position(
            position=self.torch.from_numpy(joints),
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        position = numpy_value(self.robot.get_dofs_position(self.motor_dofs)).astype(np.float32)
        velocity = numpy_value(self.robot.get_dofs_velocity(self.motor_dofs)).astype(np.float32)
        return position.reshape(self.num_envs, ACTION_DIM), velocity.reshape(self.num_envs, ACTION_DIM)

    def observe(self, last_action: np.ndarray) -> dict[str, np.ndarray]:
        last_action = np.asarray(last_action, dtype=np.float32).reshape(self.num_envs, ACTION_DIM)
        position, velocity = self._joint_state()
        root_position = numpy_value(self.robot.get_pos()).astype(np.float32).reshape(self.num_envs, 3)
        root_quaternion = numpy_value(self.robot.get_quat()).astype(np.float32).reshape(self.num_envs, 4)
        world_linear = numpy_value(self.robot.get_vel()).astype(np.float32).reshape(self.num_envs, 3)
        world_angular = numpy_value(self.robot.get_ang()).astype(np.float32).reshape(self.num_envs, 3)
        body_linear = quat_inverse_rotate(root_quaternion, world_linear).astype(np.float32)
        body_angular = quat_inverse_rotate(root_quaternion, world_angular).astype(np.float32)
        projected_gravity = quat_inverse_rotate(
            root_quaternion,
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (self.num_envs, 3)),
        ).astype(np.float32)
        root_rpy = quat_to_rpy(root_quaternion).astype(np.float32)
        state = np.concatenate(
            (
                body_angular,
                projected_gravity,
                position[:, POLICY_TO_SCENE] - DEFAULT_JOINT_POS[None, :],
                velocity[:, POLICY_TO_SCENE],
                last_action,
            ),
            axis=1,
        ).astype(np.float32)
        contact_force = numpy_value(self.robot.get_links_net_contact_force()).astype(np.float32)
        contact_force = contact_force.reshape(self.num_envs, -1, 3)
        foot_contact = np.linalg.norm(contact_force[:, self.foot_link_indices, :], axis=-1) > CONTACT_FORCE_TERMINATION_N
        base_contact = np.linalg.norm(contact_force[:, self.base_link_index, :], axis=-1) > CONTACT_FORCE_TERMINATION_N
        position_policy = position[:, POLICY_TO_SCENE]
        lower_policy = self.joint_lower_scene[POLICY_TO_SCENE]
        upper_policy = self.joint_upper_scene[POLICY_TO_SCENE]
        return {
            "state": state,
            "root_position": root_position,
            "root_quaternion": root_quaternion,
            "root_rpy": root_rpy,
            "world_linear_velocity": world_linear,
            "world_angular_velocity": world_angular,
            "body_linear_velocity": body_linear,
            "body_angular_velocity": body_angular,
            "joint_position": position_policy.astype(np.float32),
            "joint_velocity": velocity[:, POLICY_TO_SCENE].astype(np.float32),
            "foot_contacts": foot_contact.astype(bool),
            "base_contact": base_contact.astype(bool),
            "body_height": root_position[:, 2].copy(),
            "joint_limit_margin": np.minimum(position_policy - lower_policy[None, :], upper_policy[None, :] - position_policy),
        }

    def step(self, target_policy: np.ndarray) -> np.ndarray:
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(self.num_envs, ACTION_DIM)
        current_position, current_velocity = self._joint_state()
        current_policy = current_position[:, POLICY_TO_SCENE]
        current_velocity_policy = current_velocity[:, POLICY_TO_SCENE]
        applied = np.clip(target_policy, self.joint_lower_scene[POLICY_TO_SCENE], self.joint_upper_scene[POLICY_TO_SCENE])
        torque_policy = KP[None, :] * (applied - current_policy) - KD[None, :] * current_velocity_policy
        torque_policy = np.clip(torque_policy, -EFFORT_LIMIT[None, :], EFFORT_LIMIT[None, :]).astype(np.float32)
        self.robot.control_dofs_force(self.torch.from_numpy(torque_policy[:, POLICY_TO_SCENE]), self.motor_dofs)
        for _ in range(DECIMATION):
            self.scene.step()
        return torque_policy

    def safe_hold(self) -> None:
        """Apply one fixed-PD damping step after navigation has terminated."""

        current_position, current_velocity = self._joint_state()
        current_velocity_policy = current_velocity[:, POLICY_TO_SCENE]
        torque_policy = np.clip(-KD[None, :] * current_velocity_policy, -EFFORT_LIMIT[None, :], EFFORT_LIMIT[None, :]).astype(np.float32)
        self.robot.control_dofs_force(self.torch.from_numpy(torque_policy[:, POLICY_TO_SCENE]), self.motor_dofs)
        for _ in range(DECIMATION):
            self.scene.step()

    def viewer_alive(self) -> bool:
        viewer = getattr(self.scene, "viewer", None)
        return bool(viewer is not None and viewer.is_alive())

    def close(self) -> None:
        viewer = getattr(self.scene, "viewer", None)
        if viewer is not None:
            close = getattr(viewer, "close", None)
            if callable(close):
                close()
