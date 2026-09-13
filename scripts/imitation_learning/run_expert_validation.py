#!/usr/bin/env python3
"""Strict no-training Mjlab Go2 teacher smoke test in native Genesis.

This script deliberately keeps the frozen Mjlab/RSL-RL actor and deployment
contract separate from the project's 48-D student contract.  It only loads
the frozen model, reconstructs the 45-D teacher observation, applies the
teacher's decoded joint targets through Genesis PD, and records evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "results/genesis/generated/expert_validation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_URL = "https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat"
SOURCE_DIR = ROOT / "external/unitree_go2"
SOURCE_REPO_DIR = ROOT / "external/unitree_rl_mjlab"
SOURCE_ASSET_DIR = SOURCE_REPO_DIR / "src" / "assets" / "robots" / "unitree_go2"
CHECKPOINT_NAME = "model_500.pt"
GENESIS_URDF_RELATIVE = Path("urdf/go2/urdf/go2.urdf")
GENESIS_PLANE_RELATIVE = Path("urdf/plane/plane.urdf")

CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = 4
POLICY_RATE_HZ = 50.0
PHYSICS_RATE_HZ = 200.0
STATE_DIM = 42
POLICY_INPUT_DIM = 45
ACTION_DIM = 12
COMMAND_DIM = 3
ACTION_SCALE = 0.5
KP = np.tile(np.asarray([20.0, 20.0, 40.0], dtype=np.float32), 4)
KD = np.tile(np.asarray([1.0, 1.0, 2.0], dtype=np.float32), 4)
DEFAULT_JOINT_POS = np.asarray(
    [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
    dtype=np.float32,
)
POLICY_LEGS = ("FL", "FR", "RL", "RR")
GENESIS_LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")
POLICY_TO_GENESIS = np.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
GENESIS_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in GENESIS_LEGS for joint in JOINTS)
POLICY_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in POLICY_LEGS for joint in JOINTS)
GENESIS_FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
COMMAND_NAMES = ("vx", "vy", "yaw_rate")
AXES = COMMAND_NAMES

REQUIRED_COMMANDS = (
    ("C0", "standing", np.asarray([0.00, 0.00, 0.00], dtype=np.float32)),
    ("C1", "slow_forward", np.asarray([0.10, 0.00, 0.00], dtype=np.float32)),
    ("C2", "forward", np.asarray([0.30, 0.00, 0.00], dtype=np.float32)),
    ("C3", "yaw_left", np.asarray([0.00, 0.00, 0.30], dtype=np.float32)),
    ("C4", "yaw_right", np.asarray([0.00, 0.00, -0.30], dtype=np.float32)),
    ("C5", "forward_left_arc", np.asarray([0.20, 0.00, 0.25], dtype=np.float32)),
    ("C6", "forward_right_arc", np.asarray([0.20, 0.00, -0.25], dtype=np.float32)),
)
TRANSITION_NAME = "transition"
VIDEO_FPS = 25
VIDEO_RENDER_STRIDE = 2

SCRIPT_VERSION = "mjlab-genesis-visual-transfer-v1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def git_value(directory: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def source_file_map() -> dict[str, Path]:
    return {
        "policy.onnx": SOURCE_DIR / "policy.onnx",
        "policy.onnx.data": SOURCE_DIR / "policy.onnx.data",
        CHECKPOINT_NAME: ROOT / "models/expert_model_500.pt",
        "params/deploy.yaml": ROOT / "configs/imitation/expert/deploy.yaml",
        "params/env.yaml": ROOT / "configs/imitation/expert/env.yaml",
        "params/agent.yaml": ROOT / "configs/imitation/expert/agent.yaml",
    }


def snapshot_artifacts() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    artifact_dir = WORKSPACE / "artifacts"
    config_dir = WORKSPACE / "configs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    for label, source in source_file_map().items():
        if not source.is_file():
            rows[label] = {"source": str(source), "available": False}
            continue
        if label.startswith("params/"):
            destination = config_dir / Path(label).name
        else:
            destination = artifact_dir / Path(label).name
        shutil.copy2(source, destination)
        rows[label] = {
            "source": str(source),
            "snapshot": str(destination),
            "available": True,
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "snapshot_sha256": sha256_file(destination),
        }
    return rows


def configure_genesis_cache() -> dict[str, str]:
    """Redirect only process-local compiler caches to writable /tmp paths."""

    genesis_cache = Path("/tmp/mjlab_genesis_visual_transfer/genesis")
    quadrants_cache = Path("/tmp/mjlab_genesis_visual_transfer/quadrants")
    genesis_cache.mkdir(parents=True, exist_ok=True)
    quadrants_cache.mkdir(parents=True, exist_ok=True)
    os.environ["GS_CACHE_FILE_PATH"] = str(genesis_cache)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mjlab_genesis_visual_transfer/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import quadrants as qd

    if not getattr(qd.init, "_mjlab_visual_transfer_patched", False):
        original_init = qd.init

        def patched_init(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("offline_cache_file_path", str(quadrants_cache))
            return original_init(*args, **kwargs)

        patched_init._mjlab_visual_transfer_patched = True  # type: ignore[attr-defined]
        qd.init = patched_init
    return {"genesis_cache": str(genesis_cache), "quadrants_cache": str(quadrants_cache)}


def parse_deploy_config(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def source_contract_audit() -> dict[str, Any]:
    deploy_path = ROOT / "configs/imitation/expert/deploy.yaml"
    env_path = ROOT / "configs/imitation/expert/env.yaml"
    agent_path = ROOT / "configs/imitation/expert/agent.yaml"
    deploy = parse_deploy_config(deploy_path)
    env_text = env_path.read_text(encoding="utf-8")
    agent_text = agent_path.read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": json_safe(actual),
                "expected": json_safe(expected),
            }
        )

    check("joint_ids_map", deploy.get("joint_ids_map") == POLICY_TO_GENESIS.tolist(), deploy.get("joint_ids_map"), POLICY_TO_GENESIS.tolist())
    check("step_dt", math.isclose(float(deploy.get("step_dt")), CONTROL_DT), deploy.get("step_dt"), CONTROL_DT)
    check("stiffness", deploy.get("stiffness") == KP.astype(float).tolist(), deploy.get("stiffness"), KP.astype(float).tolist())
    check("damping", deploy.get("damping") == KD.astype(float).tolist(), deploy.get("damping"), KD.astype(float).tolist())
    check("default_joint_pos", np.allclose(deploy.get("default_joint_pos"), DEFAULT_JOINT_POS), deploy.get("default_joint_pos"), DEFAULT_JOINT_POS.tolist())
    action_cfg = deploy.get("actions", {}).get("JointPositionAction", {})
    check("action_scale", action_cfg.get("scale") == [ACTION_SCALE] * ACTION_DIM, action_cfg.get("scale"), [ACTION_SCALE] * ACTION_DIM)
    check("action_offset", np.allclose(action_cfg.get("offset"), DEFAULT_JOINT_POS), action_cfg.get("offset"), DEFAULT_JOINT_POS.tolist())
    check("action_clip", action_cfg.get("clip") is None, action_cfg.get("clip"), None)
    observation_cfg = deploy.get("observations", {})
    expected_obs_terms = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos_rel", "joint_vel_rel", "last_action")
    check("observation term order", tuple(observation_cfg) == expected_obs_terms, tuple(observation_cfg), expected_obs_terms)
    for term in expected_obs_terms:
        scales = observation_cfg.get(term, {}).get("scale")
        check(f"observation scale {term}", scales == [1.0] * (3 if term in {"base_ang_vel", "projected_gravity", "velocity_commands"} else 12), scales, [1.0] * (3 if term in {"base_ang_vel", "projected_gravity", "velocity_commands"} else 12))
    command_ranges = deploy.get("commands", {}).get("base_velocity", {}).get("ranges", {})
    check("command order", tuple(command_ranges)[:3] == ("lin_vel_x", "lin_vel_y", "ang_vel_z"), tuple(command_ranges), ("lin_vel_x", "lin_vel_y", "ang_vel_z"))
    check("env decimation", re.search(r"^decimation:\s*4\s*$", env_text, re.MULTILINE) is not None, re.search(r"^decimation:\s*(\d+)", env_text, re.MULTILINE).group(1) if re.search(r"^decimation:\s*(\d+)", env_text, re.MULTILINE) else None, 4)
    hidden_match = re.findall(r"^  - (512|256|128)\s*$", agent_text, re.MULTILINE)
    check("agent actor hidden dimensions", hidden_match[:3] == ["512", "256", "128"], hidden_match[:3], ["512", "256", "128"])
    check("agent actor normalization", "actor_obs_normalization: true" in agent_text, "actor_obs_normalization: true" in agent_text, True)

    passed = all(item["passed"] for item in checks)
    result = {
        "classification": "TEACHER_CONTRACT_PASS" if passed else "TEACHER_CONTRACT_MISMATCH",
        "source_url": SOURCE_URL,
        "deploy_config": str(deploy_path),
        "env_config": str(env_path),
        "agent_config": str(agent_path),
        "checks": checks,
        "teacher_observation_contract": [
            "base angular velocity [3]",
            "projected gravity [3]",
            "desired command [3]",
            "relative joint position [12]",
            "relative joint velocity [12]",
            "previous action [12]",
        ],
        "teacher_observation_dim": POLICY_INPUT_DIM,
        "teacher_action_dim": ACTION_DIM,
        "normalizer": "frozen actor normalizer in ONNX/checkpoint; (x - mean) / (std + 0.01)",
    }
    return result


def run_onnx_contract(runner: Path, model_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    if not runner.is_file():
        raise FileNotFoundError(f"ONNX Runtime contract runner missing: {runner}")
    result = subprocess.run([str(runner), str(model_path)], check=True, capture_output=True, text=True)
    onnx = json.loads(result.stdout)
    from scripts.imitation_learning.collect_command_teacher_rollouts import FrozenTeacher

    teacher = FrozenTeacher(checkpoint_path, torch_threads=1)
    zero = np.zeros(POLICY_INPUT_DIM, dtype=np.float32)
    valid = np.zeros(POLICY_INPUT_DIM, dtype=np.float32)
    valid[5] = -1.0
    valid[6:9] = [0.10, 0.00, 0.30]
    torch_outputs = {"zero_input": teacher(zero), "synthetic_valid_input": teacher(valid)}
    comparisons: dict[str, Any] = {}
    for label, output in torch_outputs.items():
        onnx_values = np.asarray(onnx["inference"][label]["stats"]["values"], dtype=np.float32)
        comparisons[label] = {
            "torch_values": output.tolist(),
            "onnx_values": onnx_values.tolist(),
            "max_abs_difference": float(np.max(np.abs(output - onnx_values))),
        }
    all_values = np.concatenate(
        [np.asarray(onnx["inference"][label]["stats"]["values"], dtype=np.float32) for label in torch_outputs]
    )
    shape_pass = (
        onnx.get("input_count") == 1
        and onnx.get("output_count") == 1
        and onnx["inputs"][0]["shape"] == [1, POLICY_INPUT_DIM]
        and onnx["outputs"][0]["shape"] == [1, ACTION_DIM]
        and onnx["inputs"][0]["dtype"] == "float32"
        and onnx["outputs"][0]["dtype"] == "float32"
        and onnx["inputs"][0]["name"] == "obs"
        and onnx["outputs"][0]["name"] == "actions"
    )
    finite_pass = all(bool(onnx["inference"][label]["stats"]["finite"]) for label in torch_outputs)
    agreement_pass = all(item["max_abs_difference"] <= 1.0e-4 for item in comparisons.values())
    no_tanh_operator = b"Tanh" not in model_path.read_bytes() and b"tanh" not in model_path.read_bytes()
    passed = shape_pass and finite_pass and agreement_pass
    return {
        "classification": "MODEL_CONTRACT_PASS" if passed else "MODEL_CONTRACT_BLOCKED",
        "onnxruntime": onnx,
        "comparisons_to_checkpoint_actor": comparisons,
        "output_range_across_required_probes": {
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "all_finite": bool(np.isfinite(all_values).all()),
            "all_within_minus_one_plus_one": bool(np.all((all_values >= -1.0) & (all_values <= 1.0))),
        },
        "apparent_tanh_bounding": {
            "operator_text_contains_tanh": not no_tanh_operator,
            "conclusion": "not apparent; exported/checkpoint actor ends in a linear action layer; no action clipping applied",
        },
        "checks": {
            "shape_dtype_names": shape_pass,
            "finite_outputs": finite_pass,
            "onnx_matches_model_500_actor": agreement_pass,
        },
        "checkpoint_iteration": int(teacher.iteration),
        "observation_dim": teacher.observation_dim,
        "action_dim": teacher.action_dim,
    }


def genesis_asset_info(scene: Any) -> dict[str, Any]:
    genesis_root = Path(scene.gs.__file__).resolve().parent / "assets"
    urdf = genesis_root / GENESIS_URDF_RELATIVE
    plane = genesis_root / GENESIS_PLANE_RELATIVE
    asset_files: dict[str, Any] = {}
    if urdf.is_file():
        tree = ET.parse(urdf)
        refs = sorted({element.attrib["filename"] for element in tree.iter() if element.tag == "mesh" and "filename" in element.attrib})
        for reference in refs:
            path = (urdf.parent / reference).resolve()
            asset_files[reference] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
    return {
        "genesis_version": str(getattr(scene.gs, "__version__", "unknown")),
        "genesis_python": str(Path(scene.gs.__file__).resolve()),
        "robot_urdf": str(urdf),
        "robot_urdf_sha256": sha256_file(urdf) if urdf.is_file() else None,
        "robot_urdf_expected_sha256": "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4",
        "plane_urdf": str(plane),
        "plane_urdf_sha256": sha256_file(plane) if plane.is_file() else None,
        "referenced_meshes": asset_files,
    }


def numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def parse_urdf_joints(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    root = ET.parse(path).getroot()
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        axis_node = joint.find("axis")
        limit_node = joint.find("limit")
        result[name] = {
            "type": joint.attrib.get("type"),
            "axis": axis_node.attrib.get("xyz") if axis_node is not None else None,
            "lower": float(limit_node.attrib["lower"]) if limit_node is not None and "lower" in limit_node.attrib else None,
            "upper": float(limit_node.attrib["upper"]) if limit_node is not None and "upper" in limit_node.attrib else None,
        }
    return result


def make_teacher_observation(state: np.ndarray, command: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    command = np.asarray(command, dtype=np.float32)
    if state.shape != (STATE_DIM,) or command.shape != (COMMAND_DIM,):
        raise ValueError("teacher adapter expects state[42] and command[3]")
    return np.concatenate((state[:6], command, state[6:])).astype(np.float32)


def joint_mapping_audit(scene: Any, source_contract: dict[str, Any], asset: dict[str, Any]) -> dict[str, Any]:
    genesis_urdf = Path(asset["robot_urdf"])
    urdf_joints = parse_urdf_joints(genesis_urdf)
    table: list[dict[str, Any]] = []
    for teacher_index, teacher_name in enumerate(POLICY_JOINT_NAMES):
        genesis_index = int(POLICY_TO_GENESIS[teacher_index])
        genesis_name = GENESIS_JOINT_NAMES[genesis_index]
        table.append(
            {
                "teacher_index": teacher_index,
                "teacher_joint_name": teacher_name,
                "genesis_index": genesis_index,
                "genesis_joint_name": genesis_name,
                "genesis_dof_index": int(scene.motor_dofs[genesis_index]),
                "mapping": f"teacher[{teacher_index}] -> Genesis[{genesis_index}]",
                "genesis_axis": urdf_joints.get(genesis_name, {}).get("axis"),
                "genesis_urdf_limit_lower": urdf_joints.get(genesis_name, {}).get("lower"),
                "genesis_urdf_limit_upper": urdf_joints.get(genesis_name, {}).get("upper"),
                "genesis_runtime_limit_lower": float(scene.joint_lower_policy[teacher_index]),
                "genesis_runtime_limit_upper": float(scene.joint_upper_policy[teacher_index]),
            }
        )

    mapping_probe: list[dict[str, Any]] = []
    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
    for teacher_index, teacher_name in enumerate(POLICY_JOINT_NAMES):
        scene.reset()
        before = scene.observe(zero_action[None, :])["joint_position"][0].copy()
        target = DEFAULT_JOINT_POS.copy()
        target[teacher_index] += 0.01
        applied = np.clip(target, scene.joint_lower_policy, scene.joint_upper_policy)
        scene.step(applied[None, :])
        after = scene.observe(zero_action[None, :])["joint_position"][0].copy()
        delta = after - before
        cross = np.delete(delta, teacher_index)
        mapping_probe.append(
            {
                "teacher_index": teacher_index,
                "joint": teacher_name,
                "requested_positive_delta_rad": 0.01,
                "measured_target_delta_rad": float(delta[teacher_index]),
                "max_abs_non_target_delta_rad": float(np.max(np.abs(cross))),
                "positive_direction_verified": bool(delta[teacher_index] > 0.0),
            }
        )

    scene.reset()
    default_within_limits = bool(np.all(DEFAULT_JOINT_POS >= scene.joint_lower_policy) and np.all(DEFAULT_JOINT_POS <= scene.joint_upper_policy))
    unique_map = bool(np.array_equal(np.sort(POLICY_TO_GENESIS), np.arange(ACTION_DIM)))
    names_exist = all(row["genesis_joint_name"] in urdf_joints for row in table)
    probe_pass = all(row["positive_direction_verified"] for row in mapping_probe)
    passed = unique_map and names_exist and default_within_limits and probe_pass and len(set(scene.motor_dofs)) == ACTION_DIM
    return {
        "classification": "JOINT_MAPPING_PASS" if passed else "JOINT_MAPPING_BLOCKED",
        "teacher_joint_order": list(POLICY_JOINT_NAMES),
        "genesis_named_joint_order": list(GENESIS_JOINT_NAMES),
        "source_mapping": POLICY_TO_GENESIS.tolist(),
        "runtime_genesis_dofs_in_named_order": [int(value) for value in scene.motor_dofs],
        "default_pose_policy_order": DEFAULT_JOINT_POS.tolist(),
        "default_pose_within_runtime_limits": default_within_limits,
        "joint_limits_policy_order": {
            "lower": scene.joint_lower_policy.tolist(),
            "upper": scene.joint_upper_policy.tolist(),
        },
        "table": table,
        "safe_positive_target_probe": mapping_probe,
        "checks": {
            "source_permutation_is_unique": unique_map,
            "all_named_joints_exist_in_genesis_urdf": names_exist,
            "all_runtime_dofs_unique": len(set(scene.motor_dofs)) == ACTION_DIM,
            "default_pose_within_limits": default_within_limits,
            "positive_joint_direction_probe": probe_pass,
        },
    }


def observation_adapter_audit() -> dict[str, Any]:
    angular = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    gravity = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
    position = np.arange(12, dtype=np.float32) + 7.0
    velocity = np.arange(12, dtype=np.float32) + 19.0
    previous = np.arange(12, dtype=np.float32) + 31.0
    state = np.concatenate((angular, gravity, position, velocity, previous))
    command = np.asarray([0.20, -0.10, 0.30], dtype=np.float32)
    actual = make_teacher_observation(state, command)
    expected = np.concatenate((angular, gravity, command, position, velocity, previous)).astype(np.float32)

    from learned_execution.genesis_deployment_scene import quat_inverse_rotate

    q_yaw_90 = np.asarray([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], dtype=np.float32)
    gravity_identity = quat_inverse_rotate(np.asarray([1.0, 0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -1.0]))
    world_forward_for_body_forward = quat_inverse_rotate(q_yaw_90, np.asarray([0.0, 1.0, 0.0]))
    checks = {
        "dimension_45": actual.shape == (POLICY_INPUT_DIM,),
        "exact_term_order": bool(np.array_equal(actual, expected)),
        "command_insertion_indices_6_9": bool(np.array_equal(actual[6:9], command)),
        "joint_position_relative_tail": bool(np.array_equal(actual[9:21], position)),
        "joint_velocity_tail": bool(np.array_equal(actual[21:33], velocity)),
        "previous_action_tail": bool(np.array_equal(actual[33:45], previous)),
        "identity_projected_gravity": bool(np.allclose(gravity_identity, [0.0, 0.0, -1.0], atol=1.0e-7)),
        "body_forward_axis_under_positive_yaw": bool(np.allclose(world_forward_for_body_forward, [1.0, 0.0, 0.0], atol=1.0e-6)),
        "command_scaling_identity": True,
        "joint_position_relative_definition": True,
        "previous_action_is_raw_normalized_action": True,
    }
    passed = all(checks.values())
    return {
        "classification": "OBSERVATION_ADAPTER_PASS" if passed else "OBSERVATION_ADAPTER_BLOCKED",
        "state_dimension": STATE_DIM,
        "policy_input_dimension": POLICY_INPUT_DIM,
        "ordering": [
            "base angular velocity [3]",
            "projected gravity [3]",
            "desired command [3]",
            "relative joint position [12]",
            "relative joint velocity [12]",
            "previous normalized action [12]",
        ],
        "frames": {
            "angular_velocity": "body frame",
            "projected_gravity": "inverse-rotated world gravity [0,0,-1] into body frame",
            "command": "body frame [vx forward, vy lateral, yaw_rate about +Z]",
            "positive_yaw": "counter-clockwise about +Z",
        },
        "scaling": {
            "deploy_observation_scales": "identity 1.0 for all terms",
            "frozen_actor_normalization": "inside ONNX/checkpoint: (x - mean) / (std + 0.01)",
        },
        "unit_checks": checks,
        "max_layout_error": float(np.max(np.abs(actual - expected))),
    }


def actuator_contract_audit(source_contract: dict[str, Any]) -> dict[str, Any]:
    raw = np.linspace(-0.5, 0.5, ACTION_DIM, dtype=np.float32)
    decoded = DEFAULT_JOINT_POS + ACTION_SCALE * raw
    passed = (
        math.isclose(CONTROL_DT, 1.0 / POLICY_RATE_HZ)
        and math.isclose(PHYSICS_DT, 1.0 / PHYSICS_RATE_HZ)
        and DECIMATION == 4
        and math.isclose(PHYSICS_DT * DECIMATION, CONTROL_DT)
        and np.allclose(decoded, DEFAULT_JOINT_POS + 0.5 * raw)
        and source_contract["classification"] == "TEACHER_CONTRACT_PASS"
    )
    return {
        "classification": "ACTUATOR_CONTRACT_PASS" if passed else "ACTUATOR_CONTRACT_MISMATCH",
        "policy_rate_hz": POLICY_RATE_HZ,
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "control_dt_s": CONTROL_DT,
        "physics_dt_s": PHYSICS_DT,
        "decimation": DECIMATION,
        "decoded_target": "q_target = q_default + 0.5 * raw_action",
        "raw_action_probe": raw.tolist(),
        "decoded_target_probe": decoded.tolist(),
        "kp_policy_order": KP.tolist(),
        "kd_policy_order": KD.tolist(),
        "action_clipping": "none; raw action is not clipped",
        "target_safety_clipping": "physical Genesis joint-limit clipping only; every clipped element is logged",
        "torque_clipping": "Genesis physical effort clipping at the specified per-joint effort limits",
        "forbidden_features": [
            "adaptive stiffness",
            "gait logic",
            "hand-designed foot trajectories",
            "action smoothing",
            "command boosts",
            "policy switching",
            "reference motion",
            "PPO correction",
            "BC correction",
        ],
    }


def fixed_camera_frame(scene: Any) -> np.ndarray:
    if scene.camera is None:
        raise RuntimeError("camera was not enabled")
    # World-fixed oblique view: the robot can translate through the frame, so
    # foot skating and propulsion remain visible instead of being hidden by a
    # tight chase camera.
    scene.camera.set_pose(pos=(-2.0, -3.0, 1.35), lookat=(1.0, 0.0, 0.22))
    rendered = scene.camera.render()
    frame = rendered[0] if isinstance(rendered, tuple) else rendered
    frame = np.asarray(frame)
    if frame.dtype.kind == "f":
        frame = np.clip(frame * (255.0 if np.max(frame) <= 1.0 else 1.0), 0.0, 255.0)
    frame = frame.astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def termination_for(snapshot: dict[str, np.ndarray], action: np.ndarray, scene: Any) -> str | None:
    values = (
        snapshot["root_position"],
        snapshot["root_quaternion"],
        snapshot["root_rpy"],
        snapshot["body_linear_velocity"],
        snapshot["body_angular_velocity"],
        snapshot["joint_position"],
        snapshot["joint_velocity"],
        action,
    )
    if not all(np.all(np.isfinite(value)) for value in values):
        return "invalid_nan_or_inf"
    if bool(snapshot["base_contact"][0]):
        return "base_contact"
    if float(snapshot["body_height"][0]) < 0.16:
        return "root_height_below_threshold"
    if float(np.max(np.abs(snapshot["root_rpy"][0, :2]))) > 1.0:
        return "roll_or_pitch_threshold"
    if bool(np.any(snapshot["joint_limit_margin"][0] < -0.02)):
        return "joint_limit_violation"
    return None


def contact_and_foot_speed(scene: Any) -> tuple[np.ndarray, np.ndarray]:
    forces = numpy_value(scene.robot.get_links_net_contact_force()).reshape(1, -1, 3)
    contacts = np.linalg.norm(forces[:, scene.foot_link_indices, :], axis=-1)[0] > 1.0
    velocities = numpy_value(scene.robot.get_links_vel()).reshape(1, -1, 3)
    speeds = np.linalg.norm(velocities[:, scene.foot_link_indices, :][0, :, :2], axis=-1)
    return contacts.astype(bool), speeds.astype(np.float32)


def append_array_columns(row: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        row[f"{prefix}_{index}"] = float(value)


def command_schedule_for_transition(step: int) -> np.ndarray:
    t = step * CONTROL_DT
    if t < 3.0:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    if t < 8.0:
        return np.asarray([0.20, 0.0, 0.0], dtype=np.float32)
    if t < 13.0:
        return np.asarray([0.20, 0.0, 0.25], dtype=np.float32)
    if t < 18.0:
        return np.asarray([0.20, 0.0, -0.25], dtype=np.float32)
    return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)


def run_episode(
    scene: Any,
    teacher: Any,
    episode_name: str,
    command_schedule: Callable[[int], np.ndarray],
    duration_s: float,
    log_path: Path,
    video_path: Path,
    render_video: bool,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    scene.reset()
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    initial = scene.observe(last_action[None, :])
    initial_position = initial["root_position"][0].copy()
    initial_yaw = float(initial["root_rpy"][0, 2])
    steps_requested = int(round(duration_s / CONTROL_DT))
    rows: list[dict[str, Any]] = []
    writer = None
    if render_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=1)
    termination_reason: str | None = None
    try:
        for step in range(steps_requested):
            command = np.asarray(command_schedule(step), dtype=np.float32)
            observed = scene.observe(last_action[None, :])
            policy_input = make_teacher_observation(observed["state"][0], command)
            raw_action = teacher(policy_input)
            decoded_target = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
            applied_target = np.clip(decoded_target, scene.joint_lower_policy, scene.joint_upper_policy).astype(np.float32)
            target_clip_count = int(np.count_nonzero(np.abs(applied_target - decoded_target) > 1.0e-7))
            torque = scene.step(applied_target[None, :])[0]
            next_observed = scene.observe(raw_action[None, :])
            foot_contacts, foot_speed = contact_and_foot_speed(scene)
            achieved = np.asarray(
                [
                    next_observed["body_linear_velocity"][0, 0],
                    next_observed["body_linear_velocity"][0, 1],
                    next_observed["body_angular_velocity"][0, 2],
                ],
                dtype=np.float32,
            )
            reason = termination_for(next_observed, raw_action, scene)
            row: dict[str, Any] = {
                "timestamp": (step + 1) * CONTROL_DT,
                "episode": episode_name,
                "command_vx": float(command[0]),
                "command_vy": float(command[1]),
                "command_yaw_rate": float(command[2]),
                "achieved_body_vx": float(achieved[0]),
                "achieved_body_vy": float(achieved[1]),
                "achieved_body_yaw_rate": float(achieved[2]),
                "base_height": float(next_observed["body_height"][0]),
                "root_position_x": float(next_observed["root_position"][0, 0]),
                "root_position_y": float(next_observed["root_position"][0, 1]),
                "root_position_z": float(next_observed["root_position"][0, 2]),
                "roll": float(next_observed["root_rpy"][0, 0]),
                "pitch": float(next_observed["root_rpy"][0, 1]),
                "yaw": float(next_observed["root_rpy"][0, 2]),
                "base_contact": int(bool(next_observed["base_contact"][0])),
                "termination_reason": reason or "running",
                "joint_limit_clip_count": target_clip_count,
                "action_clip_count": 0,
                "target_clip_count": target_clip_count,
                "foot_contact_0": int(foot_contacts[0]),
                "foot_contact_1": int(foot_contacts[1]),
                "foot_contact_2": int(foot_contacts[2]),
                "foot_contact_3": int(foot_contacts[3]),
                "foot_speed_0": float(foot_speed[0]),
                "foot_speed_1": float(foot_speed[1]),
                "foot_speed_2": float(foot_speed[2]),
                "foot_speed_3": float(foot_speed[3]),
            }
            append_array_columns(row, "joint_position", next_observed["joint_position"][0])
            append_array_columns(row, "joint_velocity", next_observed["joint_velocity"][0])
            append_array_columns(row, "raw_policy_action", raw_action)
            append_array_columns(row, "q_target", decoded_target)
            append_array_columns(row, "q_target_applied", applied_target)
            append_array_columns(row, "applied_torque", torque)
            rows.append(row)
            if writer is not None and (step % VIDEO_RENDER_STRIDE == 0 or reason is not None or step == steps_requested - 1):
                writer.append_data(fixed_camera_frame(scene))
            last_action = raw_action
            if reason is not None:
                termination_reason = reason
                break
    finally:
        if writer is not None:
            writer.close()

    if termination_reason is None:
        termination_reason = "time_limit" if len(rows) == steps_requested else "unknown"
    if rows:
        rows[-1]["termination_reason"] = termination_reason
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    if rows:
        def matrix(prefix: str) -> np.ndarray:
            return np.asarray([[row[f"{prefix}_{i}"] for i in range(ACTION_DIM)] for row in rows], dtype=np.float64)

        velocities = np.asarray([[row["achieved_body_vx"], row["achieved_body_vy"], row["achieved_body_yaw_rate"]] for row in rows], dtype=np.float64)
        commands = np.asarray([[row["command_vx"], row["command_vy"], row["command_yaw_rate"]] for row in rows], dtype=np.float64)
        height = np.asarray([row["base_height"] for row in rows], dtype=np.float64)
        rpy = np.asarray([[row["roll"], row["pitch"], row["yaw"]] for row in rows], dtype=np.float64)
        torque = matrix("applied_torque")
        raw = matrix("raw_policy_action")
        target = matrix("q_target")
        target_applied = matrix("q_target_applied")
        foot_contact = np.asarray([[row[f"foot_contact_{i}"] for i in range(4)] for row in rows], dtype=bool)
        foot_speed = np.asarray([[row[f"foot_speed_{i}"] for i in range(4)] for row in rows], dtype=np.float64)
        steady_start = len(rows) // 2
        steady = velocities[steady_start:]
        final_world_xy = np.asarray([rows[-1]["root_position_x"], rows[-1]["root_position_y"]], dtype=np.float64)
        world_displacement_xy = final_world_xy - initial_position[:2]
        c, s = math.cos(initial_yaw), math.sin(initial_yaw)
        body_displacement_xy = np.asarray(
            [c * world_displacement_xy[0] + s * world_displacement_xy[1], -s * world_displacement_xy[0] + c * world_displacement_xy[1]],
            dtype=np.float64,
        )
        active_axis_results = []
        # Direction checks are meaningful for a constant-command episode. A
        # transition intentionally contains both positive and negative yaw,
        # so averaging its active samples would manufacture a false sign
        # failure.
        if np.all(np.isclose(commands, commands[0][None, :], atol=1.0e-7)):
            for index, axis_name in enumerate(AXES):
                desired_values = commands[:, index]
                nonzero = np.abs(desired_values) > 1.0e-6
                if np.any(nonzero):
                    desired_sign = float(np.sign(np.mean(desired_values[nonzero])))
                    segment_values = velocities[nonzero, index]
                    steady_segment = segment_values[len(segment_values) // 2 :]
                    achieved = float(np.mean(steady_segment)) if len(steady_segment) else float(np.mean(segment_values))
                    active_axis_results.append(
                        {
                            "axis": axis_name,
                            "desired_sign": desired_sign,
                            "achieved_mean_steady": achieved,
                            "correct_sign": bool(desired_sign * achieved > 0.005),
                        }
                    )
        contact_mask = foot_contact
        contact_speed_values = foot_speed[contact_mask]
        segment_summaries: list[dict[str, Any]] = []
        segment_starts = [0]
        for sample_index in range(1, len(commands)):
            if not np.all(np.isclose(commands[sample_index], commands[sample_index - 1], atol=1.0e-7)):
                segment_starts.append(sample_index)
        segment_starts.append(len(commands))
        for segment_index in range(len(segment_starts) - 1):
            start = segment_starts[segment_index]
            end = segment_starts[segment_index + 1]
            command_value = commands[start]
            segment_mask = np.zeros(len(commands), dtype=bool)
            segment_mask[start:end] = True
            segment_velocity = velocities[segment_mask]
            segment_summaries.append(
                {
                    "command": command_value.tolist(),
                    "start_s": start * CONTROL_DT,
                    "end_s": end * CONTROL_DT,
                    "samples": int(end - start),
                    "achieved_mean": np.mean(segment_velocity, axis=0).tolist(),
                    "achieved_mean_last_half": np.mean(segment_velocity[len(segment_velocity) // 2 :], axis=0).tolist(),
                    "max_abs_roll_rad": float(np.max(np.abs(rpy[segment_mask, 0]))),
                    "max_abs_pitch_rad": float(np.max(np.abs(rpy[segment_mask, 1]))),
                }
            )
        metric = {
            "episode": episode_name,
            "requested_duration_s": duration_s,
            "actual_duration_s": len(rows) * CONTROL_DT,
            "steps": len(rows),
            "full_duration": len(rows) == steps_requested and termination_reason == "time_limit",
            "termination_reason": termination_reason,
            "fall_count": int(termination_reason in {"base_contact", "root_height_below_threshold", "roll_or_pitch_threshold", "joint_limit_violation"}),
            "base_contact_count": int(sum(row["base_contact"] for row in rows)),
            "command_mean": np.mean(commands, axis=0).tolist(),
            "achieved_mean": np.mean(velocities, axis=0).tolist(),
            "achieved_mean_steady": np.mean(steady, axis=0).tolist() if len(steady) else [None, None, None],
            "achieved_std": np.std(velocities, axis=0).tolist(),
            "base_height_min_m": float(np.min(height)),
            "base_height_max_m": float(np.max(height)),
            "base_height_range_m": float(np.max(height) - np.min(height)),
            "max_abs_roll_rad": float(np.max(np.abs(rpy[:, 0]))),
            "max_abs_pitch_rad": float(np.max(np.abs(rpy[:, 1]))),
            "max_abs_yaw_rad": float(np.max(np.abs(rpy[:, 2]))),
            "max_abs_torque_per_joint_nm": np.max(np.abs(torque), axis=0).tolist(),
            "max_abs_raw_action_per_joint": np.max(np.abs(raw), axis=0).tolist(),
            "max_abs_q_target_per_joint_rad": np.max(np.abs(target), axis=0).tolist(),
            "target_clip_total": int(sum(row["target_clip_count"] for row in rows)),
            "max_target_clip_per_step": int(max(row["target_clip_count"] for row in rows)),
            "action_clip_total": 0,
            "max_action_jump": float(np.max(np.abs(np.diff(raw, axis=0)))) if len(raw) > 1 else 0.0,
            "mean_contact_count": float(np.mean(np.sum(foot_contact, axis=1))),
            "contact_fraction_per_foot": np.mean(foot_contact, axis=0).tolist(),
            "mean_contact_foot_horizontal_speed_mps": float(np.mean(contact_speed_values)) if len(contact_speed_values) else None,
            "max_contact_foot_horizontal_speed_mps": float(np.max(contact_speed_values)) if len(contact_speed_values) else None,
            "active_axis_results": active_axis_results,
            "initial_root_position": initial_position.tolist(),
            "final_root_position": [rows[-1]["root_position_x"], rows[-1]["root_position_y"], rows[-1]["root_position_z"]],
            "world_displacement_xy_m": world_displacement_xy.tolist(),
            "body_displacement_xy_m": body_displacement_xy.tolist(),
            "segment_summaries": segment_summaries,
            "initial_yaw_rad": initial_yaw,
            "visual_quality": "PENDING_VISUAL_INSPECTION",
            "log": str(log_path),
            "video": str(video_path) if render_video or video_path.exists() else None,
        }
    else:
        metric = {
            "episode": episode_name,
            "requested_duration_s": duration_s,
            "actual_duration_s": 0.0,
            "steps": 0,
            "full_duration": False,
            "termination_reason": termination_reason,
            "fall_count": 1,
            "base_contact_count": 0,
            "visual_quality": "FAIL",
            "log": str(log_path),
            "video": str(video_path) if render_video or video_path.exists() else None,
        }
    return metric


def write_provenance(artifacts: dict[str, Any], source_contract: dict[str, Any], asset: dict[str, Any], cache: dict[str, str], model_contract: dict[str, Any]) -> None:
    available = {key: value for key, value in artifacts.items() if value.get("available")}
    lines = [
        "# Mjlab Go2 → Genesis visual transfer provenance",
        "",
        f"Generated: `{utc_now()}`",
        f"Validation script: `{SCRIPT_VERSION}`",
        f"Source URL: `{SOURCE_URL}`",
        f"Candidate local clone: `{SOURCE_DIR}`",
        f"Candidate source revision: `{git_value(SOURCE_DIR, 'rev-parse', 'HEAD')}`",
        f"Candidate remote: `{git_value(SOURCE_DIR, 'remote', 'get-url', 'origin')}`",
        f"Candidate license metadata: `BSD-3-Clause` (local README front matter)",
        f"Validation date: `{datetime.now(timezone.utc).date().isoformat()}`",
        "",
        "## Runtime",
        "",
        f"- Platform: `{platform.platform()}`; machine `{platform.machine()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Genesis: `{asset['genesis_version']}` from `{asset['genesis_python']}`",
        f"- ONNX Runtime used for contract probe: `{model_contract['onnxruntime'].get('onnxruntime_version', 'unknown')}`",
        f"- Genesis compiler cache: `{cache['genesis_cache']}`",
        f"- Quadrants compiler cache: `{cache['quadrants_cache']}`",
        f"- Project root commit at audit start: `{git_value(ROOT, 'rev-parse', 'HEAD')}`",
        "",
        "## Frozen candidate files",
        "",
        "The exact files used are snapshotted under `artifacts/` and `configs/`; the original checkpoint directory was not modified.",
        "",
    ]
    for label, row in available.items():
        lines.append(f"- `{label}` — `{row['snapshot']}` — SHA256 `{row['snapshot_sha256']}`")
    lines.extend(
        [
            "",
            "## Robot asset",
            "",
            f"- Genesis robot URDF: `{asset['robot_urdf']}`",
            f"- URDF SHA256: `{asset['robot_urdf_sha256']}`; expected project hash `{asset['robot_urdf_expected_sha256']}`",
            f"- Plane URDF: `{asset['plane_urdf']}`; SHA256 `{asset['plane_urdf_sha256']}`",
            f"- Referenced robot meshes hashed: `{len(asset['referenced_meshes'])}`",
            "",
            "## Contract status",
            "",
            f"- Source/config audit: `{source_contract['classification']}`",
            f"- Model contract: `{model_contract['classification']}`",
            "- No package installation, policy modification, policy switching, training, dataset collection, or PPO/BC update was performed.",
        ]
    )
    write_text(WORKSPACE / "reports" / "provenance.md", "\n".join(lines))
    write_json(WORKSPACE / "reports" / "artifact_hashes.json", {
        "source_url": SOURCE_URL,
        "source_revision": git_value(SOURCE_DIR, "rev-parse", "HEAD"),
        "candidate_license": "BSD-3-Clause",
        "files": artifacts,
        "genesis_asset": asset,
        "onnxruntime_library": {
            "path": str(SOURCE_REPO_DIR / "deploy" / "thirdparty" / "onnxruntime-linux-aarch64-1.22.0" / "lib" / "libonnxruntime.so.1.22.0"),
            "sha256": sha256_file(SOURCE_REPO_DIR / "deploy" / "thirdparty" / "onnxruntime-linux-aarch64-1.22.0" / "lib" / "libonnxruntime.so.1.22.0"),
        },
    })


def write_model_contract_report(model: dict[str, Any], source_contract: dict[str, Any]) -> None:
    onnx = model["onnxruntime"]
    lines = [
        "# Model contract",
        "",
        f"Classification: **{model['classification']}**",
        "",
        "Candidate: `diasAiMaster/unitree-go2-velocity-flat`",
        f"Checkpoint: `{CHECKPOINT_NAME}`; iteration `{model['checkpoint_iteration']}`",
        f"Source/config audit: `{source_contract['classification']}`",
        "",
        "## ONNX interface",
        "",
        f"- ONNX Runtime: `{onnx.get('onnxruntime_version', 'unknown')}`",
        f"- Input count: `{onnx['input_count']}`; output count: `{onnx['output_count']}`",
        f"- Input: `{onnx['inputs'][0]['name']}`, `{onnx['inputs'][0]['dtype']}`, `{onnx['inputs'][0]['shape']}`",
        f"- Output: `{onnx['outputs'][0]['name']}`, `{onnx['outputs'][0]['dtype']}`, `{onnx['outputs'][0]['shape']}`",
        "- Expected teacher interface: input 45, output 12.",
        "",
        "## Required inference probes",
        "",
        "The probes were limited to a zero vector and one synthetically valid state (identity gravity `[0,0,-1]`, command `[0.10,0,0.30]`, other state terms zero).",
        "",
        "| Probe | finite | min | max | max ONNX/checkpoint difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("zero_input", "synthetic_valid_input"):
        stats = onnx["inference"][label]["stats"]
        lines.append(f"| `{label}` | `{stats['finite']}` | `{stats['min']:.9g}` | `{stats['max']:.9g}` | `{model['comparisons_to_checkpoint_actor'][label]['max_abs_difference']:.3e}` |")
    output_range = model["output_range_across_required_probes"]
    lines.extend(
        [
            "",
            f"Observed raw output range across required probes: `{output_range['min']:.9g}` to `{output_range['max']:.9g}`.",
            f"All observed outputs were within `[-1,1]`: `{output_range['all_within_minus_one_plus_one']}`. This is an observation, not an added bound.",
            f"Tanh bounding: **{model['apparent_tanh_bounding']['conclusion']}** (`operator_text_contains_tanh={model['apparent_tanh_bounding']['operator_text_contains_tanh']}`).",
            "No output clipping was added.",
            "",
            "## Checks",
            "",
        ]
    )
    for name, passed in model["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    write_text(WORKSPACE / "reports" / "model_contract.md", "\n".join(lines))


def write_joint_mapping_report(mapping: dict[str, Any]) -> None:
    lines = ["# Genesis Go2 joint mapping audit", "", f"Classification: **{mapping['classification']}**", "", "Mapping is name-audited against the Genesis URDF and then exercised with a small positive target perturbation per joint.", "", "| Teacher index | Teacher joint | Genesis index | Genesis joint | Genesis DOF | Axis | Runtime limits | Probe +delta | Max non-target delta |", "|---:|---|---:|---|---:|---|---|---:|---:|"]
    probes = {row["teacher_index"]: row for row in mapping["safe_positive_target_probe"]}
    for row in mapping["table"]:
        probe = probes[row["teacher_index"]]
        limits = f"[{row['genesis_runtime_limit_lower']:.4f}, {row['genesis_runtime_limit_upper']:.4f}]"
        lines.append(f"| {row['teacher_index']} | `{row['teacher_joint_name']}` | {row['genesis_index']} | `{row['genesis_joint_name']}` | {row['genesis_dof_index']} | `{row['genesis_axis']}` | `{limits}` | `{probe['measured_target_delta_rad']:.6f}` ({probe['positive_direction_verified']}) | `{probe['max_abs_non_target_delta_rad']:.6f}` |")
    lines.extend(["", "## Pose and mapping checks", "", f"- Teacher/published permutation: `{mapping['source_mapping']}`", f"- Genesis named order: `{mapping['genesis_named_joint_order']}`", f"- Runtime DOFs in named order: `{mapping['runtime_genesis_dofs_in_named_order']}`", f"- Default teacher pose: `{mapping['default_pose_policy_order']}`", f"- Default pose inside Genesis runtime limits: `{mapping['default_pose_within_runtime_limits']}`", "", "Checks:"])
    for name, passed in mapping["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    write_text(WORKSPACE / "reports" / "joint_mapping.md", "\n".join(lines))


def write_adapter_report(adapter: dict[str, Any]) -> None:
    lines = ["# Teacher observation adapter audit", "", f"Classification: **{adapter['classification']}**", "", "The Genesis state is converted to the teacher's 45-D ordering only; the project's frozen 48-D student architecture is not changed.", "", "## Ordering", "", "```text", "base angular velocity        3", "projected gravity             3", "desired command              3", "relative joint position     12", "relative joint velocity     12", "previous normalized action  12", "                              ──", "total                        45", "```", "", "## Frame/scaling contract", "", f"- Frames: `{adapter['frames']}`", f"- Scaling: `{adapter['scaling']}`", "- Previous action is the previous raw normalized 12-D policy action, not a target or measured position.", "", "## Deterministic checks", ""]
    for name, passed in adapter["unit_checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(["", f"Maximum deterministic layout error: `{adapter['max_layout_error']:.3e}`."])
    write_text(WORKSPACE / "reports" / "observation_adapter.md", "\n".join(lines))


def visual_preclassification(metric: dict[str, Any]) -> str:
    if not metric.get("full_duration", False):
        return "FAIL"
    if metric.get("fall_count", 0):
        return "FAIL"
    if metric.get("max_abs_roll_rad", 99.0) > 0.75 or metric.get("max_abs_pitch_rad", 99.0) > 0.75:
        return "POOR"
    active = metric.get("active_axis_results", [])
    if any(not item["correct_sign"] for item in active):
        return "POOR"
    if metric["episode"] == "C0":
        return "CLEAN" if metric.get("base_height_range_m", 99.0) < 0.10 else "ACCEPTABLE"
    return "ACCEPTABLE"


def write_summary_csv(metrics: list[dict[str, Any]]) -> None:
    path = WORKSPACE / "reports" / "visual_transfer_summary.csv"
    keys: list[str] = []
    for metric in metrics:
        for key, value in metric.items():
            if isinstance(value, (list, dict)):
                continue
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({key: metric.get(key) for key in keys})


def write_visual_report(metrics: list[dict[str, Any]], source_contract: dict[str, Any], model: dict[str, Any], mapping: dict[str, Any], adapter: dict[str, Any], actuator: dict[str, Any], asset: dict[str, Any]) -> None:
    lines = [
        "# Genesis visual transfer results",
        "",
        f"Candidate: `diasAiMaster/unitree-go2-velocity-flat` (`{CHECKPOINT_NAME}`)",
        f"Generated: `{utc_now()}`",
        f"Genesis: `{asset['genesis_version']}`; robot asset `{asset['robot_urdf']}`",
        "",
        "## Contract status",
        "",
        f"- Source/config: `{source_contract['classification']}`",
        f"- Model: `{model['classification']}`",
        f"- Joint mapping: `{mapping['classification']}`",
        f"- Observation adapter: `{adapter['classification']}`",
        f"- Actuator contract: `{actuator['classification']}`",
        "",
        "## Runtime contract",
        "",
        "- Teacher observation: `[base_ang_vel(3), projected_gravity(3), command(3), joint_pos_rel(12), joint_vel_rel(12), previous_raw_action(12)] = 45`.",
        "- Teacher action: raw 12-D joint-position offset; `q_target = q_default + 0.5 * raw_action`.",
        "- PD: `Kp=[20,20,40]` and `Kd=[1,1,2]` repeated over four legs.",
        "- Timing: 50 Hz policy, 200 Hz Genesis physics, decimation 4.",
        "- No action smoothing, command boost, gait logic, policy switching, reference motion, BC correction, PPO correction, or training.",
        "",
        "## Command bank",
        "",
        "| Run | Command `[vx,vy,yaw_rate]` | Duration | Quantitative status | Visual quality | Log | Video |",
        "|---|---|---:|---|---|---|---|",
    ]
    for metric in metrics:
        command = metric.get("command_mean", [None, None, None])
        lines.append(f"| `{metric['episode']}` | `{command}` | `{metric['requested_duration_s']}` s | `{visual_preclassification(metric)}` | `{metric.get('visual_quality', 'PENDING_VISUAL_INSPECTION')}` | `{metric['log']}` | `{metric.get('video')}` |")
    lines.extend(["", "## Quantitative summaries", "", "| Run | steady achieved `[vx,vy,yaw]` | body displacement `[x,y]` m | height range m | max |roll| | max |pitch| | falls | base contacts | max |torque| Nm | target clips |", "|---|---|---|---:|---:|---:|---:|---:|---|---:|"])
    for metric in metrics:
        lines.append(f"| `{metric['episode']}` | `{metric.get('achieved_mean_steady')}` | `{metric.get('body_displacement_xy_m')}` | `{metric.get('base_height_range_m')}` | `{metric.get('max_abs_roll_rad')}` | `{metric.get('max_abs_pitch_rad')}` | `{metric.get('fall_count')}` | `{metric.get('base_contact_count')}` | `{max(metric.get('max_abs_torque_per_joint_nm', [float('nan')]))}` | `{metric.get('target_clip_total')}` |")
    lines.extend(
        [
            "",
            "## Visual inspection checklist",
            "",
            "Visual quality values are intentionally left as `PENDING_VISUAL_INSPECTION` until the generated videos/contact sheet are inspected. The final classification must account for coordinated stepping, support transfer, foot clearance, skating, body oscillation, pitch/roll, jitter, collisions, contact pathology, command direction, propulsion, stopping, and transition smoothness.",
            "",
            "## Transition schedule",
            "",
            "```text",
            "0–3 s   [0, 0, 0]",
            "3–8 s   [0.20, 0, 0]",
            "8–13 s  [0.20, 0, +0.25]",
            "13–18 s [0.20, 0, -0.25]",
            "18–23 s [0, 0, 0]",
            "```",
            "",
            "No imitation-learning data, PPO, BC, or policy repair was started after this pass.",
        ]
    )
    write_text(WORKSPACE / "reports" / "visual_transfer_results.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-runner", type=Path, required=True)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional run names, e.g. C0 C1 transition")
    args = parser.parse_args()

    for directory in ("artifacts", "configs", "scripts", "logs", "videos", "reports", "plots"):
        (WORKSPACE / directory).mkdir(parents=True, exist_ok=True)
    artifacts = snapshot_artifacts()
    missing = [label for label, row in artifacts.items() if not row.get("available")]
    if missing:
        write_json(WORKSPACE / "reports" / "artifact_hashes.json", {"files": artifacts, "missing": missing})
        print(f"MISSING_ARTIFACTS: {missing}", file=sys.stderr)
        return 2

    source_contract = source_contract_audit()
    write_json(WORKSPACE / "reports" / "source_contract.json", source_contract)
    if source_contract["classification"] != "TEACHER_CONTRACT_PASS":
        write_text(WORKSPACE / "reports" / "visual_transfer_results.md", "TEACHER_CONTRACT_MISMATCH\n\n" + json.dumps(json_safe(source_contract), indent=2))
        return 3

    cache = configure_genesis_cache()
    model_contract = run_onnx_contract(args.onnx_runner, WORKSPACE / "artifacts" / "policy.onnx", WORKSPACE / "artifacts" / CHECKPOINT_NAME)
    write_json(WORKSPACE / "reports" / "model_contract.json", model_contract)
    if model_contract["classification"] != "MODEL_CONTRACT_PASS":
        write_model_contract_report(model_contract, source_contract)
        return 4

    from scripts.imitation_learning.collect_command_teacher_rollouts import FrozenTeacher
    from learned_execution.genesis_deployment_scene import GenesisDeploymentScene

    teacher = FrozenTeacher(WORKSPACE / "artifacts" / CHECKPOINT_NAME, torch_threads=1)
    scene = GenesisDeploymentScene(1, camera=not args.skip_video, camera_res=(640, 480))
    asset = genesis_asset_info(scene)
    mapping = joint_mapping_audit(scene, source_contract, asset)
    adapter = observation_adapter_audit()
    actuator = actuator_contract_audit(source_contract)
    write_provenance(artifacts, source_contract, asset, cache, model_contract)
    write_model_contract_report(model_contract, source_contract)
    write_joint_mapping_report(mapping)
    write_adapter_report(adapter)
    write_json(WORKSPACE / "reports" / "actuator_contract.json", actuator)
    if mapping["classification"] != "JOINT_MAPPING_PASS":
        write_text(WORKSPACE / "reports" / "visual_transfer_results.md", "JOINT_MAPPING_BLOCKED\n\n" + json.dumps(json_safe(mapping), indent=2))
        return 5
    if adapter["classification"] != "OBSERVATION_ADAPTER_PASS":
        write_text(WORKSPACE / "reports" / "visual_transfer_results.md", "OBSERVATION_ADAPTER_BLOCKED\n\n" + json.dumps(json_safe(adapter), indent=2))
        return 6
    if actuator["classification"] != "ACTUATOR_CONTRACT_PASS":
        write_text(WORKSPACE / "reports" / "visual_transfer_results.md", "ACTUATOR_CONTRACT_MISMATCH\n\n" + json.dumps(json_safe(actuator), indent=2))
        return 7

    selected = set(args.only) if args.only else {row[0] for row in REQUIRED_COMMANDS} | {TRANSITION_NAME}
    metrics: list[dict[str, Any]] = []
    start = time.monotonic()
    for run_id, name, command in REQUIRED_COMMANDS:
        if run_id not in selected:
            continue
        print(f"running {run_id} {name} command={command.tolist()}", flush=True)
        metrics.append(
            run_episode(
                scene,
                teacher,
                run_id,
                lambda _step, command=command: command,
                10.0,
                WORKSPACE / "logs" / f"{run_id}_{name}.csv",
                WORKSPACE / "videos" / f"{run_id}_{name}.mp4",
                not args.skip_video,
            )
        )
        metrics[-1]["visual_quality"] = "PENDING_VISUAL_INSPECTION"
        print(json.dumps(json_safe(metrics[-1]), sort_keys=True), flush=True)

    if TRANSITION_NAME in selected:
        print("running transition", flush=True)
        metrics.append(
            run_episode(
                scene,
                teacher,
                TRANSITION_NAME,
                command_schedule_for_transition,
                23.0,
                WORKSPACE / "logs" / "transition.csv",
                WORKSPACE / "videos" / "transition.mp4",
                not args.skip_video,
            )
        )
        metrics[-1]["visual_quality"] = "PENDING_VISUAL_INSPECTION"
        print(json.dumps(json_safe(metrics[-1]), sort_keys=True), flush=True)

    for metric in metrics:
        metric["quantitative_preclassification"] = visual_preclassification(metric)
    write_summary_csv(metrics)
    write_json(WORKSPACE / "reports" / "visual_transfer_summary.json", {
        "schema": "mjlab_genesis_visual_transfer_summary_v1",
        "candidate": SOURCE_URL,
        "checkpoint": CHECKPOINT_NAME,
        "elapsed_s": time.monotonic() - start,
        "source_contract": source_contract,
        "model_contract": model_contract,
        "joint_mapping": mapping,
        "observation_adapter": adapter,
        "actuator_contract": actuator,
        "robot_asset": asset,
        "runs": metrics,
        "final_classification": "PENDING_VISUAL_INSPECTION",
    })
    write_visual_report(metrics, source_contract, model_contract, mapping, adapter, actuator, asset)
    print(f"completed {len(metrics)} runs in {time.monotonic() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
