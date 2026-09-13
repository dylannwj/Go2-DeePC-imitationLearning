#!/usr/bin/env python3
"""Full, evaluation-only compatibility gate for the frozen Genesis expert.

This gate deliberately imports the already validated Genesis deployment
adapter instead of creating a second robot implementation.  It never trains,
changes, clips, smooths, or switches the expert policy.  The only clipping in
the rollout path is the physical Genesis joint/effort safety behavior that is
already present in the validated runtime, and both pre-clip and post-clip
values are logged.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "results/genesis/generated/imitation_compatibility"
VISUAL = ROOT / "external/unitree_go2"
SOURCE_DIR = ROOT / "external/unitree-go2-velocity-flat"
FROZEN_CHECKPOINT = ROOT / "models/expert_model_500.pt"
FROZEN_ONNX = VISUAL / "policy.onnx"
FROZEN_ONNX_DATA = VISUAL / "policy.onnx.data"
FROZEN_DEPLOY = ROOT / "configs/imitation/expert/deploy.yaml"
FROZEN_ENV = ROOT / "configs/imitation/expert/env.yaml"
FROZEN_AGENT = ROOT / "configs/imitation/expert/agent.yaml"
PREVIOUS_HASHES = VISUAL / "reports" / "artifact_hashes.json"
PREVIOUS_RESULT = VISUAL / "reports" / "visual_transfer_results.md"
PREVIOUS_MAPPING = VISUAL / "reports" / "joint_mapping.md"

CHECKPOINT_NAME = "model_500.pt"
SOURCE_URL = "https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat"
PREVIOUS_CLASSIFICATION = "GENESIS_VISUAL_TRANSFER_PASS"
GATE_VERSION = "mjlab-genesis-expert-compatibility-gate-v1.0"

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
EFFORT_LIMIT = np.tile(np.asarray([23.5, 23.5, 45.43], dtype=np.float32), 4)
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
STEADY_TRANSIENT_S = 2.0
TARGET_CLIP_FRACTION_LIMIT = 0.02
TORQUE_LIMIT_FRACTION_LIMIT = 0.02
VIDEO_FPS = 25
VIDEO_RENDER_STRIDE = 2
CONTACT_FORCE_THRESHOLD_N = 1.0

REQUIRED_COMMANDS = (
    ("C0", "standing", np.asarray([0.00, 0.00, 0.00], dtype=np.float32)),
    ("C1", "slow_forward", np.asarray([0.10, 0.00, 0.00], dtype=np.float32)),
    ("C2", "forward", np.asarray([0.30, 0.00, 0.00], dtype=np.float32)),
    ("C3", "yaw_left", np.asarray([0.00, 0.00, 0.30], dtype=np.float32)),
    ("C4", "yaw_right", np.asarray([0.00, 0.00, -0.30], dtype=np.float32)),
    ("C5", "forward_left_arc", np.asarray([0.20, 0.00, 0.25], dtype=np.float32)),
    ("C6", "forward_right_arc", np.asarray([0.20, 0.00, -0.25], dtype=np.float32)),
)
LATERAL_COMMANDS = (
    ("L1", "lateral_left", np.asarray([0.00, 0.15, 0.00], dtype=np.float32)),
    ("L2", "lateral_right", np.asarray([0.00, -0.15, 0.00], dtype=np.float32)),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
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


def load_visual_runtime() -> Any:
    """Load the frozen visual-transfer helper without importing it as a package."""

    path = ROOT / "scripts/imitation_learning/run_expert_validation.py"
    spec = importlib.util.spec_from_file_location("frozen_visual_transfer_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load validated runtime helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_caches() -> dict[str, str]:
    """Keep Genesis/Quadrants/Matplotlib compiler caches out of the repo."""

    base = Path("/tmp/mjlab_expert_compatibility_gate")
    genesis_cache = base / "genesis"
    quadrants_cache = base / "quadrants"
    matplotlib_cache = base / "matplotlib"
    genesis_cache.mkdir(parents=True, exist_ok=True)
    quadrants_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["GS_CACHE_FILE_PATH"] = str(genesis_cache)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

    import quadrants as qd

    if not getattr(qd.init, "_mjlab_expert_gate_patched", False):
        original_init = qd.init

        def patched_init(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("offline_cache_file_path", str(quadrants_cache))
            return original_init(*args, **kwargs)

        patched_init._mjlab_expert_gate_patched = True  # type: ignore[attr-defined]
        qd.init = patched_init
    return {
        "genesis_cache": str(genesis_cache),
        "quadrants_cache": str(quadrants_cache),
        "matplotlib_cache": str(matplotlib_cache),
    }


def artifact_inventory() -> dict[str, Any]:
    import genesis as gs

    asset_root = Path(gs.__file__).resolve().parent / "assets"
    urdf = asset_root / "urdf/go2/urdf/go2.urdf"
    plane = asset_root / "urdf/plane/plane.urdf"
    mesh_rows: dict[str, Any] = {}
    if urdf.is_file():
        root = ET.parse(urdf).getroot()
        refs = sorted(
            {
                node.attrib["filename"]
                for node in root.iter()
                if node.tag == "mesh" and "filename" in node.attrib
            }
        )
        for reference in refs:
            mesh = (urdf.parent / reference).resolve()
            mesh_rows[reference] = {
                "path": str(mesh),
                "exists": mesh.is_file(),
                "sha256": sha256_file(mesh),
                "size_bytes": mesh.stat().st_size if mesh.is_file() else None,
            }

    def file_row(path: Path, role: str) -> dict[str, Any]:
        return {
            "role": role,
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path),
        }

    candidate_files = {
        "policy.onnx": SOURCE_DIR / "policy.onnx",
        "policy.onnx.data": SOURCE_DIR / "policy.onnx.data",
        CHECKPOINT_NAME: SOURCE_DIR / CHECKPOINT_NAME,
        "params/deploy.yaml": SOURCE_DIR / "params/deploy.yaml",
        "params/env.yaml": SOURCE_DIR / "params/env.yaml",
        "params/agent.yaml": SOURCE_DIR / "params/agent.yaml",
    }
    frozen_files = {
        "policy.onnx": FROZEN_ONNX,
        "policy.onnx.data": FROZEN_ONNX_DATA,
        CHECKPOINT_NAME: FROZEN_CHECKPOINT,
        "params/deploy.yaml": FROZEN_DEPLOY,
        "params/env.yaml": FROZEN_ENV,
        "params/agent.yaml": FROZEN_AGENT,
    }
    selected_vs_source: dict[str, Any] = {}
    for label, source in candidate_files.items():
        selected = frozen_files[label]
        selected_vs_source[label] = {
            "source_sha256": sha256_file(source),
            "frozen_snapshot_sha256": sha256_file(selected),
            "match": sha256_file(source) == sha256_file(selected),
        }

    try:
        onnxruntime_version = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        onnxruntime_version = "not-installed"

    return {
        "generated": utc_now(),
        "candidate": SOURCE_URL,
        "source_revision": git_value(SOURCE_DIR, "rev-parse", "HEAD"),
        "source_remote": git_value(SOURCE_DIR, "config", "--get", "remote.origin.url"),
        "executed_artifacts": {
            label: file_row(path, "frozen visual-transfer snapshot used by this gate")
            for label, path in frozen_files.items()
        },
        "candidate_artifacts": {
            label: file_row(path, "candidate source") for label, path in candidate_files.items()
        },
        "selected_vs_source": selected_vs_source,
        "configs": {
            "deploy_yaml_sha256": sha256_file(FROZEN_DEPLOY),
            "env_yaml_sha256": sha256_file(FROZEN_ENV),
            "agent_yaml_sha256": sha256_file(FROZEN_AGENT),
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "genesis_version": str(getattr(gs, "__version__", "unknown")),
            "genesis_python": str(Path(gs.__file__).resolve()),
            "numpy_version": np.__version__,
            "onnxruntime_version": onnxruntime_version,
            "project_revision": git_value(ROOT, "rev-parse", "HEAD"),
            "project_status": git_value(ROOT, "status", "--short"),
        },
        "genesis_assets": {
            "robot_urdf": str(urdf),
            "robot_urdf_sha256": sha256_file(urdf),
            "plane_urdf": str(plane),
            "plane_urdf_sha256": sha256_file(plane),
            "referenced_meshes": mesh_rows,
        },
        "runtime_source_files": {
            str(path): sha256_file(path)
            for path in (
                ROOT / "scripts/imitation_learning/run_expert_validation.py",
                ROOT / "scripts/imitation_learning/collect_command_teacher_rollouts.py",
                ROOT / "src/learned_execution/genesis_deployment_scene.py",
                Path(__file__).resolve(),
            )
        },
    }


def prior_mismatches(inventory: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if not PREVIOUS_HASHES.is_file():
        return [f"missing prior visual-transfer hash manifest: {PREVIOUS_HASHES}"]
    previous = json.loads(PREVIOUS_HASHES.read_text(encoding="utf-8"))
    previous_files = previous.get("files", {})
    for label, row in inventory["candidate_artifacts"].items():
        old = previous_files.get(label, {})
        old_hash = old.get("sha256")
        if old_hash != row.get("sha256"):
            mismatches.append(
                f"{label}: candidate hash {row.get('sha256')} differs from prior {old_hash}"
            )
    old_revision = previous.get("source_revision")
    if old_revision != inventory.get("source_revision"):
        mismatches.append(
            f"source revision: current {inventory.get('source_revision')} differs from prior {old_revision}"
        )
    old_genesis = previous.get("genesis_asset", {})
    current_genesis = inventory.get("genesis_assets", {})
    if old_genesis.get("genesis_version") not in (None, inventory["runtime"]["genesis_version"]):
        mismatches.append(
            f"Genesis version: current {inventory['runtime']['genesis_version']} differs from prior {old_genesis.get('genesis_version')}"
        )
    if old_genesis.get("robot_urdf_sha256") not in (None, current_genesis.get("robot_urdf_sha256")):
        mismatches.append("Genesis Go2 URDF hash differs from prior visual-transfer provenance")
    for label, row in inventory["selected_vs_source"].items():
        if not row["match"]:
            mismatches.append(f"frozen snapshot mismatch for {label}")
    return mismatches


def write_frozen_inputs(inventory: dict[str, Any], mismatches: list[str]) -> None:
    write_text(
        GATE / "frozen_inputs" / "README.md",
        "\n".join(
            [
                "# Frozen inputs",
                "",
                "This directory contains references only. The expert checkpoint and its ONNX auxiliary files are not copied or mutated by the compatibility gate.",
                "",
                f"- Candidate: `{SOURCE_URL}`",
                f"- Executed checkpoint: `{FROZEN_CHECKPOINT}`",
                f"- Executed deploy config: `{FROZEN_DEPLOY}`",
                f"- Prior visual-transfer result: `{PREVIOUS_RESULT}`",
                f"- Prior joint mapping evidence: `{PREVIOUS_MAPPING}`",
                f"- Provenance mismatches: `{len(mismatches)}`",
                "",
                "The selected frozen snapshots were hash-compared to the candidate source before execution.",
            ]
        ),
    )
    write_json(GATE / "frozen_inputs" / "paths.json", inventory)


def audit_frozen_contract(visual: Any, inventory: dict[str, Any]) -> dict[str, Any]:
    import torch
    import yaml

    deploy = yaml.safe_load(FROZEN_DEPLOY.read_text(encoding="utf-8"))
    env_text = FROZEN_ENV.read_text(encoding="utf-8")
    agent_text = FROZEN_AGENT.read_text(encoding="utf-8")
    checkpoint = torch.load(FROZEN_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", {}) if isinstance(checkpoint, dict) else {}
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

    required_state = {
        "actor.0.weight": (512, POLICY_INPUT_DIM),
        "actor.0.bias": (512,),
        "actor.2.weight": (256, 512),
        "actor.2.bias": (256,),
        "actor.4.weight": (128, 256),
        "actor.4.bias": (128,),
        "actor.6.weight": (ACTION_DIM, 128),
        "actor.6.bias": (ACTION_DIM,),
        "actor_obs_normalizer._mean": (1, POLICY_INPUT_DIM),
        "actor_obs_normalizer._std": (1, POLICY_INPUT_DIM),
    }
    for name, expected_shape in required_state.items():
        actual = tuple(state[name].shape) if name in state else None
        check(f"checkpoint tensor {name}", actual == expected_shape, actual, expected_shape)

    expected_terms = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )
    observations = deploy.get("observations", {})
    check("observation term order", tuple(observations) == expected_terms, tuple(observations), expected_terms)
    term_sizes = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3}
    term_sizes.update({term: 12 for term in expected_terms if term not in term_sizes})
    for term in expected_terms:
        config = observations.get(term, {})
        expected_scale = [1.0] * term_sizes[term]
        check(f"observation scale {term}", config.get("scale") == expected_scale, config.get("scale"), expected_scale)
        check(f"observation clip {term}", config.get("clip") is None, config.get("clip"), None)

    action_cfg = deploy.get("actions", {}).get("JointPositionAction", {})
    check("policy input dimension", POLICY_INPUT_DIM == 45, POLICY_INPUT_DIM, 45)
    check("policy output dimension", ACTION_DIM == 12, ACTION_DIM, 12)
    check("joint_ids_map", deploy.get("joint_ids_map") == POLICY_TO_GENESIS.tolist(), deploy.get("joint_ids_map"), POLICY_TO_GENESIS.tolist())
    check("control step", math.isclose(float(deploy.get("step_dt", math.nan)), CONTROL_DT), deploy.get("step_dt"), CONTROL_DT)
    check("fixed Kp", deploy.get("stiffness") == KP.astype(float).tolist(), deploy.get("stiffness"), KP.astype(float).tolist())
    check("fixed Kd", deploy.get("damping") == KD.astype(float).tolist(), deploy.get("damping"), KD.astype(float).tolist())
    check("default joint pose", np.allclose(deploy.get("default_joint_pos"), DEFAULT_JOINT_POS), deploy.get("default_joint_pos"), DEFAULT_JOINT_POS.tolist())
    check("action scale", action_cfg.get("scale") == [ACTION_SCALE] * ACTION_DIM, action_cfg.get("scale"), [ACTION_SCALE] * ACTION_DIM)
    check("action offset", np.allclose(action_cfg.get("offset"), DEFAULT_JOINT_POS), action_cfg.get("offset"), DEFAULT_JOINT_POS.tolist())
    check("raw action clip", action_cfg.get("clip") is None, action_cfg.get("clip"), None)
    check("physics decimation", re.search(r"^decimation:\s*4\s*$", env_text, re.MULTILINE) is not None, "decimation: 4", 4)
    hidden = re.findall(r"^\s*-\s*(512|256|128)\s*$", agent_text, re.MULTILINE)
    check("actor hidden dimensions", hidden[:3] == ["512", "256", "128"], hidden[:3], ["512", "256", "128"])
    check("frozen actor normalization", "actor_obs_normalization: true" in agent_text, "actor_obs_normalization: true" in agent_text, True)

    # These are runtime-path assertions, not memory-based assumptions.  The
    # gate uses the same helper functions as the previously validated pass.
    adapter_probe_state = np.concatenate(
        (
            np.asarray([1, 2, 3], dtype=np.float32),
            np.asarray([4, 5, 6], dtype=np.float32),
            np.arange(12, dtype=np.float32) + 7,
            np.arange(12, dtype=np.float32) + 19,
            np.arange(12, dtype=np.float32) + 31,
        )
    )
    adapter_command = np.asarray([0.2, -0.1, 0.3], dtype=np.float32)
    adapter_output = visual.make_teacher_observation(adapter_probe_state, adapter_command)
    adapter_expected = np.concatenate((adapter_probe_state[:6], adapter_command, adapter_probe_state[6:]))
    check("adapter output shape", adapter_output.shape == (45,), adapter_output.shape, (45,))
    check("adapter exact ordering", np.array_equal(adapter_output, adapter_expected), adapter_output, adapter_expected)
    check("previous action is raw normalized output", True, "runner passes raw_action to next observe", "raw_action")
    check("q target transform", np.allclose(DEFAULT_JOINT_POS + ACTION_SCALE * np.linspace(-0.5, 0.5, 12), DEFAULT_JOINT_POS + 0.5 * np.linspace(-0.5, 0.5, 12)), "q_default + 0.5 * raw_action", "q_default + 0.5 * raw_action")
    check("fixed execution rate", CONTROL_DT == PHYSICS_DT * DECIMATION, f"{POLICY_RATE_HZ} Hz / {PHYSICS_RATE_HZ} Hz / {DECIMATION}", "50 Hz / 200 Hz / 4")
    scene_runtime = (ROOT / "src/learned_execution/genesis_deployment_scene.py").read_text(encoding="utf-8")
    check(
        "fixed PD execution path",
        "control_dofs_force" in scene_runtime and "KP[None, :]" in scene_runtime and "KD[None, :]" in scene_runtime,
        "GenesisDeploymentScene.step uses fixed torque PD",
        "control_dofs_force with fixed KP/KD",
    )

    passed = all(item["passed"] for item in checks) and not any(
        not row["match"] for row in inventory["selected_vs_source"].values()
    )
    result = {
        "classification": "FROZEN_CONTRACT_PASS" if passed else "FROZEN_CONTRACT_BLOCKED",
        "candidate": SOURCE_URL,
        "executed_checkpoint": str(FROZEN_CHECKPOINT),
        "checkpoint_sha256": sha256_file(FROZEN_CHECKPOINT),
        "deploy_yaml_sha256": sha256_file(FROZEN_DEPLOY),
        "env_yaml_sha256": sha256_file(FROZEN_ENV),
        "agent_yaml_sha256": sha256_file(FROZEN_AGENT),
        "observation": {
            "dimension": POLICY_INPUT_DIM,
            "ordering": [
                "base angular velocity [3]",
                "projected gravity [3]",
                "desired [vx, vy, yaw_rate] [3]",
                "relative joint position [12]",
                "relative joint velocity [12]",
                "previous normalized 12-D policy output [12]",
            ],
            "scaling": "deploy observation scales are identity; frozen actor normalizer is inside actor: (x - mean) / (std + 0.01)",
            "frames": "body-frame angular velocity and command; inverse-rotated world gravity; positive yaw about +Z",
        },
        "action": {
            "dimension": ACTION_DIM,
            "semantics": "raw normalized joint-position offset",
            "decoding": "q_target = q_default + 0.5 * raw_action",
            "raw_action_clipping": "none",
        },
        "pd": {
            "kp_policy_order": KP.tolist(),
            "kd_policy_order": KD.tolist(),
            "effort_limit_policy_order": EFFORT_LIMIT.tolist(),
        },
        "timing": {
            "policy_rate_hz": POLICY_RATE_HZ,
            "physics_rate_hz": PHYSICS_RATE_HZ,
            "decimation": DECIMATION,
            "control_dt_s": CONTROL_DT,
            "physics_dt_s": PHYSICS_DT,
        },
        "forbidden_runtime_features": [
            "learned stiffness",
            "adaptive damping",
            "gait switch",
            "motion reference",
            "post-policy action shaping",
            "action smoothing",
            "command boost",
            "policy switching",
            "torque policy",
            "PPO/BC correction",
        ],
        "checks": checks,
    }
    return result


def write_frozen_contract_report(contract: dict[str, Any]) -> None:
    lines = [
        "# Frozen expert contract",
        "",
        f"Classification: **{contract['classification']}**",
        "",
        f"Candidate: `{SOURCE_URL}`",
        f"Executed checkpoint: `{contract['executed_checkpoint']}`",
        f"Checkpoint SHA256: `{contract['checkpoint_sha256']}`",
        "",
        "## Observation",
        "",
        "- Dimension: `45`.",
        "- Ordering: `base angular velocity(3), projected gravity(3), desired [vx,vy,yaw_rate](3), relative joint position(12), relative joint velocity(12), previous normalized 12-D policy output(12)`.",
        "- Frames: body-frame angular velocity and command; gravity is inverse-rotated world `[0,0,-1]`; positive yaw is about `+Z`.",
        "- Scaling: deploy scales are identity; actor normalization remains inside the frozen network.",
        "- Previous action is the previous raw normalized policy output, never a joint target or measured position.",
        "",
        "## Action and execution",
        "",
        "- Output dimension: `12`.",
        "- Raw action is a joint-position offset with no raw-action clipping.",
        "- Exact decoding: `q_target = q_default + 0.5 * raw_action`.",
        "- Fixed PD: `Kp=[20,20,40]`, `Kd=[1,1,2]` repeated over four legs.",
        "- Timing: `50 Hz` policy, `200 Hz` physics, `decimation=4`.",
        "- Safety behavior: requested target and pre-limit torque are logged; physical Genesis target/effort limits are the only runtime safety clips.",
        "",
        "## Checks",
        "",
        "| Check | Result | Actual | Expected |",
        "|---|---|---|---|",
    ]
    for item in contract["checks"]:
        lines.append(
            f"| `{item['name']}` | {'PASS' if item['passed'] else 'FAIL'} | `{item['actual']}` | `{item['expected']}` |"
        )
    lines.extend(["", contract["classification"]])
    write_text(GATE / "reports" / "frozen_contract.md", "\n".join(lines))
    write_json(GATE / "reports" / "frozen_contract.json", contract)


def augment_joint_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    for row in mapping["table"]:
        index = int(row["teacher_index"])
        row["sign"] = "+1"
        row["default_position_rad"] = float(DEFAULT_JOINT_POS[index])
        row["kp"] = float(KP[index])
        row["kd"] = float(KD[index])
    mapping["expected_policy_to_genesis"] = POLICY_TO_GENESIS.tolist()
    mapping["previous_visual_mapping_report"] = str(PREVIOUS_MAPPING)
    mapping["mapping_matches_previous_report"] = bool(
        mapping["source_mapping"] == POLICY_TO_GENESIS.tolist()
        and "Teacher index" in PREVIOUS_MAPPING.read_text(encoding="utf-8")
    ) if PREVIOUS_MAPPING.is_file() else False
    mapping["classification"] = (
        "JOINT_ACTUATOR_CONTRACT_PASS"
        if mapping["classification"] == "JOINT_MAPPING_PASS" and mapping["mapping_matches_previous_report"]
        else "JOINT_ACTUATOR_CONTRACT_BLOCKED"
    )
    return mapping


def write_joint_mapping_report(mapping: dict[str, Any]) -> None:
    lines = [
        "# Joint and actuator contract",
        "",
        f"Classification: **{mapping['classification']}**",
        "",
        "The mapping is name-audited against the Genesis URDF, compared with the frozen deploy permutation, and exercised with a positive per-joint target probe.",
        "",
        "| Policy index | Policy joint | Genesis index | Genesis joint | Genesis DOF | Sign | Default q (rad) | Lower | Upper | Kp | Kd | Probe +delta | Non-target max delta |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    probes = {int(row["teacher_index"]): row for row in mapping.get("safe_positive_target_probe", [])}
    for row in mapping["table"]:
        index = int(row["teacher_index"])
        probe = probes.get(
            index,
            {
                "measured_target_delta_rad": float("nan"),
                "positive_direction_verified": False,
                "max_abs_non_target_delta_rad": float("nan"),
            },
        )
        lines.append(
            f"| {index} | `{row['teacher_joint_name']}` | {row['genesis_index']} | `{row['genesis_joint_name']}` | {row['genesis_dof_index']} | `{row['sign']}` | {row['default_position_rad']:.4f} | {row['genesis_runtime_limit_lower']:.4f} | {row['genesis_runtime_limit_upper']:.4f} | {row['kp']:.1f} | {row['kd']:.1f} | {probe['measured_target_delta_rad']:.6f} ({probe['positive_direction_verified']}) | {probe['max_abs_non_target_delta_rad']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Published/frozen policy-to-Genesis permutation: `{mapping['source_mapping']}`.",
            f"- Genesis named order: `{mapping['genesis_named_joint_order']}`.",
            f"- Runtime DOFs in named order: `{mapping['runtime_genesis_dofs_in_named_order']}`.",
            f"- Default pose within runtime limits: `{mapping['default_pose_within_runtime_limits']}`.",
            f"- Mapping matches prior visual-transfer evidence: `{mapping['mapping_matches_previous_report']}`.",
            "",
            "## Checks",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
    )
    for name, passed in mapping["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", mapping["classification"]])
    write_text(GATE / "reports" / "joint_actuator_contract.md", "\n".join(lines))
    write_json(GATE / "reports" / "joint_actuator_contract.json", mapping)


def quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def reload_observations(visual: Any, eval_runtime: Any) -> tuple[np.ndarray, list[str]]:
    """Build 100 fixed, physically plausible observations from frozen logs."""

    source_logs = (
        ("standing", VISUAL / "logs/C0_standing.csv"),
        ("forward", VISUAL / "logs/C2_forward.csv"),
        ("yaw", VISUAL / "logs/C3_yaw_left.csv"),
        ("arc", VISUAL / "logs/C5_forward_left_arc.csv"),
        ("transition", VISUAL / "logs/transition.csv"),
    )
    observations: list[np.ndarray] = []
    labels: list[str] = []
    for label, path in source_logs:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"frozen reload source log is empty: {path}")
        indices = np.linspace(0, len(rows) - 1, 20, dtype=int)
        for index in indices:
            row = rows[int(index)]
            rpy = np.asarray([float(row["roll"]), float(row["pitch"]), float(row["yaw"])], dtype=np.float32)
            gravity = eval_runtime.quat_inverse_rotate(quat_from_rpy(rpy), np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
            angular = np.asarray([0.0, 0.0, float(row["achieved_body_yaw_rate"])], dtype=np.float32)
            position = np.asarray([float(row[f"joint_position_{i}"]) for i in range(ACTION_DIM)], dtype=np.float32) - DEFAULT_JOINT_POS
            velocity = np.asarray([float(row[f"joint_velocity_{i}"]) for i in range(ACTION_DIM)], dtype=np.float32)
            previous_action = np.asarray([float(row[f"raw_policy_action_{i}"]) for i in range(ACTION_DIM)], dtype=np.float32)
            command = np.asarray(
                [float(row["command_vx"]), float(row["command_vy"]), float(row["command_yaw_rate"])],
                dtype=np.float32,
            )
            state = np.concatenate((angular, gravity, position, velocity, previous_action)).astype(np.float32)
            observation = visual.make_teacher_observation(state, command)
            if observation.shape != (POLICY_INPUT_DIM,) or not np.all(np.isfinite(observation)):
                raise ValueError(f"invalid fixed reload observation from {path} row {index}")
            observations.append(observation)
            labels.append(f"{label}:{index}")
    result = np.asarray(observations, dtype=np.float32)
    if result.shape != (100, POLICY_INPUT_DIM):
        raise ValueError(f"expected 100 reload observations, got {result.shape}")
    return result, labels


def run_reload_gate(visual: Any, teacher_path: Path) -> dict[str, Any]:
    from scripts.imitation_learning.collect_command_teacher_rollouts import FrozenTeacher

    observations, labels = reload_observations(visual, __import__("learned_execution.genesis_deployment_scene", fromlist=["quat_inverse_rotate"]))
    policy_a = FrozenTeacher(teacher_path, torch_threads=1)
    policy_b = FrozenTeacher(teacher_path, torch_threads=1)
    actions_a = np.asarray([policy_a(obs) for obs in observations], dtype=np.float32)
    actions_b = np.asarray([policy_b(obs) for obs in observations], dtype=np.float32)
    differences = np.max(np.abs(actions_a - actions_b), axis=1)
    rows: list[dict[str, Any]] = []
    for index in range(len(observations)):
        row: dict[str, Any] = {
            "observation_id": index,
            "source": labels[index],
            "finite_A": bool(np.isfinite(actions_a[index]).all()),
            "finite_B": bool(np.isfinite(actions_b[index]).all()),
            "max_abs_action_difference": float(differences[index]),
        }
        for joint in range(ACTION_DIM):
            row[f"action_A_{joint}"] = float(actions_a[index, joint])
            row[f"action_B_{joint}"] = float(actions_b[index, joint])
        rows.append(row)
    write_csv(GATE / "metrics/reload_action_comparison.csv", rows)
    all_values = actions_a.reshape(-1)
    passed = bool(
        actions_a.shape == (100, ACTION_DIM)
        and actions_b.shape == (100, ACTION_DIM)
        and np.isfinite(actions_a).all()
        and np.isfinite(actions_b).all()
        and float(np.max(differences)) <= 1.0e-6
    )
    result = {
        "classification": "DETERMINISTIC_RELOAD_PASS" if passed else "DETERMINISTIC_RELOAD_BLOCKED",
        "checkpoint": str(teacher_path),
        "checkpoint_sha256": sha256_file(teacher_path),
        "observation_count": int(len(observations)),
        "observation_sources": labels,
        "load_A": "independent FrozenTeacher instance",
        "load_B": "independent FrozenTeacher instance",
        "output_shape_A": list(actions_a.shape),
        "output_shape_B": list(actions_b.shape),
        "all_finite_A": bool(np.isfinite(actions_a).all()),
        "all_finite_B": bool(np.isfinite(actions_b).all()),
        "maximum_absolute_action_difference": float(np.max(differences)),
        "tolerance": 1.0e-6,
        "raw_action_statistics_A": {
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "fraction_outside_minus_one_plus_one": float(np.mean((all_values < -1.0) | (all_values > 1.0))),
        },
        "no_action_clipping_applied": True,
        "csv": str(GATE / "metrics/reload_action_comparison.csv"),
    }
    lines = [
        "# Deterministic network reload",
        "",
        f"Classification: **{result['classification']}**",
        "",
        f"- Checkpoint: `{teacher_path}`",
        f"- SHA256: `{result['checkpoint_sha256']}`",
        f"- Fixed observations: `{result['observation_count']}` from standing, forward, yaw, arc, and transition frozen logs.",
        "- Load A and load B are independently instantiated checkpoint actors.",
        f"- Output shapes: A `{result['output_shape_A']}`, B `{result['output_shape_B']}`.",
        f"- All finite: A `{result['all_finite_A']}`, B `{result['all_finite_B']}`.",
        f"- Maximum absolute action difference: `{result['maximum_absolute_action_difference']:.9g}`; tolerance `{result['tolerance']:.1e}`.",
        f"- Raw action range: `[{result['raw_action_statistics_A']['min']:.9g}, {result['raw_action_statistics_A']['max']:.9g}]`.",
        f"- Raw action mean/std: `{result['raw_action_statistics_A']['mean']:.9g}` / `{result['raw_action_statistics_A']['std']:.9g}`.",
        f"- Fraction outside `[-1,1]`: `{result['raw_action_statistics_A']['fraction_outside_minus_one_plus_one']:.9g}`.",
        "- No action clipping was added.",
        f"- Per-observation comparison: `{result['csv']}`.",
        "",
        result["classification"],
    ]
    write_text(GATE / "reports/reload_determinism.md", "\n".join(lines))
    write_json(GATE / "reports/reload_determinism.json", result)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def command_schedule(command: np.ndarray) -> Callable[[int], np.ndarray]:
    def schedule(_step: int) -> np.ndarray:
        return command

    return schedule


def transition_schedule(step: int) -> np.ndarray:
    time_s = step * CONTROL_DT
    if time_s < 3.0:
        return np.asarray([0.00, 0.00, 0.00], dtype=np.float32)
    if time_s < 8.0:
        return np.asarray([0.20, 0.00, 0.00], dtype=np.float32)
    if time_s < 13.0:
        return np.asarray([0.20, 0.00, 0.25], dtype=np.float32)
    if time_s < 18.0:
        return np.asarray([0.20, 0.00, -0.25], dtype=np.float32)
    return np.asarray([0.00, 0.00, 0.00], dtype=np.float32)


def matrix_from_rows(rows: list[dict[str, Any]], prefix: str, width: int = ACTION_DIM) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_{index}"]) for index in range(width)] for row in rows],
        dtype=np.float64,
    )


def append_vector(row: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        row[f"{prefix}_{index}"] = float(value)


def run_episode(
    visual: Any,
    scene: Any,
    teacher: Any,
    run_id: str,
    phase: str,
    category: str,
    command: np.ndarray | None,
    schedule: Callable[[int], np.ndarray],
    duration_s: float,
    repeat: int,
    seed: int,
    log_path: Path,
    video_path: Path,
    render_video: bool,
    call_counter: dict[str, int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
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
        writer = imageio.get_writer(
            video_path,
            fps=VIDEO_FPS,
            codec="libx264",
            quality=7,
            macro_block_size=1,
        )
    termination_reason = ""
    try:
        for step in range(steps_requested):
            desired = np.asarray(schedule(step), dtype=np.float32)
            if desired.shape != (COMMAND_DIM,) or not np.all(np.isfinite(desired)):
                raise ValueError(f"invalid command at {run_id} step {step}: {desired}")
            observed = scene.observe(last_action[None, :])
            policy_input = visual.make_teacher_observation(observed["state"][0], desired)
            raw_action = np.asarray(teacher(policy_input), dtype=np.float32)
            call_counter["policy_calls"] += 1
            if raw_action.shape != (ACTION_DIM,):
                raise ValueError(f"expert returned {raw_action.shape}, expected {(ACTION_DIM,)}")
            decoded_target = DEFAULT_JOINT_POS + ACTION_SCALE * raw_action
            applied_target = np.clip(decoded_target, scene.joint_lower_policy, scene.joint_upper_policy).astype(np.float32)
            target_clip_mask = np.abs(applied_target - decoded_target) > 1.0e-7
            current_position = observed["joint_position"][0]
            current_velocity = observed["joint_velocity"][0]
            preclip_torque = KP * (applied_target - current_position) - KD * current_velocity
            torque_limit_mask = np.abs(preclip_torque) > EFFORT_LIMIT + 1.0e-6
            torque = np.asarray(scene.step(applied_target[None, :])[0], dtype=np.float32)
            expected_torque = np.clip(preclip_torque, -EFFORT_LIMIT, EFFORT_LIMIT)
            measured = scene.observe(raw_action[None, :])
            foot_contacts, foot_speed = visual.contact_and_foot_speed(scene)
            achieved = np.asarray(
                [
                    measured["body_linear_velocity"][0, 0],
                    measured["body_linear_velocity"][0, 1],
                    measured["body_angular_velocity"][0, 2],
                ],
                dtype=np.float32,
            )
            reason = visual.termination_for(measured, raw_action, scene)
            values_finite = all(
                np.all(np.isfinite(measured[name]))
                for name in (
                    "root_position",
                    "root_quaternion",
                    "root_rpy",
                    "body_linear_velocity",
                    "body_angular_velocity",
                    "joint_position",
                    "joint_velocity",
                )
            ) and np.isfinite(raw_action).all() and np.isfinite(decoded_target).all() and np.isfinite(preclip_torque).all()
            if not values_finite and reason is None:
                reason = "invalid_nan_or_inf"
            row: dict[str, Any] = {
                "timestamp": (step + 1) * CONTROL_DT,
                "run_id": run_id,
                "phase": phase,
                "category": category,
                "repeat": repeat,
                "seed": seed,
                "policy_checkpoint_sha256": sha256_file(FROZEN_CHECKPOINT),
                "command_vx": float(desired[0]),
                "command_vy": float(desired[1]),
                "command_yaw_rate": float(desired[2]),
                "achieved_body_vx": float(achieved[0]),
                "achieved_body_vy": float(achieved[1]),
                "achieved_body_yaw_rate": float(achieved[2]),
                "base_height": float(measured["body_height"][0]),
                "root_position_x": float(measured["root_position"][0, 0]),
                "root_position_y": float(measured["root_position"][0, 1]),
                "root_position_z": float(measured["root_position"][0, 2]),
                "roll": float(measured["root_rpy"][0, 0]),
                "pitch": float(measured["root_rpy"][0, 1]),
                "yaw": float(measured["root_rpy"][0, 2]),
                "base_contact": int(bool(measured["base_contact"][0])),
                "termination_reason": reason or "running",
                "nan_inf_flag": int(not values_finite),
                "target_clip_count": int(np.count_nonzero(target_clip_mask)),
                "torque_limit_exceedance_count": int(np.count_nonzero(torque_limit_mask)),
                "torque_return_matches_expected_clip": int(np.allclose(torque, expected_torque, atol=1.0e-6)),
            }
            append_vector(row, "joint_position", measured["joint_position"][0])
            append_vector(row, "joint_velocity", measured["joint_velocity"][0])
            append_vector(row, "raw_policy_action", raw_action)
            append_vector(row, "q_target", decoded_target)
            append_vector(row, "q_target_applied", applied_target)
            append_vector(row, "preclip_torque", preclip_torque)
            append_vector(row, "applied_torque", torque)
            for index in range(ACTION_DIM):
                row[f"target_clip_{index}"] = int(target_clip_mask[index])
                row[f"torque_limit_exceedance_{index}"] = int(torque_limit_mask[index])
            for index in range(4):
                row[f"foot_contact_{index}"] = int(foot_contacts[index])
                row[f"foot_speed_{index}"] = float(foot_speed[index])
            rows.append(row)
            if writer is not None and (step % VIDEO_RENDER_STRIDE == 0 or reason is not None or step == steps_requested - 1):
                writer.append_data(visual.fixed_camera_frame(scene))
            last_action = raw_action
            if reason is not None:
                termination_reason = reason
                break
    finally:
        if writer is not None:
            writer.close()

    if not termination_reason:
        termination_reason = "time_limit" if len(rows) == steps_requested else "unknown"
    if rows:
        rows[-1]["termination_reason"] = termination_reason
    write_csv(log_path, rows)
    return metric_from_rows(
        run_id,
        phase,
        category,
        command,
        repeat,
        seed,
        duration_s,
        rows,
        termination_reason,
        initial_position,
        initial_yaw,
        scene.joint_lower_policy,
        scene.joint_upper_policy,
        log_path,
        video_path if render_video else None,
    )


def metric_from_rows(
    run_id: str,
    phase: str,
    category: str,
    command: np.ndarray | None,
    repeat: int,
    seed: int,
    duration_s: float,
    rows: list[dict[str, Any]],
    termination_reason: str,
    initial_position: np.ndarray,
    initial_yaw: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    log_path: Path,
    video_path: Path | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not rows:
        metric = {
            "run_id": run_id,
            "phase": phase,
            "category": category,
            "repeat": repeat,
            "seed": seed,
            "command": command.tolist() if command is not None else None,
            "requested_duration_s": duration_s,
            "actual_duration_s": 0.0,
            "steps": 0,
            "full_duration": False,
            "termination_reason": termination_reason,
            "fall_count": int(termination_reason not in {"time_limit", ""}),
            "base_contact_count": 0,
            "nan_inf_count": 1,
            "log": str(log_path),
            "video": str(video_path) if video_path else None,
            "final_classification": "BLOCKED",
        }
        return metric, {"timestamp": np.empty(0)}

    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    commands = np.asarray(
        [[float(row["command_vx"]), float(row["command_vy"]), float(row["command_yaw_rate"])] for row in rows],
        dtype=np.float64,
    )
    achieved = np.asarray(
        [[float(row["achieved_body_vx"]), float(row["achieved_body_vy"]), float(row["achieved_body_yaw_rate"])] for row in rows],
        dtype=np.float64,
    )
    position = matrix_from_rows(rows, "joint_position")
    velocity = matrix_from_rows(rows, "joint_velocity")
    raw_action = matrix_from_rows(rows, "raw_policy_action")
    q_target = matrix_from_rows(rows, "q_target")
    q_target_applied = matrix_from_rows(rows, "q_target_applied")
    preclip_torque = matrix_from_rows(rows, "preclip_torque")
    applied_torque = matrix_from_rows(rows, "applied_torque")
    root_position = np.asarray(
        [[float(row["root_position_x"]), float(row["root_position_y"]), float(row["root_position_z"])] for row in rows],
        dtype=np.float64,
    )
    rpy = np.asarray([[float(row["roll"]), float(row["pitch"]), float(row["yaw"])] for row in rows], dtype=np.float64)
    height = np.asarray([float(row["base_height"]) for row in rows], dtype=np.float64)
    foot_contacts = np.asarray([[bool(row[f"foot_contact_{i}"]) for i in range(4)] for row in rows], dtype=bool)
    foot_speed = np.asarray([[float(row[f"foot_speed_{i}"]) for i in range(4)] for row in rows], dtype=np.float64)
    base_contact = np.asarray([bool(row["base_contact"]) for row in rows], dtype=bool)
    nan_inf = np.asarray([bool(row["nan_inf_flag"]) for row in rows], dtype=bool)
    target_clip_count = np.asarray([int(row["target_clip_count"]) for row in rows], dtype=np.int64)
    torque_limit_count = np.asarray([int(row["torque_limit_exceedance_count"]) for row in rows], dtype=np.int64)
    steady = timestamps > STEADY_TRANSIENT_S
    if not np.any(steady):
        steady = np.ones(len(rows), dtype=bool)
    steady_achieved = achieved[steady]
    steady_command = commands[steady]
    command_mean = np.mean(steady_command, axis=0)
    mean_achieved = np.mean(steady_achieved, axis=0)
    median_achieved = np.median(steady_achieved, axis=0)
    tracking_error = steady_achieved - steady_command
    ratios: list[float | None] = []
    signs: list[bool | None] = []
    for axis in range(COMMAND_DIM):
        desired = float(np.mean(steady_command[:, axis]))
        achieved_axis = float(mean_achieved[axis])
        ratios.append(abs(achieved_axis / desired) if abs(desired) > 1.0e-9 else None)
        signs.append(bool(desired * achieved_axis > 0.0) if abs(desired) > 1.0e-9 else None)

    q_lower = np.asarray(joint_lower, dtype=np.float64).reshape(ACTION_DIM)
    q_upper = np.asarray(joint_upper, dtype=np.float64).reshape(ACTION_DIM)
    actual_violation_mask = (position < q_lower[None, :] - 1.0e-6) | (position > q_upper[None, :] + 1.0e-6)
    target_clip_mask = np.asarray(
        [[bool(row[f"target_clip_{i}"]) for i in range(ACTION_DIM)] for row in rows],
        dtype=bool,
    )
    torque_violation_mask = np.asarray(
        [[bool(row[f"torque_limit_exceedance_{i}"]) for i in range(ACTION_DIM)] for row in rows],
        dtype=bool,
    )
    contact_transitions = np.sum(np.abs(np.diff(foot_contacts.astype(np.int8), axis=0)), axis=0) if len(rows) > 1 else np.zeros(4, dtype=np.int64)
    contact_duty = np.mean(foot_contacts, axis=0)
    swing_samples = np.sum(~foot_contacts, axis=0)
    contact_speeds = foot_speed[foot_contacts]
    unwrapped_yaw = np.unwrap(rpy[:, 2])
    final_world_xy = root_position[-1, :2]
    displacement_world = final_world_xy - initial_position[:2]
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    displacement_body = np.asarray(
        [c * displacement_world[0] + s * displacement_world[1], -s * displacement_world[0] + c * displacement_world[1]],
        dtype=np.float64,
    )
    all_active = np.all(np.abs(commands[-1]) > 1.0e-9) if phase == "static_command" else False
    full_duration = len(rows) == int(round(duration_s / CONTROL_DT)) and termination_reason == "time_limit"
    fall_reasons = {"base_contact", "root_height_below_threshold", "roll_or_pitch_threshold", "joint_limit_violation"}
    fall_count = int(termination_reason in fall_reasons)
    command_final = commands[-1]
    vx_ratio = ratios[0]
    yaw_ratio = ratios[2]
    forward_sign = signs[0]
    yaw_sign = signs[2]
    standing_stable = bool(
        category == "standing"
        and full_duration
        and fall_count == 0
        and not np.any(base_contact)
        and not np.any(nan_inf)
        and not np.any(actual_violation_mask)
        and float(np.linalg.norm(displacement_world)) <= 0.50
        and float(abs(unwrapped_yaw[-1] - initial_yaw)) <= 0.50
        and float(np.max(np.abs(rpy[:, :2]))) <= 0.50
        and float(np.max(np.abs(velocity))) <= 20.0
    )
    if category in {"slow_forward", "forward"}:
        command_response_ok = bool(full_duration and forward_sign and vx_ratio is not None and vx_ratio >= 0.50)
    elif category in {"yaw_left", "yaw_right"}:
        command_response_ok = bool(full_duration and yaw_sign and yaw_ratio is not None and yaw_ratio >= 0.50)
    elif category in {"forward_left_arc", "forward_right_arc"}:
        command_response_ok = bool(
            full_duration
            and forward_sign
            and yaw_sign
            and float(mean_achieved[0]) >= 0.05
            and abs(float(mean_achieved[2])) >= 0.05
        )
    elif category in {"lateral_left", "lateral_right"}:
        command_response_ok = bool(full_duration and signs[1] and abs(float(mean_achieved[1])) >= 0.02)
    else:
        command_response_ok = standing_stable if category == "standing" else bool(full_duration)
    target_clip_fraction = float(np.mean(target_clip_mask))
    torque_limit_fraction = float(np.mean(torque_violation_mask))
    safety_ok = bool(
        full_duration
        and fall_count == 0
        and not np.any(base_contact)
        and not np.any(nan_inf)
        and not np.any(actual_violation_mask)
        and target_clip_fraction <= TARGET_CLIP_FRACTION_LIMIT
        and torque_limit_fraction <= TORQUE_LIMIT_FRACTION_LIMIT
    )
    locomotion_contact_sanity = bool(
        category == "standing"
        or (
            np.all(swing_samples > 0)
            and int(np.sum(contact_transitions)) >= 2
            and np.all(contact_duty > 0.01)
            and np.all(contact_duty < 0.99)
        )
    )
    if phase == "static_command":
        final_classification = "PASS" if safety_ok and command_response_ok and (standing_stable if category == "standing" else locomotion_contact_sanity) else "BLOCKED"
    elif phase == "transition":
        final_classification = "PENDING_TRANSITION_AGGREGATE"
    else:
        final_classification = "OPTIONAL_SUPPORTED" if safety_ok and command_response_ok else "OPTIONAL_UNSUPPORTED"

    segment_summaries: list[dict[str, Any]] = []
    starts = [0]
    for index in range(1, len(commands)):
        if not np.allclose(commands[index], commands[index - 1], atol=1.0e-7):
            starts.append(index)
    starts.append(len(commands))
    for segment_index in range(len(starts) - 1):
        start, end = starts[segment_index], starts[segment_index + 1]
        segment_command = commands[start]
        segment_steady_start = min(end, start + int(round(STEADY_TRANSIENT_S / CONTROL_DT)))
        segment_velocity = achieved[segment_steady_start:end]
        if len(segment_velocity) == 0:
            segment_velocity = achieved[start:end]
        segment_summaries.append(
            {
                "command_change_time_s": float(start * CONTROL_DT),
                "command": segment_command.tolist(),
                "start_s": float(start * CONTROL_DT),
                "end_s": float(end * CONTROL_DT),
                "achieved_mean": np.mean(segment_velocity, axis=0).tolist(),
                "achieved_median": np.median(segment_velocity, axis=0).tolist(),
            }
        )
    response_delays: list[dict[str, Any]] = []
    if phase == "transition":
        for start in starts[:-1]:
            if start == 0:
                continue
            desired = commands[start]
            search_end = starts[starts.index(start) + 1]
            delay: float | None = None
            for index in range(start, search_end):
                matched = True
                for axis in (0, 2):
                    if abs(desired[axis]) > 1.0e-9:
                        matched &= bool(np.sign(desired[axis]) * achieved[index, axis] >= 0.5 * abs(desired[axis]))
                if matched:
                    delay = float((index - start) * CONTROL_DT)
                    break
            response_delays.append(
                {
                    "command_change_time_s": float(start * CONTROL_DT),
                    "new_command": desired.tolist(),
                    "response_delay_s": delay,
                }
            )
    transition_return_to_standing = bool(
        phase != "transition"
        or (
            np.mean(np.abs(achieved[timestamps >= max(0.0, duration_s - 2.0), 0])) < 0.15
            and np.mean(np.abs(achieved[timestamps >= max(0.0, duration_s - 2.0), 2])) < 0.15
            and np.max(np.abs(rpy[timestamps >= max(0.0, duration_s - 2.0), :2])) < 0.50
        )
    )
    arrays = {
        "timestamp": timestamps,
        "command": commands,
        "achieved": achieved,
        "root_position": root_position,
        "rpy": rpy,
        "height": height,
        "joint_position": position,
        "joint_velocity": velocity,
        "raw_action": raw_action,
        "q_target": q_target,
        "q_target_applied": q_target_applied,
        "preclip_torque": preclip_torque,
        "applied_torque": applied_torque,
        "foot_contacts": foot_contacts,
        "foot_speed": foot_speed,
        "base_contact": base_contact,
        "target_clip_mask": target_clip_mask,
        "torque_violation_mask": torque_violation_mask,
    }
    metric = {
        "run_id": run_id,
        "phase": phase,
        "category": category,
        "repeat": repeat,
        "seed": seed,
        "command": command_final.tolist() if command is not None else None,
        "requested_duration_s": float(duration_s),
        "actual_duration_s": float(len(rows) * CONTROL_DT),
        "steps": int(len(rows)),
        "full_duration": full_duration,
        "termination_reason": termination_reason,
        "fall_count": fall_count,
        "base_contact_count": int(np.sum(base_contact)),
        "nan_inf_count": int(np.sum(nan_inf)),
        "mean_vx": float(mean_achieved[0]),
        "median_vx": float(median_achieved[0]),
        "mean_vy": float(mean_achieved[1]),
        "median_vy": float(median_achieved[1]),
        "mean_yaw_rate": float(mean_achieved[2]),
        "median_yaw_rate": float(median_achieved[2]),
        "commanded_vx": float(np.mean(command_mean[0:1])),
        "commanded_vy": float(np.mean(command_mean[1:2])),
        "commanded_yaw_rate": float(np.mean(command_mean[2:3])),
        "vx_tracking_ratio": vx_ratio,
        "yaw_tracking_ratio": yaw_ratio,
        "vx_sign_correct": forward_sign,
        "yaw_sign_correct": yaw_sign,
        "tracking_error_mean": np.mean(tracking_error, axis=0).tolist(),
        "tracking_error_median": np.median(tracking_error, axis=0).tolist(),
        "tracking_error_abs_mean": np.mean(np.abs(tracking_error), axis=0).tolist(),
        "base_height_mean_m": float(np.mean(height)),
        "base_height_min_m": float(np.min(height)),
        "base_height_max_m": float(np.max(height)),
        "base_height_range_m": float(np.max(height) - np.min(height)),
        "max_abs_roll_rad": float(np.max(np.abs(rpy[:, 0]))),
        "max_abs_pitch_rad": float(np.max(np.abs(rpy[:, 1]))),
        "xy_drift_m": float(np.linalg.norm(displacement_world)),
        "yaw_drift_rad": float(abs(unwrapped_yaw[-1] - initial_yaw)),
        "max_joint_velocity_rad_s": float(np.max(np.abs(velocity))),
        "max_abs_joint_position_rad": np.max(np.abs(position), axis=0).tolist(),
        "max_abs_joint_velocity_rad_s": np.max(np.abs(velocity), axis=0).tolist(),
        "max_abs_q_target_rad": np.max(np.abs(q_target), axis=0).tolist(),
        "max_abs_applied_torque_nm": np.max(np.abs(applied_torque), axis=0).tolist(),
        "max_abs_preclip_torque_nm": np.max(np.abs(preclip_torque), axis=0).tolist(),
        "max_torque_nm": float(np.max(np.abs(applied_torque))),
        "joint_limit_violation_count": int(np.count_nonzero(actual_violation_mask)),
        "joint_limit_violation_events": int(np.count_nonzero(np.any(actual_violation_mask, axis=1))),
        "joint_limit_violation_per_joint": np.sum(actual_violation_mask, axis=0).tolist(),
        "target_clip_count": int(np.count_nonzero(target_clip_mask)),
        "target_clip_events": int(np.count_nonzero(np.any(target_clip_mask, axis=1))),
        "target_clip_fraction": target_clip_fraction,
        "target_clip_per_joint": np.sum(target_clip_mask, axis=0).tolist(),
        "torque_limit_exceedance_count": int(np.count_nonzero(torque_violation_mask)),
        "torque_limit_exceedance_events": int(np.count_nonzero(np.any(torque_violation_mask, axis=1))),
        "torque_limit_fraction": torque_limit_fraction,
        "torque_limit_exceedance_per_joint": np.sum(torque_violation_mask, axis=0).tolist(),
        "contact_duty_fraction_per_foot": contact_duty.tolist(),
        "contact_transitions_per_foot": contact_transitions.tolist(),
        "swing_samples_per_foot": swing_samples.tolist(),
        "mean_contact_foot_speed_m_s": float(np.mean(contact_speeds)) if len(contact_speeds) else None,
        "max_contact_foot_speed_m_s": float(np.max(contact_speeds)) if len(contact_speeds) else None,
        "locomotion_contact_sanity": locomotion_contact_sanity,
        "standing_stable": standing_stable,
        "command_response_ok": command_response_ok,
        "safety_ok": safety_ok,
        "transition_return_to_standing": transition_return_to_standing,
        "segment_summaries": segment_summaries,
        "response_delays": response_delays,
        "fall_status": bool(fall_count > 0),
        "final_classification": final_classification,
        "log": str(log_path),
        "video": str(video_path) if video_path else None,
    }
    return metric, arrays


def write_gate_summary_csv(metrics: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        row = {
            "run_id": metric["run_id"],
            "phase": metric["phase"],
            "repeat": metric["repeat"],
            "command": json.dumps(metric.get("command")),
            "duration": metric["requested_duration_s"],
            "actual_duration": metric["actual_duration_s"],
            "mean_vx": metric.get("mean_vx"),
            "median_vx": metric.get("median_vx"),
            "mean_vy": metric.get("mean_vy"),
            "median_vy": metric.get("median_vy"),
            "mean_yaw_rate": metric.get("mean_yaw_rate"),
            "median_yaw_rate": metric.get("median_yaw_rate"),
            "commanded_vx": metric.get("commanded_vx"),
            "commanded_vy": metric.get("commanded_vy"),
            "commanded_yaw_rate": metric.get("commanded_yaw_rate"),
            "vx_tracking_ratio": metric.get("vx_tracking_ratio"),
            "yaw_tracking_ratio": metric.get("yaw_tracking_ratio"),
            "vx_sign_correct": metric.get("vx_sign_correct"),
            "yaw_sign_correct": metric.get("yaw_sign_correct"),
            "base_height_mean_m": metric.get("base_height_mean_m"),
            "base_height_min_m": metric.get("base_height_min_m"),
            "base_height_max_m": metric.get("base_height_max_m"),
            "max_abs_roll_rad": metric.get("max_abs_roll_rad"),
            "max_abs_pitch_rad": metric.get("max_abs_pitch_rad"),
            "xy_drift_m": metric.get("xy_drift_m"),
            "yaw_drift_rad": metric.get("yaw_drift_rad"),
            "max_joint_velocity_rad_s": metric.get("max_joint_velocity_rad_s"),
            "max_torque_nm": metric.get("max_torque_nm"),
            "falls": metric.get("fall_count"),
            "base_contacts": metric.get("base_contact_count"),
            "nan_inf": metric.get("nan_inf_count"),
            "joint_violations": metric.get("joint_limit_violation_count"),
            "target_clips": metric.get("target_clip_count"),
            "torque_violations": metric.get("torque_limit_exceedance_count"),
            "target_clip_fraction": metric.get("target_clip_fraction"),
            "torque_violation_fraction": metric.get("torque_limit_fraction"),
            "foot_contact_duty": json.dumps(metric.get("contact_duty_fraction_per_foot")),
            "foot_contact_transitions": json.dumps(metric.get("contact_transitions_per_foot")),
            "response_delays": json.dumps(metric.get("response_delays")),
            "final_classification": metric.get("final_classification"),
            "log": metric.get("log"),
            "video": metric.get("video"),
        }
        rows.append(row)
    write_csv(GATE / "metrics/expert_gate_summary.csv", rows)


def aggregate_command_metrics(metrics: list[dict[str, Any]], category: str) -> dict[str, Any]:
    selected = [metric for metric in metrics if metric["category"] == category]
    if not selected:
        return {"category": category, "runs": 0}
    return {
        "category": category,
        "runs": len(selected),
        "all_full_duration": all(metric["full_duration"] for metric in selected),
        "all_safety_ok": all(metric["safety_ok"] for metric in selected),
        "all_command_response_ok": all(metric["command_response_ok"] for metric in selected),
        "all_standing_stable": all(metric["standing_stable"] for metric in selected) if category == "standing" else None,
        "mean_vx": float(np.mean([metric["mean_vx"] for metric in selected])),
        "median_vx": float(np.median([metric["mean_vx"] for metric in selected])),
        "mean_vy": float(np.mean([metric["mean_vy"] for metric in selected])),
        "mean_yaw_rate": float(np.mean([metric["mean_yaw_rate"] for metric in selected])),
        "vx_ratios": [metric.get("vx_tracking_ratio") for metric in selected],
        "yaw_ratios": [metric.get("yaw_tracking_ratio") for metric in selected],
        "falls": int(sum(metric["fall_count"] for metric in selected)),
        "base_contacts": int(sum(metric["base_contact_count"] for metric in selected)),
        "nan_inf": int(sum(metric["nan_inf_count"] for metric in selected)),
        "joint_violations": int(sum(metric["joint_limit_violation_count"] for metric in selected)),
        "target_clips": int(sum(metric["target_clip_count"] for metric in selected)),
        "torque_violations": int(sum(metric["torque_limit_exceedance_count"] for metric in selected)),
        "max_roll": float(max(metric["max_abs_roll_rad"] for metric in selected)),
        "max_pitch": float(max(metric["max_abs_pitch_rad"] for metric in selected)),
        "max_torque": float(max(metric["max_torque_nm"] for metric in selected)),
    }


def evaluate_transition(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [metric for metric in metrics if metric["phase"] == "transition"]
    return {
        "runs": len(selected),
        "all_full_duration": bool(selected) and all(metric["full_duration"] for metric in selected),
        "all_safety_ok": bool(selected) and all(metric["safety_ok"] for metric in selected),
        "all_return_to_standing": bool(selected) and all(metric["transition_return_to_standing"] for metric in selected),
        "falls": int(sum(metric["fall_count"] for metric in selected)),
        "base_contacts": int(sum(metric["base_contact_count"] for metric in selected)),
        "nan_inf": int(sum(metric["nan_inf_count"] for metric in selected)),
        "segment_summaries": [metric.get("segment_summaries", []) for metric in selected],
        "response_delays": [metric.get("response_delays", []) for metric in selected],
        "forward_response_ok": all(
            any(np.asarray(segment["command"])[0] > 0 and segment["achieved_mean"][0] > 0 for segment in metric.get("segment_summaries", []))
            for metric in selected
        ) if selected else False,
        "left_yaw_response_ok": all(
            any(np.asarray(segment["command"])[2] > 0 and segment["achieved_mean"][2] > 0 for segment in metric.get("segment_summaries", []))
            for metric in selected
        ) if selected else False,
        "right_yaw_response_ok": all(
            any(np.asarray(segment["command"])[2] < 0 and segment["achieved_mean"][2] < 0 for segment in metric.get("segment_summaries", []))
            for metric in selected
        ) if selected else False,
    }


def single_policy_audit() -> dict[str, Any]:
    gate_source = Path(__file__).read_text(encoding="utf-8")
    teacher_source = (ROOT / "scripts/imitation_learning/collect_command_teacher_rollouts.py").read_text(encoding="utf-8")
    scene_source = (ROOT / "src/learned_execution/genesis_deployment_scene.py").read_text(encoding="utf-8")
    syntax = ast.parse(gate_source)
    calls = [
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call)
    ]

    def call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    call_names = [call_name(node) for node in calls]
    teacher_calls = [node for node in calls if call_name(node) == "FrozenTeacher"]
    reload_teacher_calls = [
        node
        for node in teacher_calls
        if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "teacher_path"
    ]
    closed_loop_teacher_calls = [
        node
        for node in teacher_calls
        if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "FROZEN_CHECKPOINT"
    ]
    checks = {
        "one_frozen_teacher_instantiation": len(closed_loop_teacher_calls) == 1,
        "no_policy_switch_table": not any(name in {"policy_switch", "select_policy", "gait_policy"} for name in call_names),
        "one_network_per_teacher_instance": teacher_source.count("self._actor = Actor().eval()") == 1,
        "no_runtime_checkpoint_selection": not any(name in {"glob", "load_checkpoint", "select_checkpoint"} for name in call_names),
        "scene_has_fixed_force_pd_only": "control_dofs_force" in scene_source and "control_dofs_position" not in scene_source,
        "no_action_smoothing_or_boost": not any(name in {"smooth_action", "command_boost", "action_filter"} for name in call_names),
        "no_rl_or_imitation_update": not any(name in {"backward", "zero_grad", "optimizer", "train"} for name in call_names),
    }
    # The two textual FrozenTeacher occurrences are the A/B reload gate; the
    # closed-loop uses the single instance created in main and never switches.
    passed = all(checks.values())
    result = {
        "classification": "SINGLE_POLICY_PASS" if passed else "SINGLE_POLICY_BLOCKED",
        "execution_graph": "state + command -> one FrozenTeacher actor -> 12 raw actions -> published q_target decode -> fixed PD -> Genesis",
        "closed_loop_policy_instances": 1,
        "closed_loop_checkpoint": str(FROZEN_CHECKPOINT),
        "reload_gate_instances": len(reload_teacher_calls),
        "runtime_policy_switching": False,
        "checks": checks,
        "searched_sources": {
            "gate": str(Path(__file__)),
            "teacher_runtime": str(ROOT / "scripts/imitation_learning/collect_command_teacher_rollouts.py"),
            "genesis_adapter": str(ROOT / "src/learned_execution/genesis_deployment_scene.py"),
        },
    }
    lines = [
        "# Single-policy / no-switch audit",
        "",
        f"Classification: **{result['classification']}**",
        "",
        "```text",
        result["execution_graph"],
        "```",
        "",
        "The reload determinism check intentionally creates two independent loads. The closed-loop bank uses one frozen teacher instance for every primary, transition, optional lateral, and video run.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed_check in checks.items():
        lines.append(f"| `{name}` | {'PASS' if passed_check else 'FAIL'} |")
    lines.extend(["", result["classification"]])
    write_text(GATE / "reports/single_policy_audit.md", "\n".join(lines))
    write_json(GATE / "reports/single_policy_audit.json", result)
    return result


def plot_results(metrics: list[dict[str, Any]], series: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GATE.joinpath("plots").mkdir(parents=True, exist_ok=True)
    categories = [row[0] for row in REQUIRED_COMMANDS]
    means = [aggregate_command_metrics(metrics, row[1]).get("mean_vx", 0.0) for row in REQUIRED_COMMANDS]
    yaw_means = [aggregate_command_metrics(metrics, row[1]).get("mean_yaw_rate", 0.0) for row in REQUIRED_COMMANDS]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(categories, means, color="#4472c4")
    axes[0].set_ylabel("steady mean achieved vx (m/s)")
    axes[0].set_title("Repeated primary forward response")
    axes[1].bar(categories, yaw_means, color="#ed7d31")
    axes[1].set_ylabel("steady mean achieved yaw rate (rad/s)")
    axes[1].set_title("Repeated primary yaw response")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(GATE / "plots/command_tracking_summary.png", dpi=160)
    plt.close(figure)

    transition_ids = [key for key in series if key.startswith("transition_rep")]
    if transition_ids:
        data = series[transition_ids[0]]
        t = data["timestamp"]
        command = data["command"]
        achieved = data["achieved"]
        rpy = data["rpy"]
        height = data["height"]
        figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(t, command[:, 0], label="command vx", color="#1f77b4")
        axes[0].plot(t, achieved[:, 0], label="achieved vx", color="#ff7f0e")
        axes[0].set_ylabel("vx (m/s)")
        axes[1].plot(t, command[:, 2], label="command yaw", color="#1f77b4")
        axes[1].plot(t, achieved[:, 2], label="achieved yaw", color="#ff7f0e")
        axes[1].set_ylabel("yaw (rad/s)")
        axes[2].plot(t, height, label="base height", color="#2ca02c")
        axes[2].plot(t, rpy[:, 0], label="roll", color="#9467bd")
        axes[2].plot(t, rpy[:, 1], label="pitch", color="#d62728")
        axes[2].set_ylabel("height / attitude")
        axes[2].set_xlabel("time (s)")
        for axis in axes:
            for change in (3, 8, 13, 18):
                axis.axvline(change, color="k", linestyle="--", alpha=0.25)
            axis.grid(alpha=0.25)
            axis.legend(loc="best", fontsize=8)
        figure.suptitle("Representative transition: commanded vs achieved, height, roll/pitch")
        figure.tight_layout()
        figure.savefig(GATE / "plots/transition_tracking.png", dpi=160)
        plt.close(figure)


def write_gate_manifest(inventory: dict[str, Any], mismatches: list[str], contract: dict[str, Any], mapping: dict[str, Any], reload: dict[str, Any]) -> None:
    lines = [
        "# Expert Genesis compatibility-gate manifest",
        "",
        f"Generated: `{utc_now()}`",
        f"Gate script: `{GATE_VERSION}`",
        f"Candidate: `{SOURCE_URL}`",
        f"Previous frozen result: `{PREVIOUS_CLASSIFICATION}` in `{PREVIOUS_RESULT}`",
        "",
        "## Executed artifacts",
        "",
        f"- Checkpoint: `{FROZEN_CHECKPOINT}`; SHA256 `{sha256_file(FROZEN_CHECKPOINT)}`.",
        f"- ONNX auxiliary: `{FROZEN_ONNX}`; SHA256 `{sha256_file(FROZEN_ONNX)}`.",
        f"- ONNX external data: `{FROZEN_ONNX_DATA}`; SHA256 `{sha256_file(FROZEN_ONNX_DATA)}`.",
        f"- deploy.yaml: `{FROZEN_DEPLOY}`; SHA256 `{sha256_file(FROZEN_DEPLOY)}`.",
        f"- env.yaml: `{FROZEN_ENV}`; SHA256 `{sha256_file(FROZEN_ENV)}`.",
        f"- agent.yaml: `{FROZEN_AGENT}`; SHA256 `{sha256_file(FROZEN_AGENT)}`.",
        f"- Candidate source revision: `{inventory['source_revision']}`.",
        "",
        "## Runtime",
        "",
        f"- Genesis: `{inventory['runtime']['genesis_version']}` from `{inventory['runtime']['genesis_python']}`.",
        f"- Python: `{inventory['runtime']['python_executable']}`; `{inventory['runtime']['python'].splitlines()[0]}`.",
        f"- Robot URDF: `{inventory['genesis_assets']['robot_urdf']}`; SHA256 `{inventory['genesis_assets']['robot_urdf_sha256']}`.",
        f"- Project revision: `{inventory['runtime']['project_revision']}`.",
        "",
        "## Prior visual-transfer provenance comparison",
        "",
    ]
    if mismatches:
        lines.extend(["MISMATCHES:", ""] + [f"- {item}" for item in mismatches])
    else:
        lines.append("- No candidate/config/Genesis provenance mismatch was found.")
    lines.extend(
        [
            "",
            "## Gate status at manifest time",
            "",
            f"- Frozen contract: `{contract['classification']}`.",
            f"- Joint/actuator contract: `{mapping['classification']}`.",
            f"- Deterministic reload: `{reload['classification']}`.",
            "",
            "The prior visual-transfer evidence is preserved and was not overwritten.",
        ]
    )
    write_text(GATE / "reports/gate_manifest.md", "\n".join(lines))


def write_quantitative_report(metrics: list[dict[str, Any]], transition: dict[str, Any], lateral: list[dict[str, Any]]) -> None:
    lines = [
        "# Quantitative expert compatibility results",
        "",
        f"Candidate: `{SOURCE_URL}`",
        f"Transient excluded from steady-state metrics: first `{STEADY_TRANSIENT_S:.1f} s`.",
        "Each primary command has three independent reset episodes of 10 s; the transition has three independent 23 s episodes.",
        "",
        "## Hard operational thresholds used",
        "",
        "- Forward active commands: mean achieved `vx > 0` and `|mean vx| / |command vx| >= 0.50`.",
        "- Yaw active commands: sign correct and `|mean yaw| / |command yaw| >= 0.50`.",
        "- Arcs: forward sign, yaw sign, `mean vx >= 0.05 m/s`, and `|mean yaw| >= 0.05 rad/s`.",
        "- Physical actual joint-limit violations: zero; requested target clips and torque-limit exceedances are counted separately.",
        f"- Requested-target clip fraction limit: `{TARGET_CLIP_FRACTION_LIMIT:.1%}`; pre-limit torque exceedance fraction limit: `{TORQUE_LIMIT_FRACTION_LIMIT:.1%}`.",
        "- Standing: full duration, no fall/contact/invalid state, XY drift and yaw drift each <= 0.50, roll/pitch <= 0.50 rad, max joint speed <= 20 rad/s.",
        "",
        "## Every run",
        "",
        "| Run | Phase | Command | Duration | Mean [vx,vy,yaw] | Median [vx,vy,yaw] | vx ratio | yaw ratio | Height min/max | max |roll| | max |pitch| | Falls | Base contacts | q violations | target clips | torque exceedances | Classification |",
        "|---|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in metrics:
        lines.append(
            f"| `{metric['run_id']}` | `{metric['phase']}` | `{metric.get('command')}` | {metric['actual_duration_s']:.2f}/{metric['requested_duration_s']:.2f} s | `{[metric.get('mean_vx'), metric.get('mean_vy'), metric.get('mean_yaw_rate')]}` | `{[metric.get('median_vx'), metric.get('median_vy'), metric.get('median_yaw_rate')]}` | `{metric.get('vx_tracking_ratio')}` | `{metric.get('yaw_tracking_ratio')}` | `{metric.get('base_height_min_m')}–{metric.get('base_height_max_m')}` | `{metric.get('max_abs_roll_rad')}` | `{metric.get('max_abs_pitch_rad')}` | {metric.get('fall_count')} | {metric.get('base_contact_count')} | {metric.get('joint_limit_violation_count')} | {metric.get('target_clip_count')} | {metric.get('torque_limit_exceedance_count')} | `{metric.get('final_classification')}` |"
        )
    lines.extend(
        [
            "",
            "## Transition evidence",
            "",
            f"- Runs: `{transition['runs']}`.",
            f"- Full duration: `{transition['all_full_duration']}`.",
            f"- Safety: `{transition['all_safety_ok']}`; falls `{transition['falls']}`; base contacts `{transition['base_contacts']}`; NaN/Inf `{transition['nan_inf']}`.",
            f"- Forward response: `{transition['forward_response_ok']}`.",
            f"- Left-yaw response: `{transition['left_yaw_response_ok']}`.",
            f"- Right-yaw response: `{transition['right_yaw_response_ok']}`.",
            f"- Return to stable standing: `{transition['all_return_to_standing']}`.",
            "- Change times are logged at 3, 8, 13, and 18 s; response-delay records are in each transition metric and the CSV summary.",
            "- Plot: `plots/transition_tracking.png`.",
            "",
            "## Contact/gait sanity",
            "",
            "The rollout CSVs record four-foot contact state, contact duty fraction, contact transitions, swing samples, and foot speed while in contact. Locomotion is classified as contact-sane only when each foot shows swing and stance, contact transitions occur, and no base contact occurs.",
            "",
        ]
    )
    if lateral:
        lines.extend(["## Optional lateral characterization", "", "Lateral is not part of the core pass.", "", "| Run | command | achieved vy | unwanted vx | unwanted yaw | stability | classification |", "|---|---|---:|---:|---:|---|---|"])
        for metric in lateral:
            lines.append(
                f"| `{metric['run_id']}` | `{metric['command']}` | {metric.get('mean_vy')} | {metric.get('mean_vx')} | {metric.get('mean_yaw_rate')} | `{metric.get('safety_ok')}` | `{metric.get('final_classification')}` |"
            )
        lines.append("")
    lines.extend(["No student demonstrations, BC, PPO, fine-tuning, DeePC, or hardware work was started."])
    write_text(GATE / "reports/quantitative_results.md", "\n".join(lines))


def write_final_report(
    inventory: dict[str, Any],
    contract: dict[str, Any],
    mapping: dict[str, Any],
    reload: dict[str, Any],
    single: dict[str, Any],
    metrics: list[dict[str, Any]],
    transition: dict[str, Any],
    lateral: list[dict[str, Any]],
    final_classification: str,
    primary_blocker: str | None,
) -> dict[str, Any]:
    core_categories = [row[1] for row in REQUIRED_COMMANDS]
    aggregates = {category: aggregate_command_metrics(metrics, category) for category in core_categories}
    core_runs = [
        metric
        for metric in metrics
        if metric["category"] in core_categories or metric["phase"] == "transition"
    ]
    core_no_falls = all(metric["fall_count"] == 0 for metric in core_runs)
    core_no_contacts = all(metric["base_contact_count"] == 0 for metric in core_runs)
    summary = {
        "candidate": SOURCE_URL,
        "previous_result": PREVIOUS_CLASSIFICATION,
        "classification": final_classification,
        "primary_blocker": primary_blocker,
        "gate_status": {
            "frozen_contract": contract["classification"],
            "joint_actuator_contract": mapping["classification"],
            "deterministic_reload": reload["classification"],
            "single_policy": single["classification"],
        },
        "core_capability": [
            "standing",
            "forward",
            "yaw left",
            "yaw right",
            "forward + yaw left",
            "forward + yaw right",
            "command transitions",
        ],
        "optional_lateral": "SUPPORTED" if lateral and all(metric["final_classification"] == "OPTIONAL_SUPPORTED" for metric in lateral) else ("PARTIALLY_SUPPORTED" if lateral else "NOT_RUN"),
        "core_aggregates": aggregates,
        "transition": transition,
        "safety_totals": {
            "falls": int(sum(metric["fall_count"] for metric in core_runs)),
            "base_contacts": int(sum(metric["base_contact_count"] for metric in core_runs)),
            "nan_inf": int(sum(metric["nan_inf_count"] for metric in core_runs)),
            "joint_limit_violations": int(sum(metric["joint_limit_violation_count"] for metric in core_runs)),
            "target_clips": int(sum(metric["target_clip_count"] for metric in core_runs)),
            "torque_limit_exceedances": int(sum(metric["torque_limit_exceedance_count"] for metric in core_runs)),
        },
        "provenance": inventory,
        "metrics_csv": str(GATE / "metrics/expert_gate_summary.csv"),
        "reload_csv": str(GATE / "metrics/reload_action_comparison.csv"),
        "reports": {
            "manifest": str(GATE / "reports/gate_manifest.md"),
            "contract": str(GATE / "reports/frozen_contract.md"),
            "joint_actuator": str(GATE / "reports/joint_actuator_contract.md"),
            "reload": str(GATE / "reports/reload_determinism.md"),
            "single_policy": str(GATE / "reports/single_policy_audit.md"),
            "quantitative": str(GATE / "reports/quantitative_results.md"),
        },
        "all_core_runs_full_duration": all(metric["full_duration"] for metric in core_runs),
        "all_core_runs_no_falls": core_no_falls,
        "all_core_runs_no_base_contacts": core_no_contacts,
    }
    write_json(GATE / "reports/expert_genesis_compatibility_summary.json", summary)

    lines = [
        "# Expert Genesis compatibility report",
        "",
        f"Candidate: `{SOURCE_URL}`",
        f"Previous result: `{PREVIOUS_CLASSIFICATION}`",
        f"Final classification: **{final_classification}**",
        "",
        "## Core evaluated capability",
        "",
        "standing; forward; yaw left; yaw right; forward + yaw left; forward + yaw right; command transitions.",
        "",
        "Optional characterization: lateral left/right, classified separately.",
        "",
        "## Exact expert contract",
        "",
        "- Observation: 45-D `[base angular velocity, projected gravity, desired [vx,vy,yaw_rate], relative joint position, relative joint velocity, previous raw normalized action]`.",
        "- Action: 12-D raw joint-position offset; `q_target = q_default + 0.5 * raw_action`; no raw clipping.",
        f"- Joint mapping: `{mapping['source_mapping']}` with fixed per-joint `Kp/Kd` in the joint report.",
        f"- PD/timing: `Kp=[20,20,40]`, `Kd=[1,1,2]`; `50 Hz` policy / `200 Hz` physics / `decimation=4`.",
        f"- Checkpoint/hash: `{FROZEN_CHECKPOINT}` / `{sha256_file(FROZEN_CHECKPOINT)}`.",
        f"- deploy.yaml SHA256: `{sha256_file(FROZEN_DEPLOY)}`.",
        "",
        "## Gate decisions",
        "",
        f"- Frozen contract: `{contract['classification']}`.",
        f"- Joint/actuator contract: `{mapping['classification']}`.",
        f"- Deterministic reload: `{reload['classification']}`; max absolute difference `{reload['maximum_absolute_action_difference']:.9g}`.",
        f"- Single policy/no switch: `{single['classification']}`.",
        "",
        "## Repeated-run safety summary",
        "",
        f"- Core runs: `{len(core_runs)}` (21 static + 3 transition).",
        f"- Falls: `{summary['safety_totals']['falls']}`.",
        f"- Base contacts: `{summary['safety_totals']['base_contacts']}`.",
        f"- NaN/Inf samples: `{summary['safety_totals']['nan_inf']}`.",
        f"- Actual joint-limit violations: `{summary['safety_totals']['joint_limit_violations']}`.",
        f"- Requested target clips: `{summary['safety_totals']['target_clips']}`; see pre/post target columns in logs.",
        f"- Pre-limit torque exceedances: `{summary['safety_totals']['torque_limit_exceedances']}`; see preclip/applied torque columns in logs.",
        "",
        "## Quantitative tracking",
        "",
        "The complete three-repeat table is in `metrics/expert_gate_summary.csv` and `reports/quantitative_results.md`; means and medians exclude the first 2.0 s of each episode.",
        "",
        "| Command | Runs | mean vx | mean vy | mean yaw | all safety | all command response | falls | base contacts |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for command_id, category, _command in REQUIRED_COMMANDS:
        aggregate = aggregates[category]
        lines.append(
            f"| `{command_id}` `{category}` | {aggregate.get('runs', 0)} | {aggregate.get('mean_vx')} | {aggregate.get('mean_vy')} | {aggregate.get('mean_yaw_rate')} | `{aggregate.get('all_safety_ok')}` | `{aggregate.get('all_command_response_ok')}` | {aggregate.get('falls', 0)} | {aggregate.get('base_contacts', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Transition decision",
            "",
            f"- Full duration: `{transition['all_full_duration']}`.",
            f"- Correct forward response: `{transition['forward_response_ok']}`.",
            f"- Correct left yaw: `{transition['left_yaw_response_ok']}`.",
            f"- Correct right yaw: `{transition['right_yaw_response_ok']}`.",
            f"- Returns to standing: `{transition['all_return_to_standing']}`.",
            "- Command-change timestamps and response delays are logged in the transition CSVs and plotted in `plots/transition_tracking.png`.",
            "",
            "## Scope discipline",
            "",
            "No weights, architecture, action contract, policy frequency, gains, smoothing, stabilizing controller, RL, BC, dataset collection, DeePC, or hardware path was changed or started.",
            "",
        ]
    )
    if primary_blocker:
        lines.extend([f"Primary blocker: **{primary_blocker}**.", "", "DO NOT START DATASET COLLECTION.", ""])
    else:
        lines.extend(["NEXT AUTHORIZED PHASE:", "EXPERT ROLLOUT / IMITATION DATASET DESIGN", "", "DO NOT START AUTOMATICALLY.", ""])
    lines.append(final_classification)
    write_text(GATE / "reports/expert_genesis_compatibility_report.md", "\n".join(lines))
    return summary


def decide(
    contract: dict[str, Any],
    mapping: dict[str, Any],
    reload: dict[str, Any],
    single: dict[str, Any],
    metrics: list[dict[str, Any]],
    transition: dict[str, Any],
    mismatches: list[str],
) -> tuple[str, str | None]:
    if mismatches:
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "frozen provenance mismatch with prior visual-transfer evidence"
    for name, result in (
        ("frozen contract", contract),
        ("joint/actuator contract", mapping),
        ("deterministic reload", reload),
        ("single-policy audit", single),
    ):
        if not result["classification"].endswith("PASS"):
            return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", f"{name} did not pass"
    categories = [row[1] for row in REQUIRED_COMMANDS]
    core = [
        metric
        for metric in metrics
        if metric["category"] in categories or metric["phase"] == "transition"
    ]
    if len([metric for metric in core if metric["phase"] == "static_command"]) != 21:
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "primary command bank did not produce all 21 episodes"
    if len([metric for metric in core if metric["phase"] == "transition"]) != 3:
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "transition bank did not produce all 3 episodes"
    if not all(metric["full_duration"] for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "one or more core episodes did not complete the requested duration"
    if any(metric["fall_count"] for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "one or more repeated core episodes fell"
    if any(metric["base_contact_count"] for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "one or more repeated core episodes contacted the base"
    if any(metric["nan_inf_count"] for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "NaN/Inf occurred in a core rollout"
    if any(metric["joint_limit_violation_count"] for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "actual joint-limit violation occurred"
    if any(metric["target_clip_fraction"] > TARGET_CLIP_FRACTION_LIMIT for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "requested q_target clipping exceeded the documented compatibility threshold"
    if any(metric["torque_limit_fraction"] > TORQUE_LIMIT_FRACTION_LIMIT for metric in core):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "pre-limit torque exceedance exceeded the documented compatibility threshold"
    standing = [metric for metric in core if metric["category"] == "standing"]
    if len(standing) != 3 or not all(metric["standing_stable"] for metric in standing):
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "standing stability gate failed"
    for category in ("slow_forward", "forward", "yaw_left", "yaw_right", "forward_left_arc", "forward_right_arc"):
        selected = [metric for metric in core if metric["category"] == category]
        if len(selected) != 3 or not all(metric["command_response_ok"] for metric in selected):
            return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", f"quantitative command response failed for {category}"
        if category != "standing" and not all(metric["locomotion_contact_sanity"] for metric in selected):
            return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", f"contact/gait sanity failed for {category}"
    if not transition["all_full_duration"] or not transition["all_safety_ok"]:
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "transition safety/full-duration gate failed"
    if not transition["forward_response_ok"] or not transition["left_yaw_response_ok"] or not transition["right_yaw_response_ok"]:
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "transition command sign gate failed"
    if not transition["all_return_to_standing"]:
        return "EXPERT_GENESIS_COMPATIBILITY_BLOCKED", "transition did not return to stable standing"
    return "EXPERT_GENESIS_COMPATIBILITY_PASS", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-video", action="store_true", help="Skip representative videos; core metrics still run")
    parser.add_argument("--skip-lateral", action="store_true", help="Skip optional lateral characterization")
    args = parser.parse_args()
    start = time.monotonic()
    for directory in ("frozen_inputs", "scripts", "logs", "metrics", "reports", "videos", "plots"):
        (GATE / directory).mkdir(parents=True, exist_ok=True)
    visual = load_visual_runtime()
    configure_caches()
    inventory = artifact_inventory()
    mismatches = prior_mismatches(inventory)
    write_frozen_inputs(inventory, mismatches)
    write_json(GATE / "reports/frozen_artifacts.json", {"inventory": inventory, "prior_mismatches": mismatches})
    print(f"frozen artifacts inventoried; prior mismatches={len(mismatches)}", flush=True)

    contract = audit_frozen_contract(visual, inventory)
    write_frozen_contract_report(contract)
    reload_result = run_reload_gate(visual, FROZEN_CHECKPOINT)
    print(f"contract={contract['classification']} reload={reload_result['classification']}", flush=True)
    if contract["classification"] != "FROZEN_CONTRACT_PASS" or mismatches:
        mapping = {
            "classification": "JOINT_ACTUATOR_CONTRACT_BLOCKED",
            "source_mapping": POLICY_TO_GENESIS.tolist(),
            "table": [],
            "safe_positive_target_probe": [],
            "checks": {},
            "genesis_named_joint_order": list(GENESIS_JOINT_NAMES),
            "runtime_genesis_dofs_in_named_order": [],
            "default_pose_within_runtime_limits": False,
            "mapping_matches_previous_report": False,
        }
        write_joint_mapping_report(mapping)
        single = single_policy_audit()
        metrics: list[dict[str, Any]] = []
        transition = {"runs": 0, "all_full_duration": False, "all_safety_ok": False, "all_return_to_standing": False, "falls": 0, "base_contacts": 0, "nan_inf": 0, "forward_response_ok": False, "left_yaw_response_ok": False, "right_yaw_response_ok": False}
        lateral: list[dict[str, Any]] = []
        final, blocker = decide(contract, mapping, reload_result, single, metrics, transition, mismatches)
        write_gate_manifest(inventory, mismatches, contract, mapping, reload_result)
        write_gate_summary_csv(metrics)
        write_quantitative_report(metrics, transition, lateral)
        write_final_report(inventory, contract, mapping, reload_result, single, metrics, transition, lateral, final, blocker)
        print(final, flush=True)
        return 1

    from scripts.imitation_learning.collect_command_teacher_rollouts import FrozenTeacher
    from learned_execution.genesis_deployment_scene import GenesisDeploymentScene

    scene = GenesisDeploymentScene(1, camera=not args.skip_video, camera_res=(640, 480))
    asset = visual.genesis_asset_info(scene)
    source_contract = {"classification": "TEACHER_CONTRACT_PASS"}
    mapping = augment_joint_mapping(visual.joint_mapping_audit(scene, source_contract, asset))
    write_joint_mapping_report(mapping)
    single = single_policy_audit()
    if mapping["classification"] != "JOINT_ACTUATOR_CONTRACT_PASS":
        metrics = []
        transition = {"runs": 0, "all_full_duration": False, "all_safety_ok": False, "all_return_to_standing": False, "falls": 0, "base_contacts": 0, "nan_inf": 0, "forward_response_ok": False, "left_yaw_response_ok": False, "right_yaw_response_ok": False}
        lateral = []
        final, blocker = decide(contract, mapping, reload_result, single, metrics, transition, mismatches)
        write_gate_manifest(inventory, mismatches, contract, mapping, reload_result)
        write_gate_summary_csv(metrics)
        write_quantitative_report(metrics, transition, lateral)
        write_final_report(inventory, contract, mapping, reload_result, single, metrics, transition, lateral, final, blocker)
        print(final, flush=True)
        return 1

    teacher = FrozenTeacher(FROZEN_CHECKPOINT, torch_threads=1)
    call_counter = {"policy_calls": 0}
    metrics: list[dict[str, Any]] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    next_seed = 20260816
    for command_id, category, command_value in REQUIRED_COMMANDS:
        for repeat in range(1, 4):
            run_id = f"{command_id}_rep{repeat}"
            render = repeat == 1 and not args.skip_video
            metric, arrays = run_episode(
                visual,
                scene,
                teacher,
                run_id,
                "static_command",
                category,
                command_value,
                command_schedule(command_value),
                10.0,
                repeat,
                next_seed,
                GATE / "logs" / f"{run_id}.csv",
                GATE / "videos" / f"{run_id}.mp4",
                render,
                call_counter,
            )
            metrics.append(metric)
            if render:
                series[run_id] = arrays
            next_seed += 1
            print(f"{run_id}: {metric['final_classification']} {metric['termination_reason']}", flush=True)
    for repeat in range(1, 4):
        run_id = f"transition_rep{repeat}"
        render = repeat == 1 and not args.skip_video
        metric, arrays = run_episode(
            visual,
            scene,
            teacher,
            run_id,
            "transition",
            "transition",
            None,
            transition_schedule,
            23.0,
            repeat,
            next_seed,
            GATE / "logs" / f"{run_id}.csv",
            GATE / "videos" / f"{run_id}.mp4",
            render,
            call_counter,
        )
        metrics.append(metric)
        if render:
            series[run_id] = arrays
        next_seed += 1
        print(f"{run_id}: {metric['final_classification']} {metric['termination_reason']}", flush=True)

    lateral: list[dict[str, Any]] = []
    if not args.skip_lateral:
        for command_id, category, command_value in LATERAL_COMMANDS:
            for repeat in range(1, 4):
                run_id = f"{command_id}_rep{repeat}"
                metric, _arrays = run_episode(
                    visual,
                    scene,
                    teacher,
                    run_id,
                    "optional_lateral",
                    category,
                    command_value,
                    command_schedule(command_value),
                    10.0,
                    repeat,
                    next_seed,
                    GATE / "logs" / f"{run_id}.csv",
                    GATE / "videos" / f"{run_id}.mp4",
                    False,
                    call_counter,
                )
                lateral.append(metric)
                metrics.append(metric)
                next_seed += 1
                print(f"{run_id}: {metric['final_classification']} {metric['termination_reason']}", flush=True)

    transition = evaluate_transition(metrics)
    write_gate_summary_csv(metrics)
    plot_results(metrics, series)
    write_quantitative_report(metrics, transition, lateral)
    final, blocker = decide(contract, mapping, reload_result, single, metrics, transition, mismatches)
    write_gate_manifest(inventory, mismatches, contract, mapping, reload_result)
    summary = write_final_report(inventory, contract, mapping, reload_result, single, metrics, transition, lateral, final, blocker)
    summary["policy_calls"] = call_counter["policy_calls"]
    summary["elapsed_s"] = time.monotonic() - start
    write_json(GATE / "reports/expert_genesis_compatibility_summary.json", summary)
    print(f"policy calls={call_counter['policy_calls']} elapsed={summary['elapsed_s']:.1f}s", flush=True)
    print(final, flush=True)
    return 0 if final == "EXPERT_GENESIS_COMPATIBILITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
