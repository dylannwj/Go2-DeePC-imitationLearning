#!/usr/bin/env python3
"""Read-only operating-envelope benchmark for the final Genesis champion.

This benchmark intentionally has no controller knobs.  It freezes the current
C2 + vx implementation, runs the fixed target bank through the same
``LiveProductRuntime`` used by the UI, and records observed limitations.
The historical C2-only freeze is not loaded or compared here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = ROOT / "src"

for import_path in (
    ROOT,
    SRC_ROOT,
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


EVIDENCE_ROOT = ROOT / "results/genesis"
REPORT_ROOT = Path(os.environ.get("DEEPC_BENCHMARK_OUTPUT", str(EVIDENCE_ROOT / "generated")))
FREEZE_PATH = EVIDENCE_ROOT / "final_genesis_champion_freeze.json"
RESULT_JSON_PATH = REPORT_ROOT / "final_champion_benchmark.json"
RESULT_CSV_PATH = REPORT_ROOT / "final_champion_benchmark.csv"
RESULT_MD_PATH = REPORT_ROOT / "final_champion_benchmark.md"

CANDIDATE_SOURCE = ROOT / "src/learned_execution/candidate_qp.py"
CANDIDATE_CONFIG = ROOT / "configs/genesis/candidate_config.json"
DEEPC_SOURCE = ROOT / "src/deepc/genesis_recovered_deepc.py"
DEEPC_CONFIG = ROOT / "configs/genesis/deepc_final_recovered.json"
HANKEL_PATH = ROOT / "data/imitation/hankel/final_student_deepc_hankel_v2_stride2.npz"
STUDENT_PATH = ROOT / "models/go2_observable_student_bc_linear_v1.pt"
STUDENT_RUNTIME_SOURCE = ROOT / "src/learned_execution/direct_student_runtime.py"
FROZEN_CONTROLLER_SOURCE = ROOT / "src/learned_execution/frozen_champion_controller.py"
GENESIS_SCENE_SOURCE = ROOT / "src/learned_execution/genesis_window.py"
GENESIS_EVALUATOR_SOURCE = ROOT / "src/learned_execution/genesis_deployment_scene.py"
LINEAR_CONSTANTS_SOURCE = ROOT / "src/learned_execution/linear_bc_common.py"
LIVE_RUNTIME_SOURCE = ROOT / "scripts/genesis/run_live_product.py"
UI_STATE_SOURCE = ROOT / "src/learned_execution/ui_state.py"
UI_BRIDGE_SOURCE = ROOT / "src/learned_execution/ui_runtime_bridge.py"
UI_TARGET_SOURCE = ROOT / "src/learned_execution/final_deepc_ui.py"
UI_CONFIG = ROOT / "configs/genesis/ui_config.json"
RUNTIME_CONTRACT = ROOT / "configs/genesis/frozen_runtime_contract.json"

TARGET_POSITION_TOLERANCE_M = 0.06
TARGET_YAW_TOLERANCE_RAD = 0.1
TARGET_YAW_TOLERANCE_DEG = math.degrees(TARGET_YAW_TOLERANCE_RAD)
CONFIRMATION_INTERVALS = 3
MAX_INTERVALS = 100
EPISODE_WALL_TIMEOUT_S = 240.0
ACTIVITY_MARGIN_TOLERANCE = 1.0e-6

C2_LIMIT = 0.28800664310564905
VX_LOWER_LIMIT = -0.8
VX_UPPER_LIMIT = 0.8

# These are the only benchmark target choices.  Distance is Euclidean target
# distance in the world XY plane; diagonal components are normalized equally.
TRANSLATION_DISTANCES_M = (0.25, 0.75, 1.20)
DIRECTION_VECTORS = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "forward-left": (1.0, 1.0),
    "forward-right": (1.0, -1.0),
    "backward-left": (-1.0, 1.0),
    "backward-right": (-1.0, -1.0),
}
HEADING_SWEEP_DEG = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0, -180.0)

TERMINAL_STATES = {"TARGET REACHED", "STOPPED", "SAFETY STOP", "SOLVER FAILURE", "INTERFACE FAILURE"}
HARD_SAFETY_KEYS = ("fall", "base_contact", "nan_inf", "actual_joint_limit_violation", "torque_limit_exceedance")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_version(module: Any) -> str:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else "unknown"


def source_hash_entries() -> dict[str, dict[str, str]]:
    paths = {
        "candidate_qp_source": CANDIDATE_SOURCE,
        "deepc_source": DEEPC_SOURCE,
        "frozen_controller_adapter": FROZEN_CONTROLLER_SOURCE,
        "student_runtime_source": STUDENT_RUNTIME_SOURCE,
        "genesis_scene_source": GENESIS_SCENE_SOURCE,
        "genesis_evaluator_source": GENESIS_EVALUATOR_SOURCE,
        "student_linear_constants_source": LINEAR_CONSTANTS_SOURCE,
        "live_runtime_source": LIVE_RUNTIME_SOURCE,
        "ui_state_source": UI_STATE_SOURCE,
        "ui_bridge_source": UI_BRIDGE_SOURCE,
        "ui_target_source": UI_TARGET_SOURCE,
        "benchmark_harness_source": Path(__file__).resolve(),
    }
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def artifact_hash_entries(genesis_module: Any) -> dict[str, dict[str, str]]:
    genesis_root = Path(genesis_module.__file__).resolve().parent
    paths = {
        "learned_student_checkpoint": STUDENT_PATH,
        "hankel_data_bank": HANKEL_PATH,
        "deepc_configuration": DEEPC_CONFIG,
        "candidate_configuration": CANDIDATE_CONFIG,
        "ui_target_configuration": UI_CONFIG,
        "genesis_runtime_contract": RUNTIME_CONTRACT,
        "go2_urdf": genesis_root / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf",
        "plane_urdf": genesis_root / "assets" / "urdf" / "plane" / "plane.urdf",
    }
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def collect_identity() -> dict[str, Any]:
    """Read the current final implementation identity without running it."""

    import cvxpy
    import genesis
    import numpy
    import osqp
    import quadrants
    import torch

    from learned_execution.direct_student_runtime import ACTION_SCALE, EFFORT_LIMIT, KD, KP, Q_DEFAULT_CANONICAL
    from learned_execution import frozen_champion_controller as fc
    from learned_execution.frozen_champion_controller import FinalChampionDeePC
    from learned_execution import final_deepc_ui
    from learned_execution import genesis_window
    from learned_execution import ui_runtime_bridge
    from learned_execution import ui_state
    from scripts.genesis.run_live_product import (
        CONFIRMATION_INTERVALS as LIVE_CONFIRMATION_INTERVALS,
        DEEPC_DT,
        MAX_INTERVALS as LIVE_MAX_INTERVALS,
        STUDENT_ACTIONS_PER_SAMPLE,
        TARGET_POSITION_TOLERANCE_M as LIVE_POSITION_TOLERANCE_M,
        TARGET_YAW_TOLERANCE_DEG as LIVE_YAW_TOLERANCE_DEG,
    )

    candidate_source_text = CANDIDATE_SOURCE.read_text(encoding="utf-8")
    frozen_controller_text = FROZEN_CONTROLLER_SOURCE.read_text(encoding="utf-8")
    live_runtime_text = LIVE_RUNTIME_SOURCE.read_text(encoding="utf-8")
    candidate = read_json(CANDIDATE_CONFIG)
    deepc = read_json(DEEPC_CONFIG)
    ui_config = read_json(UI_CONFIG)
    runtime_contract = read_json(RUNTIME_CONTRACT)
    genesis_urdf = Path(genesis.__file__).resolve().parent / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"
    plane_urdf = Path(genesis.__file__).resolve().parent / "assets" / "urdf" / "plane" / "plane.urdf"

    active_constraints = {
        "c2": {
            "expression": "future_input[t, vy] + future_input[t, yaw_rate] <= 0.28800664310564905",
            "source_line": "constraints = [future_input[1::3] + future_input[2::3] <= float(limit)]",
            "limit": C2_LIMIT,
            "scope": "every planned horizon step t=0..N-1",
            "inside_optimization": True,
        },
        "vx": {
            "expression": "-0.8 <= future_input[t, vx] <= 0.8",
            "source_lines": [
                "constraints.append(future_input[0::3] >= lower)",
                "constraints.append(future_input[0::3] <= upper)",
            ],
            "lower": VX_LOWER_LIMIT,
            "upper": VX_UPPER_LIMIT,
            "scope": "every planned horizon step t=0..N-1",
            "inside_optimization": True,
        },
        "post_solver_command_rewriting": False,
    }

    completion = {
        "position_tolerance_m": TARGET_POSITION_TOLERANCE_M,
        "yaw_tolerance_deg": TARGET_YAW_TOLERANCE_DEG,
        "yaw_tolerance_rad": TARGET_YAW_TOLERANCE_RAD,
        "confirmation_intervals": CONFIRMATION_INTERVALS,
        "max_replans": MAX_INTERVALS,
        "criterion": "measured position and wrapped yaw errors inside tolerance for three confirmation intervals",
    }
    solver = {
        "normal": dict(deepc["qp"]),
        "retry": {
            "trigger": "post-solver C2 or vx violation greater than 2e-8",
            "solver": deepc["qp"]["solver"],
            "solver_eps_abs": fc.C2_RETRY_EPS_ABS,
            "solver_eps_rel": fc.C2_RETRY_EPS_REL,
            "solver_max_iter": fc.C2_RETRY_MAX_ITER,
            "warm_start": True,
            "one_retry_only": True,
        },
    }
    runtime_selected = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "genesis": package_version(genesis),
        "quadrants": package_version(quadrants),
        "cvxpy": cvxpy.__version__,
        "osqp": osqp.__version__,
        "physics_timestep_s": runtime_contract["physics_contract"]["simulation_timestep_s"],
        "control_timestep_s": runtime_contract["physics_contract"]["control_timestep_s"],
        "physics_steps_per_policy_action": runtime_contract["physics_contract"]["physics_steps_per_policy_action"],
        "constraint_solver": runtime_contract["physics_contract"]["constraint_solver"],
        "collision_enabled": runtime_contract["physics_contract"]["collision_enabled"],
        "joint_limit_enabled": runtime_contract["physics_contract"]["joint_limit_enabled"],
        "viewer": "disabled for benchmark",
        "seed": 20260804,
        "go2_urdf": str(genesis_urdf),
        "go2_urdf_sha256": sha256_file(genesis_urdf),
        "plane_urdf": str(plane_urdf),
        "plane_urdf_sha256": sha256_file(plane_urdf),
    }
    ui_target_handling = {
        "target_frame": "world [x_m, y_m, yaw_rad]",
        "numeric_entry": "Pose3.from_degrees(x_m, y_m, yaw_deg), then normalize yaw",
        "canvas_xy": "target yaw defaults to current measured yaw",
        "canvas_yaw_aim": "yaw = atan2(pointer_y - target_y, pointer_x - target_x)",
        "target_locked_while_running": True,
        "config_default_target_id": ui_config["default_target_id"],
        "target_presets": ui_config["target_presets"],
    }
    source_assertions = {
        "candidate_has_exact_c2_source_line": candidate_source_text.count(active_constraints["c2"]["source_line"]) == 1,
        "candidate_has_vx_lower_source_line": candidate_source_text.count(active_constraints["vx"]["source_lines"][0]) == 1,
        "candidate_has_vx_upper_source_line": candidate_source_text.count(active_constraints["vx"]["source_lines"][1]) == 1,
        "candidate_accepts_vx_bounds": "vx_lower: float | None = None" in candidate_source_text and "vx_upper: float | None = None" in candidate_source_text,
        "adapter_passes_vx_bounds": "vx_lower=self.vx_lower_limit" in frozen_controller_text and "vx_upper=self.vx_upper_limit" in frozen_controller_text,
        "candidate_config_post_solver_rewriting_false": candidate["constraint"]["post_solver_rewriting"] is False,
        "deepc_runtime_projection_false": deepc["qp"]["runtime_command_projection"] is False,
        "live_runtime_post_solver_rewriting_false": "post_solver_rewriting=False" in live_runtime_text,
        "live_runtime_delivers_first_planned_input": "deepc_command = np.asarray(future_u[0]" in live_runtime_text,
        "student_command_boundary_exact": "student_received_command = last_step.student_received_command[0].copy()" in live_runtime_text,
        "imported_ui_modules_match": all(module.__file__ is not None for module in (FinalChampionDeePC, final_deepc_ui, genesis_window, ui_runtime_bridge, ui_state) if hasattr(module, "__file__")),
        "live_completion_matches_frozen": (
            DEEPC_DT == 0.10
            and STUDENT_ACTIONS_PER_SAMPLE == 5
            and LIVE_POSITION_TOLERANCE_M == TARGET_POSITION_TOLERANCE_M
            and LIVE_YAW_TOLERANCE_DEG == TARGET_YAW_TOLERANCE_DEG
            and LIVE_CONFIRMATION_INTERVALS == CONFIRMATION_INTERVALS
            and LIVE_MAX_INTERVALS == MAX_INTERVALS
        ),
    }
    identity = {
        "final_controller": "FINAL_GENESIS_DEEPC_LEARNED_EXECUTION_CHAMPION",
        "designated_variant": "C2_PLUS_VX",
        "historical_c2_only_used": False,
        "source_hashes": source_hash_entries(),
        "artifact_hashes": artifact_hash_entries(genesis),
        "active_constraints": active_constraints,
        "deePC": {
            "T_ini": deepc["deepc"]["T_ini"],
            "N": deepc["deepc"]["N"],
            "sample_dt_s": deepc["deepc"]["sample_dt_s"],
            "reference_gain": deepc["deepc"]["reference_gain"],
            "input": deepc["deepc"]["input"],
            "output": deepc["deepc"]["output"],
            "hankel_stride": deepc["deepc"]["hankel_stride"],
        },
        "solver_settings": solver,
        "completion_criterion": completion,
        "genesis_runtime": runtime_selected,
        "genesis_runtime_contract_sha256": sha256_file(RUNTIME_CONTRACT),
        "ui_target_handling": ui_target_handling,
        "student": {
            "checkpoint_sha256": sha256_file(STUDENT_PATH),
            "architecture": [48, 256, 256, 12],
            "action_scale": np.asarray(ACTION_SCALE).tolist(),
            "q_default_canonical": np.asarray(Q_DEFAULT_CANONICAL).tolist(),
            "kp": np.asarray(KP).tolist(),
            "kd": np.asarray(KD).tolist(),
            "effort_limit": np.asarray(EFFORT_LIMIT).tolist(),
        },
        "genesis_safety_contract": runtime_contract["safety_contract"],
        "source_assertions": source_assertions,
    }
    core = {
        key: identity[key]
        for key in (
            "final_controller",
            "designated_variant",
            "source_hashes",
            "artifact_hashes",
            "active_constraints",
            "deePC",
            "solver_settings",
            "completion_criterion",
            "genesis_runtime",
            "genesis_runtime_contract_sha256",
            "ui_target_handling",
            "student",
            "source_assertions",
        )
    }
    identity["fingerprint_sha256"] = canonical_digest(core)
    return identity


BENCHMARK_PROTOCOL = {
    "translation_distances_m": list(TRANSLATION_DISTANCES_M),
    "directions": list(DIRECTION_VECTORS),
    "direction_vector_convention": "unit world XY vector multiplied by distance",
    "translation_target_yaw_deg": 0.0,
    "heading_sweep_deg": list(HEADING_SWEEP_DEG),
    "heading_target_position_m": [0.0, 0.0],
    "heading_sweep_is_pure_yaw": True,
        "completion_criterion": {
            "position_tolerance_m": TARGET_POSITION_TOLERANCE_M,
            "yaw_tolerance_deg": TARGET_YAW_TOLERANCE_DEG,
            "yaw_tolerance_rad": TARGET_YAW_TOLERANCE_RAD,
            "confirmation_intervals": CONFIRMATION_INTERVALS,
            "max_replans": MAX_INTERVALS,
        },
    "episode_wall_timeout_s": EPISODE_WALL_TIMEOUT_S,
    "historical_c2_only_comparison": "excluded",
}


def freeze_or_verify() -> dict[str, Any]:
    identity = collect_identity()
    if not all(bool(value) for value in identity["source_assertions"].values()):
        failed = [key for key, value in identity["source_assertions"].items() if not value]
        raise RuntimeError(f"final champion identity assertion failed: {failed}")
    if FREEZE_PATH.exists():
        frozen = read_json(FREEZE_PATH)
        if frozen.get("status") != "FROZEN" or frozen.get("final_controller") != identity["final_controller"]:
            raise RuntimeError("existing final champion freeze is not the C2_PLUS_VX freeze")
        if frozen.get("designated_variant") != "C2_PLUS_VX":
            raise RuntimeError("existing final champion freeze has a different variant")
        if frozen.get("benchmark_protocol") != BENCHMARK_PROTOCOL:
            raise RuntimeError("existing final champion freeze has a different benchmark protocol")
        if frozen.get("identity", {}).get("fingerprint_sha256") != identity["fingerprint_sha256"]:
            raise RuntimeError(
                "final champion hash/config mismatch against the existing freeze: "
                f"{identity['fingerprint_sha256']} != {frozen.get('identity', {}).get('fingerprint_sha256')}"
            )
        return frozen
    freeze = {
        "schema": "final-genesis-deepc-learned-execution-champion-c2-plus-vx-freeze-v1",
        "status": "FROZEN",
        "final_controller": identity["final_controller"],
        "designated_variant": "C2_PLUS_VX",
        "generated_utc": utc_now(),
        "post_freeze_controller_changes": "NONE",
        "historical_c2_only_used": False,
        "identity": identity,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "benchmark_semantics": "PASS means the frozen benchmark completed with unchanged artifacts; target reach and safety outcomes are reported empirically and are not formal safety claims.",
    }
    write_json(FREEZE_PATH, freeze)
    return freeze


def target_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    index = 0
    for direction, vector in DIRECTION_VECTORS.items():
        unit = np.asarray(vector, dtype=np.float64)
        unit /= np.linalg.norm(unit)
        for distance in TRANSLATION_DISTANCES_M:
            xy = unit * distance
            specs.append(
                {
                    "episode_id": f"E{index:03d}_{direction.replace('-', '_')}_{distance:.2f}m",
                    "family": "translation",
                    "direction": direction,
                    "distance_m": distance,
                    "heading_deg": 0.0,
                    "target": [float(xy[0]), float(xy[1]), 0.0],
                }
            )
            index += 1
    for heading_deg in HEADING_SWEEP_DEG:
        specs.append(
            {
                "episode_id": f"E{index:03d}_heading_{heading_deg:+06.1f}deg".replace("+", "p").replace("-", "m").replace(".", "d"),
                "family": "heading",
                "direction": "heading",
                "distance_m": 0.0,
                "heading_deg": heading_deg,
                "target": [0.0, 0.0, math.radians(heading_deg)],
            }
        )
        index += 1
    return specs


def install_step_audit() -> None:
    """Attach passive telemetry accounting to the existing student adapter."""

    from scripts.genesis import run_live_product

    step_type = run_live_product.DirectStudentBatch
    original = step_type.step
    if getattr(original, "_final_champion_benchmark_wrapper", False):
        return

    def audited_step(self: Any, command: np.ndarray) -> Any:
        result = original(self, command)
        stats = getattr(self, "_final_champion_benchmark_stats", None)
        if stats is None:
            stats = {
                "step_count": 0,
                "safety_counts": {key: 0 for key in HARD_SAFETY_KEYS},
                "other_safety_counts": {"q_target_raw_outside_joint_limits": 0, "q_target_clip_flag": 0},
                "requested_torque_limit_violations": 0,
                "applied_torque_limit_violations": 0,
                "direct_command_mismatches": 0,
                "max_abs_requested_torque_N": 0.0,
                "max_abs_applied_torque_N": 0.0,
                "max_abs_raw_action": 0.0,
                "max_abs_joint_velocity": 0.0,
            }
        stats["step_count"] += int(np.asarray(result.requested_command).shape[0])
        for key in HARD_SAFETY_KEYS:
            stats["safety_counts"][key] += int(np.asarray(result.safety[key], dtype=bool).sum())
        for key in stats["other_safety_counts"]:
            stats["other_safety_counts"][key] += int(np.asarray(result.safety[key], dtype=bool).sum())
        stats["requested_torque_limit_violations"] += int(np.asarray(result.safety["torque_limit_exceedance"], dtype=bool).sum())
        # The adapter clips applied torque to EFFORT_LIMIT before Genesis, so
        # use the same immutable limit directly for an explicit readback check.
        from learned_execution.direct_student_runtime import EFFORT_LIMIT

        stats["applied_torque_limit_violations"] += int(
            np.any(np.abs(np.asarray(result.torque_applied, dtype=np.float64)) > np.asarray(EFFORT_LIMIT, dtype=np.float64)[None, :] + 1.0e-6)
        )
        stats["direct_command_mismatches"] += int(
            not np.array_equal(np.asarray(result.requested_command), np.asarray(result.student_received_command))
        )
        stats["max_abs_requested_torque_N"] = max(stats["max_abs_requested_torque_N"], float(np.max(np.abs(result.torque_raw))))
        stats["max_abs_applied_torque_N"] = max(stats["max_abs_applied_torque_N"], float(np.max(np.abs(result.torque_applied))))
        stats["max_abs_raw_action"] = max(stats["max_abs_raw_action"], float(np.max(np.abs(result.raw_action))))
        stats["max_abs_joint_velocity"] = max(stats["max_abs_joint_velocity"], float(np.max(np.abs(result.after["joint_velocity"]))))
        self._final_champion_benchmark_stats = stats
        return result

    audited_step._final_champion_benchmark_wrapper = True  # type: ignore[attr-defined]
    step_type.step = audited_step


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute exactly one target in an isolated Genesis process."""

    from learned_execution.ui_runtime_bridge import UIRuntimeBridge
    from learned_execution.ui_state import Pose3
    from scripts.genesis.run_live_product import LiveProductRuntime

    install_step_audit()
    identity = collect_identity()
    bridge = UIRuntimeBridge()
    target = Pose3(float(spec["target"][0]), float(spec["target"][1]), float(spec["target"][2])).normalized()
    bridge.request_target(target, source="final frozen champion benchmark", target_id=spec["episode_id"])
    runtime = LiveProductRuntime(bridge, max_intervals=MAX_INTERVALS, show_viewer=False)
    runtime.start()
    run_requested = False
    wall_timeout = EPISODE_WALL_TIMEOUT_S
    deadline = time.monotonic() + wall_timeout
    while time.monotonic() < deadline:
        state = bridge.snapshot()
        if state.controller_state == "READY" and not run_requested:
            run_requested = bool(bridge.request_run())
        if state.controller_state in TERMINAL_STATES and (run_requested or state.controller_state in {"SAFETY STOP", "INTERFACE FAILURE"}):
            break
        time.sleep(0.05)
    timed_out = time.monotonic() >= deadline and bridge.snapshot().controller_state not in TERMINAL_STATES
    if timed_out:
        bridge.request_emergency_stop("Benchmark wall timeout")
    runtime.stop()
    runtime.join(timeout=20.0)
    state = bridge.snapshot()
    stats = getattr(runtime.direct_runtime, "_final_champion_benchmark_stats", None) or {
        "step_count": 0,
        "safety_counts": {key: 0 for key in HARD_SAFETY_KEYS},
        "other_safety_counts": {"q_target_raw_outside_joint_limits": 0, "q_target_clip_flag": 0},
        "requested_torque_limit_violations": 0,
        "applied_torque_limit_violations": 0,
        "direct_command_mismatches": 0,
        "max_abs_requested_torque_N": 0.0,
        "max_abs_applied_torque_N": 0.0,
        "max_abs_raw_action": 0.0,
        "max_abs_joint_velocity": 0.0,
    }
    log_path = ROOT / "results/genesis/generated/logs" / f"{state.run_id}.jsonl" if state.run_id else None
    return {
        "spec": spec,
        "identity_fingerprint_sha256": identity["fingerprint_sha256"],
        "state": {
            "controller_state": state.controller_state,
            "status_detail": state.status_detail,
            "run_id": state.run_id,
            "replan_count": state.replan_count,
            "iteration": state.iteration,
            "exit_reason": state.exit_reason,
            "safety_ok": state.safety_state.ok,
            "safety_reason": state.safety_state.reason,
            "elapsed_s": state.elapsed_s,
        },
        "log_path": str(log_path) if log_path is not None else None,
        "stats": stats,
        "timed_out": timed_out,
    }


def load_log(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def classify_timeout(spec: dict[str, Any], state: dict[str, Any], rows: list[dict[str, Any]], stats: dict[str, Any]) -> tuple[str, str]:
    outcome = state.get("controller_state", "")
    if state.get("timed_out"):
        return "harness_timeout", "external benchmark wall timeout before the product reached a terminal state"
    if outcome == "SOLVER FAILURE" or state.get("exit_reason") == "SOLVER_FAILURE":
        return "solver_failure", str(state.get("status_detail") or "frozen controller solver failure")
    safety_total = sum(int(value) for value in stats.get("safety_counts", {}).values())
    if outcome == "SAFETY STOP" or safety_total:
        return "hard_safety_failure", str(state.get("safety_reason") or state.get("status_detail") or "hard safety event")
    if outcome == "TARGET REACHED":
        return "target_reached", "measured position/yaw criterion confirmed for three intervals"
    iterations = [row for row in rows if row.get("event") == "receding_horizon_iteration"]
    if not iterations:
        return "no_progress_timeout", "no completed measured-feedback replan was recorded"
    first = iterations[0]
    last = iterations[-1]
    first_pos = float(first.get("position_error_m", first.get("position_error", math.inf)))
    last_pos = float(last.get("position_error_m", last.get("position_error", math.inf)))
    first_yaw = abs(float(first.get("yaw_error_rad", first.get("yaw_error", math.inf))))
    last_yaw = abs(float(last.get("yaw_error_rad", last.get("yaw_error", math.inf))))
    pos_ok = last_pos <= TARGET_POSITION_TOLERANCE_M
    yaw_ok = last_yaw <= TARGET_YAW_TOLERANCE_RAD
    if pos_ok != yaw_ok:
        return "position_yaw_conflict", f"one final error was inside tolerance and the other was not (position={last_pos:.6f}, yaw={last_yaw:.6f})"
    signed_yaw = [wrap_angle(float(row.get("target_yaw", 0.0)) - float(row.get("actual_next_pose", [0.0, 0.0, 0.0])[2])) for row in iterations if row.get("actual_next_pose")]
    signs = [1 if value > 1.0e-4 else -1 if value < -1.0e-4 else 0 for value in signed_yaw]
    signs = [value for value in signs if value]
    sign_changes = sum(left != right for left, right in zip(signs, signs[1:]))
    if sign_changes >= 4 and last_yaw > TARGET_YAW_TOLERANCE_RAD:
        return "oscillation", f"signed yaw error changed side {sign_changes} times without reaching the criterion"
    target_heading = abs(float(spec.get("heading_deg", 0.0)))
    commands = [row.get("deepc_command", [0.0, 0.0, 0.0]) for row in iterations]
    max_yaw_command = max((abs(float(command[2])) for command in commands), default=0.0)
    if target_heading >= 60.0 and last_yaw > 0.5 and last_yaw >= 0.75 * first_yaw and max_yaw_command < 0.15:
        return "insufficient_yaw_authority", f"large-heading yaw error remained high while commanded |yaw_rate| stayed below 0.15 (max={max_yaw_command:.6f})"
    pos_reduction = first_pos - last_pos
    yaw_reduction = first_yaw - last_yaw
    if pos_reduction > 0.05 or yaw_reduction > 0.05:
        return "safe_progress_timeout", f"no hard safety event; errors reduced (position {first_pos:.6f}->{last_pos:.6f}, yaw {first_yaw:.6f}->{last_yaw:.6f})"
    return "no_progress_timeout", f"no hard safety event and little measured error reduction (position {first_pos:.6f}->{last_pos:.6f}, yaw {first_yaw:.6f}->{last_yaw:.6f})"


def summarize_worker(worker: dict[str, Any]) -> dict[str, Any]:
    spec = worker["spec"]
    state = dict(worker["state"])
    state["timed_out"] = worker.get("timed_out", False)
    rows = load_log(worker.get("log_path"))
    finish = next((row for row in reversed(rows) if row.get("event") == "run_finished"), {})
    iterations = [row for row in rows if row.get("event") == "receding_horizon_iteration"]
    measurements = [row for row in rows if row.get("event") == "measured_pose"]
    target = np.asarray(spec["target"], dtype=np.float64)
    final_pose = np.asarray(finish.get("final_pose") or (iterations[-1].get("actual_next_pose") if iterations else [math.nan, math.nan, math.nan]), dtype=np.float64)
    if final_pose.shape != (3,):
        final_pose = np.full(3, math.nan)
    final_position_error = float(math.hypot(target[0] - final_pose[0], target[1] - final_pose[1])) if np.all(np.isfinite(final_pose)) else None
    final_yaw_error = abs(wrap_angle(target[2] - final_pose[2])) if np.all(np.isfinite(final_pose)) else None
    stats = worker.get("stats", {})
    commands = [np.asarray(row.get("deepc_command", [math.nan, math.nan, math.nan]), dtype=np.float64) for row in iterations]
    commands = [command for command in commands if command.shape == (3,) and np.all(np.isfinite(command))]
    c2_values = [float(command[1] + command[2]) for command in commands]
    c2_margins = [C2_LIMIT - value for value in c2_values]
    vx_margins = [min(float(command[0]) - VX_LOWER_LIMIT, VX_UPPER_LIMIT - float(command[0])) for command in commands]
    retry_rows = [row for row in iterations if row.get("solver_retry_used") is True]
    hard_counts = {key: int(stats.get("safety_counts", {}).get(key, 0)) for key in HARD_SAFETY_KEYS}
    hard_total = sum(hard_counts.values())
    classification, basis = classify_timeout(spec, state, rows, stats)
    outcome = finish.get("outcome", state.get("controller_state", "UNKNOWN"))
    reached = outcome == "TARGET REACHED" and classification == "target_reached"
    direction_vector = np.asarray(DIRECTION_VECTORS.get(spec.get("direction", ""), (0.0, 0.0)), dtype=np.float64)
    if np.linalg.norm(direction_vector) > 0:
        direction_vector /= np.linalg.norm(direction_vector)
        directional_projection = float(np.dot(final_pose[:2], direction_vector)) if np.all(np.isfinite(final_pose)) else None
    else:
        directional_projection = None
    actual_backward_reached = None
    if spec["family"] == "translation" and str(spec["direction"]).startswith("backward"):
        actual_backward_reached = bool(reached and directional_projection is not None and directional_projection > 0.06)
    signed_yaw = [wrap_angle(float(row.get("target_yaw", target[2])) - float(row.get("actual_next_pose", [0.0, 0.0, 0.0])[2])) for row in iterations if row.get("actual_next_pose")]
    return {
        "episode_id": spec["episode_id"],
        "family": spec["family"],
        "direction": spec["direction"],
        "distance_m": spec["distance_m"],
        "target_x_m": spec["target"][0],
        "target_y_m": spec["target"][1],
        "target_yaw_deg": spec["heading_deg"],
        "target_reached": reached,
        "outcome": outcome,
        "classification": classification,
        "classification_basis": basis,
        "final_x_m": float(final_pose[0]) if np.isfinite(final_pose[0]) else None,
        "final_y_m": float(final_pose[1]) if np.isfinite(final_pose[1]) else None,
        "final_yaw_rad": float(final_pose[2]) if np.isfinite(final_pose[2]) else None,
        "final_position_error_m": final_position_error,
        "final_yaw_error_rad": final_yaw_error,
        "final_yaw_error_deg": math.degrees(final_yaw_error) if final_yaw_error is not None else None,
        "completion_time_s": float(finish.get("elapsed_s", state.get("elapsed_s", 0.0))),
        "replans": int(finish.get("replan_count", state.get("replan_count", len(iterations)))),
        "solver_failures": int(outcome == "SOLVER FAILURE" or state.get("exit_reason") == "SOLVER_FAILURE"),
        "hard_safety_events": hard_total,
        "hard_safety_event_breakdown": hard_counts,
        "requested_torque_limit_violations": int(stats.get("requested_torque_limit_violations", 0)),
        "applied_torque_limit_violations": int(stats.get("applied_torque_limit_violations", 0)),
        "joint_limit_violations": hard_counts["actual_joint_limit_violation"],
        "falls": hard_counts["fall"],
        "base_contacts": hard_counts["base_contact"],
        "nan_inf": hard_counts["nan_inf"],
        "direct_command_mismatches": int(stats.get("direct_command_mismatches", 0)),
        "max_abs_requested_torque_N": float(stats.get("max_abs_requested_torque_N", 0.0)),
        "max_abs_applied_torque_N": float(stats.get("max_abs_applied_torque_N", 0.0)),
        "max_abs_raw_action": float(stats.get("max_abs_raw_action", 0.0)),
        "max_abs_joint_velocity": float(stats.get("max_abs_joint_velocity", 0.0)),
        "c2_active_replans": int(sum(margin <= ACTIVITY_MARGIN_TOLERANCE for margin in c2_margins)),
        "c2_min_margin": min(c2_margins) if c2_margins else None,
        "c2_violation_replans": int(sum(margin < -2.0e-8 for margin in c2_margins)),
        "vx_active_replans": int(sum(margin <= ACTIVITY_MARGIN_TOLERANCE for margin in vx_margins)),
        "vx_min_margin": min(vx_margins) if vx_margins else None,
        "vx_violation_replans": int(sum(margin < -2.0e-8 for margin in vx_margins)),
        "solver_retry_replans": len(retry_rows),
        "negative_vx_replans": int(sum(float(command[0]) < -1.0e-6 for command in commands)),
        "positive_vx_replans": int(sum(float(command[0]) > 1.0e-6 for command in commands)),
        "min_vx": min((float(command[0]) for command in commands), default=None),
        "max_vx": max((float(command[0]) for command in commands), default=None),
        "max_abs_yaw_rate_command": max((abs(float(command[2])) for command in commands), default=None),
        "signed_yaw_error_sign_changes": sum(
            left != right
            for left, right in zip(
                [1 if value > 1.0e-4 else -1 if value < -1.0e-4 else 0 for value in signed_yaw if abs(value) > 1.0e-4],
                [1 if value > 1.0e-4 else -1 if value < -1.0e-4 else 0 for value in signed_yaw if abs(value) > 1.0e-4][1:],
            )
        ),
        "directional_projection_m": directional_projection,
        "actual_backward_target_reached": actual_backward_reached,
        "log_path": worker.get("log_path"),
        "identity_fingerprint_sha256": worker.get("identity_fingerprint_sha256"),
        "measured_rows": len(measurements),
        "iteration_rows": len(iterations),
    }


def run_episode(spec: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    current = collect_identity()
    if current["fingerprint_sha256"] != frozen["identity"]["fingerprint_sha256"]:
        raise RuntimeError(f"frozen champion changed before {spec['episode_id']}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-spec",
        json.dumps(spec, separators=(",", ":")),
    ]
    env = dict(os.environ)
    cache_root = Path(tempfile.gettempdir()) / "deepc_thesis_cache"
    env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env.setdefault("GS_CACHE_FILE_PATH", str(cache_root / "genesis"))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=EPISODE_WALL_TIMEOUT_S + 30.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "spec": spec,
            "state": {"controller_state": "HARNESS TIMEOUT", "status_detail": str(exc), "exit_reason": "HARNESS_TIMEOUT"},
            "timed_out": True,
            "stats": {},
            "identity_fingerprint_sha256": frozen["identity"]["fingerprint_sha256"],
            "log_path": None,
        }
    marker = "FINAL_CHAMPION_WORKER_RESULT="
    worker = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            worker = json.loads(line[len(marker) :])
            break
    if worker is None:
        return {
            "spec": spec,
            "state": {"controller_state": "HARNESS ERROR", "status_detail": completed.stderr[-2000:], "exit_reason": "HARNESS_ERROR"},
            "timed_out": False,
            "stats": {},
            "identity_fingerprint_sha256": frozen["identity"]["fingerprint_sha256"],
            "log_path": None,
        }
    return worker


def aggregate(episodes: list[dict[str, Any]], freeze: dict[str, Any], start_identity: dict[str, Any], end_identity: dict[str, Any] | None, completed: bool) -> dict[str, Any]:
    success_count = sum(bool(row["target_reached"]) for row in episodes)
    solver_failures = sum(int(row["solver_failures"]) for row in episodes)
    hard_events = sum(int(row["hard_safety_events"]) for row in episodes)
    backward_rows = [row for row in episodes if row["family"] == "translation" and str(row["direction"]).startswith("backward")]
    lateral_rows = [row for row in episodes if row["family"] == "translation" and row["direction"] in {"left", "right"}]
    diagonal_rows = [row for row in episodes if row["family"] == "translation" and "-" in str(row["direction"])]
    forward_rows = [row for row in episodes if row["family"] == "translation" and row["direction"] == "forward"]
    direction_summary: dict[str, Any] = {}
    for direction in DIRECTION_VECTORS:
        rows = [row for row in episodes if row["family"] == "translation" and row["direction"] == direction]
        direction_summary[direction] = {
            "successes": sum(bool(row["target_reached"]) for row in rows),
            "episodes": len(rows),
            "results": [{"distance_m": row["distance_m"], "target_reached": row["target_reached"], "final_position_error_m": row["final_position_error_m"], "final_yaw_error_rad": row["final_yaw_error_rad"], "classification": row["classification"]} for row in rows],
        }
    heading_rows = [row for row in episodes if row["family"] == "heading"]
    reached_headings = [abs(float(row["target_yaw_deg"])) for row in heading_rows if row["target_reached"]]
    all_finite_yaw_errors = [float(row["final_yaw_error_rad"]) for row in episodes if row["final_yaw_error_rad"] is not None]
    final_hash_unchanged = end_identity is not None and end_identity["fingerprint_sha256"] == freeze["identity"]["fingerprint_sha256"]
    classifications = Counter(row["classification"] for row in episodes)
    return {
        "final_champion_reproduced": bool(start_identity["fingerprint_sha256"] == freeze["identity"]["fingerprint_sha256"] and final_hash_unchanged),
        "final_champion_hashes": {
            "source_hashes": freeze["identity"]["source_hashes"],
            "artifact_hashes": freeze["identity"]["artifact_hashes"],
            "identity_fingerprint_sha256": freeze["identity"]["fingerprint_sha256"],
        },
        "target_reaching_success": {"successes": success_count, "episodes": len(episodes)},
        "forward_target_reaching": {"successes": sum(bool(row["target_reached"]) for row in forward_rows), "episodes": len(forward_rows)},
        "backward_target_reaching": {
            "status": "YES" if backward_rows and all(bool(row.get("actual_backward_target_reached")) for row in backward_rows) else "PARTIAL" if any(bool(row.get("actual_backward_target_reached")) for row in backward_rows) else "NO",
            "actual_motion_verified_successes": sum(bool(row.get("actual_backward_target_reached")) for row in backward_rows),
            "episodes": len(backward_rows),
        },
        "lateral_target_reaching": {"successes": sum(bool(row["target_reached"]) for row in lateral_rows), "episodes": len(lateral_rows)},
        "diagonal_target_reaching": {"successes": sum(bool(row["target_reached"]) for row in diagonal_rows), "episodes": len(diagonal_rows)},
        "final_yaw_empirical_envelope": {
            "criterion": "pure-yaw heading sweep, position tolerance 0.06 m, yaw tolerance 0.1 rad (5.729577951 degrees), three confirmations",
            "sweep_episodes": len(heading_rows),
            "reached_heading_abs_deg_max": max(reached_headings) if reached_headings else None,
            "reached_headings_deg": sorted(set(reached_headings)),
            "worst_final_yaw_error_rad": max(all_finite_yaw_errors) if all_finite_yaw_errors else None,
            "results": [{"target_yaw_deg": row["target_yaw_deg"], "target_reached": row["target_reached"], "final_yaw_error_rad": row["final_yaw_error_rad"], "classification": row["classification"]} for row in heading_rows],
        },
        "solver_failures": solver_failures,
        "hard_safety_events": hard_events,
        "hard_safety_event_episodes": sum(bool(row["hard_safety_events"]) for row in episodes),
        "classification_counts": dict(classifications),
        "episodes_completed": len(episodes),
        "episodes_expected": len(target_specs()),
        "benchmark_completed": bool(completed and len(episodes) == len(target_specs())),
        "final_champion_benchmark": "PASS" if completed and len(episodes) == len(target_specs()) and start_identity["fingerprint_sha256"] == freeze["identity"]["fingerprint_sha256"] and final_hash_unchanged else "FAIL",
    }


CSV_FIELDS = [
    "episode_id", "family", "direction", "distance_m", "target_yaw_deg", "target_reached", "outcome", "classification", "classification_basis",
    "final_position_error_m", "final_yaw_error_rad", "final_yaw_error_deg", "completion_time_s", "replans", "solver_failures", "hard_safety_events",
    "falls", "base_contacts", "joint_limit_violations", "requested_torque_limit_violations", "applied_torque_limit_violations", "nan_inf",
    "max_abs_requested_torque_N", "max_abs_applied_torque_N", "c2_active_replans", "c2_min_margin", "c2_violation_replans", "vx_active_replans",
    "vx_min_margin", "vx_violation_replans", "solver_retry_replans", "negative_vx_replans", "min_vx", "max_vx", "max_abs_yaw_rate_command",
    "directional_projection_m", "actual_backward_target_reached", "log_path",
]


def write_csv(episodes: list[dict[str, Any]]) -> None:
    RESULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)


def result_markdown(freeze: dict[str, Any], summary: dict[str, Any], episodes: list[dict[str, Any]]) -> str:
    identity = freeze["identity"]
    lines = [
        "# Final Genesis DeePC + learned-locomotion champion benchmark",
        "",
        "`FINAL_GENESIS_DEEPC_LEARNED_EXECUTION_CHAMPION = C2_PLUS_VX`",
        "",
        "This is a benchmark of the current frozen C2 + vx implementation only. The historical C2-only implementation and its 0.05 rad completion protocol were not recovered, rerun, or compared.",
        "",
        "## Freeze and result",
        "",
        f"- `FINAL_CHAMPION_REPRODUCED = {'YES' if summary['final_champion_reproduced'] else 'NO'}`",
        f"- `TARGET_REACHING_SUCCESS = {summary['target_reaching_success']['successes']}/{summary['target_reaching_success']['episodes']}`",
        f"- `SOLVER_FAILURES = {summary['solver_failures']}`",
        f"- `HARD_SAFETY_EVENTS = {summary['hard_safety_events']}` (event count; episodes with any: {summary['hard_safety_event_episodes']})",
        f"- `FINAL_CHAMPION_BENCHMARK = {summary['final_champion_benchmark']}`",
        f"- Freeze fingerprint: `{identity['fingerprint_sha256']}`",
        "",
        "## Active frozen configuration",
        "",
        f"- C2: `{identity['active_constraints']['c2']['expression']}`",
        f"- vx: `{identity['active_constraints']['vx']['expression']}`",
        f"- DeePC: `T_ini={identity['deePC']['T_ini']}`, `N={identity['deePC']['N']}`, `dt={identity['deePC']['sample_dt_s']} s`, `reference_gain={identity['deePC']['reference_gain']}`",
        f"- Normal solver: `{identity['solver_settings']['normal']['solver']}`, eps_abs `{identity['solver_settings']['normal']['solver_eps_abs']}`, eps_rel `{identity['solver_settings']['normal']['solver_eps_rel']}`, max_iter `{identity['solver_settings']['normal']['solver_max_iter']}`",
        f"- One-shot retry: eps_abs `{identity['solver_settings']['retry']['solver_eps_abs']}`, eps_rel `{identity['solver_settings']['retry']['solver_eps_rel']}`, max_iter `{identity['solver_settings']['retry']['solver_max_iter']}`",
        f"- Completion: position <= `{identity['completion_criterion']['position_tolerance_m']} m`, yaw <= `{identity['completion_criterion']['yaw_tolerance_deg']}°`, `{identity['completion_criterion']['confirmation_intervals']}` confirmations, max `{identity['completion_criterion']['max_replans']}` replans",
        f"- Post-solver command rewriting: `{identity['active_constraints']['post_solver_command_rewriting']}`",
        "",
        "## Directional target reaching",
        "",
        "| Direction | Reached | Episodes |",
        "|---|---:|---:|",
    ]
    for direction in DIRECTION_VECTORS:
        row = summary["forward_target_reaching"] if direction == "forward" else None
        direction_row = next((item for item in [summary.get("direction_summary", {}).get(direction)] if item), None)
        if direction_row is None:
            direction_rows = [episode for episode in episodes if episode["family"] == "translation" and episode["direction"] == direction]
            reached = sum(bool(item["target_reached"]) for item in direction_rows)
            total = len(direction_rows)
        else:
            reached = direction_row["successes"]
            total = direction_row["episodes"]
        lines.append(f"| {direction} | {reached} | {total} |")
    lines.extend(
        [
            "",
            f"- Forward target reaching: `{summary['forward_target_reaching']['successes']}/{summary['forward_target_reaching']['episodes']}`",
            f"- Backward target reaching: `{summary['backward_target_reaching']['status']}`; actual directional motion verified `{summary['backward_target_reaching']['actual_motion_verified_successes']}/{summary['backward_target_reaching']['episodes']}`",
            f"- Lateral target reaching: `{summary['lateral_target_reaching']['successes']}/{summary['lateral_target_reaching']['episodes']}`",
            f"- Diagonal target reaching: `{summary['diagonal_target_reaching']['successes']}/{summary['diagonal_target_reaching']['episodes']}`",
            "",
            "## Final-heading sweep",
            "",
            f"Pure-yaw targets at position `[0, 0]`; sweep uses the frozen 0.1-rad (5.7296°) criterion. Empirical maximum reached absolute heading: `{summary['final_yaw_empirical_envelope']['reached_heading_abs_deg_max']}°`.",
            "",
            "| Target yaw (deg) | Reached | Final yaw error (rad) | Classification |",
            "|---:|---|---:|---|",
        ]
    )
    for row in [episode for episode in episodes if episode["family"] == "heading"]:
        final_error = "n/a" if row["final_yaw_error_rad"] is None else f"{row['final_yaw_error_rad']:.6f}"
        lines.append(f"| {row['target_yaw_deg']:+.1f} | {'YES' if row['target_reached'] else 'NO'} | {final_error} | {row['classification']} |")
    lines.extend(
        [
            "",
            "## Per-episode measurements",
            "",
            "The CSV contains the full machine-readable row set, including exact final errors, constraint activity, solver retries, and safety counters.",
            "",
            "| Episode | Family / direction | Reached | Final position error (m) | Final yaw error (rad) | Time (s) | Classification |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in episodes:
        position = "n/a" if row["final_position_error_m"] is None else f"{row['final_position_error_m']:.6f}"
        yaw = "n/a" if row["final_yaw_error_rad"] is None else f"{row['final_yaw_error_rad']:.6f}"
        lines.append(f"| {row['episode_id']} | {row['family']} / {row['direction']} | {'YES' if row['target_reached'] else 'NO'} | {position} | {yaw} | {row['completion_time_s']:.3f} | {row['classification']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Timeout classifications are empirical labels based on the recorded error trajectory. They are not formal diagnoses or safety guarantees. Any solver or hard-safety event is reported as observed; the controller was not changed after the boundary was observed.",
            "",
            "## Source hashes",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
        ]
    )
    for name, entry in {**identity["source_hashes"], **identity["artifact_hashes"]}.items():
        lines.append(f"| {name} | `{entry['sha256']}` |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--preflight-only", action="store_true", help="Freeze/verify the champion without running Genesis episodes")
    args = parser.parse_args(argv)
    if args.worker_spec:
        try:
            worker = run_worker(json.loads(args.worker_spec))
            print("FINAL_CHAMPION_WORKER_RESULT=" + json.dumps(json_safe(worker), sort_keys=True))
            return 0
        except Exception as exc:
            print("FINAL_CHAMPION_WORKER_ERROR=" + json.dumps({"error": str(exc)}, sort_keys=True))
            return 2

    freeze = freeze_or_verify()
    start_identity = collect_identity()
    if start_identity["fingerprint_sha256"] != freeze["identity"]["fingerprint_sha256"]:
        raise RuntimeError("final champion did not reproduce at benchmark start")
    if args.preflight_only:
        print(json.dumps({"freeze": freeze, "preflight": "PASS"}, indent=2, sort_keys=True))
        return 0
    episodes: list[dict[str, Any]] = []
    completed = True
    specs = target_specs()
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec['episode_id']}", flush=True)
        worker = run_episode(spec, freeze)
        row = summarize_worker(worker)
        episodes.append(row)
        print(
            f"  {row['outcome']} · reached={row['target_reached']} · "
            f"pos={row['final_position_error_m']} · yaw={row['final_yaw_error_rad']} · "
            f"classification={row['classification']}",
            flush=True,
        )
        if row["classification"] == "harness_error" or row["classification"] == "harness_timeout":
            completed = False
            break
        current = collect_identity()
        if current["fingerprint_sha256"] != freeze["identity"]["fingerprint_sha256"]:
            completed = False
            raise RuntimeError(f"final champion changed after {spec['episode_id']}")
    end_identity = collect_identity()
    summary = aggregate(episodes, freeze, start_identity, end_identity, completed)
    summary["direction_summary"] = {}
    for direction in DIRECTION_VECTORS:
        rows = [row for row in episodes if row["family"] == "translation" and row["direction"] == direction]
        summary["direction_summary"][direction] = {"successes": sum(bool(row["target_reached"]) for row in rows), "episodes": len(rows)}
    report = {
        "schema": "final-genesis-champion-operating-envelope-benchmark-v1",
        "generated_utc": utc_now(),
        "final_controller": "FINAL_GENESIS_DEEPC_LEARNED_EXECUTION_CHAMPION",
        "designated_variant": "C2_PLUS_VX",
        "freeze_path": str(FREEZE_PATH),
        "protocol": BENCHMARK_PROTOCOL,
        "summary": summary,
        "episodes": episodes,
    }
    write_json(RESULT_JSON_PATH, report)
    write_csv(episodes)
    RESULT_MD_PATH.write_text(result_markdown(freeze, {**summary, "direction_summary": summary["direction_summary"]}, episodes), encoding="utf-8")
    print(json.dumps({"summary": summary, "reports": {"json": str(RESULT_JSON_PATH), "csv": str(RESULT_CSV_PATH), "markdown": str(RESULT_MD_PATH), "freeze": str(FREEZE_PATH)}}, indent=2, sort_keys=True))
    return 0 if summary["final_champion_benchmark"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
