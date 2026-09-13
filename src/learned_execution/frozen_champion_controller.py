"""Adapter exposing the recovered DeePC with solver-side C2 and vx limits.

The QP implementation is loaded from the candidate source named by
``final_controller_freeze.json``.  This module owns the UI integration
boundary, measured-history bookkeeping, and the explicit forward-speed box
passed into that QP; it does not rewrite commands after solving.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FINAL_FREEZE = PROJECT_ROOT / "results/genesis/final_genesis_champion_freeze.json"
FINAL_CONSTRAINT_AUDIT = PROJECT_ROOT / "results/genesis/audits/final_constraint_audit.json"
FINAL_VALIDATION_DECISION = PROJECT_ROOT / "results/genesis/audits/final_validation_decision.json"
CANDIDATE_CONFIG = PROJECT_ROOT / "configs/genesis/candidate_config.json"
CANDIDATE_DECISION = PROJECT_ROOT / "results/genesis/audits/candidate_decision.json"
CANDIDATE_QP_SOURCE = PROJECT_ROOT / "src/learned_execution/candidate_qp.py"
BASE_CONTROLLER_SOURCE = PROJECT_ROOT / "src/deepc/genesis_recovered_deepc.py"
BASE_CONFIG = PROJECT_ROOT / "configs/genesis/deepc_final_recovered.json"
HANKEL_PATH = PROJECT_ROOT / "data/imitation/hankel/final_student_deepc_hankel_v2_stride2.npz"
STUDENT_PATH = PROJECT_ROOT / "models/go2_observable_student_bc_linear_v1.pt"

EXPECTED_STUDENT_SHA256 = "6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc"
EXPECTED_HANKEL_SHA256 = "5ece444b2d82a75df6b751809d2b5f5c1aacfe06cc69a0919fa21532804bfb7b"
C2_DISPLAY_LIMIT = "0.2880066431"
EXACT_C2_SOURCE_LINE = "constraints = [future_input[1::3] + future_input[2::3] <= float(limit)]"
C2_POST_SOLVER_TOLERANCE = 2.0e-8
C2_RETRY_EPS_ABS = 1.0e-8
C2_RETRY_EPS_REL = 1.0e-8
C2_RETRY_MAX_ITER = 100_000
VX_LOWER_LIMIT = -0.8
VX_UPPER_LIMIT = 0.8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE_MODULE: ModuleType | None = None
_CANDIDATE_MODULE: ModuleType | None = None


def _base_module() -> ModuleType:
    global _BASE_MODULE
    if _BASE_MODULE is None:
        _BASE_MODULE = _load_module(BASE_CONTROLLER_SOURCE, "_final_frozen_pre_s6_controller")
    return _BASE_MODULE


def _candidate_module() -> ModuleType:
    global _CANDIDATE_MODULE
    if _CANDIDATE_MODULE is None:
        _CANDIDATE_MODULE = _load_module(CANDIDATE_QP_SOURCE, "_final_vx_limited_qp_source")
    return _CANDIDATE_MODULE


def frozen_artifact_audit() -> dict[str, Any]:
    """Verify that the runtime inputs still equal the frozen artifacts."""

    freeze = _read_json(FINAL_FREEZE)
    constraint = _read_json(FINAL_CONSTRAINT_AUDIT)
    candidate = _read_json(CANDIDATE_CONFIG)
    decision = _read_json(FINAL_VALIDATION_DECISION)
    candidate_source = inspect.getsource(_candidate_module().candidate_solve_qp)
    target_definitions = freeze.get("target_definitions", [])
    hashes = {
        "controller_source": sha256_file(BASE_CONTROLLER_SOURCE),
        "candidate_qp_source": sha256_file(CANDIDATE_QP_SOURCE),
        "base_config": sha256_file(BASE_CONFIG),
        "candidate_config": sha256_file(CANDIDATE_CONFIG),
        "candidate_decision": sha256_file(CANDIDATE_DECISION),
        "final_validation_decision": sha256_file(FINAL_VALIDATION_DECISION),
        "hankel": sha256_file(HANKEL_PATH),
        "student": sha256_file(STUDENT_PATH),
    }
    frozen_parameters = candidate.get("frozen_deepc_parameters", {})
    checks = {
        "freeze_status": freeze.get("status") == "FROZEN",
        "controller_identity": freeze.get("final_controller") == "FINAL_GENESIS_DEEPC_LEARNED_EXECUTION_CHAMPION",
        "controller_source": hashes["controller_source"] == freeze.get("controller_source_hash"),
        "candidate_qp_source": hashes["candidate_qp_source"] == freeze.get("candidate_qp_source_hash"),
        "base_config": hashes["base_config"] == freeze.get("base_config_hash"),
        "candidate_config": hashes["candidate_config"] == freeze.get("candidate_config_hash"),
        "candidate_decision": hashes["candidate_decision"] == freeze.get("candidate_decision_hash"),
        "hankel": hashes["hankel"] == freeze.get("hankel_hash") == EXPECTED_HANKEL_SHA256,
        "student": hashes["student"] == freeze.get("student_hash") == EXPECTED_STUDENT_SHA256,
        "candidate_parameters_equal_base": frozen_parameters == _read_json(BASE_CONFIG),
        "target_definitions": _canonical_digest(target_definitions) == freeze.get("target_definitions_sha256"),
        "candidate_target_independent": candidate.get("constraint", {}).get("target_independent") is True,
        "candidate_post_solver_rewriting_false": candidate.get("constraint", {}).get("post_solver_rewriting") is False,
        "constraint_audit_c2": constraint.get("C2_CONSTRAINT_ACTIVE") == "YES",
        "constraint_audit_c1": constraint.get("C1_ACTIVE") == "NO",
        "constraint_audit_c3": constraint.get("C3_ACTIVE") == "NO",
        "constraint_audit_s6": constraint.get("S6_ACTIVE") == "NO",
        "constraint_audit_canonical_set": constraint.get("CANONICAL_CONTINUOUS_SAFE_SET_ACTIVE") == "NO",
        "constraint_audit_post_solver": constraint.get("post_solver_command_rewriting") == "NO",
        "exact_c2_source_line": candidate_source.count(EXACT_C2_SOURCE_LINE) == 1,
        "candidate_source_has_no_later_safety_path": not any(
            token in candidate_source for token in ("CanonicalSafeSet", "S6", "C1", "C3", "safe_set")
        ),
        "final_validation_pass": decision.get("outcome") == "FINAL_MINIMAL_SAFETY_CANDIDATE_PASS",
    }
    return {
        "schema": "final-genesis-deepc-champion-artifact-audit-v1",
        "final_controller": freeze.get("final_controller"),
        "candidate": candidate.get("candidate_name"),
        "paths": {
            "final_controller_freeze": FINAL_FREEZE,
            "final_constraint_audit": FINAL_CONSTRAINT_AUDIT,
            "final_validation_decision": FINAL_VALIDATION_DECISION,
            "candidate_config": CANDIDATE_CONFIG,
            "candidate_decision": CANDIDATE_DECISION,
            "candidate_qp_source": CANDIDATE_QP_SOURCE,
            "base_controller_source": BASE_CONTROLLER_SOURCE,
            "base_config": BASE_CONFIG,
            "hankel": HANKEL_PATH,
            "student": STUDENT_PATH,
        },
        "hashes": hashes,
        "expected_hashes": {
            "controller_source": freeze.get("controller_source_hash"),
            "candidate_qp_source": freeze.get("candidate_qp_source_hash"),
            "base_config": freeze.get("base_config_hash"),
            "candidate_config": freeze.get("candidate_config_hash"),
            "candidate_decision": freeze.get("candidate_decision_hash"),
            "final_validation_decision": hashes["final_validation_decision"],
            "hankel": freeze.get("hankel_hash"),
            "student": freeze.get("student_hash"),
        },
        "c2": {
            "display_limit": C2_DISPLAY_LIMIT,
            "effective_limit": float(freeze.get("candidate_threshold_effective")),
            "constraint_source_line": EXACT_C2_SOURCE_LINE,
            "inside_optimization": True,
        },
        "vx_box": {
            "lower_limit": VX_LOWER_LIMIT,
            "upper_limit": VX_UPPER_LIMIT,
            "inside_optimization": True,
            "applies_to": "future_input[0::3] for every planned horizon step",
        },
        "target_definitions": target_definitions,
        "target_definitions_sha256": freeze.get("target_definitions_sha256"),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


_BASE_CLASS: type[Any] | None = None


def _base_class() -> type[Any]:
    global _BASE_CLASS
    if _BASE_CLASS is None:
        _BASE_CLASS = _base_module().FinalRecedingDeePC
    return _BASE_CLASS


_base_mod = _base_module()
FinalDeePCError = _base_mod.FinalDeePCError
DeePCUpdate = _base_mod.DeePCUpdate
PoseTarget = _base_mod.PoseTarget
qp_config_from_json = _base_mod.qp_config_from_json
reference_gain_from_json = _base_mod.reference_gain_from_json

C2_LIMIT = float(_read_json(CANDIDATE_CONFIG)["constraint"]["limit"])


class FinalChampionDeePC(_base_class()):
    """Frozen DeePC plus solver-side C2 and forward-speed constraints."""

    def __init__(
        self,
        hankel: Any,
        *,
        qp_config: Any,
        reference_gain: float,
        c2_limit: float = C2_LIMIT,
        vx_lower_limit: float = VX_LOWER_LIMIT,
        vx_upper_limit: float = VX_UPPER_LIMIT,
    ):
        self.c2_limit = float(c2_limit)
        if not np.isfinite(self.c2_limit) or self.c2_limit != C2_LIMIT:
            raise ValueError(f"C2 limit is not the frozen value: {self.c2_limit!r}")
        self.vx_lower_limit = float(vx_lower_limit)
        self.vx_upper_limit = float(vx_upper_limit)
        if not np.isfinite(self.vx_lower_limit) or not np.isfinite(self.vx_upper_limit):
            raise ValueError("vx limits must be finite")
        if self.vx_lower_limit > self.vx_upper_limit:
            raise ValueError("vx_lower_limit must not exceed vx_upper_limit")
        super().__init__(
            hankel,
            qp_config=qp_config,
            reference_gain=reference_gain,
        )
        self.last_solver_retry_used = False
        self.last_solver_retry_reason = ""
        self.last_c2_max_excess = None
        self.last_c2_retry_max_excess = None
        self.last_vx_max_excess = None
        self.last_vx_retry_max_excess = None

    def _tighter_retry_config(self) -> Any:
        """Return a one-shot higher-accuracy configuration for a constraint retry."""

        return replace(
            self.qp_config,
            solver_eps_abs=min(float(self.qp_config.solver_eps_abs), C2_RETRY_EPS_ABS),
            solver_eps_rel=min(float(self.qp_config.solver_eps_rel), C2_RETRY_EPS_REL),
            solver_max_iter=max(int(self.qp_config.solver_max_iter), C2_RETRY_MAX_ITER),
        )

    def _validate_planned_input(self, solution: Any) -> np.ndarray:
        planned = np.asarray(solution.planned_input, dtype=np.float64)
        if planned.shape != (self.hankel.N, 3) or not np.all(np.isfinite(planned)):
            raise FinalDeePCError("DEEPC_SOLVER_FAILURE: invalid planned input")
        return planned

    def _c2_max_excess(self, planned: np.ndarray) -> float:
        values = planned[:, 1] + planned[:, 2] - self.c2_limit
        return float(np.max(values))

    def _vx_max_excess(self, planned: np.ndarray) -> float:
        lower_excess = self.vx_lower_limit - planned[:, 0]
        upper_excess = planned[:, 0] - self.vx_upper_limit
        return float(max(np.max(lower_excess), np.max(upper_excess)))

    def solve(self, measured_pose: np.ndarray, target: Any) -> Any:
        """Run the archived candidate QP and return the normal DeePC update."""

        pose = np.asarray(measured_pose, dtype=np.float64)
        target_pose = target.as_array() if isinstance(target, PoseTarget) else np.asarray(target, dtype=np.float64)
        base = _base_module()
        error = base._pose_error(pose, target_pose)
        reference = base._reference_from_error(self.reference_gain * error, self.hankel.N)
        import time

        self.last_solver_retry_used = False
        self.last_solver_retry_reason = ""
        self.last_c2_max_excess = None
        self.last_c2_retry_max_excess = None
        self.last_vx_max_excess = None
        self.last_vx_retry_max_excess = None

        started = time.perf_counter()
        try:
            solution = _candidate_module().candidate_solve_qp(
                self.hankel,
                self.u_history,
                self.y_history,
                reference,
                self.qp_config,
                self.c2_limit,
                vx_lower=self.vx_lower_limit,
                vx_upper=self.vx_upper_limit,
            )
        except (base.DeePCSolverError, ValueError, FloatingPointError) as exc:
            raise FinalDeePCError(f"DEEPC_SOLVER_FAILURE: {exc}") from exc
        elapsed = time.perf_counter() - started
        planned = self._validate_planned_input(solution)
        initial_c2_excess = self._c2_max_excess(planned)
        initial_vx_excess = self._vx_max_excess(planned)
        self.last_c2_max_excess = initial_c2_excess
        self.last_vx_max_excess = initial_vx_excess
        violations = []
        if initial_c2_excess > C2_POST_SOLVER_TOLERANCE:
            violations.append("C2")
        if initial_vx_excess > C2_POST_SOLVER_TOLERANCE:
            violations.append("VX")
        if violations:
            self.last_solver_retry_used = True
            self.last_solver_retry_reason = "_AND_".join(
                f"{name}_POST_SOLVER_VIOLATION" for name in violations
            )
            retry_started = time.perf_counter()
            try:
                solution = _candidate_module().candidate_solve_qp(
                    self.hankel,
                    self.u_history,
                    self.y_history,
                    reference,
                    self._tighter_retry_config(),
                    self.c2_limit,
                    vx_lower=self.vx_lower_limit,
                    vx_upper=self.vx_upper_limit,
                )
            except (base.DeePCSolverError, ValueError, FloatingPointError) as exc:
                raise FinalDeePCError(
                    "DEEPC_SOLVER_FAILURE: frozen constraint violation; "
                    f"tighter retry failed: {exc}"
                ) from exc
            elapsed += time.perf_counter() - retry_started
            planned = self._validate_planned_input(solution)
            retry_c2_excess = self._c2_max_excess(planned)
            retry_vx_excess = self._vx_max_excess(planned)
            self.last_c2_retry_max_excess = retry_c2_excess
            self.last_vx_retry_max_excess = retry_vx_excess
            if (
                retry_c2_excess > C2_POST_SOLVER_TOLERANCE
                or retry_vx_excess > C2_POST_SOLVER_TOLERANCE
            ):
                raise FinalDeePCError(
                    "DEEPC_SOLVER_FAILURE: frozen constraint violation after one tighter retry; "
                    f"initial_max_excess={initial_c2_excess:.3e}, "
                    f"retry_max_excess={retry_c2_excess:.3e}, "
                    f"initial_vx_max_excess={initial_vx_excess:.3e}, "
                    f"retry_vx_max_excess={retry_vx_excess:.3e}"
                )
        command = np.asarray(planned[0], dtype=np.float32)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise FinalDeePCError("DEEPC_SOLVER_FAILURE: non-finite first command")
        if np.any(np.abs(command.astype(np.float64)) > self.catastrophic_command_abs_limit):
            raise FinalDeePCError("DEEPC_SOLVER_FAILURE: catastrophic command; terminate without substitution")
        predicted_next = base._pose_after_local_increment(pose, solution.predicted_output[0])
        self.solve_count += 1
        self.last_solution = solution
        return DeePCUpdate(
            solve_index=self.solve_count,
            command=command.copy(),
            solution=solution,
            measured_pose=pose.copy(),
            target_pose=target_pose.copy(),
            body_error=error.copy(),
            reference=reference.copy(),
            predicted_next_pose=predicted_next,
            solve_time_s=float(elapsed),
            history_revision=self.history_revision,
        )

    def audit_state(self) -> dict[str, Any]:
        state = super().audit_state()
        state.update(
            {
                "controller_identity": "FINAL_GENESIS_DEEPC_LEARNED_EXECUTION_CHAMPION",
                "c2_limit_display": C2_DISPLAY_LIMIT,
                "c2_limit_effective": self.c2_limit,
                "c2_inside_optimization": True,
                "post_solver_command_rewriting": False,
                "c2_retry_enabled": True,
                "c2_retry_eps_abs": C2_RETRY_EPS_ABS,
                "c2_retry_eps_rel": C2_RETRY_EPS_REL,
                "c2_retry_max_iter": C2_RETRY_MAX_ITER,
                "vx_lower_limit": self.vx_lower_limit,
                "vx_upper_limit": self.vx_upper_limit,
                "vx_inside_optimization": True,
                "last_solver_retry_used": self.last_solver_retry_used,
                "last_solver_retry_reason": self.last_solver_retry_reason,
                "last_c2_max_excess": self.last_c2_max_excess,
                "last_c2_retry_max_excess": self.last_c2_retry_max_excess,
                "last_vx_max_excess": self.last_vx_max_excess,
                "last_vx_retry_max_excess": self.last_vx_retry_max_excess,
            }
        )
        return state
