"""Unified 3-input, measured-feedback receding-horizon DeePC controller.

The controller owns only the DeePC optimization and history update.  It does
not know how to walk, select a gait, switch modes, project commands, or fall
back to another policy.  A caller must execute the returned first command,
measure Genesis, and call :meth:`append_measured_response` before solving
again.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deepc.geometry import wrap_angle  # noqa: E402
from deepc.hankel import HankelData, load_hankel  # noqa: E402
from deepc.qp import DeePCSolution, DeePCSolverError, QPConfig, solve_qp  # noqa: E402

COMMAND_NAMES = ("vx", "vy", "yaw_rate")
OUTPUT_NAMES = ("forward_step_m", "left_step_m", "yaw_step_rad")


class FinalDeePCError(RuntimeError):
    """A solver or contract failure that must terminate the experiment."""


@dataclass(frozen=True)
class PoseTarget:
    x_m: float
    y_m: float
    yaw_rad: float

    def as_array(self) -> np.ndarray:
        value = np.asarray([self.x_m, self.y_m, self.yaw_rad], dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("target pose must be finite")
        return value


@dataclass(frozen=True)
class DeePCUpdate:
    solve_index: int
    command: np.ndarray
    solution: DeePCSolution
    measured_pose: np.ndarray
    target_pose: np.ndarray
    body_error: np.ndarray
    reference: np.ndarray
    predicted_next_pose: np.ndarray
    solve_time_s: float
    history_revision: int


def _pose_error(measured_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(measured_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    if pose.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(target)):
        raise ValueError("poses must be finite shape (3,)")
    dx, dy = target[:2] - pose[:2]
    yaw = float(pose[2])
    return np.asarray(
        [
            np.cos(yaw) * dx + np.sin(yaw) * dy,
            -np.sin(yaw) * dx + np.cos(yaw) * dy,
            wrap_angle(float(target[2] - pose[2])),
        ],
        dtype=np.float64,
    )


def _reference_from_error(error: np.ndarray, horizon: int) -> np.ndarray:
    """Build a linear cumulative pose reference for local motion outputs."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    value = np.asarray(error, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("body error must be finite shape (3,)")
    fractions = np.arange(1, horizon + 1, dtype=np.float64)[:, None] / float(horizon)
    return fractions * value[None, :]


def _pose_after_local_increment(pose: np.ndarray, increment: np.ndarray) -> np.ndarray:
    state = np.asarray(pose, dtype=np.float64)
    step = np.asarray(increment, dtype=np.float64)
    if state.shape != (3,) or step.shape != (3,) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(step)):
        raise ValueError("pose and increment must be finite shape (3,)")
    forward, left, yaw_step = step
    yaw = state[2]
    result = np.asarray(
        [
            state[0] + np.cos(yaw) * forward - np.sin(yaw) * left,
            state[1] + np.sin(yaw) * forward + np.cos(yaw) * left,
            state[2] + yaw_step,
        ],
        dtype=np.float64,
    )
    return result


class FinalRecedingDeePC:
    """History-aware textbook DeePC with explicit measured feedback."""

    def __init__(
        self,
        hankel: HankelData | str | Path,
        *,
        qp_config: QPConfig | None = None,
        catastrophic_command_abs_limit: float = 2.0,
        reference_gain: float = 1.0,
    ):
        self.hankel = load_hankel(hankel) if isinstance(hankel, (str, Path)) else hankel
        if self.hankel.u_dim != 3 or self.hankel.y_dim != 3:
            raise ValueError("final controller requires a 3-input/3-output Hankel package")
        if tuple(self.hankel.u_labels) != COMMAND_NAMES:
            raise ValueError(f"unexpected input labels: {self.hankel.u_labels}")
        if tuple(self.hankel.y_labels) != OUTPUT_NAMES:
            raise ValueError(f"unexpected output labels: {self.hankel.y_labels}")
        self.qp_config = qp_config or QPConfig(
            tracking_weights=(40.0, 40.0, 20.0),
            input_weights=(0.10, 0.10, 0.10),
            past_input_weight=50.0,
            past_output_weight=200.0,
            delta_input_weight=1.0,
            g_weight=1.0e-3,
            # These are deliberately unset.  The characterization region is
            # identification metadata, not a runtime command guard.
            input_lower=None,
            input_upper=None,
            max_delta_input=None,
            solver="OSQP",
            solver_eps_abs=1.0e-6,
            solver_eps_rel=1.0e-6,
            solver_max_iter=100_000,
        )
        self.catastrophic_command_abs_limit = float(catastrophic_command_abs_limit)
        if not np.isfinite(self.catastrophic_command_abs_limit) or self.catastrophic_command_abs_limit <= 0:
            raise ValueError("catastrophic_command_abs_limit must be finite and positive")
        self.reference_gain = float(reference_gain)
        if not np.isfinite(self.reference_gain) or self.reference_gain <= 0:
            raise ValueError("reference_gain must be finite and positive")
        self.u_history = np.zeros((self.hankel.T_ini, 3), dtype=np.float64)
        self.y_history = np.zeros((self.hankel.T_ini, 3), dtype=np.float64)
        self.history_revision = 0
        self.solve_count = 0
        self.measurement_update_count = 0
        self.executed_interval_count = 0
        self.last_solution: DeePCSolution | None = None

    def initialize_measured_history(self, u_history: np.ndarray, y_history: np.ndarray) -> None:
        """Install the actual measured initialization history.

        This method intentionally does not synthesize a stationary history.
        Callers should populate it by applying the defined initialization
        input and recording the resulting Genesis increments.
        """

        u = np.asarray(u_history, dtype=np.float64)
        y = np.asarray(y_history, dtype=np.float64)
        expected = (self.hankel.T_ini, 3)
        if u.shape != expected or y.shape != expected:
            raise ValueError(f"measured histories must have shape {expected}")
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(y)):
            raise FloatingPointError("measured initialization history contains NaN/Inf")
        self.u_history = u.copy()
        self.y_history = y.copy()
        self.history_revision += 1

    def current_history(self) -> tuple[np.ndarray, np.ndarray]:
        return self.u_history.copy(), self.y_history.copy()

    def solve(self, measured_pose: np.ndarray, target: PoseTarget | np.ndarray) -> DeePCUpdate:
        """Solve using only the current measured pose and latest measured history."""

        pose = np.asarray(measured_pose, dtype=np.float64)
        target_pose = target.as_array() if isinstance(target, PoseTarget) else np.asarray(target, dtype=np.float64)
        error = _pose_error(pose, target_pose)
        reference = _reference_from_error(self.reference_gain * error, self.hankel.N)
        start = time.perf_counter()
        try:
            solution = solve_qp(
                self.hankel,
                self.u_history,
                self.y_history,
                reference,
                self.qp_config,
            )
        except (DeePCSolverError, ValueError, FloatingPointError) as exc:
            raise FinalDeePCError(f"DEEPC_SOLVER_FAILURE: {exc}") from exc
        elapsed = time.perf_counter() - start
        command = np.asarray(solution.planned_input[0], dtype=np.float32)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise FinalDeePCError("DEEPC_SOLVER_FAILURE: non-finite first command")
        if np.any(np.abs(command.astype(np.float64)) > self.catastrophic_command_abs_limit):
            raise FinalDeePCError(
                "DEEPC_SOLVER_FAILURE: catastrophic command; terminate without substitution"
            )
        predicted_next = _pose_after_local_increment(pose, solution.predicted_output[0])
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

    def append_measured_response(self, executed_command: np.ndarray, measured_output: np.ndarray) -> None:
        """Append exactly the executed command and actual measured output."""

        command = np.asarray(executed_command, dtype=np.float64)
        output = np.asarray(measured_output, dtype=np.float64)
        if command.shape != (3,) or output.shape != (3,):
            raise ValueError("measured response samples must have shape (3,)")
        if not np.all(np.isfinite(command)) or not np.all(np.isfinite(output)):
            raise FloatingPointError("measured response contains NaN/Inf")
        self.u_history = np.vstack((self.u_history[1:], command))
        self.y_history = np.vstack((self.y_history[1:], output))
        self.history_revision += 1
        self.measurement_update_count += 1
        self.executed_interval_count += 1

    def audit_state(self) -> dict[str, Any]:
        return {
            "T_ini": self.hankel.T_ini,
            "N": self.hankel.N,
            "u_dim": self.hankel.u_dim,
            "y_dim": self.hankel.y_dim,
            "u_labels": list(self.hankel.u_labels),
            "y_labels": list(self.hankel.y_labels),
            "history_revision": self.history_revision,
            "solve_count": self.solve_count,
            "measurement_update_count": self.measurement_update_count,
            "executed_interval_count": self.executed_interval_count,
            "input_lower": self.qp_config.input_lower,
            "input_upper": self.qp_config.input_upper,
            "max_delta_input": self.qp_config.max_delta_input,
            "reference_gain": self.reference_gain,
        }


def qp_config_from_json(path: str | Path) -> QPConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    qp = payload["qp"]
    return QPConfig(
        tracking_weights=tuple(qp["tracking_weights"]),
        input_weights=tuple(qp["input_weights"]),
        past_input_weight=float(qp["past_input_weight"]),
        past_output_weight=float(qp["past_output_weight"]),
        delta_input_weight=float(qp["delta_input_weight"]),
        g_weight=float(qp["g_weight"]),
        input_lower=None,
        input_upper=None,
        max_delta_input=None,
        solver=str(qp.get("solver", "OSQP")),
        solver_eps_abs=float(qp.get("solver_eps_abs", 1.0e-6)),
        solver_eps_rel=float(qp.get("solver_eps_rel", 1.0e-6)),
        solver_max_iter=int(qp.get("solver_max_iter", 100_000)),
    )


def reference_gain_from_json(path: str | Path) -> float:
    """Read the target-reference gain without adding a runtime command rule."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    gain = float(payload.get("deepc", {}).get("reference_gain", 1.0))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("deepc.reference_gain must be finite and positive")
    return gain
