#!/usr/bin/env python3
"""Launch the Genesis viewer and the final target-pose Tkinter UI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Thread
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = ROOT / "src"
RUNTIME_OUTPUT_ROOT = Path(os.environ.get("DEEPC_RUNTIME_OUTPUT", str(ROOT / "results/genesis/generated")))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learned_execution.final_deepc_ui import launch_ui, load_ui_config  # noqa: E402
from learned_execution.genesis_window import GenesisViewerScene  # noqa: E402
from learned_execution.ui_runtime_bridge import UIRuntimeBridge  # noqa: E402
from learned_execution.ui_state import Pose3, pose_errors, wrap_angle  # noqa: E402
from learned_execution.direct_student_runtime import (  # noqa: E402
    DirectStudentBatch,
    FrozenObservableStudent,
    body_increment,
    pose_from_snapshot,
    sha256_file,
)
from learned_execution.frozen_champion_controller import (  # noqa: E402
    FinalDeePCError,
    FinalChampionDeePC,
    PoseTarget,
    qp_config_from_json,
    reference_gain_from_json,
)
from learned_execution.genesis_guard import assert_genesis_only  # noqa: E402

from deepc.hankel import load_hankel  # noqa: E402

DEEPC_DT = 0.10
STUDENT_ACTIONS_PER_SAMPLE = 5
TARGET_POSITION_TOLERANCE_M = 0.06
TARGET_YAW_TOLERANCE_RAD = 0.1
TARGET_YAW_TOLERANCE_DEG = math.degrees(TARGET_YAW_TOLERANCE_RAD)
CONFIRMATION_INTERVALS = 3
MAX_INTERVALS = 100

HANKEL_PATH = ROOT / "data/imitation/hankel/final_student_deepc_hankel_v2_stride2.npz"
CONFIG_PATH = ROOT / "configs/genesis/deepc_final_recovered.json"
CHECKPOINT_PATH = ROOT / "models/go2_observable_student_bc_linear_v1.pt"
EXPECTED_CHECKPOINT_SHA256 = "6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc"


def _pose_update(unwrapped: np.ndarray, snapshot: dict[str, np.ndarray]) -> np.ndarray:
    current = pose_from_snapshot(snapshot)[0]
    updated = np.asarray(unwrapped, dtype=np.float64).copy()
    updated[:2] = current[:2]
    updated[2] += math.atan2(math.sin(float(current[2] - unwrapped[2])), math.cos(float(current[2] - unwrapped[2])))
    return updated


def _pose3(value: np.ndarray) -> Pose3:
    return Pose3(float(value[0]), float(value[1]), wrap_angle(float(value[2])))


def _safety_failure(result: Any) -> str | None:
    checks = (
        ("fall", "fall"),
        ("base_contact", "base contact"),
        ("nan_inf", "NaN/Inf"),
        ("actual_joint_limit_violation", "actual joint-limit violation"),
        ("torque_limit_exceedance", "torque-limit exceedance"),
    )
    for key, label in checks:
        if bool(result.safety[key][0]):
            return label
    return None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Pose3):
        return value.as_tuple()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class RunLogger:
    """Append one JSON object per measured sample or replan event."""

    def __init__(self, root: Path, run_id: str, target: Pose3, target_id: str | None = None) -> None:
        self.path = root / "logs" / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self.write(
            {
                "event": "run_started",
                "run_id": run_id,
                "target_x": target.x_m,
                "target_y": target.y_m,
                "target_yaw": target.yaw_rad,
                "target_yaw_deg": target.yaw_deg,
                "target_id": target_id or "",
                "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
                "hankel": str(HANKEL_PATH),
                "config": str(CONFIG_PATH),
            }
        )

    def write(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(_json_value(payload), sort_keys=True) + "\n")
        self._stream.flush()

    def close(
        self,
        outcome: str,
        *,
        final_pose: Pose3 | None = None,
        elapsed_s: float = 0.0,
        replans: int = 0,
        exit_reason: str = "",
        failure_detail: str = "",
    ) -> None:
        self.write(
            {
                "event": "run_finished",
                "outcome": outcome,
                "final_pose": final_pose,
                "elapsed_s": elapsed_s,
                "replan_count": replans,
                "exit_reason": exit_reason,
                "failure_detail": failure_detail,
            }
        )
        self._stream.close()


@dataclass
class RuntimeSession:
    controller: FinalChampionDeePC
    measured_pose: np.ndarray
    history_ready: bool = True


class LiveProductRuntime:
    """Run the recovered controller behind a non-blocking UI bridge."""

    def __init__(
        self,
        bridge: UIRuntimeBridge,
        *,
        max_intervals: int = MAX_INTERVALS,
        viewer_res: tuple[int, int] = (880, 760),
        show_viewer: bool = True,
    ) -> None:
        self.bridge = bridge
        self.max_intervals = int(max_intervals)
        self.viewer_res = viewer_res
        self.show_viewer = bool(show_viewer)
        self.scene: Any = None
        self.student: FrozenObservableStudent | None = None
        self.direct_runtime: DirectStudentBatch | None = None
        self.session: RuntimeSession | None = None
        self._thread: Thread | None = None
        self._thread_lock = RLock()
        self._measured_pose = np.zeros(3, dtype=np.float64)
        self._logger: RunLogger | None = None
        self._run_started_at = 0.0
        self.bridge.bind_run_starter(self.start)

    def start(self) -> bool:
        """Start the single Genesis worker, if it is not already running."""

        with self._thread_lock:
            if self.bridge.shutdown_event.is_set():
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = Thread(target=self._worker, name="genesis-deepc-runtime", daemon=True)
            self._thread.start()
            return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        self.bridge.request_shutdown()

    def _worker(self) -> None:
        try:
            assert_genesis_only("pre_genesis_scene")
            self.scene = GenesisViewerScene(viewer_res=self.viewer_res) if self.show_viewer else self._make_headless_scene()
            assert_genesis_only("post_genesis_scene")
            self.bridge.set_viewer_available(bool(self.show_viewer and self.scene.viewer_alive()))
            self.student = FrozenObservableStudent()
            self.direct_runtime = DirectStudentBatch(self.scene, self.student)
            self._prepare_session(reset_scene=True)
            self._command_loop()
        except Exception as exc:  # startup errors are visible in the product state
            self.bridge.publish_safety(False, f"Runtime startup failure: {exc}")
            self.bridge.publish_controller_state("INTERFACE FAILURE", detail=str(exc))
        finally:
            if self._logger is not None:
                self._logger.close(
                    "shutdown",
                    final_pose=_pose3(self._measured_pose),
                    replans=self._replans(),
                    exit_reason="WORKER_SHUTDOWN",
                )
                self._logger = None
            if self.scene is not None:
                with suppress(Exception):
                    if self.bridge.snapshot().controller_state in {"RUNNING", "STOPPED", "SAFETY STOP"}:
                        self.scene.safe_hold()
                with suppress(Exception):
                    self.scene.close()

    def _make_headless_scene(self) -> Any:
        from learned_execution.direct_student_runtime import make_genesis_scene

        return make_genesis_scene(1, camera=False)

    def _command_loop(self) -> None:
        while not self.bridge.shutdown_event.is_set():
            request = self.bridge.consume_request(timeout=0.05)
            if request is None:
                if self.bridge.snapshot().controller_state == "RUNNING" and self.scene is not None and not self.scene.viewer_alive() and self.show_viewer:
                    self.bridge.request_emergency_stop("Genesis viewer closed")
                continue
            if request.kind == "shutdown":
                return
            if request.kind == "reset":
                self._prepare_session(reset_scene=True)
                continue
            if request.kind == "run" and request.target_pose is not None:
                self._execute_target(request.target_pose, target_id=request.target_id)
                continue
            if request.kind in {"stop", "emergency_stop"}:
                continue

    def _prepare_session(self, *, reset_scene: bool) -> None:
        assert_genesis_only("session_prepare")
        if self.scene is None or self.direct_runtime is None:
            raise RuntimeError("Genesis scene is not initialized")
        if reset_scene:
            self.scene.reset()
        self.direct_runtime.reset()
        self._measured_pose = pose_from_snapshot(self.direct_runtime.observe())[0].astype(np.float64)
        self.bridge.publish_actual_pose(_pose3(self._measured_pose), elapsed_s=0.0)
        hankel = load_hankel(HANKEL_PATH)
        controller = FinalChampionDeePC(
            hankel,
            qp_config=qp_config_from_json(CONFIG_PATH),
            reference_gain=reference_gain_from_json(CONFIG_PATH),
        )
        init_u: list[np.ndarray] = []
        init_y: list[np.ndarray] = []
        for _ in range(hankel.T_ini):
            if self.bridge.shutdown_event.is_set() or self.bridge.stop_event.is_set():
                self.bridge.publish_controller_state("STOPPED", detail="Initialization stopped")
                self.bridge.publish_safety(False, "Initialization interrupted")
                return
            start_pose = self._measured_pose.copy()
            command = np.zeros((1, 3), dtype=np.float32)
            last_step = None
            for _ in range(STUDENT_ACTIONS_PER_SAMPLE):
                last_step = self.direct_runtime.step(command)
                self._measured_pose = _pose_update(self._measured_pose, last_step.after)
                self.bridge.publish_actual_pose(_pose3(self._measured_pose), elapsed_s=0.0)
                failure = _safety_failure(last_step)
                if failure is not None:
                    self.bridge.publish_safety(False, failure)
                    self.bridge.publish_controller_state("SAFETY STOP", detail=failure)
                    return
            assert last_step is not None
            # The final observation after the hold is the Genesis measurement
            # installed in the DeePC history; no prediction is used here.
            self._measured_pose = _pose_update(self._measured_pose, self.direct_runtime.observe())
            init_u.append(command[0].astype(np.float64))
            init_y.append(body_increment(start_pose[None, :], self._measured_pose[None, :])[0].astype(np.float64))
        controller.initialize_measured_history(np.asarray(init_u), np.asarray(init_y))
        self.session = RuntimeSession(controller=controller, measured_pose=self._measured_pose.copy())
        self.bridge.publish_ready(actual_pose=_pose3(self._measured_pose))

    def _execute_target(self, target: Pose3, *, target_id: str | None = None) -> None:
        if self.session is None or not self.session.history_ready:
            self._prepare_session(reset_scene=False)
        if self.session is None:
            return
        target_obj = PoseTarget(target.x_m, target.y_m, target.yaw_rad)
        run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._run_started_at = time.monotonic()
        self.bridge.set_run_metadata(run_id)
        self._logger = RunLogger(RUNTIME_OUTPUT_ROOT, run_id, target, target_id=target_id)
        confirmation = 0
        iteration = 0
        run_active = True
        outcome = "STOPPED"
        exit_reason = "UNKNOWN"
        failure_detail = ""
        try:
            while run_active:
                requested_exit = self._requested_exit_reason()
                if requested_exit is not None:
                    exit_reason = requested_exit
                    outcome = "SAFETY STOP" if requested_exit == "EMERGENCY_STOP" else "STOPPED"
                    run_active = False
                    self.bridge.publish_controller_state(
                        outcome,
                        detail="Emergency stop requested" if requested_exit == "EMERGENCY_STOP" else "User stop requested",
                        elapsed_s=self._elapsed(),
                        run_active=False,
                        exit_reason=exit_reason,
                    )
                    break
                if iteration >= self.max_intervals:
                    # max_intervals is an explicit experiment timeout.  A
                    # completed hold never ends this loop by itself.
                    exit_reason = "EXPERIMENT_TIMEOUT"
                    outcome = "STOPPED"
                    run_active = False
                    self.bridge.publish_controller_state(
                        "STOPPED",
                        detail=f"Explicit experiment timeout after {iteration} replans",
                        elapsed_s=self._elapsed(),
                        run_active=False,
                        exit_reason=exit_reason,
                    )
                    break
                if self.show_viewer and self.scene is not None and not self.scene.viewer_alive():
                    self.bridge.request_emergency_stop("Genesis viewer closed")
                    outcome = "SAFETY STOP"
                    exit_reason = "EMERGENCY_STOP"
                    run_active = False
                    break
                # Read the latest actual Genesis pose immediately before every
                # solve.  The controller's u_ini/y_ini are the measured
                # histories maintained by the frozen recovered controller.
                assert_genesis_only("before_deepc_solve")
                self._measured_pose = _pose_update(self._measured_pose, self.direct_runtime.observe())
                self.session.measured_pose = self._measured_pose.copy()
                try:
                    update = self.session.controller.solve(self._measured_pose, target_obj)
                except FinalDeePCError as exc:
                    outcome = "SOLVER FAILURE"
                    exit_reason = "SOLVER_FAILURE"
                    failure_detail = str(exc)
                    run_active = False
                    self.bridge.publish_controller_state(
                        "SOLVER FAILURE",
                        detail=str(exc),
                        elapsed_s=self._elapsed(),
                        run_active=False,
                        exit_reason=exit_reason,
                    )
                    break

                # ``planned_input`` is the frozen controller's future_u.  The
                # live product takes only its first row and sends that exact
                # array to the student; no runtime command transform exists.
                solver_selected_command = np.asarray(update.solution.command, dtype=np.float64).reshape(3).copy()
                future_u = np.asarray(update.solution.planned_input)
                deepc_command = np.asarray(future_u[0], dtype=np.float32).copy()
                if not np.array_equal(deepc_command, np.asarray(update.command, dtype=np.float32)):
                    raise AssertionError("controller command is not future_u[0]")
                solver_delivery_ok = bool(np.allclose(solver_selected_command, deepc_command, rtol=0.0, atol=1.0e-6))
                if not solver_delivery_ok:
                    raise AssertionError("solver-selected command changed before delivery")
                c2_margin = float(self.session.controller.c2_limit - solver_selected_command[1] - solver_selected_command[2])
                if c2_margin < -2.0e-8:
                    raise AssertionError("solver-selected command violates frozen C2")
                vx_lower_limit = float(self.session.controller.vx_lower_limit)
                vx_upper_limit = float(self.session.controller.vx_upper_limit)
                vx_command = float(solver_selected_command[0])
                if vx_command < vx_lower_limit - 2.0e-8 or vx_command > vx_upper_limit + 2.0e-8:
                    raise AssertionError("solver-selected command violates frozen vx box")
                predicted_next_pose = np.asarray(update.predicted_next_pose, dtype=np.float64).copy()
                start_pose = self._measured_pose.copy()
                last_step = None
                direct_ok = True
                student_received_command: np.ndarray | None = None
                failure: str | None = None
                hold_interrupted = False
                for _ in range(STUDENT_ACTIONS_PER_SAMPLE):
                    last_step = self.direct_runtime.step(deepc_command[None, :])
                    student_received_command = last_step.student_received_command[0].copy()
                    direct_ok = bool(np.array_equal(deepc_command, student_received_command))
                    self._measured_pose = _pose_update(self._measured_pose, last_step.after)
                    self.bridge.publish_actual_pose(_pose3(self._measured_pose), elapsed_s=self._elapsed())
                    self._log_measurement(
                        target,
                        deepc_command,
                        student_received_command,
                        direct_ok and solver_delivery_ok,
                        failure=None,
                        solver_selected_command=solver_selected_command,
                        c2_margin=c2_margin,
                    )
                    if not direct_ok:
                        failure = "DIRECT_DEEPC_TO_STUDENT_MISMATCH"
                        break
                    failure = _safety_failure(last_step)
                    if failure is not None:
                        break
                    if self._requested_exit_reason() is not None:
                        hold_interrupted = True
                        break
                if failure is not None:
                    outcome = "INTERFACE FAILURE" if not direct_ok else "SAFETY STOP"
                    exit_reason = "INTERFACE_FAILURE" if not direct_ok else "SAFETY_STOP"
                    run_active = False
                    if direct_ok:
                        self.bridge.publish_safety(False, failure)
                    self.bridge.publish_controller_state(
                        outcome,
                        detail=failure,
                        elapsed_s=self._elapsed(),
                        run_active=False,
                        exit_reason=exit_reason,
                    )
                    break
                if hold_interrupted or last_step is None or student_received_command is None:
                    requested_exit = self._requested_exit_reason() or "USER_STOP"
                    exit_reason = requested_exit
                    outcome = "SAFETY STOP" if requested_exit == "EMERGENCY_STOP" else "STOPPED"
                    run_active = False
                    self.bridge.publish_controller_state(
                        outcome,
                        detail="Run interrupted during the DeePC hold interval",
                        elapsed_s=self._elapsed(),
                        run_active=False,
                        exit_reason=exit_reason,
                    )
                    break

                # Measure the actual resulting Genesis state after all five
                # 50 Hz student actions.  This measured pose, never the
                # predicted pose, is converted to the history output.
                self._measured_pose = _pose_update(self._measured_pose, self.direct_runtime.observe())
                self.bridge.publish_actual_pose(_pose3(self._measured_pose), elapsed_s=self._elapsed())
                actual_next_pose = self._measured_pose.copy()
                actual_output = body_increment(start_pose[None, :], actual_next_pose[None, :])[0]
                before_revision = self.session.controller.history_revision
                self.session.controller.append_measured_response(deepc_command, actual_output)
                measured_feedback_ok = self.session.controller.history_revision == before_revision + 1
                iteration += 1
                replan_count = self.session.controller.measurement_update_count
                if not (
                    replan_count == iteration
                    and self.session.controller.solve_count == replan_count
                    and self.session.controller.executed_interval_count == replan_count
                ):
                    raise AssertionError("solve/execute/measured replan counts diverged")
                self.session.measured_pose = actual_next_pose.copy()
                position_error, yaw_error = pose_errors(_pose3(actual_next_pose), target)
                assert position_error is not None and yaw_error is not None
                if position_error <= TARGET_POSITION_TOLERANCE_M and yaw_error <= TARGET_YAW_TOLERANCE_RAD:
                    confirmation += 1
                else:
                    confirmation = 0
                target_reached = confirmation >= CONFIRMATION_INTERVALS
                if target_reached:
                    outcome = "TARGET REACHED"
                    exit_reason = "TARGET_REACHED"
                    run_active = False
                else:
                    # The next iteration starts immediately at the top of the
                    # worker-owned while loop with the newly measured history.
                    run_active = True
                self.bridge.publish_deepc(
                    deepc_command,
                    student_received_command=student_received_command,
                    solver_selected_command=solver_selected_command,
                    delivered_command=deepc_command,
                    replan_count=replan_count,
                    solver_status=update.solution.status,
                    solve_time_s=update.solve_time_s,
                    elapsed_s=self._elapsed(),
                    direct_command_ok=direct_ok,
                    measured_feedback_ok=measured_feedback_ok,
                    iteration=iteration,
                    predicted_next_pose=_pose3(predicted_next_pose),
                    actual_next_pose=_pose3(actual_next_pose),
                    position_error=position_error,
                    yaw_error=yaw_error,
                    c2_margin=c2_margin,
                    c2_limit=self.session.controller.c2_limit,
                    post_solver_rewriting=False,
                    run_active=run_active,
                    exit_reason=exit_reason if target_reached else "",
                )
                self._log_iteration(
                    target,
                    iteration=iteration,
                    replan_count=replan_count,
                    deepc_command=deepc_command,
                    student_received_command=student_received_command,
                    predicted_next_pose=predicted_next_pose,
                    actual_next_pose=actual_next_pose,
                    position_error=position_error,
                    yaw_error=yaw_error,
                    run_active=run_active,
                    exit_reason=exit_reason if target_reached else "",
                    update=update,
                    actual_output=actual_output,
                    direct_ok=direct_ok,
                    solver_delivery_ok=solver_delivery_ok,
                    solver_selected_command=solver_selected_command,
                    c2_margin=c2_margin,
                    measured_feedback_ok=measured_feedback_ok,
                    history_revision_before=before_revision,
                    history_revision_after=self.session.controller.history_revision,
                )
                if target_reached:
                    self.bridge.publish_controller_state(
                        "TARGET REACHED",
                        detail=f"Measured tolerance confirmed · {replan_count} replans",
                        elapsed_s=self._elapsed(),
                        run_active=False,
                        exit_reason=exit_reason,
                    )
                    break
        except Exception as exc:
            outcome = "INTERFACE FAILURE"
            exit_reason = "INTERFACE_FAILURE"
            failure_detail = str(exc)
            run_active = False
            self.bridge.publish_controller_state(
                "INTERFACE FAILURE",
                detail=str(exc),
                elapsed_s=self._elapsed(),
                run_active=False,
                exit_reason=exit_reason,
            )
        finally:
            if self._logger is not None:
                self._logger.close(
                    outcome,
                    final_pose=_pose3(self._measured_pose),
                    elapsed_s=self._elapsed(),
                    replans=self._replans(),
                    exit_reason=exit_reason,
                    failure_detail=failure_detail,
                )
                self._logger = None
            if self.session is not None:
                self.session.history_ready = False
            self.bridge.stop_event.clear()
            if outcome in {"STOPPED", "SAFETY STOP", "SOLVER FAILURE", "INTERFACE FAILURE"}:
                with suppress(Exception):
                    self.scene.safe_hold()

    def _requested_exit_reason(self) -> str | None:
        """Return only an externally requested termination reason."""

        if self.bridge.shutdown_event.is_set() or self.bridge.stop_event.is_set():
            return "EMERGENCY_STOP" if not self.bridge.snapshot().safety_state.ok else "USER_STOP"
        return None

    def _log_measurement(
        self,
        target: Pose3,
        command: np.ndarray,
        received: np.ndarray,
        direct_ok: bool,
        failure: str | None,
        *,
        solver_selected_command: np.ndarray | None = None,
        c2_margin: float | None = None,
    ) -> None:
        if self._logger is None:
            return
        position_error, yaw_error = pose_errors(_pose3(self._measured_pose), target)
        self._logger.write(
            {
                "event": "measured_pose",
                "phase": "genesis_50hz",
                "run_id": self.bridge.snapshot().run_id,
                "timestamp_s": self._elapsed(),
                "target_x": target.x_m,
                "target_y": target.y_m,
                "target_yaw": target.yaw_rad,
                "actual_x": self._measured_pose[0],
                "actual_y": self._measured_pose[1],
                "actual_yaw": self._measured_pose[2],
                "position_error_m": position_error,
                "yaw_error_rad": yaw_error,
                "deepc_command": command,
                "solver_selected_command": solver_selected_command,
                "delivered_command": command,
                "student_received_command": received,
                "command_equal": direct_ok,
                "post_solver_rewriting": False,
                "c2_margin": c2_margin,
                "solver_retry_used": bool(getattr(self.session.controller, "last_solver_retry_used", False)),
                "solver_retry_reason": getattr(self.session.controller, "last_solver_retry_reason", ""),
                "c2_max_excess_before_retry": getattr(self.session.controller, "last_c2_max_excess", None),
                "c2_max_excess_after_retry": getattr(self.session.controller, "last_c2_retry_max_excess", None),
                "vx_lower_limit": getattr(self.session.controller, "vx_lower_limit", None),
                "vx_upper_limit": getattr(self.session.controller, "vx_upper_limit", None),
                "vx_margin": (
                    min(
                        float(command[0]) - float(self.session.controller.vx_lower_limit),
                        float(self.session.controller.vx_upper_limit) - float(command[0]),
                    )
                    if self.session is not None
                    else None
                ),
                "vx_max_excess_before_retry": getattr(self.session.controller, "last_vx_max_excess", None),
                "vx_max_excess_after_retry": getattr(self.session.controller, "last_vx_retry_max_excess", None),
                "replan_count": self._replans(),
                "solver_status": self.bridge.snapshot().solver_status,
                "solve_time_s": self.bridge.snapshot().solve_time_s,
                "safety_ok": self.bridge.snapshot().safety_state.ok,
                "safety_reason": failure or self.bridge.snapshot().safety_state.reason,
            }
        )

    def _log_iteration(
        self,
        target: Pose3,
        *,
        iteration: int,
        replan_count: int,
        deepc_command: np.ndarray,
        student_received_command: np.ndarray,
        solver_selected_command: np.ndarray,
        c2_margin: float,
        solver_delivery_ok: bool,
        predicted_next_pose: np.ndarray,
        actual_next_pose: np.ndarray,
        position_error: float,
        yaw_error: float,
        run_active: bool,
        exit_reason: str,
        update: Any,
        actual_output: np.ndarray,
        direct_ok: bool,
        measured_feedback_ok: bool,
        history_revision_before: int,
        history_revision_after: int,
    ) -> None:
        if self._logger is None:
            return
        self._logger.write(
            {
                "event": "receding_horizon_iteration",
                "phase": "deePC_0p10s",
                "run_id": self.bridge.snapshot().run_id,
                "timestamp_s": self._elapsed(),
                "iteration": iteration,
                "replan_count": replan_count,
                "target_x": target.x_m,
                "target_y": target.y_m,
                "target_yaw": target.yaw_rad,
                "deepc_command": deepc_command,
                "solver_selected_command": solver_selected_command,
                "delivered_command": deepc_command,
                "student_received_command": student_received_command,
                "command_equal": direct_ok and solver_delivery_ok,
                "solver_delivery_ok": solver_delivery_ok,
                "post_solver_rewriting": False,
                "c2_limit": self.session.controller.c2_limit if self.session is not None else None,
                "c2_margin": c2_margin,
                "solver_retry_used": bool(getattr(self.session.controller, "last_solver_retry_used", False)),
                "solver_retry_reason": getattr(self.session.controller, "last_solver_retry_reason", ""),
                "c2_max_excess_before_retry": getattr(self.session.controller, "last_c2_max_excess", None),
                "c2_max_excess_after_retry": getattr(self.session.controller, "last_c2_retry_max_excess", None),
                "vx_lower_limit": getattr(self.session.controller, "vx_lower_limit", None),
                "vx_upper_limit": getattr(self.session.controller, "vx_upper_limit", None),
                "vx_margin": min(
                    float(deepc_command[0]) - float(self.session.controller.vx_lower_limit),
                    float(self.session.controller.vx_upper_limit) - float(deepc_command[0]),
                ),
                "vx_max_excess_before_retry": getattr(self.session.controller, "last_vx_max_excess", None),
                "vx_max_excess_after_retry": getattr(self.session.controller, "last_vx_retry_max_excess", None),
                "actual_output": actual_output,
                "predicted_next_pose": predicted_next_pose,
                "actual_next_pose": actual_next_pose,
                "position_error": position_error,
                "yaw_error": yaw_error,
                "position_error_m": position_error,
                "yaw_error_rad": yaw_error,
                "run_active": run_active,
                "exit_reason": exit_reason,
                "solver_status": update.solution.status,
                "solve_time_s": update.solve_time_s,
                "safety_ok": self.bridge.snapshot().safety_state.ok,
                "safety_reason": self.bridge.snapshot().safety_state.reason,
                "measured_feedback_replanning": measured_feedback_ok,
                "predicted_pose_used_as_truth": False,
                "history_revision_before": history_revision_before,
                "history_revision_after": history_revision_after,
            }
        )

    def _elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._run_started_at) if self._run_started_at else 0.0

    def _replans(self) -> int:
        return self.session.controller.solve_count if self.session is not None else 0


def _default_target(config: dict[str, Any]) -> tuple[Pose3, str]:
    target_id = str(config.get("default_target_id", "P1"))
    for item in config.get("target_presets", []):
        if str(item.get("target_id")) == target_id:
            return Pose3(float(item["x_m"]), float(item["y_m"]), float(item["yaw_rad"])).normalized(), target_id
    return Pose3.from_degrees(0.25, 0.0, 0.0), "P1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/genesis/ui_config.json")
    parser.add_argument("--max-intervals", type=int, default=MAX_INTERVALS)
    parser.add_argument("--headless", action="store_true", help="Use the validated headless Genesis scene; Tkinter remains visible")
    args = parser.parse_args(argv)

    config = load_ui_config(args.config)
    bridge = UIRuntimeBridge(max_trail_points=int(config.get("map", {}).get("max_trail_points", 25_000)))
    default_target, default_target_id = _default_target(config)
    bridge.request_target(default_target, source=f"default preset {default_target_id}", target_id=default_target_id)
    runtime_cfg = config.get("runtime", {})
    viewer_res = tuple(runtime_cfg.get("genesis_viewer_res", [880, 760]))
    runtime = LiveProductRuntime(
        bridge,
        max_intervals=args.max_intervals,
        viewer_res=(int(viewer_res[0]), int(viewer_res[1])),
        show_viewer=not args.headless,
    )
    runtime.start()

    def on_close() -> None:
        runtime.stop()
        runtime.join(timeout=2.0)

    ui = launch_ui(bridge, config=config, on_close=on_close)
    try:
        ui.mainloop()
    finally:
        runtime.stop()
        runtime.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
