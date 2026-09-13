"""Thread-safe target/telemetry bridge between Tkinter and the live runtime."""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, RLock
from typing import Literal

from .ui_state import (
    ControllerState,
    LiveSystemState,
    Pose3,
    SafetyState,
    command_tuple,
    pose_errors,
)

RequestKind = Literal["run", "stop", "reset", "emergency_stop", "shutdown"]


@dataclass(frozen=True)
class RuntimeRequest:
    kind: RequestKind
    target_pose: Pose3 | None = None
    target_id: str | None = None
    created_at: float = 0.0


class UIRuntimeBridge:
    """Own the small amount of synchronization required by the product.

    Tkinter writes target and button requests.  The Genesis thread publishes
    measured pose, DeePC telemetry, and safety state.  The UI never waits for
    a solver or a physics step.
    """

    def __init__(self, *, max_trail_points: int = 25_000) -> None:
        if max_trail_points < 100:
            raise ValueError("max_trail_points must leave room for a useful trail")
        self._lock = RLock()
        self._requests: queue.Queue[RuntimeRequest] = queue.Queue()
        self._stop_event = Event()
        self._shutdown_event = Event()
        self._max_trail_points = int(max_trail_points)
        self._state = LiveSystemState()
        self._run_starter: Callable[[], object] | None = None

    def bind_run_starter(self, starter: Callable[[], object]) -> None:
        """Bind the worker starter used by RUN.

        The callback is invoked outside the bridge lock.  It must be
        idempotent: RUN is a UI request, while the runtime owns the worker
        thread and the receding-horizon loop.
        """

        with self._lock:
            self._run_starter = starter

    @property
    def stop_event(self) -> Event:
        return self._stop_event

    @property
    def shutdown_event(self) -> Event:
        return self._shutdown_event

    def snapshot(self) -> LiveSystemState:
        with self._lock:
            return self._state

    def _set_state(self, **changes: object) -> None:
        with self._lock:
            self._state = self._state.__class__(**{**self._state.__dict__, **changes})

    def set_viewer_available(self, available: bool) -> None:
        self._set_state(viewer_available=bool(available))

    def request_target(self, target_pose: Pose3, *, source: str = "numeric", target_id: str | None = None) -> bool:
        target = target_pose.normalized()
        with self._lock:
            if self._state.controller_state == "RUNNING":
                return False
            position_error, yaw_error = pose_errors(self._state.actual_pose, target)
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "target_pose": target,
                    "active_target_pose": None,
                    "position_error_m": position_error,
                    "yaw_error_rad": yaw_error,
                    "target_source": source,
                    "target_id": "" if target_id is None else str(target_id),
                    "target_committed": False,
                    "status_detail": "Target pose ready",
                }
            )
        return True

    def request_run(self) -> bool:
        starter: Callable[[], object] | None
        with self._lock:
            target = self._state.target_pose
            target_id = self._state.target_id
            if target is None or self._state.controller_state == "RUNNING":
                return False
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "active_target_pose": target,
                    "target_committed": True,
                    "controller_state": "RUNNING",
                    "status_detail": "Target committed; waiting for measured feedback",
                    "run_active": True,
                    "iteration": 0,
                    "replan_count": 0,
                    "exit_reason": "",
                }
            )
            starter = self._run_starter
        self._requests.put(
            RuntimeRequest(
                "run",
                target_pose=target,
                target_id=target_id,
                created_at=time.monotonic(),
            )
        )
        if starter is not None:
            # Starting the worker is deliberately separate from advancing the
            # controller.  The worker consumes the queued RUN request and
            # owns every solve/hold/measure/replan transition.
            starter()
        return True

    def request_stop(self) -> bool:
        with self._lock:
            if self._state.controller_state != "RUNNING":
                return False
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "controller_state": "STOPPED",
                    "status_detail": "User stop requested",
                    "run_active": False,
                    "exit_reason": "USER_STOP",
                }
            )
        self._stop_event.set()
        self._requests.put(RuntimeRequest("stop", created_at=time.monotonic()))
        return True

    def request_emergency_stop(self, reason: str = "User emergency stop") -> None:
        with self._lock:
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "controller_state": "SAFETY STOP",
                    "safety_state": SafetyState(False, reason),
                    "status_detail": reason,
                    "run_active": False,
                    "exit_reason": "EMERGENCY_STOP",
                }
            )
        self._stop_event.set()
        self._requests.put(RuntimeRequest("emergency_stop", created_at=time.monotonic()))

    def request_reset(self) -> bool:
        with self._lock:
            if self._state.controller_state == "RUNNING":
                return False
            viewer_available = self._state.viewer_available
            self._state = LiveSystemState(
                controller_state="STARTING",
                safety_state=SafetyState(),
                status_detail="Resetting Genesis and measured history",
                viewer_available=viewer_available,
                exit_reason="RESET",
            )
        self._stop_event.clear()
        self._requests.put(RuntimeRequest("reset", created_at=time.monotonic()))
        return True

    def request_shutdown(self) -> None:
        self._shutdown_event.set()
        self._stop_event.set()
        self._requests.put(RuntimeRequest("shutdown", created_at=time.monotonic()))

    def request_clear_trail(self) -> None:
        with self._lock:
            self._state = self._state.__class__(**{**self._state.__dict__, "trail": tuple()})

    def consume_request(self, timeout: float = 0.05) -> RuntimeRequest | None:
        try:
            return self._requests.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish_actual_pose(self, pose: Pose3, *, elapsed_s: float | None = None) -> None:
        actual = pose.normalized()
        with self._lock:
            target = self._state.active_target_pose or self._state.target_pose
            position_error, yaw_error = pose_errors(actual, target)
            trail = (*self._state.trail, actual)
            if len(trail) > self._max_trail_points:
                trail = trail[-self._max_trail_points :]
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "actual_pose": actual,
                    "trail": trail,
                    "position_error_m": position_error,
                    "yaw_error_rad": yaw_error,
                    "actual_sample_count": self._state.actual_sample_count + 1,
                    "elapsed_s": self._state.elapsed_s if elapsed_s is None else float(elapsed_s),
                }
            )

    def publish_deepc(
        self,
        command: object,
        *,
        student_received_command: object,
        solver_selected_command: object | None = None,
        delivered_command: object | None = None,
        replan_count: int,
        solver_status: str,
        solve_time_s: float,
        elapsed_s: float,
        direct_command_ok: bool,
        measured_feedback_ok: bool,
        iteration: int | None = None,
        predicted_next_pose: Pose3 | None = None,
        actual_next_pose: Pose3 | None = None,
        position_error: float | None = None,
        yaw_error: float | None = None,
        c2_margin: float | None = None,
        c2_limit: float | None = None,
        post_solver_rewriting: bool | None = None,
        run_active: bool | None = None,
        exit_reason: str | None = None,
    ) -> None:
        requested = command_tuple(command)
        received = command_tuple(student_received_command)
        selected = requested if solver_selected_command is None else command_tuple(solver_selected_command)
        delivered = received if delivered_command is None else command_tuple(delivered_command)
        with self._lock:
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "deepc_command": requested,
                    "student_received_command": received,
                    "solver_selected_command": selected,
                    "delivered_command": delivered,
                    "direct_command_ok": bool(direct_command_ok),
                    "replan_count": int(replan_count),
                    "solver_status": str(solver_status),
                    "solve_time_s": float(solve_time_s),
                    "elapsed_s": float(elapsed_s),
                    "measured_feedback_ok": bool(measured_feedback_ok),
                    "iteration": self._state.iteration if iteration is None else int(iteration),
                    "predicted_next_pose": predicted_next_pose,
                    "actual_next_pose": actual_next_pose,
                    "position_error_m": self._state.position_error_m if position_error is None else float(position_error),
                    "yaw_error_rad": self._state.yaw_error_rad if yaw_error is None else float(yaw_error),
                    "c2_margin": None if c2_margin is None else float(c2_margin),
                    "c2_limit": None if c2_limit is None else float(c2_limit),
                    "post_solver_rewriting": post_solver_rewriting,
                    "run_active": self._state.run_active if run_active is None else bool(run_active),
                    "exit_reason": self._state.exit_reason if exit_reason is None else str(exit_reason),
                }
            )

    def publish_ready(self, *, actual_pose: Pose3 | None = None, detail: str = "Genesis ready") -> None:
        if actual_pose is not None:
            self.publish_actual_pose(actual_pose)
        with self._lock:
            queued_run = self._state.controller_state == "RUNNING" and self._state.active_target_pose is not None
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "controller_state": "RUNNING" if queued_run else "READY",
                    "safety_state": SafetyState(),
                    "status_detail": "Target committed; starting measured-feedback run" if queued_run else detail,
                    "target_committed": queued_run,
                    "active_target_pose": self._state.active_target_pose if queued_run else None,
                    "run_active": queued_run,
                    "exit_reason": "" if queued_run else self._state.exit_reason,
                }
            )

    def publish_controller_state(
        self,
        state: ControllerState,
        *,
        detail: str = "",
        elapsed_s: float | None = None,
        run_active: bool | None = None,
        exit_reason: str | None = None,
    ) -> None:
        active = state == "RUNNING" if run_active is None else bool(run_active)
        self._set_state(
            controller_state=state,
            status_detail=detail,
            elapsed_s=self._state.elapsed_s if elapsed_s is None else float(elapsed_s),
            run_active=active,
            exit_reason=self._state.exit_reason if exit_reason is None else str(exit_reason),
        )

    def publish_safety(self, ok: bool, reason: str = "") -> None:
        with self._lock:
            self._state = self._state.__class__(
                **{
                    **self._state.__dict__,
                    "safety_state": SafetyState(bool(ok), reason),
                    "controller_state": self._state.controller_state if ok else "SAFETY STOP",
                    "status_detail": self._state.status_detail if ok else reason,
                    "run_active": self._state.run_active if ok else False,
                    "exit_reason": self._state.exit_reason if ok else "SAFETY_STOP",
                }
            )

    def set_run_metadata(self, run_id: str, *, elapsed_s: float = 0.0) -> None:
        self._set_state(
            run_id=run_id,
            elapsed_s=float(elapsed_s),
            replan_count=0,
            iteration=0,
            run_active=True,
            exit_reason="",
            predicted_next_pose=None,
            actual_next_pose=None,
        )
