"""Small, dependency-light data model for the Genesis live product UI.

The UI state contains target input and measured telemetry only.  In
particular, there is no navigation command generation in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

ControllerState = Literal[
    "STARTING",
    "READY",
    "RUNNING",
    "TARGET REACHED",
    "STOPPED",
    "SAFETY STOP",
    "SOLVER FAILURE",
    "INTERFACE FAILURE",
]


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to the human-facing ``[-pi, pi]`` interval."""

    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


@dataclass(frozen=True)
class Pose3:
    """A world-frame ``[x, y, yaw]`` pose in metres and radians."""

    x_m: float
    y_m: float
    yaw_rad: float

    def finite(self) -> bool:
        return all(math.isfinite(value) for value in (self.x_m, self.y_m, self.yaw_rad))

    def normalized(self) -> Pose3:
        if not self.finite():
            raise ValueError("pose must contain finite values")
        return Pose3(float(self.x_m), float(self.y_m), wrap_angle(self.yaw_rad))

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.x_m), float(self.y_m), float(self.yaw_rad))

    @classmethod
    def from_degrees(cls, x_m: float, y_m: float, yaw_deg: float) -> Pose3:
        return cls(float(x_m), float(y_m), math.radians(float(yaw_deg))).normalized()

    @property
    def yaw_deg(self) -> float:
        return math.degrees(wrap_angle(self.yaw_rad))


@dataclass(frozen=True)
class SafetyState:
    ok: bool = True
    reason: str = ""


@dataclass(frozen=True)
class LiveSystemState:
    """Immutable snapshot shared between the runtime and Tkinter."""

    target_pose: Pose3 | None = None
    active_target_pose: Pose3 | None = None
    actual_pose: Pose3 | None = None
    trail: tuple[Pose3, ...] = field(default_factory=tuple)
    position_error_m: float | None = None
    yaw_error_rad: float | None = None
    deepc_command: tuple[float, float, float] | None = None
    student_received_command: tuple[float, float, float] | None = None
    direct_command_ok: bool | None = None
    replan_count: int = 0
    solver_status: str = "—"
    solve_time_s: float | None = None
    controller_state: ControllerState = "STARTING"
    safety_state: SafetyState = field(default_factory=SafetyState)
    run_id: str | None = None
    elapsed_s: float = 0.0
    actual_sample_count: int = 0
    measured_feedback_ok: bool | None = None
    iteration: int = 0
    predicted_next_pose: Pose3 | None = None
    actual_next_pose: Pose3 | None = None
    run_active: bool = False
    exit_reason: str = ""
    target_source: str = "default"
    target_id: str = ""
    status_detail: str = "Preparing Genesis"
    target_committed: bool = False
    viewer_available: bool = False
    solver_selected_command: tuple[float, float, float] | None = None
    delivered_command: tuple[float, float, float] | None = None
    c2_margin: float | None = None
    c2_limit: float | None = None
    post_solver_rewriting: bool | None = None


def pose_errors(actual: Pose3 | None, target: Pose3 | None) -> tuple[float | None, float | None]:
    """Return display-only position and wrapped-yaw errors."""

    if actual is None or target is None:
        return None, None
    position = math.hypot(target.x_m - actual.x_m, target.y_m - actual.y_m)
    yaw = abs(wrap_angle(target.yaw_rad - actual.yaw_rad))
    return position, yaw


def command_tuple(values: object) -> tuple[float, float, float]:
    """Validate a three-component telemetry command without transforming it."""

    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("command must contain three finite values") from exc
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("command must contain three finite values")
    return result  # type: ignore[return-value]
