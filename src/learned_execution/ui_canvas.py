"""Canvas-only world visualization for the Genesis live product."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .ui_state import LiveSystemState, Pose3

ViewMode = Literal["fixed_world", "robot_centered"]


@dataclass(frozen=True)
class MapStyle:
    background: str = "#10161d"
    grid: str = "#27333e"
    axis: str = "#53616d"
    trail: str = "#93a0aa"
    robot: str = "#4ea1ff"
    robot_outline: str = "#b9ddff"
    target: str = "#f39a5b"
    target_heading: str = "#be8cff"
    target_ring: str = "#9a78c9"
    text: str = "#b8c5cf"
    focus: str = "#6d8192"


class WorldMapCanvas(tk.Canvas):
    """2D world map with fixed-world and robot-centered views.

    The canvas only converts between world coordinates and pixels.  It never
    creates or modifies a navigation command.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        style: MapStyle | None = None,
        initial_span_m: float = 1.2,
        robot_view_span_m: float = 1.4,
        target_tolerance_m: float = 0.06,
        fixed_center: tuple[float, float] = (0.0, 0.0),
        **kwargs: object,
    ) -> None:
        super().__init__(master, highlightthickness=0, **kwargs)
        if initial_span_m <= 0 or robot_view_span_m <= 0 or target_tolerance_m <= 0:
            raise ValueError("map spans and target tolerance must be positive")
        self.map_style = style or MapStyle()
        self.initial_span_m = float(initial_span_m)
        self.robot_view_span_m = float(robot_view_span_m)
        self.target_tolerance_m = float(target_tolerance_m)
        self.fixed_center = (float(fixed_center[0]), float(fixed_center[1]))
        self.view_mode: ViewMode = "fixed_world"
        self._fixed_span_m = self.initial_span_m
        self._reset_view_requested = False
        self._last_state: LiveSystemState | None = None
        self._last_preview_yaw: float | None = None
        self._last_aiming = False

    def set_view_mode(self, mode: ViewMode) -> None:
        if mode not in {"fixed_world", "robot_centered"}:
            raise ValueError(f"unknown map view {mode!r}")
        self.view_mode = mode
        self.redraw(self._last_state, preview_yaw_rad=self._last_preview_yaw, aiming=self._last_aiming)

    def reset_view(self) -> None:
        self._fixed_span_m = self.initial_span_m
        self._reset_view_requested = True
        self.redraw(self._last_state, preview_yaw_rad=self._last_preview_yaw, aiming=self._last_aiming)

    def world_to_canvas(self, x_m: float, y_m: float, state: LiveSystemState | None = None) -> tuple[float, float]:
        width = max(1, int(self.winfo_width()))
        height = max(1, int(self.winfo_height()))
        center_x, center_y, scale = self._view_parameters(state)
        return (
            width * 0.5 + (float(x_m) - center_x) * scale,
            height * 0.5 - (float(y_m) - center_y) * scale,
        )

    def canvas_to_world(self, x_px: float, y_px: float, state: LiveSystemState | None = None) -> tuple[float, float]:
        width = max(1, int(self.winfo_width()))
        height = max(1, int(self.winfo_height()))
        center_x, center_y, scale = self._view_parameters(state)
        return (
            center_x + (float(x_px) - width * 0.5) / scale,
            center_y - (float(y_px) - height * 0.5) / scale,
        )

    def bind_target_handlers(
        self,
        *,
        on_left_click: Callable[[float, float], None],
        on_right_click: Callable[[float, float], None],
        on_motion: Callable[[float, float], None],
    ) -> None:
        self.bind(
            "<Button-1>",
            lambda event: on_left_click(*self.canvas_to_world(event.x, event.y, self._last_state)),
        )
        self.bind(
            "<Button-3>",
            lambda event: on_right_click(*self.canvas_to_world(event.x, event.y, self._last_state)),
        )
        self.bind(
            "<Motion>",
            lambda event: on_motion(*self.canvas_to_world(event.x, event.y, self._last_state)),
        )

    def redraw(
        self,
        state: LiveSystemState | None,
        *,
        preview_yaw_rad: float | None = None,
        aiming: bool = False,
    ) -> None:
        self._last_state = state
        self._last_preview_yaw = preview_yaw_rad
        self._last_aiming = aiming
        self.delete("all")
        if state is None:
            self.create_text(20, 20, anchor="nw", text="Waiting for Genesis…", fill=self.map_style.text)
            return
        self._ensure_fixed_view_visible(state)
        self._draw_grid(state)

        if len(state.trail) > 1:
            points: list[float] = []
            for pose in state.trail:
                x, y = self.world_to_canvas(pose.x_m, pose.y_m, state)
                points.extend((x, y))
            self.create_line(*points, fill=self.map_style.trail, width=2, smooth=True, tags="trail")

        target = state.active_target_pose or state.target_pose
        if target is not None:
            self._draw_target(state, target, preview_yaw_rad if aiming else None, aiming)
        if state.actual_pose is not None:
            self._draw_robot(state, state.actual_pose)

        self.create_text(
            16,
            16,
            anchor="nw",
            text="WORLD MAP  ·  measured trajectory",
            fill=self.map_style.text,
            font=("TkDefaultFont", 10, "bold"),
        )
        self.create_text(
            16,
            max(18, int(self.winfo_height()) - 18),
            anchor="sw",
            text="left click: target XY   ·   right click: aim final heading",
            fill=self.map_style.text,
            font=("TkDefaultFont", 9),
        )

    def _view_parameters(self, state: LiveSystemState | None) -> tuple[float, float, float]:
        width = max(1, int(self.winfo_width()))
        height = max(1, int(self.winfo_height()))
        if self.view_mode == "robot_centered" and state is not None and state.actual_pose is not None:
            center_x, center_y = state.actual_pose.x_m, state.actual_pose.y_m
            span = self.robot_view_span_m
        else:
            center_x, center_y = self.fixed_center
            span = self._fixed_span_m
        scale = min(width, height) / max(span, 1.0e-6)
        return center_x, center_y, scale

    def _ensure_fixed_view_visible(self, state: LiveSystemState) -> None:
        if self.view_mode != "fixed_world":
            return
        points: list[tuple[float, float]] = [(self.fixed_center[0], self.fixed_center[1])]
        if state.actual_pose is not None:
            points.append((state.actual_pose.x_m, state.actual_pose.y_m))
        if state.target_pose is not None:
            points.append((state.target_pose.x_m, state.target_pose.y_m))
        points.extend((pose.x_m, pose.y_m) for pose in state.trail[-2000:])
        half = self._fixed_span_m * 0.5
        distance = max(
            max(abs(x - self.fixed_center[0]) for x, _ in points),
            max(abs(y - self.fixed_center[1]) for _, y in points),
        )
        if distance > half * 0.92:
            self._fixed_span_m = max(self._fixed_span_m, distance * 2.25)

    def _draw_grid(self, state: LiveSystemState) -> None:
        width = max(1, int(self.winfo_width()))
        height = max(1, int(self.winfo_height()))
        center_x, center_y, scale = self._view_parameters(state)
        span = min(width, height) / scale
        step = self._nice_step(span / 8.0)
        x_min = center_x - width / (2.0 * scale)
        x_max = center_x + width / (2.0 * scale)
        y_min = center_y - height / (2.0 * scale)
        y_max = center_y + height / (2.0 * scale)
        x = math.floor(x_min / step) * step
        while x <= x_max + step:
            screen_x, _ = self.world_to_canvas(x, center_y, state)
            color = self.map_style.axis if abs(x) < step * 0.25 else self.map_style.grid
            self.create_line(screen_x, 0, screen_x, height, fill=color, width=1)
            if abs(x) > step * 0.25:
                self.create_text(screen_x + 4, height - 8, anchor="sw", text=f"{x:.1f}", fill=self.map_style.text)
            x += step
        y = math.floor(y_min / step) * step
        while y <= y_max + step:
            _, screen_y = self.world_to_canvas(center_x, y, state)
            color = self.map_style.axis if abs(y) < step * 0.25 else self.map_style.grid
            self.create_line(0, screen_y, width, screen_y, fill=color, width=1)
            if abs(y) > step * 0.25:
                self.create_text(6, screen_y - 4, anchor="sw", text=f"{y:.1f}", fill=self.map_style.text)
            y += step

    def _draw_robot(self, state: LiveSystemState, pose: Pose3) -> None:
        x, y = self.world_to_canvas(pose.x_m, pose.y_m, state)
        radius = 11
        self.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=self.map_style.robot,
            outline=self.map_style.robot_outline,
            width=2,
            tags="robot",
        )
        self._draw_arrow(state, pose.x_m, pose.y_m, pose.yaw_rad, 0.16, self.map_style.robot, width=3)
        self.create_text(x + 15, y - 14, anchor="sw", text="Go2", fill=self.map_style.robot_outline, font=("TkDefaultFont", 9, "bold"))

    def _draw_target(self, state: LiveSystemState, pose: Pose3, preview_yaw: float | None, aiming: bool) -> None:
        x, y = self.world_to_canvas(pose.x_m, pose.y_m, state)
        tolerance_px = max(8.0, self.target_tolerance_m * self._view_parameters(state)[2])
        self.create_oval(
            x - tolerance_px,
            y - tolerance_px,
            x + tolerance_px,
            y + tolerance_px,
            outline=self.map_style.target_ring,
            width=1,
            dash=(3, 4),
        )
        self.create_line(x - 8, y, x + 8, y, fill=self.map_style.target, width=2)
        self.create_line(x, y - 8, x, y + 8, fill=self.map_style.target, width=2)
        self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=self.map_style.target, outline="")
        heading = pose.yaw_rad if preview_yaw is None else preview_yaw
        self._draw_arrow(state, pose.x_m, pose.y_m, heading, 0.19, self.map_style.target_heading, width=3)
        label = "AIM FINAL HEADING" if aiming else "TARGET POSE"
        self.create_text(x + 14, y + 14, anchor="nw", text=label, fill=self.map_style.target, font=("TkDefaultFont", 9, "bold"))

    def _draw_arrow(self, state: LiveSystemState, x_m: float, y_m: float, yaw_rad: float, length_m: float, color: str, *, width: int) -> None:
        x0, y0 = self.world_to_canvas(x_m, y_m, state)
        x1, y1 = self.world_to_canvas(
            x_m + length_m * math.cos(yaw_rad),
            y_m + length_m * math.sin(yaw_rad),
            state,
        )
        self.create_line(x0, y0, x1, y1, fill=color, width=width, arrow=tk.LAST)

    @staticmethod
    def _nice_step(raw: float) -> float:
        raw = max(float(raw), 1.0e-4)
        exponent = math.floor(math.log10(raw))
        fraction = raw / (10.0**exponent)
        if fraction < 1.5:
            nice = 1.0
        elif fraction < 3.5:
            nice = 2.0
        elif fraction < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10.0**exponent)
