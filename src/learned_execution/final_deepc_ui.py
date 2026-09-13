"""Final Genesis-specific Tkinter product UI.

This module is intentionally a presentation and input layer.  It accepts a
target pose and displays measured Genesis/DeePC telemetry; the runtime thread
owns every navigation command and every controller decision.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from tkinter import ttk
from typing import Any

from .ui_canvas import MapStyle, WorldMapCanvas
from .ui_runtime_bridge import UIRuntimeBridge
from .ui_state import LiveSystemState, Pose3

COLORS = {
    "window": "#0b1016",
    "panel": "#131b24",
    "panel_alt": "#18232e",
    "border": "#293846",
    "text": "#e8eef3",
    "muted": "#8d9aa6",
    "blue": "#4ea1ff",
    "orange": "#f39a5b",
    "purple": "#be8cff",
    "green": "#5ed39a",
    "red": "#ff6f73",
    "yellow": "#f1c75b",
}


def _fmt(value: float | None, digits: int = 3, unit: str = "") -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.{digits}f}{unit}"


def _fmt_unsigned(value: float | None, digits: int = 3, unit: str = "") -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}{unit}"


class FinalDeePCUI(tk.Tk):
    """Clean two-pane target-pose UI driven by :class:`UIRuntimeBridge`."""

    def __init__(
        self,
        bridge: UIRuntimeBridge,
        *,
        config: dict[str, Any] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.config_payload = config or {}
        self._target_presets = self._read_target_presets()
        self._preset_display_to_id = {
            value["display"]: target_id for target_id, value in self._target_presets.items()
        }
        self._on_close_callback = on_close
        self._closing = False
        self._aiming = False
        self._aim_target_xy: tuple[float, float] | None = None
        self._preview_yaw_rad: float | None = None
        self._numeric_preview: Pose3 | None = None
        self._last_state: LiveSystemState | None = None

        window = self.config_payload.get("window", {})
        width = int(window.get("width", 1320))
        height = int(window.get("height", 800))
        self.title("Final DeePC + Learned Go2")
        self.geometry(f"{width}x{height}")
        self.minsize(980, 650)
        self.configure(bg=COLORS["window"])
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._configure_style()
        self._create_variables()
        self._build_layout()
        self._bind_map()
        self.after(100, self._refresh)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        with suppress(tk.TclError):
            style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["window"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Section.TFrame", background=COLORS["panel_alt"])
        style.configure("Title.TLabel", background=COLORS["window"], foreground=COLORS["text"], font=("TkDefaultFont", 16, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["window"], foreground=COLORS["muted"], font=("TkDefaultFont", 9))
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Muted.Panel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure("Section.TLabel", background=COLORS["panel_alt"], foreground=COLORS["text"])
        style.configure("SectionMuted.TLabel", background=COLORS["panel_alt"], foreground=COLORS["muted"])
        style.configure("Value.Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("TkFixedFont", 10))
        style.configure("Value.Section.TLabel", background=COLORS["panel_alt"], foreground=COLORS["text"], font=("TkFixedFont", 10))
        style.configure("Small.Panel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=("TkDefaultFont", 8))
        style.configure("Primary.TButton", background=COLORS["blue"], foreground="#07111b", padding=(12, 7), font=("TkDefaultFont", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#79b9ff"), ("disabled", "#3b556d")])
        style.configure("Secondary.TButton", background=COLORS["panel_alt"], foreground=COLORS["text"], padding=(10, 6))
        style.map("Secondary.TButton", background=[("active", "#263747"), ("disabled", "#18232e")])
        style.configure("Danger.TButton", background="#8c343e", foreground="#ffffff", padding=(10, 7), font=("TkDefaultFont", 9, "bold"))
        style.map("Danger.TButton", background=[("active", "#c44d58")])
        style.configure("TEntry", fieldbackground="#0e151c", foreground=COLORS["text"], insertcolor=COLORS["text"], bordercolor=COLORS["border"], padding=5)
        style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure("TRadiobutton", background=COLORS["panel"], foreground=COLORS["muted"])

    def _create_variables(self) -> None:
        default_id = str(self.config_payload.get("default_target_id", "P1"))
        default = self._target_presets.get(default_id)
        if default is None and self._target_presets:
            default_id, default = next(iter(self._target_presets.items()))
        if default is None:
            default = {"x_m": 0.25, "y_m": 0.0, "yaw_rad": 0.0, "display": "Custom target"}
        self.target_preset_var = tk.StringVar(value=default["display"])
        self.target_x_var = tk.StringVar(value=f"{float(default['x_m']):+.3f}")
        self.target_y_var = tk.StringVar(value=f"{float(default['y_m']):+.3f}")
        self.target_yaw_var = tk.StringVar(value=f"{math.degrees(float(default['yaw_rad'])):+.1f}")
        self.current_x_var = tk.StringVar(value="—")
        self.current_y_var = tk.StringVar(value="—")
        self.current_yaw_var = tk.StringVar(value="—")
        self.position_error_var = tk.StringVar(value="—")
        self.yaw_error_var = tk.StringVar(value="—")
        self.vx_var = tk.StringVar(value="—")
        self.vy_var = tk.StringVar(value="—")
        self.yaw_rate_var = tk.StringVar(value="—")
        self.c2_margin_var = tk.StringVar(value="—")
        self.replans_var = tk.StringVar(value="0")
        self.iteration_var = tk.StringVar(value="0")
        self.solve_time_var = tk.StringVar(value="—")
        self.solver_var = tk.StringVar(value="—")
        self.direct_var = tk.StringVar(value="—")
        self.state_var = tk.StringVar(value="STARTING")
        self.detail_var = tk.StringVar(value="Preparing Genesis")
        self.safety_var = tk.StringVar(value="✓ OK")
        self.run_active_var = tk.StringVar(value="—")
        self.exit_reason_var = tk.StringVar(value="—")
        self.view_var = tk.StringVar(value="fixed_world")
        self.connection_var = tk.StringVar(value="GENESIS 1.3.1  ·  connecting")
        self.map_hint_var = tk.StringVar(value="Click a destination or enter a target pose")
        for variable in (self.target_x_var, self.target_y_var, self.target_yaw_var):
            variable.trace_add("write", self._on_numeric_preview_change)

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, style="App.TFrame", padding=(18, 14, 18, 16))
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        title_wrap = ttk.Frame(header, style="App.TFrame")
        title_wrap.grid(row=0, column=0, sticky="w")
        ttk.Label(title_wrap, text="Final DeePC + Learned Go2", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_wrap, text="Genesis thesis product  /  target-pose navigation", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.connection_var, style="Subtitle.TLabel").grid(row=0, column=1, sticky="e", padx=(10, 0))

        main = ttk.Frame(shell, style="App.TFrame")
        main.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=376)

        map_panel = ttk.Frame(main, style="Panel.TFrame", padding=10)
        map_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        map_panel.rowconfigure(0, weight=1)
        map_panel.columnconfigure(0, weight=1)
        map_cfg = self.config_payload.get("map", {})
        self.map_canvas = WorldMapCanvas(
            map_panel,
            background=COLORS["window"],
            style=MapStyle(),
            initial_span_m=float(map_cfg.get("initial_span_m", 1.2)),
            robot_view_span_m=float(map_cfg.get("robot_view_span_m", 1.4)),
            target_tolerance_m=float(map_cfg.get("target_tolerance_visual_m", 0.06)),
            fixed_center=tuple(map_cfg.get("fixed_center_m", [0.0, 0.0])),
        )
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        map_footer = ttk.Frame(map_panel, style="Panel.TFrame")
        map_footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        map_footer.columnconfigure(0, weight=1)
        ttk.Label(map_footer, textvariable=self.map_hint_var, style="Small.Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(map_footer, text="Fixed world is the default view", style="Small.Panel.TLabel").grid(row=0, column=1, sticky="e")

        sidebar_view = ttk.Frame(main, style="Panel.TFrame", width=376)
        sidebar_view.grid(row=0, column=1, sticky="nsew")
        sidebar_view.grid_propagate(False)
        sidebar_view.columnconfigure(0, weight=1)
        sidebar_view.rowconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(
            sidebar_view,
            background=COLORS["panel"],
            borderwidth=0,
            highlightthickness=0,
        )
        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scrollbar = ttk.Scrollbar(sidebar_view, orient="vertical", command=sidebar_canvas.yview)
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar = ttk.Frame(sidebar_canvas, style="Panel.TFrame", padding=12)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def update_sidebar_scroll_region(_event: object | None = None) -> None:
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def fit_sidebar_width(event: tk.Event[tk.Misc]) -> None:
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)

        sidebar.bind("<Configure>", update_sidebar_scroll_region)
        sidebar_canvas.bind("<Configure>", fit_sidebar_width)

        def is_sidebar_widget(widget: object) -> bool:
            current = widget
            while current is not None:
                if current is sidebar or current is sidebar_canvas:
                    return True
                current = getattr(current, "master", None)
            return False

        def scroll_sidebar(event: tk.Event[tk.Misc]) -> str | None:
            if not is_sidebar_widget(event.widget):
                return None
            event_number = getattr(event, "num", None)
            if event_number == 4:
                units = -3
            elif event_number == 5:
                units = 3
            else:
                delta = getattr(event, "delta", 0)
                units = -int(delta / 120) if delta else -1
            sidebar_canvas.yview_scroll(units, "units")
            return "break"

        self.bind_all("<MouseWheel>", scroll_sidebar, add="+")
        self.bind_all("<Button-4>", scroll_sidebar, add="+")
        self.bind_all("<Button-5>", scroll_sidebar, add="+")

        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(5, weight=1)

        self._build_target_section(sidebar, row=0)
        self._build_telemetry_section(sidebar, row=1)
        self._build_deepc_section(sidebar, row=2)
        self._build_status_section(sidebar, row=3)
        self._build_controls(sidebar, row=5)

    def _section(self, parent: ttk.Frame, title: str, row: int) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Section.TFrame", padding=(11, 9, 11, 10))
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 9))
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=title, style="Section.TLabel", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        return frame

    def _read_target_presets(self) -> dict[str, dict[str, Any]]:
        presets: dict[str, dict[str, Any]] = {}
        for item in self.config_payload.get("target_presets", []):
            try:
                target_id = str(item["target_id"])
                x_m = float(item["x_m"])
                y_m = float(item["y_m"])
                yaw_rad = float(item["yaw_rad"])
                if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad)):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            label = str(item.get("label", target_id))
            presets[target_id] = {
                "x_m": x_m,
                "y_m": y_m,
                "yaw_rad": yaw_rad,
                "display": f"{target_id}  ·  {label}",
            }
        return presets

    def _build_target_section(self, parent: ttk.Frame, *, row: int) -> None:
        frame = self._section(parent, "TARGET POSE", row)
        ttk.Label(frame, text="Preset", style="SectionMuted.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        preset_values = tuple(value["display"] for value in self._target_presets.values())
        self.target_preset_combo = ttk.Combobox(
            frame,
            textvariable=self.target_preset_var,
            values=preset_values,
            state="readonly" if preset_values else "normal",
            width=22,
        )
        self.target_preset_combo.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Button(frame, text="LOAD", style="Secondary.TButton", command=self._load_preset).grid(row=1, column=2, sticky="ew", padx=(7, 0), pady=2)
        labels = (("X", self.target_x_var, "m"), ("Y", self.target_y_var, "m"), ("Yaw", self.target_yaw_var, "deg"))
        for index, (label, variable, unit) in enumerate(labels, start=2):
            ttk.Label(frame, text=label, style="SectionMuted.TLabel").grid(row=index, column=0, sticky="w", padx=(0, 8), pady=2)
            entry = ttk.Entry(frame, textvariable=variable, width=12, justify="right")
            entry.grid(row=index, column=1, sticky="ew", pady=2)
            ttk.Label(frame, text=unit, style="SectionMuted.TLabel").grid(row=index, column=2, sticky="w", padx=(7, 0))
        ttk.Button(frame, text="APPLY TARGET", style="Primary.TButton", command=self._apply_target).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(9, 2))
        ttk.Label(frame, text="right click → aim heading → left click locks yaw", style="SectionMuted.TLabel", font=("TkDefaultFont", 8)).grid(row=6, column=0, columnspan=3, sticky="w", pady=(5, 0))

    def _build_telemetry_section(self, parent: ttk.Frame, *, row: int) -> None:
        frame = self._section(parent, "MEASURED GENESIS POSE", row)
        values = (("x", self.current_x_var), ("y", self.current_y_var), ("yaw", self.current_yaw_var))
        for index, (label, variable) in enumerate(values, start=1):
            ttk.Label(frame, text=label, style="SectionMuted.TLabel").grid(row=index, column=0, sticky="w", pady=1)
            ttk.Label(frame, textvariable=variable, style="Value.Section.TLabel", anchor="e").grid(row=index, column=1, columnspan=2, sticky="e", pady=1)
        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=7)
        ttk.Label(frame, text="TARGET ERROR", style="SectionMuted.TLabel", font=("TkDefaultFont", 8, "bold")).grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 3))
        ttk.Label(frame, text="position", style="SectionMuted.TLabel").grid(row=6, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.position_error_var, style="Value.Section.TLabel", anchor="e").grid(row=6, column=1, columnspan=2, sticky="e")
        ttk.Label(frame, text="yaw", style="SectionMuted.TLabel").grid(row=7, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.yaw_error_var, style="Value.Section.TLabel", anchor="e").grid(row=7, column=1, columnspan=2, sticky="e")

    def _build_deepc_section(self, parent: ttk.Frame, *, row: int) -> None:
        frame = self._section(parent, "DEEPC  ·  DIRECT [vx, vy, yaw_rate]", row)
        values = (("vx", self.vx_var), ("vy", self.vy_var), ("yaw_rate", self.yaw_rate_var), ("replans", self.replans_var), ("iteration", self.iteration_var), ("solve time", self.solve_time_var), ("solver", self.solver_var))
        for index, (label, variable) in enumerate(values, start=1):
            ttk.Label(frame, text=label, style="SectionMuted.TLabel").grid(row=index, column=0, sticky="w", pady=1)
            ttk.Label(frame, textvariable=variable, style="Value.Section.TLabel", anchor="e").grid(row=index, column=1, columnspan=2, sticky="e", pady=1)
        ttk.Separator(frame, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=7)
        ttk.Label(frame, text="DIRECT COMMAND", style="SectionMuted.TLabel").grid(row=9, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.direct_var, style="Value.Section.TLabel", anchor="e").grid(row=9, column=1, columnspan=2, sticky="e")
        ttk.Label(frame, text="C2 MARGIN", style="SectionMuted.TLabel").grid(row=10, column=0, sticky="w", pady=(4, 0))
        ttk.Label(frame, textvariable=self.c2_margin_var, style="Value.Section.TLabel", anchor="e").grid(row=10, column=1, columnspan=2, sticky="e", pady=(4, 0))

    def _build_status_section(self, parent: ttk.Frame, *, row: int) -> None:
        frame = self._section(parent, "RUN STATUS", row)
        ttk.Label(frame, textvariable=self.state_var, style="Value.Section.TLabel", font=("TkDefaultFont", 11, "bold")).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, textvariable=self.detail_var, style="SectionMuted.TLabel", wraplength=310).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 7))
        ttk.Label(frame, text="SAFETY", style="SectionMuted.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.safety_var, style="Value.Section.TLabel", anchor="e").grid(row=3, column=1, columnspan=2, sticky="e")
        ttk.Label(frame, text="RUN ACTIVE", style="SectionMuted.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.run_active_var, style="Value.Section.TLabel", anchor="e").grid(row=4, column=1, columnspan=2, sticky="e")
        ttk.Label(frame, text="EXIT", style="SectionMuted.TLabel").grid(row=5, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.exit_reason_var, style="Value.Section.TLabel", anchor="e").grid(row=5, column=1, columnspan=2, sticky="e")

    def _build_controls(self, parent: ttk.Frame, *, row: int) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=row, column=0, sticky="sew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        ttk.Button(frame, text="RUN", style="Primary.TButton", command=self._run).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        ttk.Button(frame, text="STOP", style="Secondary.TButton", command=self._stop).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="RESET / NEW RUN", style="Secondary.TButton", command=self._reset).grid(row=0, column=2, sticky="ew", padx=(4, 0), pady=2)
        ttk.Button(frame, text="CLEAR TRAIL", style="Secondary.TButton", command=self.bridge.request_clear_trail).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 2))
        ttk.Button(frame, text="EMERGENCY STOP", style="Danger.TButton", command=self._emergency_stop).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=3, sticky="ew", pady=9)
        ttk.Label(frame, text="VIEW", style="Muted.Panel.TLabel", font=("TkDefaultFont", 8, "bold")).grid(row=4, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Fixed world", variable=self.view_var, value="fixed_world", command=self._change_view).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frame, text="Robot centered", variable=self.view_var, value="robot_centered", command=self._change_view).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Button(frame, text="RESET VIEW", style="Secondary.TButton", command=self.map_canvas.reset_view).grid(row=5, column=2, rowspan=2, sticky="e")

    def _bind_map(self) -> None:
        self.map_canvas.bind_target_handlers(
            on_left_click=self._map_left_click,
            on_right_click=self._map_right_click,
            on_motion=self._map_motion,
        )

    def _load_preset(self) -> None:
        target_id = self._preset_display_to_id.get(self.target_preset_var.get())
        if target_id is None:
            self._show_message("Choose a frozen target preset first")
            return
        preset = self._target_presets[target_id]
        target = Pose3(float(preset["x_m"]), float(preset["y_m"]), float(preset["yaw_rad"])).normalized()
        self._set_target_fields(target)
        self._apply_target(source=f"preset {target_id}", target_id=target_id)

    def _apply_target(self, *, source: str = "numeric input", target_id: str | None = None) -> None:
        try:
            target = Pose3.from_degrees(
                float(self.target_x_var.get()),
                float(self.target_y_var.get()),
                float(self.target_yaw_var.get()),
            )
        except ValueError:
            self._show_message("Enter finite numeric X, Y, and yaw values")
            return
        if not self.bridge.request_target(target, source=source, target_id=target_id):
            self._show_message("Target is committed while RUNNING; stop before editing it")
            return
        if target_id is None:
            self.target_preset_var.set("Custom target")
        self._numeric_preview = None
        self._aiming = False
        self._aim_target_xy = None
        self._preview_yaw_rad = None
        self._show_message("Target pose applied")

    def _map_left_click(self, x_m: float, y_m: float) -> None:
        if self._aiming and self._aim_target_xy is not None:
            target_x, target_y = self._aim_target_xy
            yaw = self._preview_yaw_rad
            if yaw is None:
                yaw = self._current_yaw()
            target = Pose3(target_x, target_y, yaw).normalized()
            if self.bridge.request_target(target, source="canvas yaw aim"):
                self._set_target_fields(target)
                self.target_preset_var.set("Custom target")
                self._show_message("Final heading locked")
            self._aiming = False
            self._aim_target_xy = None
            self._preview_yaw_rad = None
            return
        yaw = self._current_yaw()
        target = Pose3(x_m, y_m, yaw).normalized()
        if self.bridge.request_target(target, source="canvas XY"):
            self._set_target_fields(target)
            self.target_preset_var.set("Custom target")
            self._show_message("Target XY set; yaw defaults to current measured heading")
        else:
            self._show_message("Target is committed while RUNNING; stop before editing it")

    def _map_right_click(self, x_m: float, y_m: float) -> None:
        yaw = self._current_yaw()
        target = Pose3(x_m, y_m, yaw).normalized()
        if not self.bridge.request_target(target, source="canvas XY / yaw aiming"):
            self._show_message("Target is committed while RUNNING; stop before editing it")
            return
        self._set_target_fields(target)
        self.target_preset_var.set("Custom target")
        self._aiming = True
        self._aim_target_xy = (x_m, y_m)
        self._preview_yaw_rad = yaw
        self._show_message("Move the pointer to preview final heading; left click to lock")

    def _map_motion(self, x_m: float, y_m: float) -> None:
        if not self._aiming or self._aim_target_xy is None:
            return
        target_x, target_y = self._aim_target_xy
        self._preview_yaw_rad = math.atan2(y_m - target_y, x_m - target_x)
        self.map_canvas.redraw(self._display_state(), preview_yaw_rad=self._preview_yaw_rad, aiming=True)

    def _current_yaw(self) -> float:
        state = self.bridge.snapshot()
        return state.actual_pose.yaw_rad if state.actual_pose is not None else 0.0

    def _set_target_fields(self, target: Pose3) -> None:
        self.target_x_var.set(f"{target.x_m:+.3f}")
        self.target_y_var.set(f"{target.y_m:+.3f}")
        self.target_yaw_var.set(f"{target.yaw_deg:+.1f}")
        self._numeric_preview = None

    def _on_numeric_preview_change(self, *_args: object) -> None:
        if self._aiming:
            return
        try:
            self._numeric_preview = Pose3.from_degrees(
                float(self.target_x_var.get()),
                float(self.target_y_var.get()),
                float(self.target_yaw_var.get()),
            )
        except ValueError:
            self._numeric_preview = None
        if self._last_state is not None and self._last_state.controller_state != "RUNNING":
            self.map_canvas.redraw(self._display_state())

    def _display_state(self) -> LiveSystemState | None:
        state = self._last_state or self.bridge.snapshot()
        if self._numeric_preview is not None and state.controller_state != "RUNNING":
            position_error = math.hypot(self._numeric_preview.x_m - (state.actual_pose.x_m if state.actual_pose else 0.0), self._numeric_preview.y_m - (state.actual_pose.y_m if state.actual_pose else 0.0)) if state.actual_pose else None
            yaw_error = abs(math.atan2(math.sin(self._numeric_preview.yaw_rad - state.actual_pose.yaw_rad), math.cos(self._numeric_preview.yaw_rad - state.actual_pose.yaw_rad))) if state.actual_pose else None
            return replace(state, target_pose=self._numeric_preview, active_target_pose=None, position_error_m=position_error, yaw_error_rad=yaw_error)
        return state

    def _change_view(self) -> None:
        self.map_canvas.set_view_mode(self.view_var.get())  # type: ignore[arg-type]

    def _run(self) -> None:
        if not self.bridge.request_run():
            self._show_message("Apply a target pose and wait for READY before RUN")

    def _stop(self) -> None:
        if not self.bridge.request_stop():
            self._show_message("No active run")

    def _reset(self) -> None:
        if not self.bridge.request_reset():
            self._show_message("Stop the active run before RESET / NEW RUN")
        else:
            self._show_message("Reset requested")

    def _emergency_stop(self) -> None:
        self.bridge.request_emergency_stop()
        self._show_message("SAFETY STOP requested")

    def _show_message(self, message: str) -> None:
        self.map_hint_var.set(message)

    def _refresh(self) -> None:
        if self._closing:
            return
        state = self.bridge.snapshot()
        self._last_state = state
        self.connection_var.set("GENESIS 1.3.1  ·  live" if state.viewer_available else "GENESIS 1.3.1  ·  starting")
        self.state_var.set(state.controller_state)
        self.detail_var.set(state.status_detail)
        self.safety_var.set("✓ OK" if state.safety_state.ok else f"SAFETY STOP  ·  {state.safety_state.reason}")
        pose = state.actual_pose
        self.current_x_var.set(_fmt(pose.x_m if pose else None, 3, " m"))
        self.current_y_var.set(_fmt(pose.y_m if pose else None, 3, " m"))
        self.current_yaw_var.set(_fmt(pose.yaw_deg if pose else None, 1, "°"))
        self.position_error_var.set(_fmt_unsigned(state.position_error_m, 3, " m"))
        self.yaw_error_var.set(_fmt_unsigned(math.degrees(state.yaw_error_rad) if state.yaw_error_rad is not None else None, 1, "°"))
        command = state.deepc_command
        self.vx_var.set(_fmt(command[0] if command else None, 3, " m/s"))
        self.vy_var.set(_fmt(command[1] if command else None, 3, " m/s"))
        self.yaw_rate_var.set(_fmt(command[2] if command else None, 3, " rad/s"))
        self.replans_var.set(str(state.replan_count))
        self.iteration_var.set(str(state.iteration))
        self.solve_time_var.set(_fmt_unsigned(state.solve_time_s * 1000.0 if state.solve_time_s is not None else None, 1, " ms"))
        solver_status = state.solver_status.lower()
        if solver_status in {"optimal", "optimal_inaccurate", "feasible"}:
            self.solver_var.set("FEASIBLE")
        elif "infeasible" in solver_status:
            self.solver_var.set("INFEASIBLE")
        else:
            self.solver_var.set(state.solver_status)
        self.c2_margin_var.set(_fmt(state.c2_margin, 6))
        if state.direct_command_ok is None:
            self.direct_var.set("—")
        else:
            self.direct_var.set("PASS" if state.direct_command_ok else "INTERFACE ERROR")
        self.run_active_var.set("YES" if state.run_active else "NO")
        self.exit_reason_var.set(state.exit_reason or "—")
        self.map_canvas.redraw(self._display_state(), preview_yaw_rad=self._preview_yaw_rad, aiming=self._aiming)
        self.after(100, self._refresh)

    def _close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._on_close_callback is not None:
            self._on_close_callback()
        else:
            self.bridge.request_shutdown()
        self.destroy()


def load_ui_config(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def launch_ui(
    bridge: UIRuntimeBridge,
    *,
    config: dict[str, Any] | None = None,
    on_close: Callable[[], None] | None = None,
) -> FinalDeePCUI:
    """Construct the window; the caller owns ``mainloop``."""

    return FinalDeePCUI(bridge, config=config, on_close=on_close)
