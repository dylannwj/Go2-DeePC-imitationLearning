#!/usr/bin/env python3

import json
import math
import threading
import time
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from std_msgs.msg import Bool, String, Float64


MAX_TARGET_DISTANCE_M = 0.80
CAUTION_TARGET_DISTANCE_M = 0.50
MOCAP_FRESH_SEC = 0.50

# Initial guess only.
# The UI will auto-calibrate this from real forward motion.
UI_HEADING_OFFSET_DEG = 98.6
UI_HEADING_OFFSET_RAD = math.radians(UI_HEADING_OFFSET_DEG)

# Dog2 center correction from live turn calibration
CENTER_RX = 0.08392210436264078
CENTER_RY = 0.02708164171004897

# Auto heading calibration from mocap trail while robot is commanded forward.
CALIBRATE_MIN_SAFE_VX = 0.05
CALIBRATE_MAX_ABS_YAW_RATE = 0.08
CALIBRATE_MIN_STEP_M = 0.05
CALIBRATION_ALPHA = 0.50


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def mocap_xy_to_m(raw_x, raw_y):
    if abs(raw_x) > 10.0 or abs(raw_y) > 10.0:
        return raw_x * 0.001, raw_y * 0.001
    return raw_x, raw_y


class DeePCUiRos(Node):
    def __init__(self):
        super().__init__("deepc_ui_node")

        self.lock = threading.Lock()

        self.pose = None
        self.last_pose_time = None

        self.safety_status = {
            "safe": False,
            "mode": "unknown",
            "reasons": ["no_status_yet"],
        }

        self.proposed_cmd = Twist()
        self.safe_cmd = Twist()

        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/mocap/go2_pose",
            self.pose_cb,
            mocap_qos,
        )

        self.safety_sub = self.create_subscription(
            String,
            "/deepc/safety_status",
            self.safety_cb,
            10,
        )

        self.proposed_sub = self.create_subscription(
            Twist,
            "/deepc/proposed_cmd",
            self.proposed_cb,
            10,
        )

        self.safe_cmd_sub = self.create_subscription(
            Twist,
            "/deepc/safe_cmd_preview",
            self.safe_cmd_cb,
            10,
        )

        self.target_pub = self.create_publisher(
            PointStamped,
            "/deepc/ui_target_point",
            10,
        )

        self.target_yaw_pub = self.create_publisher(
            Float64,
            "/deepc/ui_target_yaw",
            10,
        )

        self.emergency_pub = self.create_publisher(
            Bool,
            "/deepc/emergency_stop",
            10,
        )

        self.run_pub = self.create_publisher(
            Bool,
            "/deepc/ui_run",
            10,
        )

        self.stop_pub = self.create_publisher(
            Bool,
            "/deepc/ui_stop",
            10,
        )

        self.zero_cmd_pub = self.create_publisher(
            Twist,
            "/deepc/proposed_cmd",
            10,
        )

    def pose_cb(self, msg):
        raw_x = float(msg.pose.position.x)
        raw_y = float(msg.pose.position.y)
        x, y = mocap_xy_to_m(raw_x, raw_y)
        yaw = quat_to_yaw(msg.pose.orientation)

        # Draw/control UI from corrected robot center, not raw Dog2 body origin.
        heading = wrap_angle(yaw + UI_HEADING_OFFSET_RAD)
        c = math.cos(heading)
        ss = math.sin(heading)
        off_x = c * CENTER_RX - ss * CENTER_RY
        off_y = ss * CENTER_RX + c * CENTER_RY
        cx = x - off_x
        cy = y - off_y

        with self.lock:
            # Keep yaw raw because existing UI applies heading_offset_rad elsewhere.
            self.pose = (cx, cy, yaw)
            self.last_pose_time = time.time()

    def safety_cb(self, msg):
        try:
            data = json.loads(msg.data)
            safe = bool(data.get("safe", False))
            mode = str(data.get("mode", "unknown"))
            reasons = data.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
        except Exception:
            safe = False
            mode = "parse_error"
            reasons = [msg.data[:120]]

        with self.lock:
            self.safety_status = {
                "safe": safe,
                "mode": mode,
                "reasons": reasons,
            }

    def proposed_cb(self, msg):
        with self.lock:
            self.proposed_cmd = msg

    def safe_cmd_cb(self, msg):
        with self.lock:
            self.safe_cmd = msg

    def publish_target(self, x, y):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mocap"
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = 0.0
        self.target_pub.publish(msg)
        self.get_logger().warn(
            f"UI target published: x={x:.3f}, y={y:.3f}"
        )

    def publish_target_yaw(self, yaw_rad):
        msg = Float64()
        msg.data = float(wrap_angle(yaw_rad))
        self.target_yaw_pub.publish(msg)
        self.get_logger().warn(
            f"UI final yaw published: yaw={msg.data:+.3f} rad"
        )

    def set_emergency(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.emergency_pub.publish(msg)
        self.get_logger().warn(f"Emergency stop set to {value}")

    def send_run(self):
        msg = Bool()
        msg.data = True
        self.run_pub.publish(msg)
        self.get_logger().warn("UI sent RUN request")

    def send_stop(self):
        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

        zero = Twist()
        for _ in range(10):
            self.zero_cmd_pub.publish(zero)
            time.sleep(0.02)

        self.get_logger().warn("UI sent STOP request + zero command")


class DeePCUiApp:
    def __init__(self, root, ros_node):
        self.root = root
        self.node = ros_node

        self.root.title("Go2 DeePC UI - Target Selector")
        self.root.geometry("1160x740")

        self.target_xy = None
        self.target_local = None
        self.target_final_yaw = None
        self.aiming_yaw = False
        self.mouse_canvas_xy = None
        self.target_final_yaw = None
        self.aiming_yaw = False
        self.mouse_canvas_xy = None
        self.trail = []
        self.max_trail = 300

        self.canvas_w = 680
        self.canvas_h = 660
        self.scale_px_per_m = 320.0

        self.view_mode = tk.StringVar(value="robot")
        self.fixed_center = None
        self.estop_active = False
        self.controller_proc = None
        self.controller_lock = threading.Lock()

        # UI heading calibration.
        self.heading_offset_rad = UI_HEADING_OFFSET_RAD
        self.calib_anchor_pose = None
        self.last_forward_heading = None

        self.build_ui()
        self.update_loop()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=430)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.canvas = tk.Canvas(
            left,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="white",
            highlightthickness=1,
            highlightbackground="black",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Button-2>", self.on_canvas_right_click)
        self.canvas.bind("<Control-Button-1>", self.on_canvas_right_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        hint = ttk.Label(
            left,
            text=(
                "Left click = normal target. Right click = select target B with final yaw. "
                "After right click, move mouse to aim arrow and left click to lock final heading."
            ),
        )
        hint.pack(anchor="w", pady=(6, 0))

        title = ttk.Label(
            right,
            text="Go2 DeePC Control Panel",
            font=("Arial", 14, "bold"),
        )
        title.pack(anchor="w", pady=(0, 10))

        self.pose_label = ttk.Label(right, text="Pose: waiting...")
        self.pose_label.pack(anchor="w", pady=2)

        self.heading_label = ttk.Label(right, text="Heading offset: waiting...")
        self.heading_label.pack(anchor="w", pady=2)

        self.target_label = ttk.Label(right, text="Target B: none")
        self.target_label.pack(anchor="w", pady=2)

        self.dist_label = ttk.Label(right, text="Distance to B: n/a")
        self.dist_label.pack(anchor="w", pady=2)

        sep1 = ttk.Separator(right, orient="horizontal")
        sep1.pack(fill=tk.X, pady=10)

        view_title = ttk.Label(right, text="View mode")
        view_title.pack(anchor="w", pady=(0, 3))

        view_frame = ttk.Frame(right)
        view_frame.pack(anchor="w", pady=2)

        ttk.Radiobutton(
            view_frame,
            text="Robot-centered",
            variable=self.view_mode,
            value="robot",
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            view_frame,
            text="Fixed world",
            variable=self.view_mode,
            value="world",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Button(
            right,
            text="Set Fixed View Center = Current Go2",
            command=self.set_fixed_center_to_current,
        ).pack(fill=tk.X, pady=(4, 3))

        ttk.Button(
            right,
            text="Clear Trail",
            command=self.clear_trail,
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            right,
            text="Reset Heading Offset to -65",
            command=self.reset_heading_calibration,
        ).pack(fill=tk.X, pady=3)

        sep2 = ttk.Separator(right, orient="horizontal")
        sep2.pack(fill=tk.X, pady=10)

        self.safety_label = ttk.Label(right, text="Safety: waiting...")
        self.safety_label.pack(anchor="w", pady=2)

        self.reasons_label = ttk.Label(
            right,
            text="Reasons: n/a",
            wraplength=320,
        )
        self.reasons_label.pack(anchor="w", pady=2)

        sep3 = ttk.Separator(right, orient="horizontal")
        sep3.pack(fill=tk.X, pady=10)

        checklist_title = ttk.Label(
            right,
            text="Run readiness checklist",
            font=("Arial", 11, "bold"),
        )
        checklist_title.pack(anchor="w", pady=(0, 4))

        self.ready_mocap_label = ttk.Label(right, text="Mocap fresh: ❌")
        self.ready_mocap_label.pack(anchor="w", pady=1)

        self.ready_safety_label = ttk.Label(right, text="Safety safe: ❌")
        self.ready_safety_label.pack(anchor="w", pady=1)

        self.ready_target_label = ttk.Label(right, text="Target selected: ❌")
        self.ready_target_label.pack(anchor="w", pady=1)

        self.ready_distance_label = ttk.Label(right, text="Distance allowed: ❌")
        self.ready_distance_label.pack(anchor="w", pady=1)

        self.ready_estop_label = ttk.Label(right, text="Emergency stop off: ✅")
        self.ready_estop_label.pack(anchor="w", pady=1)

        self.ready_overall_label = ttk.Label(
            right,
            text="Ready to run: ❌ UI monitor only",
            font=("Arial", 10, "bold"),
        )
        self.ready_overall_label.pack(anchor="w", pady=(5, 1))

        sep4 = ttk.Separator(right, orient="horizontal")
        sep4.pack(fill=tk.X, pady=10)

        self.proposed_label = ttk.Label(right, text="Proposed cmd: n/a")
        self.proposed_label.pack(anchor="w", pady=2)

        self.safe_cmd_label = ttk.Label(right, text="Safe cmd: n/a")
        self.safe_cmd_label.pack(anchor="w", pady=2)

        sep5 = ttk.Separator(right, orient="horizontal")
        sep5.pack(fill=tk.X, pady=10)

        lower_controls = ttk.Frame(right)
        lower_controls.pack(fill=tk.X, pady=(0, 6))

        quick_box = ttk.LabelFrame(lower_controls, text="Quick target")
        quick_box.grid(row=0, column=0, sticky="nw", padx=(0, 8))

        action_box = ttk.LabelFrame(lower_controls, text="Actions")
        action_box.grid(row=0, column=1, sticky="new")

        lower_controls.columnconfigure(1, weight=1)

        ttk.Button(
            quick_box,
            text="Forward 30cm",
            command=lambda: self.set_relative_target(0.30, 0.0),
            width=13,
        ).grid(row=0, column=0, padx=2, pady=2)

        ttk.Button(
            quick_box,
            text="Back 30cm",
            command=lambda: self.set_relative_target(-0.30, 0.0),
            width=13,
        ).grid(row=0, column=1, padx=2, pady=2)

        ttk.Button(
            quick_box,
            text="Left 30cm",
            command=lambda: self.set_relative_target(0.0, 0.30),
            width=13,
        ).grid(row=1, column=0, padx=2, pady=2)

        ttk.Button(
            quick_box,
            text="Right 30cm",
            command=lambda: self.set_relative_target(0.0, -0.30),
            width=13,
        ).grid(row=1, column=1, padx=2, pady=2)

        ttk.Button(
            quick_box,
            text="Forward 50cm",
            command=lambda: self.set_relative_target(0.50, 0.0),
            width=13,
        ).grid(row=2, column=0, padx=2, pady=2)

        ttk.Button(
            quick_box,
            text="Back 50cm",
            command=lambda: self.set_relative_target(-0.50, 0.0),
            width=13,
        ).grid(row=2, column=1, padx=2, pady=2)

        ttk.Button(
            action_box,
            text="Run A→B",
            command=self.run_ab,
        ).pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(
            action_box,
            text="Stop Controller",
            command=self.stop_controller,
        ).pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(
            action_box,
            text="Clear Target",
            command=self.clear_target,
        ).pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(
            action_box,
            text="EMERGENCY STOP",
            command=self.activate_emergency_stop,
        ).pack(fill=tk.X, padx=4, pady=(8, 2))

        ttk.Button(
            action_box,
            text="Reset E-Stop",
            command=self.reset_emergency_stop,
        ).pack(fill=tk.X, padx=4, pady=2)

        self.disabled_run_label = ttk.Label(
            right,
            text="Run A→B publishes target + RUN to MIQP receding point controller. Safety pipeline still required.",
            foreground="gray",
            wraplength=400,
        )
        self.disabled_run_label.pack(anchor="w", pady=(4, 0))

    def get_pose_snapshot(self):
        with self.node.lock:
            pose = self.node.pose
            last_pose_time = self.node.last_pose_time
            safety = dict(self.node.safety_status)
            proposed = self.node.proposed_cmd
            safe_cmd = self.node.safe_cmd

            proposed_tuple = (
                float(proposed.linear.x),
                float(proposed.linear.y),
                float(proposed.angular.z),
            )
            safe_tuple = (
                float(safe_cmd.linear.x),
                float(safe_cmd.linear.y),
                float(safe_cmd.angular.z),
            )

        return pose, last_pose_time, safety, proposed_tuple, safe_tuple

    def corrected_heading(self, raw_yaw):
        return wrap_angle(raw_yaw + self.heading_offset_rad)

    def robot_local_to_world(self, x, y, heading, forward_m, left_m):
        dx_world = forward_m * math.cos(heading) - left_m * math.sin(heading)
        dy_world = forward_m * math.sin(heading) + left_m * math.cos(heading)
        return x + dx_world, y + dy_world

    def reset_heading_calibration(self):
        self.heading_offset_rad = UI_HEADING_OFFSET_RAD
        self.calib_anchor_pose = None
        self.last_forward_heading = None
        print(
            f"[UI heading calib] reset offset to "
            f"{math.degrees(self.heading_offset_rad):+.1f} deg"
        )

    def update_heading_calibration(self, pose, safe_cmd):
        if pose is None:
            return

        x, y, yaw = pose
        safe_vx, _, safe_wz = safe_cmd

        # Only calibrate during mostly straight forward command.
        if safe_vx < CALIBRATE_MIN_SAFE_VX or abs(safe_wz) > CALIBRATE_MAX_ABS_YAW_RATE:
            self.calib_anchor_pose = None
            return

        if self.calib_anchor_pose is None:
            self.calib_anchor_pose = (x, y, yaw)
            return

        ax, ay, ayaw = self.calib_anchor_pose
        dx = x - ax
        dy = y - ay
        dist = math.hypot(dx, dy)

        if dist < CALIBRATE_MIN_STEP_M:
            return

        measured_forward_heading = math.atan2(dy, dx)
        new_offset = wrap_angle(measured_forward_heading - ayaw)

        err = wrap_angle(new_offset - self.heading_offset_rad)
        self.heading_offset_rad = wrap_angle(
            self.heading_offset_rad + CALIBRATION_ALPHA * err
        )

        self.last_forward_heading = measured_forward_heading
        self.calib_anchor_pose = (x, y, yaw)

        print(
            f"[UI heading calib] movement_heading={math.degrees(measured_forward_heading):+.1f} deg, "
            f"raw_yaw={math.degrees(yaw):+.1f} deg, "
            f"offset={math.degrees(self.heading_offset_rad):+.1f} deg"
        )

    def set_fixed_center_to_current(self):
        pose, _, _, _, _ = self.get_pose_snapshot()
        if pose is None:
            return
        x, y, _ = pose
        self.fixed_center = (x, y)

    def clear_trail(self):
        self.trail = []

    def activate_emergency_stop(self):
        self.estop_active = True
        self.node.send_stop()
        self.node.set_emergency(True)

    def reset_emergency_stop(self):
        self.estop_active = False
        self.controller_proc = None
        self.controller_lock = threading.Lock()
        self.node.set_emergency(False)

    def set_relative_target(self, forward_m, left_m):
        pose, _, _, _, _ = self.get_pose_snapshot()
        if pose is None:
            return

        x, y, yaw = pose
        heading = self.corrected_heading(yaw)

        self.target_xy = self.robot_local_to_world(
            x,
            y,
            heading,
            forward_m,
            left_m,
        )
        self.target_local = (forward_m, left_m)
        self.target_final_yaw = heading
        self.aiming_yaw = False
        self.mouse_canvas_xy = None
        self.publish_current_target()

    def publish_current_target(self):
        if self.target_xy is None:
            return

        x, y = self.target_xy
        self.node.publish_target(x, y)
        if self.target_final_yaw is not None:
            self.node.publish_target_yaw(self.target_final_yaw)

    def clear_target(self):
        self.target_xy = None
        self.target_local = None
        self.target_final_yaw = None
        self.aiming_yaw = False
        self.mouse_canvas_xy = None
        self.target_final_yaw = None
        self.aiming_yaw = False
        self.mouse_canvas_xy = None

    def run_ab(self):
        if self.target_xy is None:
            print("Cannot run: no target selected")
            return

        self.publish_current_target()
        self.node.send_run()
        print("[UI] Sent target + RUN to online receding MIQP DeePC controller")


    def stop_controller(self):

        print("[UI] Stop requested")

        with self.controller_lock:
            proc = self.controller_proc
            self.controller_proc = None

        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        self.node.send_stop()

    def canvas_to_world(self, px, py, pose):
        current_x, current_y, yaw = pose
        heading = self.corrected_heading(yaw)

        if self.view_mode.get() == "robot":
            forward_m = -(py - self.canvas_h / 2.0) / self.scale_px_per_m
            left_m = -(px - self.canvas_w / 2.0) / self.scale_px_per_m
            return self.robot_local_to_world(
                current_x,
                current_y,
                heading,
                forward_m,
                left_m,
            ), (forward_m, left_m)

        center_x, center_y = self.get_view_center(pose)
        target_x = center_x + (px - self.canvas_w / 2.0) / self.scale_px_per_m
        target_y = center_y - (py - self.canvas_h / 2.0) / self.scale_px_per_m
        return (target_x, target_y), None

    def set_target_from_canvas(self, event, publish=True, start_yaw_select=False):
        pose, _, _, _, _ = self.get_pose_snapshot()
        if pose is None:
            return

        current_x, current_y, _ = pose
        (target_x, target_y), local = self.canvas_to_world(event.x, event.y, pose)

        dist_from_robot = math.hypot(target_x - current_x, target_y - current_y)
        if dist_from_robot > MAX_TARGET_DISTANCE_M:
            print(
                f"Target rejected: {dist_from_robot:.2f} m "
                f"is above {MAX_TARGET_DISTANCE_M:.2f} m UI limit"
            )
            return

        self.target_xy = (target_x, target_y)
        self.target_local = local

        if start_yaw_select:
            self.target_final_yaw = None
            self.aiming_yaw = True
            self.mouse_canvas_xy = (event.x, event.y)
            print("[UI] Target B selected. Move mouse to aim final yaw, then left click to lock.")
        else:
            self.aiming_yaw = False
            self.mouse_canvas_xy = None

        if publish:
            self.publish_current_target()

    def on_canvas_right_click(self, event):
        self.set_target_from_canvas(event, publish=False, start_yaw_select=True)

    def on_canvas_motion(self, event):
        if self.aiming_yaw:
            self.mouse_canvas_xy = (event.x, event.y)

    def lock_yaw_from_canvas(self, event):
        pose, _, _, _, _ = self.get_pose_snapshot()
        if pose is None or self.target_xy is None:
            return

        (mouse_x, mouse_y), _ = self.canvas_to_world(event.x, event.y, pose)
        target_x, target_y = self.target_xy

        final_yaw = math.atan2(mouse_y - target_y, mouse_x - target_x)
        self.target_final_yaw = wrap_angle(final_yaw)
        self.aiming_yaw = False
        self.mouse_canvas_xy = (event.x, event.y)

        self.publish_current_target()

        print(
            f"[UI] Locked final yaw: {math.degrees(self.target_final_yaw):+.1f} deg"
        )

    def on_canvas_click(self, event):
        if self.aiming_yaw and self.target_xy is not None:
            self.lock_yaw_from_canvas(event)
            return

        self.set_target_from_canvas(event, publish=True, start_yaw_select=False)

    def get_view_center(self, pose):
        x, y, _ = pose

        if self.view_mode.get() == "robot":
            return x, y

        if self.fixed_center is None:
            self.fixed_center = (x, y)

        return self.fixed_center

    def world_to_canvas(self, x, y, center_x, center_y, heading_rad=None):
        dx = x - center_x
        dy = y - center_y

        if self.view_mode.get() == "robot" and heading_rad is not None:
            # Rotate world into robot-local coordinates.
            # Screen up = robot forward.
            # Screen left = robot left.
            forward = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
            left = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)

            cx = self.canvas_w / 2.0 - left * self.scale_px_per_m
            cy = self.canvas_h / 2.0 - forward * self.scale_px_per_m
            return cx, cy

        cx = self.canvas_w / 2.0 + dx * self.scale_px_per_m
        cy = self.canvas_h / 2.0 - dy * self.scale_px_per_m
        return cx, cy

    def draw_grid(self):
        self.canvas.delete("all")

        cx = self.canvas_w / 2.0
        cy = self.canvas_h / 2.0
        grid_step_m = 0.25
        grid_step_px = grid_step_m * self.scale_px_per_m

        for i in range(-8, 9):
            x = cx + i * grid_step_px
            self.canvas.create_line(x, 0, x, self.canvas_h, fill="#eeeeee")
            y = cy + i * grid_step_px
            self.canvas.create_line(0, y, self.canvas_w, y, fill="#eeeeee")

        self.canvas.create_line(cx, 0, cx, self.canvas_h, fill="#cccccc")
        self.canvas.create_line(0, cy, self.canvas_w, cy, fill="#cccccc")

        r30 = 0.30 * self.scale_px_per_m
        self.canvas.create_oval(
            cx - r30,
            cy - r30,
            cx + r30,
            cy + r30,
            outline="#dddddd",
            dash=(3, 3),
        )

        r50 = 0.50 * self.scale_px_per_m
        self.canvas.create_oval(
            cx - r50,
            cy - r50,
            cx + r50,
            cy + r50,
            outline="#dddddd",
            dash=(3, 3),
        )

    def draw_scene(self, pose):
        self.draw_grid()

        if pose is None:
            self.canvas.create_text(
                self.canvas_w / 2,
                self.canvas_h / 2,
                text="Waiting for /mocap/go2_pose",
                fill="red",
                font=("Arial", 16),
            )
            return

        x, y, yaw = pose
        center_x, center_y = self.get_view_center(pose)
        view_heading = self.corrected_heading(yaw)

        self.trail.append((x, y))
        if len(self.trail) > self.max_trail:
            self.trail = self.trail[-self.max_trail:]

        # Trail
        prev = None
        for tx, ty in self.trail:
            c = self.world_to_canvas(tx, ty, center_x, center_y, view_heading)
            if prev is not None:
                self.canvas.create_line(prev[0], prev[1], c[0], c[1], fill="#999999")
            prev = c

        # Robot
        robot_cx, robot_cy = self.world_to_canvas(
            x, y, center_x, center_y, view_heading
        )

        r = 8
        self.canvas.create_oval(
            robot_cx - r,
            robot_cy - r,
            robot_cx + r,
            robot_cy + r,
            fill="blue",
            outline="black",
        )

        # Heading arrow
        heading_len = 35

        if self.view_mode.get() == "robot":
            # In robot-centered mode, corrected forward is always screen-up.
            hx = robot_cx
            hy = robot_cy - heading_len
        else:
            hx = robot_cx + heading_len * math.cos(view_heading)
            hy = robot_cy - heading_len * math.sin(view_heading)

        self.canvas.create_line(
            robot_cx,
            robot_cy,
            hx,
            hy,
            arrow=tk.LAST,
            width=2,
            fill="blue",
        )

        # Target B
        if self.target_xy is not None:
            tx, ty = self.target_xy
            tcx, tcy = self.world_to_canvas(
                tx, ty, center_x, center_y, view_heading
            )

            dist = math.hypot(tx - x, ty - y)

            if dist <= CAUTION_TARGET_DISTANCE_M:
                target_color = "green"
            elif dist <= MAX_TARGET_DISTANCE_M:
                target_color = "orange"
            else:
                target_color = "red"

            self.canvas.create_line(
                robot_cx,
                robot_cy,
                tcx,
                tcy,
                fill=target_color,
                width=2,
            )

            tr = 8
            self.canvas.create_oval(
                tcx - tr,
                tcy - tr,
                tcx + tr,
                tcy + tr,
                fill="orange",
                outline="black",
            )

            tol = 0.05 * self.scale_px_per_m
            self.canvas.create_oval(
                tcx - tol,
                tcy - tol,
                tcx + tol,
                tcy + tol,
                outline="orange",
                dash=(3, 3),
            )

        # Final yaw arrow
        if self.target_xy is not None:
            tx, ty = self.target_xy
            tcx, tcy = self.world_to_canvas(
                tx, ty, center_x, center_y, view_heading
            )

            if self.aiming_yaw and self.mouse_canvas_xy is not None:
                ax, ay = self.mouse_canvas_xy
                self.canvas.create_line(
                    tcx,
                    tcy,
                    ax,
                    ay,
                    arrow=tk.LAST,
                    width=3,
                    fill="purple",
                )
                self.canvas.create_text(
                    tcx + 10,
                    tcy - 20,
                    text="Left click to lock final yaw",
                    anchor="w",
                    fill="purple",
                    font=("Arial", 10, "bold"),
                )
            elif self.target_final_yaw is not None:
                yaw_len = 0.18
                ex = tx + yaw_len * math.cos(self.target_final_yaw)
                ey = ty + yaw_len * math.sin(self.target_final_yaw)
                ecx, ecy = self.world_to_canvas(
                    ex, ey, center_x, center_y, view_heading
                )
                self.canvas.create_line(
                    tcx,
                    tcy,
                    ecx,
                    ecy,
                    arrow=tk.LAST,
                    width=3,
                    fill="purple",
                )

        # View mode text
        self.canvas.create_text(
            10,
            10,
            text=f"View: {'Robot-centered' if self.view_mode.get() == 'robot' else 'Fixed world'}",
            anchor="nw",
            fill="black",
            font=("Arial", 10, "bold"),
        )

    def compute_readiness(self, pose, last_pose_time, safety):
        now = time.time()

        mocap_fresh = (
            pose is not None
            and last_pose_time is not None
            and (now - last_pose_time) < MOCAP_FRESH_SEC
        )

        safety_safe = bool(safety.get("safe", False))
        target_selected = self.target_xy is not None

        distance_allowed = False
        dist = None
        if pose is not None and self.target_xy is not None:
            x, y, _ = pose
            tx, ty = self.target_xy
            dist = math.hypot(tx - x, ty - y)
            distance_allowed = dist <= MAX_TARGET_DISTANCE_M

        estop_off = not self.estop_active

        ready = (
            mocap_fresh
            and safety_safe
            and target_selected
            and distance_allowed
            and estop_off
        )

        return {
            "mocap_fresh": mocap_fresh,
            "safety_safe": safety_safe,
            "target_selected": target_selected,
            "distance_allowed": distance_allowed,
            "estop_off": estop_off,
            "ready": ready,
            "dist": dist,
        }

    def update_readiness_labels(self, readiness):
        self.ready_mocap_label.config(
            text=f"Mocap fresh: {'✅' if readiness['mocap_fresh'] else '❌'}"
        )

        self.ready_safety_label.config(
            text=f"Safety safe: {'✅' if readiness['safety_safe'] else '❌'}"
        )

        self.ready_target_label.config(
            text=f"Target selected: {'✅' if readiness['target_selected'] else '❌'}"
        )

        if readiness["dist"] is None:
            dist_text = "n/a"
        else:
            dist_text = f"{readiness['dist']:.3f} m"

        self.ready_distance_label.config(
            text=(
                f"Distance allowed: "
                f"{'✅' if readiness['distance_allowed'] else '❌'} "
                f"({dist_text})"
            )
        )

        self.ready_estop_label.config(
            text=f"Emergency stop off: {'✅' if readiness['estop_off'] else '❌'}"
        )

        self.ready_overall_label.config(
            text=(
                "Ready to run: ✅ UI ready, receding point controller connected"
                if readiness["ready"]
                else "Ready to run: ❌"
            )
        )

    def update_labels(self, pose, last_pose_time, safety, proposed, safe_cmd):
        now = time.time()

        if pose is None:
            self.pose_label.config(text="Pose: waiting for /mocap/go2_pose")
            self.heading_label.config(text="Heading offset: waiting...")
        else:
            x, y, yaw = pose
            age = now - last_pose_time if last_pose_time else 999.0
            fresh = age < MOCAP_FRESH_SEC
            state = "fresh" if fresh else "STALE"
            corrected = self.corrected_heading(yaw)

            self.pose_label.config(
                text=(
                    f"Pose: x={x:+.3f} m, y={y:+.3f} m, "
                    f"yaw={yaw:+.3f} rad ({state})"
                )
            )

            self.heading_label.config(
                text=(
                    f"Heading offset: {math.degrees(self.heading_offset_rad):+.1f} deg | "
                    f"corrected heading: {math.degrees(corrected):+.1f} deg"
                )
            )

        if self.target_xy is None:
            self.target_label.config(text="Target B: none")
            self.dist_label.config(text="Distance to B: n/a")
        else:
            tx, ty = self.target_xy
            if self.target_final_yaw is None:
                yaw_text = "final yaw: not set"
            else:
                yaw_text = f"final yaw={math.degrees(self.target_final_yaw):+.1f} deg"
            self.target_label.config(
                text=f"Target B: x={tx:+.3f}, y={ty:+.3f}, {yaw_text}"
            )

            if pose is not None:
                x, y, _ = pose
                dist = math.hypot(tx - x, ty - y)

                if dist <= CAUTION_TARGET_DISTANCE_M:
                    label = "safe"
                elif dist <= MAX_TARGET_DISTANCE_M:
                    label = "caution"
                else:
                    label = "too far"

                self.dist_label.config(text=f"Distance to B: {dist:.3f} m ({label})")
            else:
                self.dist_label.config(text="Distance to B: n/a")

        safe = safety.get("safe", False)
        mode = safety.get("mode", "unknown")
        reasons = safety.get("reasons", [])

        self.safety_label.config(
            text=f"Safety: {'SAFE' if safe else 'UNSAFE'} | mode={mode}"
        )
        self.reasons_label.config(
            text="Reasons: " + (", ".join(reasons) if reasons else "[]")
        )

        readiness = self.compute_readiness(pose, last_pose_time, safety)
        self.update_readiness_labels(readiness)

        pvx, pvy, pwz = proposed
        svx, svy, swz = safe_cmd

        self.proposed_label.config(
            text=f"Proposed: vx={pvx:+.3f}, vy={pvy:+.3f}, wz={pwz:+.3f}"
        )
        self.safe_cmd_label.config(
            text=f"Safe:     vx={svx:+.3f}, vy={svy:+.3f}, wz={swz:+.3f}"
        )

    def update_loop(self):
        pose, last_pose_time, safety, proposed, safe_cmd = self.get_pose_snapshot()

        # DISABLED for mode-switched textbook DeePC: controllers use fixed -65 deg heading offset.
        # self.update_heading_calibration(pose, safe_cmd)
        self.draw_scene(pose)
        self.update_labels(pose, last_pose_time, safety, proposed, safe_cmd)

        self.root.after(100, self.update_loop)


def main():
    rclpy.init()
    node = DeePCUiRos()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    DeePCUiApp(root, node)

    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
