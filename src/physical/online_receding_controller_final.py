#!/usr/bin/env python3
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from std_msgs.msg import Bool, String, Float64


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class OnlineMIQPDeePC(Node):
    def __init__(self):
        super().__init__("miqp_online_receding_deepc_controller")

        project_root = Path(__file__).resolve().parents[2]
        self.base = project_root / "data/physical/runtime"
        self.base.mkdir(parents=True, exist_ok=True)
        self.solver = project_root / "src/physical/miqp_solver_final.py"
        self.hankel = project_root / "data/physical/hankel/dog2_vxyyaw_hankel_Tini10_N20_BALANCED_YAW_BACK_2026_07_06.npz"
        self.center_report = project_root / "results/physical/go2_center_correction_report.json"

        if not self.center_report.exists():
            raise FileNotFoundError(
                f"Missing {self.center_report}. Need center correction report from correct_center_and_rebuild_hankel.py"
            )

        report = json.loads(self.center_report.read_text())
        off = report["estimated_body_offset_in_tracked_frame_m"]
        self.center_rx = float(off["rx_forward_m"])
        self.center_ry = float(off["ry_left_m"])

        self.get_logger().warn(
            f"Loaded center correction: rx={self.center_rx:+.4f}m, ry={self.center_ry:+.4f}m"
        )

        z = np.load(self.hankel, allow_pickle=True)
        self.Tini = 10
        self.N = 20

        self.pose = None  # CENTER-corrected pose used for DeePC/control
        self.raw_pose = None  # raw mocap pose only for UI target conversion
        self.prev_sample_pose = None
        self.last_pose_time = None
        self.last_good_xy = None

        self.u_hist = deque(maxlen=self.Tini)
        self.y_hist = deque(maxlen=self.Tini)
        for _ in range(self.Tini):
            self.u_hist.append((0.0, 0.0, 0.0))
            self.y_hist.append((0.0, 0.0, 0.0))

        self.target = None
        self.target_final_yaw = None
        self.active = False
        self.position_hold = False
        self.start_time = None
        self.best_goal_dist = None

        self.current_cmd = (0.0, 0.0, 0.0)
        self.solving = False
        self.last_solve_time = 0.0
        self.solve_period = 0.35

        # The physical Go2 cannot reliably enter a 3.5 cm window before an
        # in-flight MIQP command carries it past the target. Enter position
        # hold sooner, then finish the requested yaw without translating.
        self.goal_tol = 0.06
        self.final_yaw_tol = 0.05
        self.max_time = 25.0
        self.max_target_dist = 0.70
        self.mocap_stale_sec = 0.50
        self.closest_pass_margin = 0.10

        self.lock = threading.Lock()
        self.solve_generation = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(PoseStamped, "/mocap/go2_pose", self.pose_cb, qos)
        self.create_subscription(PointStamped, "/deepc/ui_target_point", self.target_cb, 10)
        self.create_subscription(Float64, "/deepc/ui_target_yaw", self.target_yaw_cb, 10)
        self.create_subscription(Bool, "/deepc/ui_run", self.run_cb, 10)
        self.create_subscription(Bool, "/deepc/ui_stop", self.stop_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, "/deepc/proposed_cmd", 10)
        self.status_pub = self.create_publisher(String, "/deepc/ui_controller_status", 10)

        self.timer = self.create_timer(0.05, self.timer_cb)

    def norm_xy(self, x, y):
        if abs(x) > 10.0 or abs(y) > 10.0:
            return x / 1000.0, y / 1000.0
        return x, y

    def center_correct_xy(self, x, y, yaw):
        # Same correction as correct_center_and_rebuild_hankel.py:
        # center = tracked_position - R(yaw) * [rx, ry]
        c = math.cos(yaw)
        s = math.sin(yaw)
        off_x = c * self.center_rx - s * self.center_ry
        off_y = s * self.center_rx + c * self.center_ry
        return x - off_x, y - off_y

    def pose_cb(self, msg):
        raw_x, raw_y = self.norm_xy(float(msg.pose.position.x), float(msg.pose.position.y))
        yaw_raw = yaw_from_quat(msg.pose.orientation)
        yaw = wrap_pi(yaw_raw + math.radians(98.65390921916695))

        if self.last_good_xy is not None:
            lx, ly = self.last_good_xy
            if math.hypot(raw_x - lx, raw_y - ly) > 0.30:
                self.get_logger().warn("Rejected mocap jump")
                return

        cx, cy = self.center_correct_xy(raw_x, raw_y, yaw)

        with self.lock:
            self.raw_pose = (raw_x, raw_y, yaw)
            self.pose = (cx, cy, yaw)
            self.last_good_xy = (raw_x, raw_y)
            self.last_pose_time = time.time()

    def target_cb(self, msg):
        # UI now publishes center-corrected target coordinates.
        tx, ty = self.norm_xy(float(msg.point.x), float(msg.point.y))

        with self.lock:
            self.target = (tx, ty)
            if hasattr(self, "raw_target"):
                self.raw_target = (tx, ty)
            self.best_goal_dist = None

        self.get_logger().warn(
            f"Received UI target center_frame=({tx:.3f},{ty:.3f})"
        )

    def target_yaw_cb(self, msg):
        yaw = wrap_pi(float(msg.data))
        with self.lock:
            self.target_final_yaw = yaw
        self.get_logger().warn(f"Received UI final yaw={yaw:+.3f} rad")

    def run_cb(self, msg):
        if not msg.data:
            return

        with self.lock:
            if self.pose is None or self.target is None:
                self.get_logger().warn("RUN refused: missing pose/target")
                return

            x, y, _ = self.pose
            tx, ty = self.target
            d = math.hypot(tx - x, ty - y)

            if d > self.max_target_dist:
                self.get_logger().warn(f"RUN refused: target too far {d:.3f}m")
                return

            if self.target_final_yaw is None:
                self.target_final_yaw = self.pose[2]

            self.active = True
            self.position_hold = False
            self.start_time = time.time()
            self.best_goal_dist = d
            self.current_cmd = (0.0, 0.0, 0.0)
            self.solve_generation += 1
            self.last_solve_time = 0.0

        self.get_logger().warn(f"RUN accepted: goal_dist={d:.3f}m")

    def stop_cb(self, msg):
        if msg.data:
            self.stop_done("ui_stop")

    def send_cmd(self, vx, vy, wz):
        m = Twist()
        m.linear.x = float(vx)
        m.linear.y = float(vy)
        m.angular.z = float(wz)
        self.cmd_pub.publish(m)

    def send_zero(self, n=8):
        for _ in range(n):
            self.send_cmd(0.0, 0.0, 0.0)

    def status(self, mode, **kw):
        msg = String()
        data = {"active": self.active, "mode": mode}
        data.update(kw)
        msg.data = json.dumps(data)
        self.status_pub.publish(msg)

    def stop_done(self, reason, goal_dist=None, heading_error=None):
        with self.lock:
            self.active = False
            self.position_hold = False
            self.current_cmd = (0.0, 0.0, 0.0)
            self.solve_generation += 1
        self.send_zero(12)
        self.get_logger().warn(f"STOP {reason}: goal_dist={goal_dist}, heading_error={heading_error}")
        self.status(reason, goal_dist=goal_dist, heading_error=heading_error)

    def sample_history(self):
        with self.lock:
            pose = self.pose
            cmd = self.current_cmd

        if pose is None:
            return

        if self.prev_sample_pose is None:
            self.prev_sample_pose = pose
            return

        x0, y0, yaw0 = self.prev_sample_pose
        x1, y1, yaw1 = pose

        dx = x1 - x0
        dy = y1 - y0

        forward_step = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        left_step = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        yaw_step = wrap_pi(yaw1 - yaw0)

        self.y_hist.append((forward_step, left_step, yaw_step))
        self.u_hist.append(cmd)
        self.prev_sample_pose = pose

    def local_target(self):
        with self.lock:
            pose = self.pose
            target = self.target
            target_final_yaw = self.target_final_yaw
            position_hold = self.position_hold
            best = self.best_goal_dist

        if pose is None or target is None:
            return None

        x, y, yaw = pose
        tx, ty = target

        dx = tx - x
        dy = ty - y

        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        left = -dx * math.sin(yaw) + dy * math.cos(yaw)
        goal_dist = math.hypot(dx, dy)

        heading_error = math.atan2(left, max(0.03, forward))

        # Pure 3-input target with final yaw.
        # No lateral-heavy / backward special-case motion strategy.
        if position_hold:
            target_forward = 0.0
            target_left = 0.0
        else:
            target_forward = clamp(forward, -0.10, 0.14)
            target_left = clamp(left, -0.10, 0.10)

        final_yaw_error = 0.0 if target_final_yaw is None else wrap_pi(target_final_yaw - yaw)
        target_yaw = clamp(final_yaw_error, -0.45, 0.45)

        return {
            "target_forward": target_forward,
            "target_left": target_left,
            "target_yaw": target_yaw,
            "goal_dist": goal_dist,
            "heading_error": heading_error,
            "forward": forward,
            "left": left,
            "best": best,
        }

    def maybe_start_solve(self):
        if self.solving:
            return

        now = time.time()
        if now - self.last_solve_time < self.solve_period:
            return

        lt = self.local_target()
        if lt is None:
            return

        with self.lock:
            self.solving = True
            self.solve_generation += 1
            solve_generation = self.solve_generation

        self.last_solve_time = now

        th = threading.Thread(target=self.solve_worker, args=(lt, solve_generation), daemon=True)
        th.start()

    def solve_worker(self, lt, solve_generation):
        try:
            u_ini = np.array(self.u_hist, dtype=float).reshape(-1)
            y_ini = np.array(self.y_hist, dtype=float).reshape(-1)

            u_path = self.base / "miqp_online_uini.npy"
            y_path = self.base / "miqp_online_yini.npy"
            out_csv = self.base / "miqp_online_latest.csv"
            out_json = self.base / "miqp_online_latest.json"

            np.save(u_path, u_ini)
            np.save(y_path, y_ini)

            cmd = [
                sys.executable,
                str(self.solver),
                "--hankel", str(self.hankel),
                "--u-ini-file", str(u_path),
                "--y-ini-file", str(y_path),
                "--target-forward", f"{lt['target_forward']:.5f}",
                "--target-left", f"{lt['target_left']:.5f}",
                "--target-yaw", f"{lt['target_yaw']:.5f}",
                "--use-n", "20",
                "--max-cols", "900",
                "--time-limit", "2.00",
                "--mip-gap", "0.08",
                "--threads", "1",
                "--w-g", "1000",
                "--simplex-g",
                "--w-effort-vy", "10000",
                "--w-terminal-left", "1200000",
                "--w-stage-left", "70000",
                "--w-terminal-yaw", "2500000",
                "--w-stage-yaw", "120000",
                "--out-csv", str(out_csv),
                "--out-json", str(out_json),
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.base),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2.8,
            )

            if result.returncode != 0:
                self.get_logger().warn("MIQP solve failed:\n" + result.stdout[-800:])
                with self.lock:
                    self.current_cmd = (0.0, 0.0, 0.0)
                return

            rows = list(csv.DictReader(open(out_csv)))
            if not rows:
                self.get_logger().warn("MIQP wrote empty CSV")
                with self.lock:
                    self.current_cmd = (0.0, 0.0, 0.0)
                return

            vx = float(rows[0]["vx_cmd"])
            vy = float(rows[0].get("vy_cmd", 0.0))
            wz = float(rows[0]["yaw_rate_cmd"])

            with self.lock:
                if (not self.active) or solve_generation != self.solve_generation:
                    self.get_logger().warn("Discarded stale MIQP solution after stop/newer solve")
                    return
                self.current_cmd = (vx, vy, wz)

            try:
                js = json.loads(Path(out_json).read_text())
                pred = js.get("terminal_prediction", {})
                self.get_logger().info(
                    f"MIQP cmd vx={vx:+.3f}, vy={vy:+.3f}, wz={wz:+.3f}, "
                    f"target=({lt['target_forward']:+.3f},{lt['target_left']:+.3f},{lt['target_yaw']:+.3f}), "
                    f"pred={pred}"
                )
            except Exception:
                self.get_logger().info(f"MIQP cmd vx={vx:+.3f}, wz={wz:+.3f}")

        except Exception as e:
            self.get_logger().warn(f"MIQP solve exception: {e}")
            with self.lock:
                self.current_cmd = (0.0, 0.0, 0.0)
        finally:
            self.solving = False

    def timer_cb(self):
        self.sample_history()

        with self.lock:
            active = self.active
            pose = self.pose
            target = self.target
            last_pose_time = self.last_pose_time
            start_time = self.start_time
            current_cmd = self.current_cmd

        if not active:
            self.status("idle")
            return

        now = time.time()

        if pose is None or target is None:
            self.stop_done("missing_pose_or_target")
            return

        if last_pose_time is None or now - last_pose_time > self.mocap_stale_sec:
            self.stop_done("mocap_stale")
            return

        if start_time is not None and now - start_time > self.max_time:
            self.stop_done("timeout")
            return

        lt = self.local_target()
        if lt is None:
            self.stop_done("no_local_target")
            return

        goal_dist = lt["goal_dist"]
        heading_error = lt["heading_error"]

        with self.lock:
            target_final_yaw = self.target_final_yaw

        current_yaw = pose[2]
        final_yaw_error = 0.0 if target_final_yaw is None else wrap_pi(target_final_yaw - current_yaw)

        with self.lock:
            if self.best_goal_dist is None or goal_dist < self.best_goal_dist:
                self.best_goal_dist = goal_dist
            best = self.best_goal_dist

        if goal_dist <= self.goal_tol and abs(final_yaw_error) <= self.final_yaw_tol:
            self.stop_done("target_reached_pos_yaw", goal_dist, final_yaw_error)
            return

        # Latch position hold as soon as the larger physical-robot tolerance
        # is reached. Do not resume translation if yaw correction or center
        # correction makes the measured distance move back outside the window.
        if goal_dist <= self.goal_tol:
            with self.lock:
                self.position_hold = True

        if (
            best is not None
            and best < 0.12
            and goal_dist > best + self.closest_pass_margin
            and abs(final_yaw_error) <= self.final_yaw_tol
        ):
            self.stop_done("closest_approach_passed_pos_yaw", goal_dist, final_yaw_error)
            return

        self.maybe_start_solve()

        vx, vy, wz = current_cmd
        with self.lock:
            position_hold = self.position_hold
        if position_hold:
            vx = 0.0
            vy = 0.0
        self.send_cmd(vx, vy, wz)

        self.get_logger().info(
            f"publish vx={vx:+.3f}, vy={vy:+.3f}, wz={wz:+.3f}, goal={goal_dist:.3f}, "
            f"heading_err={heading_error:+.3f}, final_yaw_err={final_yaw_error:+.3f}, "
            f"local_f={lt['forward']:+.3f}, local_l={lt['left']:+.3f}"
        )

        self.status(
            "running",
            goal_dist=goal_dist,
            heading_error=heading_error,
            vx=vx,
            wz=wz,
            best_goal_dist=best,
        )


def main():
    rclpy.init()
    node = OnlineMIQPDeePC()
    try:
        rclpy.spin(node)
    finally:
        node.send_zero(20)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
