#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class DatasetPublisher(Node):
    def __init__(self):
        super().__init__("collect_dog2_vxyyaw_dataset")
        self.pub = self.create_publisher(Twist, "/deepc/proposed_cmd", 10)

    def send(self, vx, vy, wz, sec, label):
        self.get_logger().warn(f"{label}: vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f}, sec={sec}")
        t_end = time.time() + sec
        rate = 0.10
        while time.time() < t_end:
            msg = Twist()
            msg.linear.x = float(vx)
            msg.linear.y = float(vy)
            msg.angular.z = float(wz)
            self.pub.publish(msg)
            time.sleep(rate)
        self.zero(2.0)

    def zero(self, sec=2.0):
        t_end = time.time() + sec
        while time.time() < t_end:
            msg = Twist()
            self.pub.publish(msg)
            time.sleep(0.10)

def main():
    rclpy.init()
    n = DatasetPublisher()
    time.sleep(1.0)

    blocks = [
        # pure useful axes
        (0.10, 0.00, 0.00, 5.0, "forward_010"),
        (0.15, 0.00, 0.00, 5.0, "forward_015"),
        (0.00, 0.10, 0.00, 5.0, "left_010"),
        (0.00,-0.10, 0.00, 5.0, "right_010"),
        (0.00, 0.00, 0.20, 5.0, "yaw_pos_020"),
        (0.00, 0.00,-0.20, 5.0, "yaw_neg_020"),
        (0.00, 0.00, 0.30, 5.0, "yaw_pos_030"),
        (0.00, 0.00,-0.30, 5.0, "yaw_neg_030"),

        # diagonal / combined movement
        (0.10, 0.10, 0.00, 5.0, "forward_left"),
        (0.10,-0.10, 0.00, 5.0, "forward_right"),
        (0.10, 0.00, 0.20, 5.0, "curve_left"),
        (0.10, 0.00,-0.20, 5.0, "curve_right"),
        (0.10, 0.10, 0.20, 5.0, "forward_left_yaw"),
        (0.10,-0.10,-0.20, 5.0, "forward_right_yaw"),

        # repeat useful primitives
        (0.00, 0.10, 0.00, 5.0, "left_repeat"),
        (0.00,-0.10, 0.00, 5.0, "right_repeat"),
        (0.10, 0.10, 0.00, 5.0, "diag_left_repeat"),
        (0.10,-0.10, 0.00, 5.0, "diag_right_repeat"),
    ]

    for b in blocks:
        n.send(*b)

    n.zero(3.0)
    n.get_logger().warn("DONE dataset command sequence")
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
