#!/usr/bin/env python3
import math
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand

Vec3 = Tuple[float, float, float]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(max(1e-12, dot(a, a)))


def unit(a: Vec3, fallback: Vec3) -> Vec3:
    n = norm(a)
    if n < 1e-6:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def angle_between(a: Vec3, b: Vec3) -> float:
    au = unit(a, (1.0, 0.0, 0.0))
    bu = unit(b, (1.0, 0.0, 0.0))
    return math.acos(clamp(dot(au, bu), -1.0, 1.0))


class SimGeometricRetarget(Node):
    JOINT_NAMES = [
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
    ]

    def __init__(self):
        super().__init__("sim_geometric_retarget_tpose_yaw")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")

        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("tpose_hold_sec", 4.0)
        self.declare_parameter("max_joint_step_rad", 0.025)
        self.declare_parameter("yaw_switch_threshold_m", 0.045)
        self.declare_parameter("visibility_threshold", 0.25)

        # T-поза. Если будет неидеально по геометрии, потом подстроим эти 8 чисел.
        self.declare_parameter("t_right_pitch", 0.0)
        self.declare_parameter("t_right_roll", -0.65)
        self.declare_parameter("t_right_yaw_up", 1.30)
        self.declare_parameter("t_right_yaw_down", 0.35)
        self.declare_parameter("t_right_elbow", 1.20)

        self.declare_parameter("t_left_pitch", 0.0)
        self.declare_parameter("t_left_roll", 0.65)
        self.declare_parameter("t_left_yaw_up", -1.30)
        self.declare_parameter("t_left_yaw_down", -0.35)
        self.declare_parameter("t_left_elbow", 1.20)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.tpose_hold_sec = float(self.get_parameter("tpose_hold_sec").value)
        self.max_step = float(self.get_parameter("max_joint_step_rad").value)
        self.yaw_th = float(self.get_parameter("yaw_switch_threshold_m").value)
        self.vis_th = float(self.get_parameter("visibility_threshold").value)

        self.t_right_pitch = float(self.get_parameter("t_right_pitch").value)
        self.t_right_roll = float(self.get_parameter("t_right_roll").value)
        self.t_right_yaw_up = float(self.get_parameter("t_right_yaw_up").value)
        self.t_right_yaw_down = float(self.get_parameter("t_right_yaw_down").value)
        self.t_right_elbow = float(self.get_parameter("t_right_elbow").value)

        self.t_left_pitch = float(self.get_parameter("t_left_pitch").value)
        self.t_left_roll = float(self.get_parameter("t_left_roll").value)
        self.t_left_yaw_up = float(self.get_parameter("t_left_yaw_up").value)
        self.t_left_yaw_down = float(self.get_parameter("t_left_yaw_down").value)
        self.t_left_elbow = float(self.get_parameter("t_left_elbow").value)

        self.tpose = [
            self.t_right_pitch,
            self.t_right_roll,
            self.t_right_yaw_up,
            self.t_right_elbow,
            self.t_left_pitch,
            self.t_left_roll,
            self.t_left_yaw_up,
            self.t_left_elbow,
        ]

        self.current_q = list(self.tpose)
        self.target_q = list(self.tpose)

        self.right_mode = "up"
        self.left_mode = "up"

        self.last_lm: Optional[Dict[str, Vec3]] = None
        self.last_vis: Dict[str, float] = {}

        self.start_time = self.get_clock().now()

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.subscriber = self.create_subscription(
            PoseLandmarks3D,
            self.input_topic,
            self.pose_cb,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_rate_hz),
            self.timer_cb,
        )

        self.get_logger().info("============================================================")
        self.get_logger().info("SIM GEOMETRIC RETARGET T-POSE + YAW HYSTERESIS")
        self.get_logger().info(f"input_topic:      {self.input_topic}")
        self.get_logger().info(f"output_topic:     {self.output_topic}")
        self.get_logger().info(f"T-pose q:         {[round(x, 3) for x in self.tpose]}")
        self.get_logger().info(f"tpose_hold_sec:   {self.tpose_hold_sec}")
        self.get_logger().info(f"max_step:         {self.max_step}")
        self.get_logger().info("Initial elbow mode: up")
        self.get_logger().info("============================================================")

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def pose_cb(self, msg: PoseLandmarks3D):
        if not msg.valid:
            return

        lm = {}
        vis = {}

        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                lm[name] = (float(msg.x[i]), float(msg.y[i]), float(msg.z[i]))
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0

        required = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
        ]

        if all(name in lm for name in required):
            self.last_lm = lm
            self.last_vis = vis

    def body_frame(self, lm: Dict[str, Vec3]):
        ls = lm["left_shoulder"]
        rs = lm["right_shoulder"]
        lh = lm["left_hip"]
        rh = lm["right_hip"]

        shoulders = mul(add(ls, rs), 0.5)
        hips = mul(add(lh, rh), 0.5)

        body_right = unit(sub(ls, rs), (0.0, 1.0, 0.0))
        body_up = unit(sub(shoulders, hips), (0.0, 0.0, 1.0))
        body_forward = unit(cross(body_right, body_up), (1.0, 0.0, 0.0))
        body_right = unit(cross(body_up, body_forward), body_right)

        return body_right, body_up, body_forward

    def wrist_residual(self, shoulder: Vec3, elbow: Vec3, wrist: Vec3, body_up: Vec3) -> float:
        upper = sub(elbow, shoulder)
        sw = sub(wrist, shoulder)
        denom = max(1e-6, dot(upper, upper))
        t = clamp(dot(sw, upper) / denom, 0.0, 1.0)
        point_on_upper = add(shoulder, mul(upper, t))
        residual_vec = sub(wrist, point_on_upper)
        return dot(residual_vec, body_up)

    def update_mode(self, side: str, residual: float):
        if side == "right":
            old = self.right_mode
            if residual > self.yaw_th:
                self.right_mode = "up"
            elif residual < -self.yaw_th:
                self.right_mode = "down"
            if old != self.right_mode:
                self.get_logger().info(f"right yaw mode: {old} -> {self.right_mode}, residual={residual:.3f}")
        else:
            old = self.left_mode
            if residual > self.yaw_th:
                self.left_mode = "up"
            elif residual < -self.yaw_th:
                self.left_mode = "down"
            if old != self.left_mode:
                self.get_logger().info(f"left yaw mode: {old} -> {self.left_mode}, residual={residual:.3f}")

    def solve_arm(self, side: str, shoulder: Vec3, elbow: Vec3, wrist: Vec3, body_right: Vec3, body_up: Vec3, body_forward: Vec3):
        upper = sub(elbow, shoulder)
        fore = sub(wrist, elbow)

        side_sign = 1.0 if side == "left" else -1.0

        lat = side_sign * dot(upper, body_right)
        up = dot(upper, body_up)
        front = dot(upper, body_forward)

        pitch = -0.90 * math.atan2(front, max(1e-6, math.sqrt(lat * lat + up * up)))
        pitch = clamp(pitch, -1.15, 1.15)

        side_angle = math.atan2(max(0.0, lat), max(1e-6, abs(up)))
        roll = side_sign * clamp(0.45 * side_angle, 0.0, 1.10)

        bend = angle_between(upper, fore)

        if side == "right":
            yaw = self.t_right_yaw_up if self.right_mode == "up" else self.t_right_yaw_down
            elbow_q = clamp(self.t_right_elbow + 0.40 * bend, 0.65, 1.95)
        else:
            yaw = self.t_left_yaw_up if self.left_mode == "up" else self.t_left_yaw_down
            elbow_q = clamp(self.t_left_elbow + 0.40 * bend, 0.65, 1.95)

        return [pitch, roll, yaw, elbow_q]

    def compute_target(self):
        lm = self.last_lm
        if lm is None:
            return list(self.tpose)

        for name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]:
            if self.last_vis.get(name, 1.0) < self.vis_th:
                return self.target_q

        body_right, body_up, body_forward = self.body_frame(lm)

        rs, re, rw = lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"]
        ls, le, lw = lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"]

        rr = self.wrist_residual(rs, re, rw, body_up)
        lr = self.wrist_residual(ls, le, lw, body_up)

        self.update_mode("right", rr)
        self.update_mode("left", lr)

        rq = self.solve_arm("right", rs, re, rw, body_right, body_up, body_forward)
        lq = self.solve_arm("left", ls, le, lw, body_right, body_up, body_forward)

        return rq + lq

    def smooth(self):
        out = []
        for cur, tgt in zip(self.current_q, self.target_q):
            d = clamp(tgt - cur, -self.max_step, self.max_step)
            out.append(cur + d)
        self.current_q = out

    def timer_cb(self):
        # В первые секунды всегда T-поза.
        if self.elapsed() < self.tpose_hold_sec:
            self.target_q = list(self.tpose)
            self.right_mode = "up"
            self.left_mode = "up"
        else:
            self.target_q = self.compute_target()

        self.smooth()

        msg = UpperBodyCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "sim_geometric_retarget_tpose_yaw"
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(x) for x in self.current_q]
        msg.confidence = [1.0] * 8
        msg.valid = True

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SimGeometricRetarget()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
