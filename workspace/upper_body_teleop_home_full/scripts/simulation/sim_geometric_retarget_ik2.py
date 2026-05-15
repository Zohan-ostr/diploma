#!/usr/bin/env python3
import math
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unit(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    au = unit(a)
    bu = unit(b)
    if au is None or bu is None:
        return 0.0
    return math.acos(clamp(float(np.dot(au, bu)), -1.0, 1.0))


def project_perp(v: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    axis_u = unit(axis)
    if axis_u is None:
        return None
    p = v - float(np.dot(v, axis_u)) * axis_u
    return unit(p)


class ExpFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value = None

    def reset(self, value):
        self.value = np.array(value, dtype=float)

    def update(self, value):
        value = np.array(value, dtype=float)
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value
        return self.value.copy()


class H1RetargetIK2(Node):
    JOINT_NAMES = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    BASE_Q = np.array([
        -0.10, -0.015,  0.017, 1.34,
        -0.10, -0.015, -0.050, 1.32,
    ], dtype=float)

    TPOSE_Q = np.array([
        -0.10,  1.57,  0.017, 1.30,
        -0.10, -1.57, -0.050, 1.30,
    ], dtype=float)

    LOWER = np.array([
        -2.87, -0.34, -1.30, -1.25,
        -2.87, -3.11, -4.45, -1.25,
    ], dtype=float)

    UPPER = np.array([
         2.87,  3.11,  4.45,  2.61,
         2.87,  0.34,  1.30,  2.61,
    ], dtype=float)

    def __init__(self):
        super().__init__("sim_geometric_retarget_ik2")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("calibration_topic", "/upper_body/start_calibration")

        self.declare_parameter("calibration_frames", 45)
        self.declare_parameter("landmark_alpha", 0.25)
        self.declare_parameter("joint_alpha", 0.28)

        self.declare_parameter("max_joint_step", 0.055)
        self.declare_parameter("yaw_max_step", 0.220)
        self.declare_parameter("elbow_max_step", 0.110)

        self.declare_parameter("map_x_from_z", 1.0)

        self.declare_parameter("pitch_gain", 0.45)
        self.declare_parameter("elbow_gain", 1.35)

        # По официальным лимитам H1:
        # left_yaw:  -1.30 .. +4.45
        # right_yaw: -4.45 .. +1.30
        # Здесь используем только безопасную зону, где локоть сгибается вперёд.
        self.declare_parameter("left_yaw_min", -1.25)
        self.declare_parameter("left_yaw_max", 4.20)
        self.declare_parameter("right_yaw_min", -4.20)
        self.declare_parameter("right_yaw_max", 1.25)

        self.declare_parameter("yaw_grid_count", 81)
        self.declare_parameter("yaw_continuity_weight", 0.035)
        self.declare_parameter("yaw_center_weight", 0.004)
        self.declare_parameter("wrist_forward_weight", 0.00)
        self.declare_parameter("backward_block_threshold", -0.02)

        self.declare_parameter("min_shoulder_visibility", 0.10)
        self.declare_parameter("min_elbow_visibility", 0.03)
        self.declare_parameter("min_wrist_visibility", 0.01)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.calibration_topic = self.get_parameter("calibration_topic").value

        self.calibration_frames = int(self.get_parameter("calibration_frames").value)
        self.landmark_alpha = float(self.get_parameter("landmark_alpha").value)
        self.joint_alpha = float(self.get_parameter("joint_alpha").value)

        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        self.yaw_max_step = float(self.get_parameter("yaw_max_step").value)
        self.elbow_max_step = float(self.get_parameter("elbow_max_step").value)

        self.map_x_from_z = float(self.get_parameter("map_x_from_z").value)
        self.pitch_gain = float(self.get_parameter("pitch_gain").value)
        self.elbow_gain = float(self.get_parameter("elbow_gain").value)

        self.left_yaw_min = float(self.get_parameter("left_yaw_min").value)
        self.left_yaw_max = float(self.get_parameter("left_yaw_max").value)
        self.right_yaw_min = float(self.get_parameter("right_yaw_min").value)
        self.right_yaw_max = float(self.get_parameter("right_yaw_max").value)

        self.yaw_grid_count = int(self.get_parameter("yaw_grid_count").value)
        self.yaw_continuity_weight = float(self.get_parameter("yaw_continuity_weight").value)
        self.yaw_center_weight = float(self.get_parameter("yaw_center_weight").value)
        self.wrist_forward_weight = float(self.get_parameter("wrist_forward_weight").value)
        self.backward_block_threshold = float(self.get_parameter("backward_block_threshold").value)

        self.min_sh_vis = float(self.get_parameter("min_shoulder_visibility").value)
        self.min_el_vis = float(self.get_parameter("min_elbow_visibility").value)
        self.min_wr_vis = float(self.get_parameter("min_wrist_visibility").value)

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_landmarks, 10)
        self.calib_sub = self.create_subscription(Bool, self.calibration_topic, self.on_start_calibration, 10)

        self.filters: Dict[str, ExpFilter] = {}

        self.q_filter = ExpFilter(self.joint_alpha)
        self.q_filter.reset(self.BASE_Q)
        self.last_q = self.BASE_Q.copy()

        self.rest_samples = []
        self.rest_ready = False
        self.calibration_requested = False
        self.calibrating = False

        self.prev_left_yaw = float(self.BASE_Q[2])
        self.prev_right_yaw = float(self.BASE_Q[6])

        self.msg_count = 0

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("H1 RETARGET IK2: elbow IK + wrist IK")
        self.get_logger().info(f"input_topic:   {self.input_topic}")
        self.get_logger().info(f"output_topic:  {self.output_topic}")
        self.get_logger().info(f"calib_topic:   {self.calibration_topic}")
        self.get_logger().info(f"TPOSE_Q:       {np.array2string(self.TPOSE_Q, precision=3)}")
        self.get_logger().info("Algorithm:")
        self.get_logger().info("  1) shoulder pitch/roll from shoulder->elbow")
        self.get_logger().info("  2) shoulder yaw + elbow from elbow->wrist by grid IK")
        self.get_logger().info("============================================================")

    def on_start_calibration(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().info("Manual calibration command received from terminal 5.")
        self.get_logger().info("Holding T-pose while collecting neutral samples...")

        self.calibration_requested = True
        self.calibrating = True
        self.rest_ready = False
        self.rest_samples = []

        self.prev_left_yaw = float(self.BASE_Q[2])
        self.prev_right_yaw = float(self.BASE_Q[6])

    def get_points(self, msg: PoseLandmarks3D):
        raw = {}
        vis = {}
        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                raw[name] = np.array([msg.x[i], msg.y[i], msg.z[i]], dtype=float)
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0
        return raw, vis

    def mp_to_body(self, p_mp: np.ndarray, origin_mp: np.ndarray) -> np.ndarray:
        p = p_mp - origin_mp

        # body:
        # x = forward/back from MediaPipe z
        # y = left/right from MediaPipe x
        # z = up/down from MediaPipe y
        x_forward = -self.map_x_from_z * p[2]
        y_left = p[0]
        z_up = -p[1]

        return np.array([x_forward, y_left, z_up], dtype=float)

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

    def make_body_points(self, raw):
        shoulder_mid = 0.5 * (raw["left_shoulder"] + raw["right_shoulder"])
        origin = shoulder_mid

        pts = {}
        for name in [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
        ]:
            if name in raw:
                pts[name] = self.filtered_point(name, self.mp_to_body(raw[name], origin))
        return pts

    def visible_enough(self, vis):
        checks = [
            ("left_shoulder", self.min_sh_vis),
            ("right_shoulder", self.min_sh_vis),
            ("left_elbow", self.min_el_vis),
            ("right_elbow", self.min_el_vis),
            ("left_wrist", self.min_wr_vis),
            ("right_wrist", self.min_wr_vis),
        ]
        return all(vis.get(name, 0.0) >= thr for name, thr in checks)

    def calibrate_or_wait(self, pts, vis):
        if self.rest_ready:
            return True

        if not self.visible_enough(vis):
            if self.msg_count % 30 == 0:
                self.get_logger().warn(
                    "Waiting for visible landmarks: "
                    f"LS={vis.get('left_shoulder',0):.2f}, "
                    f"LE={vis.get('left_elbow',0):.2f}, "
                    f"LW={vis.get('left_wrist',0):.2f}, "
                    f"RS={vis.get('right_shoulder',0):.2f}, "
                    f"RE={vis.get('right_elbow',0):.2f}, "
                    f"RW={vis.get('right_wrist',0):.2f}"
                )
            return False

        self.rest_samples.append(1)

        if len(self.rest_samples) < self.calibration_frames:
            if self.msg_count % 15 == 0:
                self.get_logger().info(f"Calibrating neutral pose... {len(self.rest_samples)}/{self.calibration_frames}")
            return False

        self.calibrating = False
        self.calibration_requested = False
        self.rest_ready = True

        self.get_logger().info("Calibration complete. Teleoperation enabled.")
        return True

    def arm_vectors(self, pts, side: str):
        sh = pts[f"{side}_shoulder"]
        el = pts[f"{side}_elbow"]
        wr = pts[f"{side}_wrist"]

        upper = el - sh
        fore = wr - el

        u = unit(upper)
        f = unit(fore)
        if u is None or f is None:
            return None

        return sh, el, wr, upper, fore, u, f

    def solve_shoulder_pitch_roll(self, side: str, u: np.ndarray):
        side_sign = 1.0 if side == "left" else -1.0

        forward = float(u[0])
        lateral = float(side_sign * u[1])
        vertical = float(u[2])

        lateral_pos = max(0.0, lateral)

        # Down -> 0, T-pose -> pi/2, arm up -> pi.
        roll_mag = math.atan2(lateral_pos, -vertical)
        roll_mag = clamp(roll_mag, 0.0, 3.02)

        shoulder_roll = side_sign * roll_mag

        pitch_raw = math.atan2(
            forward,
            max(0.15, math.sqrt(lateral_pos * lateral_pos + vertical * vertical)),
        )

        if side == "left":
            shoulder_pitch = self.BASE_Q[0] + self.pitch_gain * pitch_raw
            shoulder_pitch = clamp(shoulder_pitch, self.LOWER[0], self.UPPER[0])
        else:
            shoulder_pitch = self.BASE_Q[4] + self.pitch_gain * pitch_raw
            shoulder_pitch = clamp(shoulder_pitch, self.LOWER[4], self.UPPER[4])

        return shoulder_pitch, shoulder_roll

    def model_bend_dir(self, side: str, u: np.ndarray, yaw_q: float):
        """
        shoulder_yaw chooses the forearm bending plane around upper-arm axis u.

        For raised arms we must not use body_forward as the main reference,
        because that makes the elbow bend in front of the robot. Instead,
        raised-arm yaw is referenced from the outward side direction.
        """
        body_forward = np.array([1.0, 0.0, 0.0], dtype=float)
        body_up = np.array([0.0, 0.0, 1.0], dtype=float)

        side_sign = 1.0 if side == "left" else -1.0
        body_outward = np.array([0.0, side_sign, 0.0], dtype=float)

        up_perp = project_perp(body_up, u)
        fwd_perp = project_perp(body_forward, u)
        outward_perp = project_perp(body_outward, u)

        # Critical fix:
        # If upper arm points upward, use outward side as zero-yaw reference.
        # This prevents "elbow bends in front" when the arm is raised.
        if float(u[2]) > 0.45 and outward_perp is not None:
            ref0 = outward_perp
        elif up_perp is not None:
            ref0 = up_perp
        elif outward_perp is not None:
            ref0 = outward_perp
        elif fwd_perp is not None:
            ref0 = fwd_perp
        else:
            return None

        ref1 = np.cross(u, ref0)
        ref1 = unit(ref1)
        if ref1 is None:
            return ref0

        # Keep orientation deterministic.
        if float(np.dot(ref1, body_forward)) < 0.0:
            ref1 = -ref1

        yaw_center = self.BASE_Q[2] if side == "left" else self.BASE_Q[6]

        # Yaw direction fix.
        # Invert rotation around upper-arm axis.
        if side == "left":
            phi = yaw_q - yaw_center
        else:
            phi = -(yaw_q - yaw_center)

        n = math.cos(phi) * ref0 + math.sin(phi) * ref1
        return unit(n)

    def solve_yaw_elbow_grid(self, side: str, u: np.ndarray, fore: np.ndarray):
        f = unit(fore)
        if f is None:
            return None

        # For yaw selection, depth is unreliable.
        # When the upper arm is raised, completely remove forward/back component
        # so the elbow plane is selected sideways/upward, not in front.
        f_for_yaw = f.copy()
        if float(u[2]) > 0.45:
            f_for_yaw[0] = 0.0
        else:
            f_for_yaw[0] *= self.wrist_forward_weight

        target_dir = project_perp(f_for_yaw, u)

        if target_dir is None:
            # Arm nearly straight. Keep previous yaw, elbow close to base.
            if side == "left":
                return self.prev_left_yaw, self.BASE_Q[3]
            return self.prev_right_yaw, self.BASE_Q[7]

        bend_angle = angle_between(u, f)

        if side == "left":
            yaw_min = self.left_yaw_min
            yaw_max = self.left_yaw_max
            prev_yaw = self.prev_left_yaw
            base_yaw = self.BASE_Q[2]
            base_elbow = self.BASE_Q[3]
            elbow_idx = 3
        else:
            yaw_min = self.right_yaw_min
            yaw_max = self.right_yaw_max
            prev_yaw = self.prev_right_yaw
            base_yaw = self.BASE_Q[6]
            base_elbow = self.BASE_Q[7]
            elbow_idx = 7

        best_q = prev_yaw
        best_cost = 1e9

        yaw_values = np.linspace(yaw_min, yaw_max, max(9, self.yaw_grid_count))

        body_forward = np.array([1.0, 0.0, 0.0], dtype=float)

        for yaw_q in yaw_values:
            model_dir = self.model_bend_dir(side, u, float(yaw_q))
            if model_dir is None:
                continue

            direction_error = 1.0 - clamp(float(np.dot(model_dir, target_dir)), -1.0, 1.0)

            # Hard block through the back side.
            # Also, when the arm is raised, strongly discourage bending in front.
            # Raised-arm elbow should choose side/up plane, not front/back plane.
            forward_score = float(np.dot(model_dir, body_forward))

            backward_penalty = 0.0
            if forward_score < self.backward_block_threshold:
                backward_penalty = 100.0 * abs(forward_score - self.backward_block_threshold)

            front_penalty = 0.0
            if float(u[2]) > 0.45:
                front_penalty = 35.0 * abs(forward_score)

            continuity = self.yaw_continuity_weight * (float(yaw_q) - prev_yaw) ** 2
            center = self.yaw_center_weight * (float(yaw_q) - base_yaw) ** 2

            cost = direction_error + backward_penalty + front_penalty + continuity + center

            if cost < best_cost:
                best_cost = cost
                best_q = float(yaw_q)

        # Elbow bend is independent from yaw branch.
        # Current working direction: BASE - gain*bend.
        elbow_q = base_elbow - self.elbow_gain * bend_angle
        elbow_q = clamp(elbow_q, self.LOWER[elbow_idx], self.UPPER[elbow_idx])

        return best_q, elbow_q

    def compute_q(self, pts):
        lv = self.arm_vectors(pts, "left")
        rv = self.arm_vectors(pts, "right")

        if lv is None or rv is None:
            return None

        _, _, _, lupper, lfore, lu, _ = lv
        _, _, _, rupper, rfore, ru, _ = rv

        q = self.BASE_Q.copy()

        l_pitch, l_roll = self.solve_shoulder_pitch_roll("left", lu)
        r_pitch, r_roll = self.solve_shoulder_pitch_roll("right", ru)

        l_yaw_elbow = self.solve_yaw_elbow_grid("left", lu, lfore)
        r_yaw_elbow = self.solve_yaw_elbow_grid("right", ru, rfore)

        if l_yaw_elbow is None or r_yaw_elbow is None:
            return None

        l_yaw, l_elbow = l_yaw_elbow
        r_yaw, r_elbow = r_yaw_elbow

        q[0] = l_pitch
        q[1] = l_roll
        q[2] = l_yaw
        q[3] = l_elbow

        q[4] = r_pitch
        q[5] = r_roll
        q[6] = r_yaw
        q[7] = r_elbow

        q = np.clip(q, self.LOWER, self.UPPER)

        delta = q - self.last_q
        step_limits = np.array([
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
        ], dtype=float)

        step = np.clip(delta, -step_limits, step_limits)
        q_limited = self.last_q + step

        q_filtered = self.q_filter.update(q_limited)
        q_filtered = np.clip(q_filtered, self.LOWER, self.UPPER)

        self.last_q = q_filtered.copy()

        self.prev_left_yaw = float(q_filtered[2])
        self.prev_right_yaw = float(q_filtered[6])

        return q_filtered

    def publish_q(self, stamp, q, valid=True, frame_id="h1_ik2"):
        msg = UpperBodyCommand()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in q]
        msg.confidence = [1.0 if valid else 0.0 for _ in q]
        msg.valid = bool(valid)
        self.pub.publish(msg)

    def smooth_towards_and_publish(self, target_q: np.ndarray, frame_id: str):
        delta = target_q - self.last_q
        step_limits = np.array([
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
        ], dtype=float)

        step = np.clip(delta, -step_limits, step_limits)
        q = self.last_q + step

        q = self.q_filter.update(q)
        q = np.clip(q, self.LOWER, self.UPPER)

        self.last_q = q.copy()

        self.publish_q(self.get_clock().now().to_msg(), q, True, frame_id)

    def on_timer(self):
        if not self.rest_ready:
            self.smooth_towards_and_publish(self.TPOSE_Q, "tpose_wait_calibration")
            return

    def on_landmarks(self, msg: PoseLandmarks3D):
        self.msg_count += 1

        if not msg.valid:
            return

        raw, vis = self.get_points(msg)

        required = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]

        if not all(name in raw for name in required):
            return

        pts = self.make_body_points(raw)

        if not self.rest_ready:
            if not self.calibration_requested:
                return

            if not self.calibrate_or_wait(pts, vis):
                return

        q = self.compute_q(pts)
        if q is None:
            return

        self.publish_q(msg.header.stamp, q, True)

        if self.msg_count % 30 == 0:
            self.get_logger().info("q=" + np.array2string(q, precision=3, suppress_small=True))


def main():
    rclpy.init()
    node = H1RetargetIK2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
