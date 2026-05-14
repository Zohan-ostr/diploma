#!/usr/bin/env python3
import math
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

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


class H1GeometricRetargetNode(Node):
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

    # Нейтральная поза примерно из твоего lowstate.
    BASE_Q = np.array([
        -0.10, -0.015,  0.017, 1.34,
        -0.10, -0.015, -0.050, 1.32,
    ], dtype=float)

    # H1 joint limits.
    LOWER = np.array([
        -2.87, -0.34, -1.30, -1.25,
        -2.87, -3.11, -4.45, -1.25,
    ], dtype=float)

    UPPER = np.array([
         2.87,  3.11,  4.45,  2.61,
         2.87,  0.34,  1.30,  2.61,
    ], dtype=float)

    def __init__(self):
        super().__init__("h1_geometric_retarget_node")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")

        self.declare_parameter("calibration_frames", 45)

        self.declare_parameter("min_shoulder_visibility", 0.10)
        self.declare_parameter("min_elbow_visibility", 0.03)
        self.declare_parameter("min_wrist_visibility", 0.01)

        self.declare_parameter("landmark_alpha", 0.25)
        self.declare_parameter("joint_alpha", 0.20)
        self.declare_parameter("max_joint_step", 0.060)

        # Пока глубину MediaPipe не используем для forward-направления.
        self.declare_parameter("map_x_from_z", 0.0)

        # Усиления.
        self.declare_parameter("pitch_gain", 1.25)
        self.declare_parameter("roll_gain", 1.20)
        self.declare_parameter("elbow_gain", 1.15)

        # Нижняя yaw-ветка: когда кисть ниже середины shoulder-elbow.
        # Это НЕ 180 градусов, а подбираемое положение yaw для H1.
        self.declare_parameter("left_yaw_down", 3.0)
        self.declare_parameter("right_yaw_down", -3.0)
        self.declare_parameter("yaw_hysteresis", 0.035)

        # Знаки. Их можно менять параметрами без переписывания кода.
        self.declare_parameter("left_pitch_sign", -1.0)
        self.declare_parameter("right_pitch_sign", -1.0)

        self.declare_parameter("left_roll_sign", 1.0)
        self.declare_parameter("right_roll_sign", -1.0)

        # Важно: локоть уже инвертирован относительно первой версии.
        self.declare_parameter("left_elbow_sign", -1.0)
        self.declare_parameter("right_elbow_sign", -1.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.calibration_frames = int(self.get_parameter("calibration_frames").value)

        self.min_sh_vis = float(self.get_parameter("min_shoulder_visibility").value)
        self.min_el_vis = float(self.get_parameter("min_elbow_visibility").value)
        self.min_wr_vis = float(self.get_parameter("min_wrist_visibility").value)

        self.landmark_alpha = float(self.get_parameter("landmark_alpha").value)
        self.joint_alpha = float(self.get_parameter("joint_alpha").value)
        self.max_joint_step = float(self.get_parameter("max_joint_step").value)

        self.map_x_from_z = float(self.get_parameter("map_x_from_z").value)

        self.pitch_gain = float(self.get_parameter("pitch_gain").value)
        self.roll_gain = float(self.get_parameter("roll_gain").value)
        self.elbow_gain = float(self.get_parameter("elbow_gain").value)

        self.left_yaw_down = float(self.get_parameter("left_yaw_down").value)
        self.right_yaw_down = float(self.get_parameter("right_yaw_down").value)
        self.yaw_hysteresis = float(self.get_parameter("yaw_hysteresis").value)

        self.left_pitch_sign = float(self.get_parameter("left_pitch_sign").value)
        self.right_pitch_sign = float(self.get_parameter("right_pitch_sign").value)
        self.left_roll_sign = float(self.get_parameter("left_roll_sign").value)
        self.right_roll_sign = float(self.get_parameter("right_roll_sign").value)
        self.left_elbow_sign = float(self.get_parameter("left_elbow_sign").value)
        self.right_elbow_sign = float(self.get_parameter("right_elbow_sign").value)

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_landmarks, 10)

        self.filters: Dict[str, ExpFilter] = {}
        self.q_filter = ExpFilter(self.joint_alpha)
        self.q_filter.reset(self.BASE_Q)

        self.last_q = self.BASE_Q.copy()

        self.rest_samples = []
        self.rest = None

        self.left_yaw_branch = 0
        self.right_yaw_branch = 0

        self.msg_count = 0

        self.get_logger().info("===== H1 GEOMETRIC RETARGET CLEAN VERSION =====")
        self.get_logger().info(f"input_topic:  {self.input_topic}")
        self.get_logger().info(f"output_topic: {self.output_topic}")
        self.get_logger().info(f"map_x_from_z: {self.map_x_from_z}")
        self.get_logger().info(f"roll signs L/R: {self.left_roll_sign} / {self.right_roll_sign}")
        self.get_logger().info(f"elbow signs L/R: {self.left_elbow_sign} / {self.right_elbow_sign}")
        self.get_logger().info(f"yaw_down L/R: {self.left_yaw_down} / {self.right_yaw_down}")

    def get_points(self, msg: PoseLandmarks3D):
        raw = {}
        vis = {}
        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                raw[name] = np.array([msg.x[i], msg.y[i], msg.z[i]], dtype=float)
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0
        return raw, vis

    def mp_to_body(self, p_mp: np.ndarray, origin_mp: np.ndarray) -> np.ndarray:
        """
        MediaPipe world → локальная система корпуса:
        x = forward
        y = left
        z = up

        По твоим данным:
        MediaPipe x — лево/право,
        MediaPipe y — вертикаль, но вверх = минус,
        MediaPipe z — глубина, пока отключена.
        """
        p = p_mp - origin_mp
        x_forward = -self.map_x_from_z * p[2]
        y_left = p[0]
        z_up = -p[1]
        return np.array([x_forward, y_left, z_up], dtype=float)

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

    def landmarks_visible_enough(self, vis: Dict[str, float]) -> bool:
        checks = [
            ("left_shoulder", self.min_sh_vis),
            ("right_shoulder", self.min_sh_vis),
            ("left_elbow", self.min_el_vis),
            ("right_elbow", self.min_el_vis),
            ("left_wrist", self.min_wr_vis),
            ("right_wrist", self.min_wr_vis),
        ]
        return all(vis.get(name, 0.0) >= thr for name, thr in checks)

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

    def arm_features(self, pts: Dict[str, np.ndarray], side: str):
        sh = pts[f"{side}_shoulder"]
        el = pts[f"{side}_elbow"]
        wr = pts[f"{side}_wrist"]

        upper = el - sh
        fore = wr - el

        u = unit(upper)
        f = unit(fore)
        if u is None or f is None:
            return None

        # elevation: насколько плечо-локоть ушло от вертикали вниз.
        horizontal = math.sqrt(u[0] * u[0] + u[1] * u[1])
        elevation = math.atan2(horizontal, max(1e-6, -u[2]))

        # roll: боковое отведение. Для правой руки знак дальше задаётся параметром.
        side_sign_for_measure = 1.0 if side == "left" else -1.0
        roll = math.atan2(side_sign_for_measure * u[1], max(1e-6, -u[2]))

        # elbow: угол между плечом-локтем и локтем-кистью.
        # При сгибании у человека этот угол обычно растёт.
        elbow = angle_between(upper, fore)

        mid_z = 0.5 * (sh[2] + el[2])

        # Гистерезис ветки.
        if side == "left":
            current = self.left_yaw_branch
        else:
            current = self.right_yaw_branch

        if current == 0:
            wrist_below_mid = wr[2] < (mid_z - self.yaw_hysteresis)
        else:
            wrist_below_mid = wr[2] < (mid_z + self.yaw_hysteresis)

        return {
            "elevation": elevation,
            "roll": roll,
            "elbow": elbow,
            "wrist_below_mid": wrist_below_mid,
            "upper": upper,
            "fore": fore,
        }

    def calibrate_or_wait(self, pts, vis):
        if self.rest is not None:
            return True

        if not self.landmarks_visible_enough(vis):
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

        lf = self.arm_features(pts, "left")
        rf = self.arm_features(pts, "right")
        if lf is None or rf is None:
            return False

        self.rest_samples.append({"left": lf, "right": rf})

        if len(self.rest_samples) < self.calibration_frames:
            if self.msg_count % 15 == 0:
                self.get_logger().info(f"Calibrating neutral pose... {len(self.rest_samples)}/{self.calibration_frames}")
            return False

        self.rest = {
            "left": {
                "elevation": float(np.mean([s["left"]["elevation"] for s in self.rest_samples])),
                "roll": float(np.mean([s["left"]["roll"] for s in self.rest_samples])),
                "elbow": float(np.mean([s["left"]["elbow"] for s in self.rest_samples])),
            },
            "right": {
                "elevation": float(np.mean([s["right"]["elevation"] for s in self.rest_samples])),
                "roll": float(np.mean([s["right"]["roll"] for s in self.rest_samples])),
                "elbow": float(np.mean([s["right"]["elbow"] for s in self.rest_samples])),
            },
        }

        self.get_logger().info("Calibration complete")
        self.get_logger().info(f"rest left:  {self.rest['left']}")
        self.get_logger().info(f"rest right: {self.rest['right']}")
        return True

    def compute_q(self, pts):
        lf = self.arm_features(pts, "left")
        rf = self.arm_features(pts, "right")
        if lf is None or rf is None or self.rest is None:
            return None

        q = self.BASE_Q.copy()

        # Левая рука.
        delev_l = lf["elevation"] - self.rest["left"]["elevation"]
        droll_l = lf["roll"] - self.rest["left"]["roll"]
        delbow_l = lf["elbow"] - self.rest["left"]["elbow"]

        self.left_yaw_branch = 1 if lf["wrist_below_mid"] else 0

        q[0] += self.left_pitch_sign * self.pitch_gain * delev_l
        q[1] += self.left_roll_sign * self.roll_gain * droll_l
        if self.left_yaw_branch:
            q[2] = self.left_yaw_down
        q[3] += self.left_elbow_sign * self.elbow_gain * delbow_l

        # Правая рука.
        delev_r = rf["elevation"] - self.rest["right"]["elevation"]
        droll_r = rf["roll"] - self.rest["right"]["roll"]
        delbow_r = rf["elbow"] - self.rest["right"]["elbow"]

        self.right_yaw_branch = 1 if rf["wrist_below_mid"] else 0

        q[4] += self.right_pitch_sign * self.pitch_gain * delev_r
        q[5] += self.right_roll_sign * self.roll_gain * droll_r
        if self.right_yaw_branch:
            q[6] = self.right_yaw_down
        q[7] += self.right_elbow_sign * self.elbow_gain * delbow_r

        q = np.clip(q, self.LOWER, self.UPPER)

        # Ограничитель скорости.
        step = np.clip(q - self.last_q, -self.max_joint_step, self.max_joint_step)
        q_limited = self.last_q + step

        q_filtered = self.q_filter.update(q_limited)
        q_filtered = np.clip(q_filtered, self.LOWER, self.UPPER)

        self.last_q = q_filtered.copy()

        return q_filtered

    def publish_q(self, stamp, q, valid=True):
        msg = UpperBodyCommand()
        msg.header.stamp = stamp
        msg.header.frame_id = "h1_geometric_retarget"
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in q]
        msg.confidence = [1.0 if valid else 0.0 for _ in q]
        msg.valid = bool(valid)
        self.pub.publish(msg)

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

        if not self.calibrate_or_wait(pts, vis):
            return

        q = self.compute_q(pts)
        if q is None:
            return

        self.publish_q(msg.header.stamp, q, True)

        if self.msg_count % 30 == 0:
            self.get_logger().info(
                "q="
                + np.array2string(q, precision=3, suppress_small=True)
                + f" branches L/R={self.left_yaw_branch}/{self.right_yaw_branch}"
            )


def main():
    rclpy.init()
    node = H1GeometricRetargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
