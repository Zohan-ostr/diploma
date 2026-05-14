#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_FILE="$PROJECT_DIR/src/h1_robot_adapter/h1_robot_adapter/h1_geometric_retarget_node.py"

echo "============================================================"
echo " PATCH H1 GEOMETRIC RETARGET: T-POSE + ELBOW YAW HYSTERESIS"
echo "============================================================"
echo "PROJECT_DIR:  $PROJECT_DIR"
echo "TARGET_FILE:  $TARGET_FILE"
echo "============================================================"

if [ ! -f "$TARGET_FILE" ]; then
  echo "ERROR: file not found: $TARGET_FILE"
  exit 1
fi

BACKUP_FILE="${TARGET_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
cp "$TARGET_FILE" "$BACKUP_FILE"
echo "Backup saved: $BACKUP_FILE"

cat > "$TARGET_FILE" <<'PY'
#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


Vec3 = Tuple[float, float, float]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Vec3) -> float:
    return math.sqrt(max(1e-12, v_dot(a, a)))


def v_unit(a: Vec3, fallback: Vec3) -> Vec3:
    n = v_norm(a)
    if n < 1e-6:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_between(a: Vec3, b: Vec3) -> float:
    au = v_unit(a, (1.0, 0.0, 0.0))
    bu = v_unit(b, (1.0, 0.0, 0.0))
    c = clamp(v_dot(au, bu), -1.0, 1.0)
    return math.acos(c)


@dataclass
class BodyFrame:
    right: Vec3
    up: Vec3
    forward: Vec3


class H1GeometricRetargetNode(Node):
    """
    Geometric retarget for Unitree H1 upper body.

    Output joint order:
      right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
      left_shoulder_pitch,  left_shoulder_roll,  left_shoulder_yaw,  left_elbow

    Key behavior:
      1. On start: publish T-pose for tpose_hold_sec.
      2. Command is internally smoothed.
      3. Shoulder yaw mode is chosen by forearm side:
         wrist above shoulder->elbow vector => elbow-up yaw;
         wrist below shoulder->elbow vector => elbow-down yaw.
      4. Hysteresis prevents flipping when arm is almost straight.
    """

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

    REQUIRED = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]

    def __init__(self):
        super().__init__("h1_geometric_retarget_node")

        self.input_topic = self.declare_parameter("input_topic", "/pose/landmarks").value
        self.output_topic = self.declare_parameter("output_topic", "/upper_body/command_geom").value

        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 30.0).value)

        # Start behavior
        self.tpose_hold_sec = float(self.declare_parameter("tpose_hold_sec", 3.0).value)
        self.calibration_sec = float(self.declare_parameter("calibration_sec", 1.5).value)

        # Smoothing per published cycle
        self.max_joint_step_rad = float(self.declare_parameter("max_joint_step_rad", 0.035).value)

        # Hysteresis for elbow yaw switching, in MediaPipe world meters projected to body up-axis
        self.yaw_switch_threshold_m = float(self.declare_parameter("yaw_switch_threshold_m", 0.045).value)

        # T-pose / safe starting pose.
        # Tune these if the mechanical T-pose on your H1 differs.
        self.t_right_pitch = float(self.declare_parameter("t_right_pitch", 0.0).value)
        self.t_right_roll = float(self.declare_parameter("t_right_roll", -0.65).value)
        self.t_right_yaw_up = float(self.declare_parameter("t_right_yaw_up", 1.30).value)
        self.t_right_yaw_down = float(self.declare_parameter("t_right_yaw_down", 0.35).value)
        self.t_right_elbow = float(self.declare_parameter("t_right_elbow", 1.20).value)

        self.t_left_pitch = float(self.declare_parameter("t_left_pitch", 0.0).value)
        self.t_left_roll = float(self.declare_parameter("t_left_roll", 0.65).value)
        self.t_left_yaw_up = float(self.declare_parameter("t_left_yaw_up", -1.30).value)
        self.t_left_yaw_down = float(self.declare_parameter("t_left_yaw_down", -0.35).value)
        self.t_left_elbow = float(self.declare_parameter("t_left_elbow", 1.20).value)

        # Mapping gains/limits
        self.pitch_gain = float(self.declare_parameter("pitch_gain", 0.90).value)
        self.roll_gain = float(self.declare_parameter("roll_gain", 0.45).value)
        self.elbow_gain = float(self.declare_parameter("elbow_gain", 0.40).value)

        self.pitch_limit = float(self.declare_parameter("pitch_limit", 1.15).value)
        self.roll_limit = float(self.declare_parameter("roll_limit", 1.10).value)
        self.elbow_min = float(self.declare_parameter("elbow_min", 0.65).value)
        self.elbow_max = float(self.declare_parameter("elbow_max", 1.95).value)

        self.visibility_threshold = float(self.declare_parameter("visibility_threshold", 0.35).value)

        self.start_time = self.get_clock().now()
        self.last_landmarks: Optional[Dict[str, Vec3]] = None
        self.last_visibility: Dict[str, float] = {}

        # Initial yaw state after calibration: elbow-up.
        self.right_elbow_mode = "up"
        self.left_elbow_mode = "up"

        self.tpose_q = [
            self.t_right_pitch,
            self.t_right_roll,
            self.t_right_yaw_up,
            self.t_right_elbow,
            self.t_left_pitch,
            self.t_left_roll,
            self.t_left_yaw_up,
            self.t_left_elbow,
        ]

        self.current_q = list(self.tpose_q)
        self.target_q = list(self.tpose_q)

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(
            PoseLandmarks3D,
            self.input_topic,
            self.landmarks_cb,
            10,
        )

        period = 1.0 / max(1.0, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info("============================================================")
        self.get_logger().info("H1 geometric retarget started")
        self.get_logger().info(f"input_topic:       {self.input_topic}")
        self.get_logger().info(f"output_topic:      {self.output_topic}")
        self.get_logger().info(f"tpose_hold_sec:    {self.tpose_hold_sec}")
        self.get_logger().info(f"calibration_sec:   {self.calibration_sec}")
        self.get_logger().info(f"yaw threshold, m:  {self.yaw_switch_threshold_m}")
        self.get_logger().info(f"T-pose q:          {[round(x, 3) for x in self.tpose_q]}")
        self.get_logger().info("Initial elbow yaw mode: right=up left=up")
        self.get_logger().info("============================================================")

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def landmarks_cb(self, msg: PoseLandmarks3D):
        if not msg.valid:
            return

        if not (len(msg.names) == len(msg.x) == len(msg.y) == len(msg.z)):
            self.get_logger().warn("Bad PoseLandmarks3D sizes")
            return

        lm: Dict[str, Vec3] = {}
        vis: Dict[str, float] = {}

        for i, name in enumerate(msg.names):
            lm[name] = (float(msg.x[i]), float(msg.y[i]), float(msg.z[i]))
            if i < len(msg.visibility):
                vis[name] = float(msg.visibility[i])
            else:
                vis[name] = 1.0

        for name in self.REQUIRED:
            if name not in lm:
                self.get_logger().warn(f"Missing landmark: {name}")
                return

        self.last_landmarks = lm
        self.last_visibility = vis

    def build_body_frame(self, lm: Dict[str, Vec3]) -> BodyFrame:
        ls = lm["left_shoulder"]
        rs = lm["right_shoulder"]
        lh = lm["left_hip"]
        rh = lm["right_hip"]

        shoulder_mid = v_mul(v_add(ls, rs), 0.5)
        hip_mid = v_mul(v_add(lh, rh), 0.5)

        # Person-left direction. In robot command we handle left/right signs explicitly.
        body_right = v_unit(v_sub(ls, rs), (0.0, 1.0, 0.0))
        body_up = v_unit(v_sub(shoulder_mid, hip_mid), (0.0, 0.0, 1.0))

        # Complete orthonormal-ish frame.
        body_forward = v_unit(v_cross(body_right, body_up), (1.0, 0.0, 0.0))
        body_right = v_unit(v_cross(body_up, body_forward), body_right)

        return BodyFrame(right=body_right, up=body_up, forward=body_forward)

    def wrist_vertical_residual(self, shoulder: Vec3, elbow: Vec3, wrist: Vec3, frame: BodyFrame) -> float:
        upper = v_sub(elbow, shoulder)
        sw = v_sub(wrist, shoulder)
        denom = max(1e-6, v_dot(upper, upper))
        t = clamp(v_dot(sw, upper) / denom, 0.0, 1.0)
        closest = v_add(shoulder, v_mul(upper, t))
        residual_vec = v_sub(wrist, closest)
        return v_dot(residual_vec, frame.up)

    def update_elbow_mode(self, side: str, residual: float):
        th = self.yaw_switch_threshold_m

        if side == "right":
            old = self.right_elbow_mode
            if residual > th:
                self.right_elbow_mode = "up"
            elif residual < -th:
                self.right_elbow_mode = "down"

            if self.right_elbow_mode != old:
                self.get_logger().info(f"right elbow yaw mode: {old} -> {self.right_elbow_mode}, residual={residual:.3f}")

        else:
            old = self.left_elbow_mode
            if residual > th:
                self.left_elbow_mode = "up"
            elif residual < -th:
                self.left_elbow_mode = "down"

            if self.left_elbow_mode != old:
                self.get_logger().info(f"left elbow yaw mode: {old} -> {self.left_elbow_mode}, residual={residual:.3f}")

    def solve_arm(self, side: str, shoulder: Vec3, elbow: Vec3, wrist: Vec3, frame: BodyFrame) -> List[float]:
        upper = v_sub(elbow, shoulder)
        fore = v_sub(wrist, elbow)

        # Side signs:
        # left arm outward is +body_right, right arm outward is -body_right.
        side_sign = 1.0 if side == "left" else -1.0

        lat = side_sign * v_dot(upper, frame.right)
        up = v_dot(upper, frame.up)
        front = v_dot(upper, frame.forward)

        # Pitch: forward/back component.
        pitch = -self.pitch_gain * math.atan2(front, max(1e-6, math.sqrt(lat * lat + up * up)))
        pitch = clamp(pitch, -self.pitch_limit, self.pitch_limit)

        # Roll: arm side lifting.
        # If the arm is near vertical/down, roll -> 0.
        side_angle = math.atan2(max(0.0, lat), max(1e-6, abs(up)))
        roll_mag = clamp(self.roll_gain * side_angle, 0.0, self.roll_limit)
        roll = side_sign * roll_mag

        # Elbow bend from geometric upper/forearm angle.
        bend = angle_between(upper, fore)
        if side == "right":
            elbow_q = self.t_right_elbow + self.elbow_gain * bend
            yaw = self.t_right_yaw_up if self.right_elbow_mode == "up" else self.t_right_yaw_down
        else:
            elbow_q = self.t_left_elbow + self.elbow_gain * bend
            yaw = self.t_left_yaw_up if self.left_elbow_mode == "up" else self.t_left_yaw_down

        elbow_q = clamp(elbow_q, self.elbow_min, self.elbow_max)

        return [pitch, roll, yaw, elbow_q]

    def compute_target_from_landmarks(self) -> Optional[List[float]]:
        lm = self.last_landmarks
        if lm is None:
            return None

        # Visibility gate. If landmarks are uncertain, keep current target.
        for name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]:
            if self.last_visibility.get(name, 1.0) < self.visibility_threshold:
                return None

        frame = self.build_body_frame(lm)

        rs = lm["right_shoulder"]
        re = lm["right_elbow"]
        rw = lm["right_wrist"]

        ls = lm["left_shoulder"]
        le = lm["left_elbow"]
        lw = lm["left_wrist"]

        right_residual = self.wrist_vertical_residual(rs, re, rw, frame)
        left_residual = self.wrist_vertical_residual(ls, le, lw, frame)

        self.update_elbow_mode("right", right_residual)
        self.update_elbow_mode("left", left_residual)

        rq = self.solve_arm("right", rs, re, rw, frame)
        lq = self.solve_arm("left", ls, le, lw, frame)

        return rq + lq

    def smooth_to_target(self, target: List[float]):
        out = []
        for cur, tgt in zip(self.current_q, target):
            delta = tgt - cur
            delta = clamp(delta, -self.max_joint_step_rad, self.max_joint_step_rad)
            out.append(cur + delta)
        self.current_q = out

    def timer_cb(self):
        t = self.elapsed()

        # Startup/calibration phase:
        # publish T-pose and initialize elbow mode as UP.
        if t < self.tpose_hold_sec:
            self.target_q = list(self.tpose_q)
            self.right_elbow_mode = "up"
            self.left_elbow_mode = "up"
        else:
            target = self.compute_target_from_landmarks()
            if target is not None:
                self.target_q = target

        self.smooth_to_target(self.target_q)
        self.publish_command()

    def publish_command(self):
        msg = UpperBodyCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "h1_geometric_retarget_tpose_yaw"
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(x) for x in self.current_q]
        msg.confidence = [1.0] * len(msg.position)
        msg.valid = True
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = H1GeometricRetargetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
PY

chmod +x "$TARGET_FILE"

echo
echo "Patch OK."
echo "Now rebuild h1_robot_adapter inside your usual Docker/container flow."
