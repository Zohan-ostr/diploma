import math
from typing import Dict, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


# Управляем только руками.
# Для текущей простой URDF-модели:
# - shoulder_roll_joint отвечает за подъём/опускание руки;
# - elbow_joint отвечает за сгибание локтя;
# - shoulder_pitch_joint и shoulder_yaw_joint пока держим в нуле.
UPPER_COMMAND_JOINTS = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',

    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
]

FULL_JOINTS = [
    'torso_joint',
    'neck_joint',
    'head_joint',

    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',

    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',

    'left_hip_joint',
    'left_knee_joint',
    'left_ankle_joint',

    'right_hip_joint',
    'right_knee_joint',
    'right_ankle_joint',
]

LIMITS = {
    'left_shoulder_pitch_joint': (0.0, 0.0),
    'left_shoulder_roll_joint': (-0.2, 1.6),
    'left_shoulder_yaw_joint': (0.0, 0.0),
    'left_elbow_joint': (0.05, 2.2),

    'right_shoulder_pitch_joint': (0.0, 0.0),
    'right_shoulder_roll_joint': (-0.2, 1.6),
    'right_shoulder_yaw_joint': (0.0, 0.0),
    'right_elbow_joint': (0.05, 2.2),
}

FULL_ZERO = {
    'torso_joint': 0.0,
    'neck_joint': 0.0,
    'head_joint': 0.0,

    'left_shoulder_pitch_joint': 0.0,
    'left_shoulder_roll_joint': 0.0,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 0.20,

    'right_shoulder_pitch_joint': 0.0,
    'right_shoulder_roll_joint': 0.0,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 0.20,

    'left_hip_joint': 0.0,
    'left_knee_joint': 0.0,
    'left_ankle_joint': 0.0,

    'right_hip_joint': 0.0,
    'right_knee_joint': 0.0,
    'right_ankle_joint': 0.0,
}

Point = Tuple[float, float, float]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def norm(v: Point) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def angle_between(a: Point, b: Point) -> float:
    na = norm(a)
    nb = norm(b)

    if na < 1e-6 or nb < 1e-6:
        return 0.0

    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    c = dot / (na * nb)
    c = clamp(c, -1.0, 1.0)

    return math.acos(c)


def elbow_flex_2d(shoulder: Point, elbow: Point, wrist: Point) -> float:
    upper = (
        shoulder[0] - elbow[0],
        shoulder[1] - elbow[1],
        0.0,
    )

    forearm = (
        wrist[0] - elbow[0],
        wrist[1] - elbow[1],
        0.0,
    )

    joint_angle = angle_between(upper, forearm)

    # Прямая рука: angle ~= pi, flex ~= 0.
    # Согнутая рука: angle меньше, flex больше.
    flex = math.pi - joint_angle

    return clamp(flex, 0.0, 2.2)


class RetargetNode(Node):
    def __init__(self):
        super().__init__('retarget_node')

        self.alpha = float(self.declare_parameter('smoothing_alpha', 0.20).value)

        self.min_visibility = float(
            self.declare_parameter('min_visibility', 0.15).value
        )

        # Подъём руки по признаку elbow_raise.
        # Для 2D MediaPipe:
        # elbow_raise = shoulder_y - elbow_y.
        # Если локоть выше плеча, значение растёт.
        self.raise_scale = float(self.declare_parameter('raise_scale', 3.0).value)

        # Сгиб локтя.
        self.elbow_scale = float(self.declare_parameter('elbow_scale', 1.0).value)

        self.human_zero: Optional[Dict[str, float]] = None
        self.last_raw: Optional[Dict[str, float]] = None

        self.prev = dict(FULL_ZERO)
        self.is_calibrated = False
        self.frame_count = 0

        self.create_subscription(PoseLandmarks3D, '/pose/landmarks', self.on_pose, 10)
        self.create_subscription(String, '/teleop/control', self.on_control, 10)

        self.cmd_pub = self.create_publisher(UpperBodyCommand, '/upper_body/command', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.debug_pub = self.create_publisher(JointState, '/retarget/joint_states_debug', 10)
        self.status_pub = self.create_publisher(String, '/teleop/status', 10)

        self.timer = self.create_timer(0.05, self.on_timer)

        self.get_logger().info('Retarget node started.')
        self.get_logger().info('Home mode: shoulder_roll = arm raise/lower, elbow = flex.')
        self.get_logger().info('shoulder_pitch and shoulder_yaw are disabled for now.')
        self.get_logger().info('Full /joint_states is published for RViz.')

    def on_timer(self):
        status = String()
        status.data = 'ACTIVE' if self.is_calibrated else 'WAITING_FOR_CALIBRATION'
        self.status_pub.publish(status)

        if not self.is_calibrated:
            self.publish_command_and_joint_states(FULL_ZERO, valid=False)

    def on_control(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd in ('calibrate', 'c'):
            if self.last_raw is None:
                self.get_logger().warn('Cannot calibrate: no valid pose received yet.')
                return

            self.human_zero = dict(self.last_raw)
            self.prev = dict(FULL_ZERO)
            self.is_calibrated = True

            self.publish_command_and_joint_states(FULL_ZERO, valid=True)

            self.get_logger().info('Calibration saved.')
            self.get_logger().info(f'human_zero = {self.human_zero}')

        elif cmd in ('reset', 'r'):
            self.human_zero = None
            self.last_raw = None
            self.prev = dict(FULL_ZERO)
            self.is_calibrated = False

            self.publish_command_and_joint_states(FULL_ZERO, valid=False)

            self.get_logger().info('Calibration reset. Press C to calibrate again.')

    def on_pose(self, msg: PoseLandmarks3D):
        raw = self.compute_raw_features(msg)

        if raw is None:
            return

        self.last_raw = raw

        if not self.is_calibrated or self.human_zero is None:
            self.publish_command_and_joint_states(FULL_ZERO, valid=False)
            return

        target = dict(FULL_ZERO)

        # Подъём/опускание рук.
        # После исправления осей:
        # left/right shoulder_roll положительным значением поднимают руку вверх.
        left_raise_delta = raw['left_elbow_raise'] - self.human_zero['left_elbow_raise']
        right_raise_delta = raw['right_elbow_raise'] - self.human_zero['right_elbow_raise']

        target['left_shoulder_roll_joint'] = clamp(
            FULL_ZERO['left_shoulder_roll_joint'] + self.raise_scale * left_raise_delta,
            LIMITS['left_shoulder_roll_joint'][0],
            LIMITS['left_shoulder_roll_joint'][1],
        )

        target['right_shoulder_roll_joint'] = clamp(
            FULL_ZERO['right_shoulder_roll_joint'] + self.raise_scale * right_raise_delta,
            LIMITS['right_shoulder_roll_joint'][0],
            LIMITS['right_shoulder_roll_joint'][1],
        )

        # Pitch/yaw временно отключены.
        target['left_shoulder_pitch_joint'] = 0.0
        target['right_shoulder_pitch_joint'] = 0.0
        target['left_shoulder_yaw_joint'] = 0.0
        target['right_shoulder_yaw_joint'] = 0.0

        # Локти считаем абсолютно по текущему углу сгиба.
        target['left_elbow_joint'] = clamp(
            0.05 + self.elbow_scale * raw['left_elbow_flex'],
            LIMITS['left_elbow_joint'][0],
            LIMITS['left_elbow_joint'][1],
        )

        target['right_elbow_joint'] = clamp(
            0.05 + self.elbow_scale * raw['right_elbow_flex'],
            LIMITS['right_elbow_joint'][0],
            LIMITS['right_elbow_joint'][1],
        )

        # Сглаживание.
        smoothed = dict(FULL_ZERO)

        for joint in FULL_JOINTS:
            if joint in UPPER_COMMAND_JOINTS:
                smoothed[joint] = (
                    self.prev[joint] * (1.0 - self.alpha)
                    + target[joint] * self.alpha
                )
            else:
                smoothed[joint] = FULL_ZERO[joint]

            self.prev[joint] = smoothed[joint]

        self.frame_count += 1

        if self.frame_count % 20 == 0:
            self.get_logger().info(
                'raw: '
                f"L_raise={raw['left_elbow_raise']:.3f}, "
                f"R_raise={raw['right_elbow_raise']:.3f}, "
                f"L_elbow={raw['left_elbow_flex']:.3f}, "
                f"R_elbow={raw['right_elbow_flex']:.3f}; "
                'cmd: '
                f"L_roll={smoothed['left_shoulder_roll_joint']:.3f}, "
                f"R_roll={smoothed['right_shoulder_roll_joint']:.3f}, "
                f"L_elbow={smoothed['left_elbow_joint']:.3f}, "
                f"R_elbow={smoothed['right_elbow_joint']:.3f}"
            )

        self.publish_command_and_joint_states(smoothed, valid=True)

    def compute_raw_features(self, msg: PoseLandmarks3D) -> Optional[Dict[str, float]]:
        if not msg.valid or len(msg.names) == 0:
            return None

        required = [
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
        ]

        pts: Dict[str, Point] = {}
        vis: Dict[str, float] = {}

        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                pts[name] = (
                    float(msg.x[i]),
                    float(msg.y[i]),
                    float(msg.z[i]),
                )

            if i < len(msg.visibility):
                vis[name] = float(msg.visibility[i])
            else:
                vis[name] = 1.0

        if any(k not in pts for k in required):
            self.get_logger().warn(
                f'Missing required landmarks. Got: {list(pts.keys())}',
                throttle_duration_sec=2.0
            )
            return None

        must_be_visible = [
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
        ]

        if any(vis.get(k, 0.0) < self.min_visibility for k in must_be_visible):
            self.get_logger().warn(
                'Low shoulder/elbow visibility: '
                f"LS={vis.get('left_shoulder', 0.0):.2f}, "
                f"RS={vis.get('right_shoulder', 0.0):.2f}, "
                f"LE={vis.get('left_elbow', 0.0):.2f}, "
                f"RE={vis.get('right_elbow', 0.0):.2f}",
                throttle_duration_sec=2.0
            )
            return None

        lsh = pts['left_shoulder']
        rsh = pts['right_shoulder']
        lel = pts['left_elbow']
        rel = pts['right_elbow']
        lwr = pts['left_wrist']
        rwr = pts['right_wrist']

        # y вниз, поэтому shoulder_y - elbow_y растёт,
        # когда локоть поднимается выше плеча.
        left_elbow_raise = lsh[1] - lel[1]
        right_elbow_raise = rsh[1] - rel[1]

        left_elbow_flex = elbow_flex_2d(lsh, lel, lwr)
        right_elbow_flex = elbow_flex_2d(rsh, rel, rwr)

        return {
            'left_elbow_raise': left_elbow_raise,
            'right_elbow_raise': right_elbow_raise,
            'left_elbow_flex': left_elbow_flex,
            'right_elbow_flex': right_elbow_flex,
        }

    def publish_command_and_joint_states(self, joint_values: Dict[str, float], valid: bool):
        cmd = UpperBodyCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'upper_body'
        cmd.joint_names = list(UPPER_COMMAND_JOINTS)
        cmd.position = [joint_values[j] for j in UPPER_COMMAND_JOINTS]
        cmd.confidence = [1.0 if valid else 0.0 for _ in UPPER_COMMAND_JOINTS]
        cmd.valid = valid
        self.cmd_pub.publish(cmd)

        self.publish_joint_states(joint_values)

    def publish_joint_states(self, joint_values: Dict[str, float]):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(FULL_JOINTS)
        js.position = [joint_values.get(j, 0.0) for j in FULL_JOINTS]

        self.js_pub.publish(js)
        self.debug_pub.publish(js)


def main():
    rclpy.init()
    node = RetargetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
