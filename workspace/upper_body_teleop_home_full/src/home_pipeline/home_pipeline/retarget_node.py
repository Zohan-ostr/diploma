import csv
import math
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


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

# Абсолютные пределы для RViz/Home-модели.
# На реальном роботе эти значения нужно будет заменить на реальные лимиты Unitree.
LIMITS = {
    'left_shoulder_pitch_joint': (-0.8, 0.8),
    'right_shoulder_pitch_joint': (-0.8, 0.8),

    'left_shoulder_roll_joint': (-1.7, 1.7),
    'right_shoulder_roll_joint': (-1.7, 1.7),

    'left_shoulder_yaw_joint': (-3.15, 3.15),
    'right_shoulder_yaw_joint': (-3.15, 3.15),

    'left_elbow_joint': (0.05, 2.2),
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Point) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Point) -> Point:
    n = norm(a)
    if n < 1e-8:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def angle_between(a: Point, b: Point) -> float:
    na = norm(a)
    nb = norm(b)

    if na < 1e-8 or nb < 1e-8:
        return 0.0

    c = clamp(dot(a, b) / (na * nb), -1.0, 1.0)
    return math.acos(c)


def local_components(v: Point, right: Point, up: Point, front: Point) -> Point:
    return (
        dot(v, right),
        dot(v, up),
        dot(v, front),
    )


class AdaptiveSafetyLayer:
    """
    Safety layer без принудительного reset.

    Теперь safety работает как projection/clamping:
    если команда уводит руку в опасную область, она не выполняется полностью,
    а заменяется ближайшим крайним допустимым положением.

    Что ограничиваем:
    1. shoulder_roll снизу — чтобы руки не входили в корпус снизу;
    2. shoulder_roll сверху — чтобы руки не пересекались над корпусом/головой;
    3. elbow_joint адаптивно от shoulder_roll:
       - рука низко  -> локоть почти прямой;
       - рука в рабочей зоне -> локоть можно сгибать сильнее;
       - рука слишком высоко -> локоть снова ограничивается;
    4. yaw НЕ зануляем принудительно. Он продолжает работать.
    """

    def __init__(self):
        # Hard reset полностью отключён.
        # Даже при опасной команде мы просто проецируем её в безопасную область.
        self.enable_hard_reset = False

        # Крайние допустимые положения плеча по roll.
        # Если руки всё ещё пересекаются сверху/снизу — эти значения нужно ужесточить.
        self.roll_min = -1.15
        self.roll_max = 1.28

        # В зоне слишком низко/слишком высоко дополнительно ужимаем локоть.
        self.elbow_min = 0.05

    def adaptive_elbow_max(self, shoulder_roll: float) -> float:
        """
        Адаптивный максимум локтя от положения плеча.

        Идея:
        - низко: локоть почти прямой, чтобы предплечье не входило в корпус;
        - средняя зона: локоть может сгибаться широко;
        - верхняя зона: локоть снова ограничиваем, чтобы руки не пересекались сверху.
        """

        r = shoulder_roll

        if r <= -0.85:
            # Рука сильно внизу: почти прямая.
            return 0.35

        if r <= -0.25:
            # -0.85 .. -0.25 => 0.35 .. 0.85
            t = (r + 0.85) / 0.60
            return 0.35 + t * (0.85 - 0.35)

        if r <= 0.35:
            # -0.25 .. 0.35 => 0.85 .. 1.55
            t = (r + 0.25) / 0.60
            return 0.85 + t * (1.55 - 0.85)

        if r <= 0.90:
            # 0.35 .. 0.90 => 1.55 .. 2.05
            t = (r - 0.35) / 0.55
            return 1.55 + t * (2.05 - 1.55)

        if r <= self.roll_max:
            # 0.90 .. roll_max => 2.05 .. 1.05
            # Когда рука почти сверху, локоть опять ограничиваем,
            # иначе предплечье пересекает корпус/голову.
            t = (r - 0.90) / max(1e-6, (self.roll_max - 0.90))
            return 2.05 + t * (1.05 - 2.05)

        return 1.05

    def apply(self, desired: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], bool, str]:
        safe = dict(desired)

        safety_active = False

        # 1) Абсолютные лимиты из LIMITS.
        for joint in UPPER_COMMAND_JOINTS:
            lo, hi = LIMITS[joint]
            before = safe.get(joint, 0.0)
            after = clamp(before, lo, hi)
            if abs(after - before) > 1e-9:
                safety_active = True
            safe[joint] = after

        # 2) Чёткие границы shoulder_roll снизу и сверху.
        # Это ключевая защита от пересечений.
        for side in ['left', 'right']:
            roll_joint = f'{side}_shoulder_roll_joint'

            before = safe[roll_joint]
            after = clamp(before, self.roll_min, self.roll_max)

            if abs(after - before) > 1e-9:
                safety_active = True

            safe[roll_joint] = after

        # 3) Yaw больше НЕ зануляем.
        # Оставляем yaw-команду такой, какой её дал алгоритм plane flip.
        # Это возвращает проворот рук по yaw.

        # 4) Адаптивные пределы локтей.
        left_roll = safe['left_shoulder_roll_joint']
        right_roll = safe['right_shoulder_roll_joint']

        left_elbow_desired = safe['left_elbow_joint']
        right_elbow_desired = safe['right_elbow_joint']

        left_elbow_max = self.adaptive_elbow_max(left_roll)
        right_elbow_max = self.adaptive_elbow_max(right_roll)

        left_elbow_safe = clamp(left_elbow_desired, self.elbow_min, left_elbow_max)
        right_elbow_safe = clamp(right_elbow_desired, self.elbow_min, right_elbow_max)

        left_over = max(0.0, left_elbow_desired - left_elbow_max)
        right_over = max(0.0, right_elbow_desired - right_elbow_max)

        if abs(left_elbow_safe - left_elbow_desired) > 1e-9:
            safety_active = True
        if abs(right_elbow_safe - right_elbow_desired) > 1e-9:
            safety_active = True

        safe['left_elbow_joint'] = left_elbow_safe
        safe['right_elbow_joint'] = right_elbow_safe

        # 5) Reset больше не делаем.
        hard_reset = False
        reason = 'OK_PROJECTED_TO_SAFE_BOUNDARY' if safety_active else 'OK'

        return safe, {
            'left_elbow_max': left_elbow_max,
            'right_elbow_max': right_elbow_max,
            'left_elbow_over': left_over,
            'right_elbow_over': right_over,
            'safety_active': 1.0 if safety_active else 0.0,
            'hard_reset': 0.0,
        }, hard_reset, reason


class CsvCommandLogger:
    def __init__(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(base_dir, f'teleop_session_{stamp}.csv')

        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)

        header = [
            'time_sec',
            'source',
            'valid',
            'manual_enabled',
            'is_calibrated',

            'left_upper_local_x',
            'left_upper_local_y',
            'left_upper_local_z',
            'right_upper_local_x',
            'right_upper_local_y',
            'right_upper_local_z',

            'left_elevation',
            'right_elevation',
            'left_forward',
            'right_forward',

            'left_elbow_flex',
            'right_elbow_flex',

            'left_wrist_rel_y',
            'right_wrist_rel_y',
            'left_mid_upper_y',
            'right_mid_upper_y',
            'left_palm_above',
            'right_palm_above',
            'left_elbow_plane_flip',
            'right_elbow_plane_flip',

            'safety_active',
            'hard_reset',
            'left_elbow_max',
            'right_elbow_max',
            'left_elbow_over',
            'right_elbow_over',
        ] + FULL_JOINTS

        self.writer.writerow(header)
        self.file.flush()

    def write(
        self,
        time_sec: float,
        source: str,
        valid: bool,
        manual_enabled: bool,
        is_calibrated: bool,
        joint_values: Dict[str, float],
        raw: Optional[Dict[str, float]],
        safety: Optional[Dict[str, float]],
    ):
        raw = raw or {}
        safety = safety or {}

        row = [
            f'{time_sec:.6f}',
            source,
            int(valid),
            int(manual_enabled),
            int(is_calibrated),

            f"{raw.get('left_upper_local_x', 0.0):.6f}",
            f"{raw.get('left_upper_local_y', 0.0):.6f}",
            f"{raw.get('left_upper_local_z', 0.0):.6f}",
            f"{raw.get('right_upper_local_x', 0.0):.6f}",
            f"{raw.get('right_upper_local_y', 0.0):.6f}",
            f"{raw.get('right_upper_local_z', 0.0):.6f}",

            f"{raw.get('left_elevation', 0.0):.6f}",
            f"{raw.get('right_elevation', 0.0):.6f}",
            f"{raw.get('left_forward', 0.0):.6f}",
            f"{raw.get('right_forward', 0.0):.6f}",

            f"{raw.get('left_elbow_flex', 0.0):.6f}",
            f"{raw.get('right_elbow_flex', 0.0):.6f}",

            f"{raw.get('left_wrist_rel_y', 0.0):.6f}",
            f"{raw.get('right_wrist_rel_y', 0.0):.6f}",
            f"{raw.get('left_mid_upper_y', 0.0):.6f}",
            f"{raw.get('right_mid_upper_y', 0.0):.6f}",
            int(raw.get('left_palm_above', 0.0)),
            int(raw.get('right_palm_above', 0.0)),
            int(raw.get('left_elbow_plane_flip', 0.0)),
            int(raw.get('right_elbow_plane_flip', 0.0)),

            int(safety.get('safety_active', 0.0)),
            int(safety.get('hard_reset', 0.0)),
            f"{safety.get('left_elbow_max', 0.0):.6f}",
            f"{safety.get('right_elbow_max', 0.0):.6f}",
            f"{safety.get('left_elbow_over', 0.0):.6f}",
            f"{safety.get('right_elbow_over', 0.0):.6f}",
        ]

        for joint in FULL_JOINTS:
            row.append(f'{joint_values.get(joint, 0.0):.6f}')

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class RetargetNode(Node):
    def __init__(self):
        super().__init__('retarget_node')

        self.alpha = float(self.declare_parameter('smoothing_alpha', 0.20).value)

        # При срабатывании soft safety локоть приводим к пределу мягче.
        self.safety_alpha = float(self.declare_parameter('safety_alpha', 0.10).value)

        self.min_visibility = float(self.declare_parameter('min_visibility', 0.15).value)

        self.roll_scale_up = float(self.declare_parameter('roll_scale_up', 1.35).value)
        self.roll_scale_down = float(self.declare_parameter('roll_scale_down', 2.00).value)

        self.pitch_scale = float(self.declare_parameter('pitch_scale', 0.45).value)
        self.elbow_scale = float(self.declare_parameter('elbow_scale', 1.0).value)

        self.enable_pitch = bool(self.declare_parameter('enable_pitch', False).value)
        self.enable_elbow_plane_flip = bool(
            self.declare_parameter('enable_elbow_plane_flip', True).value
        )

        self.plane_flip_margin = float(self.declare_parameter('plane_flip_margin', 0.03).value)

        self.human_zero: Optional[Dict[str, float]] = None
        self.last_raw: Optional[Dict[str, float]] = None

        self.prev = dict(FULL_ZERO)
        self.is_calibrated = False

        self.manual_enabled = False
        self.manual_joint_values = dict(FULL_ZERO)

        self.left_plane_flip = False
        self.right_plane_flip = False

        self.safety_layer = AdaptiveSafetyLayer()
        self.safety_latched = False
        self.last_safety_info: Dict[str, float] = {}

        self.csv_logger = CsvCommandLogger(os.path.join(os.getcwd(), 'logs'))
        self.get_logger().info(f'Command CSV log: {self.csv_logger.path}')

        self.create_subscription(PoseLandmarks3D, '/pose/landmarks', self.on_pose, 10)
        self.create_subscription(String, '/teleop/control', self.on_control, 10)
        self.create_subscription(Bool, '/debug/manual_enabled', self.on_manual_enabled, 10)
        self.create_subscription(JointState, '/debug/manual_joint_states', self.on_manual_joint_states, 10)

        self.cmd_pub = self.create_publisher(UpperBodyCommand, '/upper_body/command', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.debug_pub = self.create_publisher(JointState, '/retarget/joint_states_debug', 10)
        self.status_pub = self.create_publisher(String, '/teleop/status', 10)
        self.safety_status_pub = self.create_publisher(String, '/teleop/safety_status', 10)

        self.timer = self.create_timer(0.05, self.on_timer)

        self.frame_count = 0

        self.get_logger().info('Retarget node started: 3D body-frame + elbow plane flip + adaptive safety.')
        self.get_logger().info('Safety: adaptive elbow max depends on shoulder roll/yaw.')
        self.get_logger().info('Soft violation: elbow is smoothly limited.')
        self.get_logger().info('Hard violation > 10 deg: default reset pose, calibration disabled.')

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_manual_enabled(self, msg: Bool):
        self.manual_enabled = bool(msg.data)

    def on_manual_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in FULL_JOINTS:
                self.manual_joint_values[name] = float(pos)

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_safety_status(self, text: str):
        msg = String()
        msg.data = text
        self.safety_status_pub.publish(msg)

    def on_timer(self):
        if self.safety_latched:
            self.publish_status('SAFETY_LATCHED_RESET_REQUIRED')
            self.publish_safety_status('SAFETY_LATCHED: press R or recalibrate with C')
            self.publish_command_and_joint_states(
                FULL_ZERO,
                valid=False,
                source='safety_latched_zero',
                raw=self.last_raw,
                safety=self.last_safety_info,
            )
            return

        if self.manual_enabled:
            self.publish_status('MANUAL_SLIDER_OVERRIDE')

            safe, safety_info, hard_reset, reason = self.safety_layer.apply(self.manual_joint_values)

            if hard_reset:
                self.trigger_safety_reset(reason, safety_info)
                return

            self.publish_safety_status('OK_MANUAL' if not safety_info.get('safety_active') else 'SOFT_LIMIT_MANUAL')

            self.publish_command_and_joint_states(
                safe,
                valid=True,
                source='manual_slider',
                raw=self.last_raw,
                safety=safety_info,
            )
            return

        self.publish_status('ACTIVE' if self.is_calibrated else 'WAITING_FOR_CALIBRATION')

        if not self.is_calibrated:
            self.publish_command_and_joint_states(
                FULL_ZERO,
                valid=False,
                source='zero_waiting',
                raw=self.last_raw,
                safety=self.last_safety_info,
            )

    def trigger_safety_reset(self, reason: str, safety_info: Dict[str, float]):
        self.get_logger().error(f'SAFETY HARD RESET: {reason}')

        self.human_zero = None
        self.is_calibrated = False
        self.prev = dict(FULL_ZERO)
        self.safety_latched = True
        self.last_safety_info = dict(safety_info)

        self.publish_safety_status(f'HARD_RESET: {reason}')

        self.publish_command_and_joint_states(
            FULL_ZERO,
            valid=False,
            source='safety_hard_reset_zero',
            raw=self.last_raw,
            safety=safety_info,
        )

    def on_control(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd in ('reset', 'r'):
            self.human_zero = None
            self.last_raw = None
            self.prev = dict(FULL_ZERO)
            self.is_calibrated = False
            self.left_plane_flip = False
            self.right_plane_flip = False
            self.safety_latched = False
            self.last_safety_info = {}

            self.publish_command_and_joint_states(
                FULL_ZERO,
                valid=False,
                source='reset_zero',
                raw=None,
                safety=None,
            )

            self.get_logger().info('Calibration and safety latch reset. Press C to calibrate again.')
            return

        if cmd in ('calibrate', 'c'):
            if self.last_raw is None:
                self.get_logger().warn('Cannot calibrate: no valid pose received yet.')
                return

            self.safety_latched = False
            self.human_zero = dict(self.last_raw)
            self.prev = dict(FULL_ZERO)
            self.is_calibrated = True

            self.left_plane_flip = bool(self.last_raw.get('left_elbow_plane_flip', 0.0))
            self.right_plane_flip = bool(self.last_raw.get('right_elbow_plane_flip', 0.0))

            self.publish_command_and_joint_states(
                FULL_ZERO,
                valid=True,
                source='calibration_zero',
                raw=self.last_raw,
                safety=None,
            )

            self.get_logger().info('Calibration saved.')
            self.get_logger().info(f'human_zero = {self.human_zero}')
            return

    def scale_elevation_delta(self, delta: float) -> float:
        if delta >= 0.0:
            return self.roll_scale_up * delta
        return self.roll_scale_down * delta

    def smooth_to_target(self, target: Dict[str, float], safety_info: Dict[str, float]) -> Dict[str, float]:
        smoothed = dict(FULL_ZERO)

        safety_active = bool(safety_info.get('safety_active', 0.0))
        alpha = self.safety_alpha if safety_active else self.alpha

        for joint in FULL_JOINTS:
            if joint in UPPER_COMMAND_JOINTS:
                smoothed[joint] = self.prev[joint] * (1.0 - alpha) + target[joint] * alpha
            else:
                smoothed[joint] = FULL_ZERO[joint]

            self.prev[joint] = smoothed[joint]

        return smoothed

    def on_pose(self, msg: PoseLandmarks3D):
        if self.manual_enabled or self.safety_latched:
            return

        raw = self.compute_raw_features(msg)

        if raw is None:
            return

        self.last_raw = raw

        if not self.is_calibrated or self.human_zero is None:
            self.publish_command_and_joint_states(
                FULL_ZERO,
                valid=False,
                source='zero_not_calibrated',
                raw=raw,
                safety=self.last_safety_info,
            )
            return

        desired = dict(FULL_ZERO)

        # Shoulder roll: arm elevation
        left_elev_delta = raw['left_elevation'] - self.human_zero['left_elevation']
        right_elev_delta = raw['right_elevation'] - self.human_zero['right_elevation']

        desired['left_shoulder_roll_joint'] = (
            FULL_ZERO['left_shoulder_roll_joint']
            + self.scale_elevation_delta(left_elev_delta)
        )

        desired['right_shoulder_roll_joint'] = (
            FULL_ZERO['right_shoulder_roll_joint']
            + self.scale_elevation_delta(right_elev_delta)
        )

        # Optional pitch
        if self.enable_pitch:
            left_forward_delta = raw['left_forward'] - self.human_zero['left_forward']
            right_forward_delta = raw['right_forward'] - self.human_zero['right_forward']

            desired['left_shoulder_pitch_joint'] = (
                FULL_ZERO['left_shoulder_pitch_joint']
                + self.pitch_scale * left_forward_delta
            )

            desired['right_shoulder_pitch_joint'] = (
                FULL_ZERO['right_shoulder_pitch_joint']
                + self.pitch_scale * right_forward_delta
            )
        else:
            desired['left_shoulder_pitch_joint'] = 0.0
            desired['right_shoulder_pitch_joint'] = 0.0

        # Yaw as elbow plane flip
        if self.enable_elbow_plane_flip:
            desired['left_shoulder_yaw_joint'] = math.pi if raw['left_elbow_plane_flip'] else 0.0

            # Для правого плеча знак противоположный.
            # Так обе руки при flip проходят через переднюю сторону корпуса,
            # а не проворачиваются в разные стороны.
            desired['right_shoulder_yaw_joint'] = -math.pi if raw['right_elbow_plane_flip'] else 0.0
        else:
            desired['left_shoulder_yaw_joint'] = 0.0
            desired['right_shoulder_yaw_joint'] = 0.0

        # Elbow absolute flex
        desired['left_elbow_joint'] = 0.05 + self.elbow_scale * raw['left_elbow_flex']
        desired['right_elbow_joint'] = 0.05 + self.elbow_scale * raw['right_elbow_flex']

        # Adaptive safety
        safe_target, safety_info, hard_reset, reason = self.safety_layer.apply(desired)
        self.last_safety_info = dict(safety_info)

        if hard_reset:
            self.trigger_safety_reset(reason, safety_info)
            return

        smoothed = self.smooth_to_target(safe_target, safety_info)

        if safety_info.get('safety_active', 0.0):
            self.publish_safety_status(
                'SOFT_LIMIT: '
                f"L_over={safety_info.get('left_elbow_over', 0.0):.3f}, "
                f"R_over={safety_info.get('right_elbow_over', 0.0):.3f}"
            )
        else:
            self.publish_safety_status('OK')

        self.frame_count += 1

        if self.frame_count % 20 == 0:
            self.get_logger().info(
                'cmd: '
                f"L_roll={smoothed['left_shoulder_roll_joint']:.3f}, "
                f"R_roll={smoothed['right_shoulder_roll_joint']:.3f}, "
                f"L_yaw={smoothed['left_shoulder_yaw_joint']:.3f}, "
                f"R_yaw={smoothed['right_shoulder_yaw_joint']:.3f}, "
                f"L_elb={smoothed['left_elbow_joint']:.3f}, "
                f"R_elb={smoothed['right_elbow_joint']:.3f}; "
                'safety: '
                f"Lmax={safety_info.get('left_elbow_max', 0.0):.3f}, "
                f"Rmax={safety_info.get('right_elbow_max', 0.0):.3f}, "
                f"active={int(safety_info.get('safety_active', 0.0))}"
            )

        self.publish_command_and_joint_states(
            smoothed,
            valid=True,
            source='mediapipe_3d_plane_flip_safety_retarget',
            raw=raw,
            safety=safety_info,
        )

    def update_plane_flip(self, current_flip: bool, wrist_rel_y: float, mid_upper_y: float) -> bool:
        if wrist_rel_y > mid_upper_y + self.plane_flip_margin:
            return False

        if wrist_rel_y < mid_upper_y - self.plane_flip_margin:
            return True

        return current_flip

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
            'left_hip',
            'right_hip',
        ]

        pts: Dict[str, Point] = {}
        vis: Dict[str, float] = {}

        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                pts[name] = (float(msg.x[i]), float(msg.y[i]), float(msg.z[i]))
            vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0

        if any(k not in pts for k in required):
            self.get_logger().warn(f'Missing landmarks. Got: {list(pts.keys())}', throttle_duration_sec=2.0)
            return None

        must_be_visible = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']

        if any(vis.get(k, 0.0) < self.min_visibility for k in must_be_visible):
            self.get_logger().warn(
                'Low visibility: '
                f"LS={vis.get('left_shoulder', 0):.2f}, RS={vis.get('right_shoulder', 0):.2f}, "
                f"LE={vis.get('left_elbow', 0):.2f}, RE={vis.get('right_elbow', 0):.2f}",
                throttle_duration_sec=2.0
            )
            return None

        lsh = pts['left_shoulder']
        rsh = pts['right_shoulder']
        lel = pts['left_elbow']
        rel = pts['right_elbow']
        lwr = pts['left_wrist']
        rwr = pts['right_wrist']
        lhip = pts['left_hip']
        rhip = pts['right_hip']

        shoulder_center = mul(add(lsh, rsh), 0.5)
        hip_center = mul(add(lhip, rhip), 0.5)

        body_right = normalize(sub(rsh, lsh))
        body_up_raw = normalize(sub(shoulder_center, hip_center))

        body_up = normalize(sub(body_up_raw, mul(body_right, dot(body_up_raw, body_right))))
        body_front = normalize(cross(body_right, body_up))

        if norm(body_front) < 1e-6:
            return None

        left_upper_vec = sub(lel, lsh)
        right_upper_vec = sub(rel, rsh)

        left_upper = normalize(left_upper_vec)
        right_upper = normalize(right_upper_vec)

        lu = local_components(left_upper, body_right, body_up, body_front)
        ru = local_components(right_upper, body_right, body_up, body_front)

        left_elevation = math.atan2(lu[1], math.sqrt(lu[0] * lu[0] + lu[2] * lu[2]))
        right_elevation = math.atan2(ru[1], math.sqrt(ru[0] * ru[0] + ru[2] * ru[2]))

        left_forward = math.atan2(lu[2], math.sqrt(lu[0] * lu[0] + lu[1] * lu[1]))
        right_forward = math.atan2(ru[2], math.sqrt(ru[0] * ru[0] + ru[1] * ru[1]))

        left_elbow_flex = clamp(math.pi - angle_between(sub(lsh, lel), sub(lwr, lel)), 0.0, 2.2)
        right_elbow_flex = clamp(math.pi - angle_between(sub(rsh, rel), sub(rwr, rel)), 0.0, 2.2)

        left_wrist_rel = local_components(sub(lwr, lsh), body_right, body_up, body_front)
        right_wrist_rel = local_components(sub(rwr, rsh), body_right, body_up, body_front)

        left_elbow_rel = local_components(sub(lel, lsh), body_right, body_up, body_front)
        right_elbow_rel = local_components(sub(rel, rsh), body_right, body_up, body_front)

        left_wrist_rel_y = left_wrist_rel[1]
        right_wrist_rel_y = right_wrist_rel[1]

        left_mid_upper_y = 0.5 * left_elbow_rel[1]
        right_mid_upper_y = 0.5 * right_elbow_rel[1]

        left_palm_above = left_wrist_rel_y > left_mid_upper_y
        right_palm_above = right_wrist_rel_y > right_mid_upper_y

        self.left_plane_flip = self.update_plane_flip(
            self.left_plane_flip,
            left_wrist_rel_y,
            left_mid_upper_y,
        )

        self.right_plane_flip = self.update_plane_flip(
            self.right_plane_flip,
            right_wrist_rel_y,
            right_mid_upper_y,
        )

        return {
            'left_upper_local_x': lu[0],
            'left_upper_local_y': lu[1],
            'left_upper_local_z': lu[2],
            'right_upper_local_x': ru[0],
            'right_upper_local_y': ru[1],
            'right_upper_local_z': ru[2],

            'left_elevation': left_elevation,
            'right_elevation': right_elevation,
            'left_forward': left_forward,
            'right_forward': right_forward,

            'left_elbow_flex': left_elbow_flex,
            'right_elbow_flex': right_elbow_flex,

            'left_wrist_rel_y': left_wrist_rel_y,
            'right_wrist_rel_y': right_wrist_rel_y,
            'left_mid_upper_y': left_mid_upper_y,
            'right_mid_upper_y': right_mid_upper_y,
            'left_palm_above': 1.0 if left_palm_above else 0.0,
            'right_palm_above': 1.0 if right_palm_above else 0.0,
            'left_elbow_plane_flip': 1.0 if self.left_plane_flip else 0.0,
            'right_elbow_plane_flip': 1.0 if self.right_plane_flip else 0.0,
        }

    def publish_command_and_joint_states(
        self,
        joint_values: Dict[str, float],
        valid: bool,
        source: str,
        raw: Optional[Dict[str, float]],
        safety: Optional[Dict[str, float]],
    ):
        cmd = UpperBodyCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'upper_body'
        cmd.joint_names = list(UPPER_COMMAND_JOINTS)
        cmd.position = [joint_values.get(j, 0.0) for j in UPPER_COMMAND_JOINTS]
        cmd.confidence = [1.0 if valid else 0.0 for _ in UPPER_COMMAND_JOINTS]
        cmd.valid = valid
        self.cmd_pub.publish(cmd)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(FULL_JOINTS)
        js.position = [joint_values.get(j, 0.0) for j in FULL_JOINTS]

        self.js_pub.publish(js)
        self.debug_pub.publish(js)

        self.csv_logger.write(
            time_sec=self.now_sec(),
            source=source,
            valid=valid,
            manual_enabled=self.manual_enabled,
            is_calibrated=self.is_calibrated,
            joint_values=joint_values,
            raw=raw,
            safety=safety,
        )

    def destroy_node(self):
        try:
            self.csv_logger.close()
        except Exception:
            pass
        super().destroy_node()


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
