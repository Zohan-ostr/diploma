import csv
import math
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


# ============================================================
# HOME / RVIZ RETARGET NODE — SEQUENTIAL IK VERSION
# ============================================================
#
# Последовательная ОЗК:
#
#   1. MediaPipe Pose даёт 3D landmarks.
#   2. Строим систему координат корпуса относительно таза.
#   3. Для каждой руки:
#        shoulder -> elbow  задаёт целевое положение локтя
#        elbow    -> wrist  задаёт целевое положение конца руки
#   4. Сначала решаем IK плеча для попадания локтем.
#   5. Потом при найденном плече решаем угол локтя для попадания кистью.
#
# Важно:
#   Это не калибровочная схема q_robot = q_zero + delta.
#   Кнопка C теперь просто сбрасывает внутренний фильтр IK.
#   Кнопка R возвращает робота в дефолтную позу.
#
# ============================================================


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


# Сейчас safety-ограничения отключены, поэтому лимиты широкие.
# Это нужно, чтобы увидеть реальное поведение IK.
JOINT_LIMITS = {
    'left_shoulder_pitch_joint': (-3.14, 3.14),
    'left_shoulder_roll_joint': (-3.14, 3.14),
    'left_shoulder_yaw_joint': (-3.14, 3.14),
    'left_elbow_joint': (-0.20, 3.14),

    'right_shoulder_pitch_joint': (-3.14, 3.14),
    'right_shoulder_roll_joint': (-3.14, 3.14),
    'right_shoulder_yaw_joint': (-3.14, 3.14),
    'right_elbow_joint': (-0.20, 3.14),
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
Mat3 = List[List[float]]


# Длины сегментов простой RViz-модели.
# Если по видео будет видно, что модель систематически недотягивается
# или перетягивается, менять надо эти два значения.
ROBOT_UPPER_ARM = 0.30
ROBOT_FOREARM = 0.30


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


def sq_dist(a: Point, b: Point) -> float:
    d = sub(a, b)
    return dot(d, d)


def angle_between(a: Point, b: Point) -> float:
    na = norm(a)
    nb = norm(b)

    if na < 1e-8 or nb < 1e-8:
        return 0.0

    c = clamp(dot(a, b) / (na * nb), -1.0, 1.0)
    return math.acos(c)


def eye3() -> Mat3:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def mat_mul(A: Mat3, B: Mat3) -> Mat3:
    return [
        [
            A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def mat_vec(R: Mat3, v: Point) -> Point:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def rot_axis(axis: Point, angle: float) -> Mat3:
    x, y, z = normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def local_components(v: Point, right: Point, up: Point, front: Point) -> Point:
    return (
        dot(v, right),
        dot(v, up),
        dot(v, front),
    )


def joint_names_for_side(side: str) -> List[str]:
    prefix = 'left' if side == 'left' else 'right'
    return [
        f'{prefix}_shoulder_pitch_joint',
        f'{prefix}_shoulder_roll_joint',
        f'{prefix}_shoulder_yaw_joint',
        f'{prefix}_elbow_joint',
    ]


class SequentialArmIKSolver:
    """
    Последовательный IK одной руки с дискретной плоскостью локтя.

    Важно:
    shoulder_yaw НЕ ищется оптимизацией.

    Почему:
        yaw — это поворот вокруг оси плечо -> локоть.
        Положение локтя почти не задаёт этот угол, поэтому непрерывная
        оптимизация yaw получается неустойчивой.

    Логика:
        1. yaw выбирается дискретно: 0 или ±pi;
        2. pitch/roll решают положение локтя;
        3. elbow решает положение кисти.
    """

    def __init__(self, side: str):
        self.side = side
        self.side_sign = 1.0 if side == 'left' else -1.0
        self.joints = joint_names_for_side(side)

    def clamp_q(self, q: List[float]) -> List[float]:
        out = []
        for joint, value in zip(self.joints, q):
            lo, hi = JOINT_LIMITS[joint]
            out.append(clamp(value, lo, hi))
        return out

    def shoulder_rotation(self, pitch: float, roll: float, yaw: float) -> Mat3:
        s = self.side_sign

        R = eye3()

        # Оси соответствуют URDF:
        # pitch around Z
        # roll around ±X
        # yaw around ±Y
        R = mat_mul(R, rot_axis((0.0, 0.0, 1.0), pitch))
        R = mat_mul(R, rot_axis((s, 0.0, 0.0), roll))
        R = mat_mul(R, rot_axis((0.0, s, 0.0), yaw))

        return R

    def fk_elbow(self, pitch: float, roll: float, yaw: float) -> Point:
        R = self.shoulder_rotation(pitch, roll, yaw)
        upper_local = (0.0, self.side_sign * ROBOT_UPPER_ARM, 0.0)
        return mat_vec(R, upper_local)

    def fk_wrist(self, q: List[float]) -> Tuple[Point, Point]:
        q = self.clamp_q(q)

        pitch, roll, yaw, elbow = q

        R = self.shoulder_rotation(pitch, roll, yaw)

        upper_local = (0.0, self.side_sign * ROBOT_UPPER_ARM, 0.0)
        elbow_pos = mat_vec(R, upper_local)

        # elbow axis: ±X
        R_elbow = mat_mul(R, rot_axis((self.side_sign, 0.0, 0.0), elbow))

        forearm_local = (0.0, self.side_sign * ROBOT_FOREARM, 0.0)
        wrist_pos = add(elbow_pos, mat_vec(R_elbow, forearm_local))

        return elbow_pos, wrist_pos

    def shoulder_loss(
        self,
        pitch_roll: List[float],
        yaw_fixed: float,
        target_elbow: Point,
        q_ref: List[float],
    ) -> float:
        pitch, roll = pitch_roll

        elbow_pos = self.fk_elbow(pitch, roll, yaw_fixed)
        err = sq_dist(elbow_pos, target_elbow)

        # Регуляризация только по pitch/roll.
        reg = (pitch - q_ref[0]) ** 2 + (roll - q_ref[1]) ** 2

        return 1.0 * err + 0.01 * reg

    def solve_shoulder_to_elbow(
        self,
        target_elbow: Point,
        yaw_fixed: float,
        q_init: List[float],
    ) -> Tuple[List[float], float]:
        # Оптимизируем только pitch и roll.
        q = [q_init[0], q_init[1]]
        q_ref = list(q_init)

        eps = 1e-3
        lr = 0.45

        pitch_joint = self.joints[0]
        roll_joint = self.joints[1]

        for _ in range(45):
            grad = [0.0, 0.0]

            for j in range(2):
                qp = list(q)
                qm = list(q)

                qp[j] += eps
                qm[j] -= eps

                lp = self.shoulder_loss(qp, yaw_fixed, target_elbow, q_ref)
                lm = self.shoulder_loss(qm, yaw_fixed, target_elbow, q_ref)

                grad[j] = (lp - lm) / (2.0 * eps)

            for j in range(2):
                q[j] -= lr * grad[j]

            q[0] = clamp(q[0], JOINT_LIMITS[pitch_joint][0], JOINT_LIMITS[pitch_joint][1])
            q[1] = clamp(q[1], JOINT_LIMITS[roll_joint][0], JOINT_LIMITS[roll_joint][1])

            lr *= 0.96

        loss = self.shoulder_loss(q, yaw_fixed, target_elbow, q_ref)
        return [q[0], q[1], yaw_fixed], loss

    def elbow_loss(
        self,
        elbow: float,
        q_shoulder: List[float],
        target_wrist: Point,
        elbow_ref: float,
    ) -> float:
        q = [q_shoulder[0], q_shoulder[1], q_shoulder[2], elbow]
        _, wrist_pos = self.fk_wrist(q)

        wrist_err = sq_dist(wrist_pos, target_wrist)
        reg = (elbow - elbow_ref) ** 2

        return wrist_err + 0.01 * reg

    def solve_elbow_to_wrist(
        self,
        q_shoulder: List[float],
        target_wrist: Point,
        elbow_init: float,
    ) -> Tuple[float, float]:
        joint = self.joints[3]
        lo, hi = JOINT_LIMITS[joint]

        elbow = clamp(elbow_init, lo, hi)
        elbow_ref = elbow

        eps = 1e-3
        lr = 0.55

        for _ in range(40):
            lp = self.elbow_loss(clamp(elbow + eps, lo, hi), q_shoulder, target_wrist, elbow_ref)
            lm = self.elbow_loss(clamp(elbow - eps, lo, hi), q_shoulder, target_wrist, elbow_ref)

            grad = (lp - lm) / (2.0 * eps)

            elbow -= lr * grad
            elbow = clamp(elbow, lo, hi)

            lr *= 0.95

        loss = self.elbow_loss(elbow, q_shoulder, target_wrist, elbow_ref)
        return elbow, loss

    def solve_sequential(
        self,
        target_elbow: Point,
        target_wrist: Point,
        yaw_fixed: float,
        q_init: List[float],
    ) -> Tuple[List[float], Dict[str, float]]:
        q_init = self.clamp_q(list(q_init))

        q_shoulder, shoulder_loss = self.solve_shoulder_to_elbow(
            target_elbow=target_elbow,
            yaw_fixed=yaw_fixed,
            q_init=q_init,
        )

        elbow, elbow_loss = self.solve_elbow_to_wrist(
            q_shoulder=q_shoulder,
            target_wrist=target_wrist,
            elbow_init=q_init[3],
        )

        q = self.clamp_q([q_shoulder[0], q_shoulder[1], q_shoulder[2], elbow])

        elbow_pos, wrist_pos = self.fk_wrist(q)

        debug = {
            'shoulder_loss': shoulder_loss,
            'elbow_loss': elbow_loss,
            'elbow_error': math.sqrt(sq_dist(elbow_pos, target_elbow)),
            'wrist_error': math.sqrt(sq_dist(wrist_pos, target_wrist)),
        }

        return q, debug


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

            'left_target_elbow_x',
            'left_target_elbow_y',
            'left_target_elbow_z',
            'left_target_wrist_x',
            'left_target_wrist_y',
            'left_target_wrist_z',

            'right_target_elbow_x',
            'right_target_elbow_y',
            'right_target_elbow_z',
            'right_target_wrist_x',
            'right_target_wrist_y',
            'right_target_wrist_z',

            'left_shoulder_loss',
            'left_elbow_loss',
            'left_elbow_error',
            'left_wrist_error',

            'right_shoulder_loss',
            'right_elbow_loss',
            'right_elbow_error',
            'right_wrist_error',
        ] + FULL_JOINTS

        self.writer.writerow(header)
        self.file.flush()

    def write(
        self,
        time_sec: float,
        source: str,
        valid: bool,
        manual_enabled: bool,
        joint_values: Dict[str, float],
        debug: Optional[Dict[str, float]] = None,
    ):
        debug = debug or {}

        row = [
            f'{time_sec:.6f}',
            source,
            int(valid),
            int(manual_enabled),

            f"{debug.get('left_target_elbow_x', 0.0):.6f}",
            f"{debug.get('left_target_elbow_y', 0.0):.6f}",
            f"{debug.get('left_target_elbow_z', 0.0):.6f}",
            f"{debug.get('left_target_wrist_x', 0.0):.6f}",
            f"{debug.get('left_target_wrist_y', 0.0):.6f}",
            f"{debug.get('left_target_wrist_z', 0.0):.6f}",

            f"{debug.get('right_target_elbow_x', 0.0):.6f}",
            f"{debug.get('right_target_elbow_y', 0.0):.6f}",
            f"{debug.get('right_target_elbow_z', 0.0):.6f}",
            f"{debug.get('right_target_wrist_x', 0.0):.6f}",
            f"{debug.get('right_target_wrist_y', 0.0):.6f}",
            f"{debug.get('right_target_wrist_z', 0.0):.6f}",

            f"{debug.get('left_shoulder_loss', 0.0):.6f}",
            f"{debug.get('left_elbow_loss', 0.0):.6f}",
            f"{debug.get('left_elbow_error', 0.0):.6f}",
            f"{debug.get('left_wrist_error', 0.0):.6f}",

            f"{debug.get('right_shoulder_loss', 0.0):.6f}",
            f"{debug.get('right_elbow_loss', 0.0):.6f}",
            f"{debug.get('right_elbow_error', 0.0):.6f}",
            f"{debug.get('right_wrist_error', 0.0):.6f}",
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

        self.alpha = float(self.declare_parameter('smoothing_alpha', 0.24).value)
        self.min_visibility = float(self.declare_parameter('min_visibility', 0.15).value)

        self.left_ik = SequentialArmIKSolver('left')
        self.right_ik = SequentialArmIKSolver('right')

        self.prev = dict(FULL_ZERO)

        self.left_q = [
            FULL_ZERO['left_shoulder_pitch_joint'],
            FULL_ZERO['left_shoulder_roll_joint'],
            FULL_ZERO['left_shoulder_yaw_joint'],
            FULL_ZERO['left_elbow_joint'],
        ]

        self.right_q = [
            FULL_ZERO['right_shoulder_pitch_joint'],
            FULL_ZERO['right_shoulder_roll_joint'],
            FULL_ZERO['right_shoulder_yaw_joint'],
            FULL_ZERO['right_elbow_joint'],
        ]

        self.left_plane_flip = False
        self.right_plane_flip = False

        self.manual_enabled = False
        self.manual_joint_values = dict(FULL_ZERO)

        self.last_debug: Dict[str, float] = {}
        self.left_plane_flip = False
        self.right_plane_flip = False


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

        self.timer = self.create_timer(0.05, self.on_timer)

        self.frame_count = 0

        self.get_logger().info('Retarget node started: sequential IK mode.')
        self.get_logger().info('Stage 1: shoulder IK -> elbow target.')
        self.get_logger().info('Stage 2: elbow IK -> wrist target relative to elbow.')
        self.get_logger().info('C = reset IK filter, R = default pose.')

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_manual_enabled(self, msg: Bool):
        self.manual_enabled = bool(msg.data)

    def on_manual_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in FULL_JOINTS:
                self.manual_joint_values[name] = float(pos)

    def reset_filter(self):
        self.prev = dict(FULL_ZERO)

        self.left_q = [
            FULL_ZERO['left_shoulder_pitch_joint'],
            FULL_ZERO['left_shoulder_roll_joint'],
            FULL_ZERO['left_shoulder_yaw_joint'],
            FULL_ZERO['left_elbow_joint'],
        ]

        self.right_q = [
            FULL_ZERO['right_shoulder_pitch_joint'],
            FULL_ZERO['right_shoulder_roll_joint'],
            FULL_ZERO['right_shoulder_yaw_joint'],
            FULL_ZERO['right_elbow_joint'],
        ]

        self.left_plane_flip = False
        self.right_plane_flip = False

    def on_control(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd in ('calibrate', 'c'):
            self.reset_filter()
            self.get_logger().info('Sequential IK filter reset. No angle calibration is used.')

        elif cmd in ('reset', 'r'):
            self.reset_filter()
            self.publish_command_and_joint_states(
                FULL_ZERO,
                valid=False,
                source='reset_zero',
                debug=self.last_debug,
            )
            self.get_logger().info('Reset to default pose.')

    def on_timer(self):
        status = String()

        if self.manual_enabled:
            status.data = 'MANUAL_SLIDER_OVERRIDE'
            self.status_pub.publish(status)

            self.publish_command_and_joint_states(
                self.manual_joint_values,
                valid=True,
                source='manual_slider',
                debug=self.last_debug,
            )
            return

        status.data = 'SEQUENTIAL_IK_ACTIVE_WAITING_FOR_POSE'
        self.status_pub.publish(status)

    def parse_pose(self, msg: PoseLandmarks3D) -> Optional[Dict[str, Point]]:
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

            if i < len(msg.visibility):
                vis[name] = float(msg.visibility[i])
            else:
                vis[name] = 1.0

        if any(k not in pts for k in required):
            self.get_logger().warn(
                f'Missing landmarks. Got: {list(pts.keys())}',
                throttle_duration_sec=2.0,
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
                'Low visibility: '
                f"LS={vis.get('left_shoulder', 0):.2f}, "
                f"RS={vis.get('right_shoulder', 0):.2f}, "
                f"LE={vis.get('left_elbow', 0):.2f}, "
                f"RE={vis.get('right_elbow', 0):.2f}",
                throttle_duration_sec=2.0,
            )
            return None

        return pts

    def build_body_frame(self, pts: Dict[str, Point]) -> Optional[Tuple[Point, Point, Point]]:
        lsh = pts['left_shoulder']
        rsh = pts['right_shoulder']
        lhip = pts['left_hip']
        rhip = pts['right_hip']

        shoulder_center = mul(add(lsh, rsh), 0.5)
        hip_center = mul(add(lhip, rhip), 0.5)

        body_right = normalize(sub(rsh, lsh))
        body_up_raw = normalize(sub(shoulder_center, hip_center))

        body_up = normalize(sub(body_up_raw, mul(body_right, dot(body_up_raw, body_right))))
        body_front = normalize(cross(body_right, body_up))

        if norm(body_right) < 1e-6 or norm(body_up) < 1e-6 or norm(body_front) < 1e-6:
            return None

        return body_right, body_up, body_front

    def map_human_vec_to_robot(self, v_human: Point, body_right: Point, body_up: Point, body_front: Point) -> Point:
        """
        human body local:
            local_right
            local_up
            local_front

        robot arm local:
            x = front
            y = side
            z = up

        y = -local_right:
            левая рука получает +Y,
            правая рука получает -Y.
        """

        local_right, local_up, local_front = local_components(v_human, body_right, body_up, body_front)

        return (
            local_front,
            -local_right,
            local_up,
        )

    def build_sequential_targets(
        self,
        side: str,
        shoulder: Point,
        elbow: Point,
        wrist: Point,
        body_right: Point,
        body_up: Point,
        body_front: Point,
    ) -> Tuple[Point, Point, bool]:
        """
        Строим target_elbow и target_wrist последовательно.

        Также определяем дискретную плоскость локтя:

            wrist выше середины shoulder->elbow  => plane_flip = False
            wrist ниже середины shoulder->elbow  => plane_flip = True

        В качестве "ладони" пока используем wrist, потому что MediaPipe Pose
        не даёт полную кисть. Позже можно заменить на MediaPipe Hands.
        """

        human_upper = sub(elbow, shoulder)
        human_forearm = sub(wrist, elbow)

        robot_upper_dir = normalize(self.map_human_vec_to_robot(human_upper, body_right, body_up, body_front))
        robot_forearm_dir = normalize(self.map_human_vec_to_robot(human_forearm, body_right, body_up, body_front))

        if norm(robot_upper_dir) < 1e-6:
            robot_upper_dir = (0.0, 1.0 if side == 'left' else -1.0, 0.0)

        if norm(robot_forearm_dir) < 1e-6:
            robot_forearm_dir = robot_upper_dir

        target_elbow = mul(robot_upper_dir, ROBOT_UPPER_ARM)
        target_wrist = add(target_elbow, mul(robot_forearm_dir, ROBOT_FOREARM))

        # Определяем положение wrist относительно середины shoulder->elbow
        # в системе координат корпуса.
        wrist_rel = local_components(sub(wrist, shoulder), body_right, body_up, body_front)
        elbow_rel = local_components(sub(elbow, shoulder), body_right, body_up, body_front)

        wrist_y = wrist_rel[1]
        mid_upper_y = 0.5 * elbow_rel[1]

        # Небольшая мёртвая зона, чтобы plane flip не дрожал на границе.
        margin = 0.03

        if side == 'left':
            previous = getattr(self, 'left_plane_flip', False)
        else:
            previous = getattr(self, 'right_plane_flip', False)

        if wrist_y < mid_upper_y - margin:
            plane_flip = True
        elif wrist_y > mid_upper_y + margin:
            plane_flip = False
        else:
            plane_flip = previous

        if side == 'left':
            self.left_plane_flip = plane_flip
        else:
            self.right_plane_flip = plane_flip

        return target_elbow, target_wrist, plane_flip


    def on_pose(self, msg: PoseLandmarks3D):
        if self.manual_enabled:
            return

        if 'world' not in msg.header.frame_id:
            self.get_logger().warn(
                f'Sequential IK needs pose_world_landmarks, got frame_id={msg.header.frame_id}',
                throttle_duration_sec=2.0,
            )
            return

        pts = self.parse_pose(msg)
        if pts is None:
            return

        body_frame = self.build_body_frame(pts)
        if body_frame is None:
            return

        body_right, body_up, body_front = body_frame

        left_target_elbow, left_target_wrist, left_flip = self.build_sequential_targets(
            'left',
            pts['left_shoulder'],
            pts['left_elbow'],
            pts['left_wrist'],
            body_right,
            body_up,
            body_front,
        )

        right_target_elbow, right_target_wrist, right_flip = self.build_sequential_targets(
            'right',
            pts['right_shoulder'],
            pts['right_elbow'],
            pts['right_wrist'],
            body_right,
            body_up,
            body_front,
        )

        # Дискретная плоскость локтя:
        # левая рука:  0 или +pi
        # правая рука: 0 или -pi
        # Так обе руки проворачиваются через переднюю сторону.
        left_yaw_fixed = -math.pi if left_flip else 0.0
        right_yaw_fixed = math.pi if right_flip else 0.0

        self.left_q, left_dbg = self.left_ik.solve_sequential(
            left_target_elbow,
            left_target_wrist,
            left_yaw_fixed,
            self.left_q,
        )

        self.right_q, right_dbg = self.right_ik.solve_sequential(
            right_target_elbow,
            right_target_wrist,
            right_yaw_fixed,
            self.right_q,
        )

        target = dict(FULL_ZERO)

        for joint, value in zip(self.left_ik.joints, self.left_q):
            target[joint] = value

        for joint, value in zip(self.right_ik.joints, self.right_q):
            target[joint] = value

        smoothed = dict(FULL_ZERO)

        for joint in FULL_JOINTS:
            if joint in UPPER_COMMAND_JOINTS:
                smoothed[joint] = self.prev[joint] * (1.0 - self.alpha) + target[joint] * self.alpha
            else:
                smoothed[joint] = FULL_ZERO[joint]

            self.prev[joint] = smoothed[joint]

        debug = {
            'left_target_elbow_x': left_target_elbow[0],
            'left_target_elbow_y': left_target_elbow[1],
            'left_target_elbow_z': left_target_elbow[2],
            'left_target_wrist_x': left_target_wrist[0],
            'left_target_wrist_y': left_target_wrist[1],
            'left_target_wrist_z': left_target_wrist[2],

            'right_target_elbow_x': right_target_elbow[0],
            'right_target_elbow_y': right_target_elbow[1],
            'right_target_elbow_z': right_target_elbow[2],
            'right_target_wrist_x': right_target_wrist[0],
            'right_target_wrist_y': right_target_wrist[1],
            'right_target_wrist_z': right_target_wrist[2],

            'left_shoulder_loss': left_dbg['shoulder_loss'],
            'left_elbow_loss': left_dbg['elbow_loss'],
            'left_elbow_error': left_dbg['elbow_error'],
            'left_wrist_error': left_dbg['wrist_error'],

            'right_shoulder_loss': right_dbg['shoulder_loss'],
            'right_elbow_loss': right_dbg['elbow_loss'],
            'right_elbow_error': right_dbg['elbow_error'],
            'right_wrist_error': right_dbg['wrist_error'],

            'left_plane_flip': 1.0 if left_flip else 0.0,
            'right_plane_flip': 1.0 if right_flip else 0.0,
            'left_yaw_fixed': left_yaw_fixed,
            'right_yaw_fixed': right_yaw_fixed,
        }

        self.last_debug = debug

        valid = (
            left_dbg['elbow_error'] < 0.12 and
            right_dbg['elbow_error'] < 0.12 and
            left_dbg['wrist_error'] < 0.18 and
            right_dbg['wrist_error'] < 0.18
        )

        self.frame_count += 1

        if self.frame_count % 20 == 0:
            self.get_logger().info(
                'SEQ IK: '
                f"L_q={[round(v, 3) for v in self.left_q]}, "
                f"R_q={[round(v, 3) for v in self.right_q]}, "
                f"L_err(e/w)=({left_dbg['elbow_error']:.3f}/{left_dbg['wrist_error']:.3f}), "
                f"R_err(e/w)=({right_dbg['elbow_error']:.3f}/{right_dbg['wrist_error']:.3f})"
            )

        self.publish_command_and_joint_states(
            smoothed,
            valid=valid,
            source='sequential_ik_elbow_then_wrist',
            debug=debug,
        )

    def publish_command_and_joint_states(
        self,
        joint_values: Dict[str, float],
        valid: bool,
        source: str,
        debug: Optional[Dict[str, float]] = None,
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
            joint_values=joint_values,
            debug=debug,
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
