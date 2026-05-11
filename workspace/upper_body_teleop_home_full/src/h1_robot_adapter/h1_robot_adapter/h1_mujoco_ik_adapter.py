#!/usr/bin/env python3

import math
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from upper_body_msgs.msg import PoseLandmarks3D
from unitree_go.msg import LowCmd, LowState


Point = Tuple[float, float, float]

ARM_ACTUATOR_INDEX = {
    'right_shoulder_pitch_joint': 12,
    'right_shoulder_roll_joint': 13,
    'right_shoulder_yaw_joint': 14,
    'right_elbow_joint': 15,

    'left_shoulder_pitch_joint': 16,
    'left_shoulder_roll_joint': 17,
    'left_shoulder_yaw_joint': 18,
    'left_elbow_joint': 19,
}

SIDE_JOINTS = {
    'left': {
        'pitch': 'left_shoulder_pitch_joint',
        'roll': 'left_shoulder_roll_joint',
        'yaw': 'left_shoulder_yaw_joint',
        'elbow': 'left_elbow_joint',
    },
    'right': {
        'pitch': 'right_shoulder_pitch_joint',
        'roll': 'right_shoulder_roll_joint',
        'yaw': 'right_shoulder_yaw_joint',
        'elbow': 'right_elbow_joint',
    },
}

SHOULDER_BODY = {
    'left': 'left_shoulder_pitch_link',
    'right': 'right_shoulder_pitch_link',
}

DISTAL_BODY = {
    'left': 'left_elbow_link_ball_hand',
    'right': 'right_elbow_link_ball_hand',
}

LANDMARKS = [
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        if fallback is None:
            return np.zeros(3, dtype=float)
        return fallback.copy()
    return v / n


class H1MujocoIKAdapter(Node):
    """
    Stable MuJoCo-specific retargeting for official Unitree H1 model.

    Main ideas:
      - Do not copy raw MediaPipe coordinates.
      - Build a human torso-local frame.
      - Retarget directions of upper arm and forearm to H1 link lengths.
      - Solve IK sequentially:
          1) shoulder -> elbow target
          2) elbow -> distal/hand target
      - Keep the wrist-below-midpoint yaw flip rule, but with hysteresis.
      - Smooth landmarks, targets, and joints.
      - Rate-limit all joint commands.
    """

    def __init__(self):
        super().__init__('h1_mujoco_ik_adapter')

        self.model_xml = str(self.declare_parameter(
            'model_xml',
            '/workspace/external/unitree_mujoco/unitree_robots/h1/scene.xml'
        ).value)

        self.output_topic = str(self.declare_parameter('output_topic', '/arm_sdk').value)

        self.kp = float(self.declare_parameter('kp', 12.0).value)
        self.kd = float(self.declare_parameter('kd', 1.2).value)

        self.min_visibility = float(self.declare_parameter('min_visibility', 0.12).value)

        # Временно глубину от MediaPipe почти не используем.
        # Для одной вебкамеры z слишком шумный.
        self.map_x_from_z = float(self.declare_parameter('map_x_from_z', 0.0).value)
        self.map_y_from_x = float(self.declare_parameter('map_y_from_x', -1.0).value)
        self.map_z_from_y = float(self.declare_parameter('map_z_from_y', -1.0).value)

        # Масштабы движения в локальной системе тела.
        self.motion_scale = float(self.declare_parameter('motion_scale', 1.0).value)
        self.upper_dir_scale = float(self.declare_parameter('upper_dir_scale', 1.0).value)
        self.fore_dir_scale = float(self.declare_parameter('fore_dir_scale', 1.0).value)

        # Фильтрация.
        self.landmark_alpha = float(self.declare_parameter('landmark_alpha', 0.25).value)
        self.target_alpha = float(self.declare_parameter('target_alpha', 0.18).value)
        self.joint_alpha = float(self.declare_parameter('joint_alpha', 0.18).value)

        # Ограничение скорости суставов на один кадр.
        self.max_joint_step = float(self.declare_parameter('max_joint_step', 0.030).value)
        self.max_yaw_step = float(self.declare_parameter('max_yaw_step', 0.020).value)
        self.max_elbow_step = float(self.declare_parameter('max_elbow_step', 0.035).value)

        # Yaw flip: оставляем твоё правило, но с гистерезисом.
        self.yaw_flip_angle = float(self.declare_parameter('yaw_flip_angle', math.pi).value)
        self.yaw_hysteresis = float(self.declare_parameter('yaw_hysteresis', 0.035).value)

        # Левый yaw разворачивался не туда — по умолчанию меняем знак.
        self.left_yaw_flip_sign = float(self.declare_parameter('left_yaw_flip_sign', -1.0).value)
        self.right_yaw_flip_sign = float(self.declare_parameter('right_yaw_flip_sign', 1.0).value)

        # Если локоть будет гнуться наоборот, эти параметры меняются при запуске.
        self.left_elbow_sign = float(self.declare_parameter('left_elbow_sign', 1.0).value)
        self.right_elbow_sign = float(self.declare_parameter('right_elbow_sign', 1.0).value)

        self.publish_enabled = bool(self.declare_parameter('publish_enabled', True).value)
        self.dry_run = bool(self.declare_parameter('dry_run', False).value)

        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)

        self.joint_id: Dict[str, int] = {}
        self.joint_qposadr: Dict[str, int] = {}
        self.joint_dofadr: Dict[str, int] = {}
        self.joint_limits: Dict[str, Tuple[float, float]] = {}

        for joint in ARM_ACTUATOR_INDEX:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                raise RuntimeError(f'Joint not found in MuJoCo model: {joint}')

            self.joint_id[joint] = int(jid)
            self.joint_qposadr[joint] = int(self.model.jnt_qposadr[jid])
            self.joint_dofadr[joint] = int(self.model.jnt_dofadr[jid])

            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                self.joint_limits[joint] = (float(lo), float(hi))
            else:
                self.joint_limits[joint] = (-math.pi, math.pi)

        self.body_id = {}
        for side in ('left', 'right'):
            sbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY[side])
            dbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, DISTAL_BODY[side])

            if sbid < 0:
                raise RuntimeError(f'Shoulder body not found: {SHOULDER_BODY[side]}')
            if dbid < 0:
                raise RuntimeError(f'Distal body not found: {DISTAL_BODY[side]}')

            self.body_id[(side, 'shoulder')] = int(sbid)
            self.body_id[(side, 'distal')] = int(dbid)

        qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(PoseLandmarks3D, '/pose/landmarks', self.on_pose, 10)
        self.create_subscription(LowState, '/lowstate', self.on_lowstate, qos_best_effort)
        self.create_subscription(String, '/teleop/control', self.on_control, 10)

        self.pub = self.create_publisher(LowCmd, self.output_topic, qos_best_effort)

        self.robot_q_current: Optional[Dict[str, float]] = None
        self.robot_q_zero: Optional[Dict[str, float]] = None
        self.q_cmd: Dict[str, float] = {j: 0.0 for j in ARM_ACTUATOR_INDEX}

        self.human_zero = None
        self.last_features = None
        self.active = False

        self.filtered_pts: Dict[str, np.ndarray] = {}
        self.filtered_targets = {
            'left': {'elbow': None, 'distal': None},
            'right': {'elbow': None, 'distal': None},
        }

        self.yaw_flip_state = {
            'left': False,
            'right': False,
        }

        self.robot_zero = {
            'shoulder_pos': {},
            'elbow_pos': {},
            'distal_pos': {},
            'upper_len': {},
            'fore_len': {},
        }

        self.counter = 0

        self.get_logger().info('H1 MuJoCo stable retargeting IK adapter started')
        self.get_logger().info(f'Model: {self.model_xml}')
        self.get_logger().info(f'Output: {self.output_topic}')
        self.get_logger().info('map_x_from_z is intentionally 0.0 by default for monocular stability')
        self.get_logger().info('Left yaw flip sign defaults to -1.0')
        self.get_logger().info('Press C in camera window to calibrate, R to reset')

    def clamp_joint(self, joint: str, q: float) -> float:
        lo, hi = self.joint_limits[joint]
        return clamp(q, lo, hi)

    def update_model_q(self, q_dict: Dict[str, float]):
        for joint, q in q_dict.items():
            if joint in self.joint_qposadr:
                self.data.qpos[self.joint_qposadr[joint]] = float(q)

    def rate_limit_joint(self, joint: str, target: float, current: float) -> float:
        if 'yaw' in joint:
            step = self.max_yaw_step
        elif 'elbow' in joint:
            step = self.max_elbow_step
        else:
            step = self.max_joint_step

        return current + clamp(target - current, -step, step)

    def smooth_joint(self, joint: str, target: float, current: float) -> float:
        limited = self.rate_limit_joint(joint, target, current)
        smoothed = current * (1.0 - self.joint_alpha) + limited * self.joint_alpha
        return self.clamp_joint(joint, smoothed)

    def read_robot_q_from_lowstate(self, msg: LowState):
        if not hasattr(msg, 'motor_state'):
            return None
        if len(msg.motor_state) <= max(ARM_ACTUATOR_INDEX.values()):
            return None

        q = {}
        for joint, idx in ARM_ACTUATOR_INDEX.items():
            q[joint] = float(msg.motor_state[idx].q)
        return q

    def capture_robot_geometry_zero(self):
        if self.robot_q_zero is None:
            return

        self.update_model_q(self.robot_q_zero)
        mujoco.mj_forward(self.model, self.data)

        for side in ('left', 'right'):
            joints = SIDE_JOINTS[side]
            elbow_joint = joints['elbow']

            shoulder_pos = np.array(self.data.xpos[self.body_id[(side, 'shoulder')]], dtype=float)
            elbow_pos = np.array(self.data.xanchor[self.joint_id[elbow_joint]], dtype=float)
            distal_pos = np.array(self.data.xpos[self.body_id[(side, 'distal')]], dtype=float)

            self.robot_zero['shoulder_pos'][side] = shoulder_pos
            self.robot_zero['elbow_pos'][side] = elbow_pos
            self.robot_zero['distal_pos'][side] = distal_pos
            self.robot_zero['upper_len'][side] = safe_norm(elbow_pos - shoulder_pos)
            self.robot_zero['fore_len'][side] = safe_norm(distal_pos - elbow_pos)

    def on_lowstate(self, msg: LowState):
        q = self.read_robot_q_from_lowstate(msg)
        if q is None:
            return

        self.robot_q_current = q

        if self.robot_q_zero is None:
            self.robot_q_zero = dict(q)
            self.q_cmd = dict(q)
            self.capture_robot_geometry_zero()

            self.get_logger().info('Captured H1 start q and geometry from /lowstate')
            for joint in [
                'right_shoulder_pitch_joint',
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint',
                'right_elbow_joint',
                'left_shoulder_pitch_joint',
                'left_shoulder_roll_joint',
                'left_shoulder_yaw_joint',
                'left_elbow_joint',
            ]:
                self.get_logger().info(f'  {joint:35s} q0={self.robot_q_zero[joint]:+.4f}')

    def parse_and_filter_pose(self, msg: PoseLandmarks3D):
        if not msg.valid:
            return None

        raw_pts = {}
        vis = {}

        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                raw_pts[name] = np.array([float(msg.x[i]), float(msg.y[i]), float(msg.z[i])], dtype=float)

            if i < len(msg.visibility):
                vis[name] = float(msg.visibility[i])
            else:
                vis[name] = 1.0

        for name in LANDMARKS:
            if name not in raw_pts:
                return None
            if vis.get(name, 0.0) < self.min_visibility:
                return None

        for name, p in raw_pts.items():
            if name not in self.filtered_pts:
                self.filtered_pts[name] = p
            else:
                self.filtered_pts[name] = (
                    self.filtered_pts[name] * (1.0 - self.landmark_alpha)
                    + p * self.landmark_alpha
                )

        pts = {name: self.filtered_pts[name].copy() for name in LANDMARKS}

        return self.build_human_features(pts)

    def build_torso_frame(self, pts: Dict[str, np.ndarray]):
        lsh = pts['left_shoulder']
        rsh = pts['right_shoulder']

        shoulder_mid = 0.5 * (lsh + rsh)

        # MediaPipe x вправо изображения.
        # Вектор right->left соответствует человеческой левой стороне.
        x_left = normalize(lsh - rsh, fallback=np.array([1.0, 0.0, 0.0]))

        # Не требуем hips: они часто имеют плохую visibility.
        # Берём "вниз экрана" и ортогонализируем относительно линии плеч.
        y_img_down = np.array([0.0, 1.0, 0.0], dtype=float)
        y_down = y_img_down - np.dot(y_img_down, x_left) * x_left
        y_down = normalize(y_down, fallback=np.array([0.0, 1.0, 0.0]))

        # Условная нормаль к плоскости туловища.
        z_forward = normalize(np.cross(x_left, y_down), fallback=np.array([0.0, 0.0, 1.0]))
        y_down = normalize(np.cross(z_forward, x_left), fallback=y_down)

        return shoulder_mid, x_left, y_down, z_forward

    def to_torso_local(self, p: np.ndarray, origin: np.ndarray, x_left: np.ndarray, y_down: np.ndarray, z_forward: np.ndarray):
        v = p - origin
        return np.array([
            float(np.dot(v, x_left)),
            float(np.dot(v, y_down)),
            float(np.dot(v, z_forward)),
        ], dtype=float)

    def human_local_to_robot_vec(self, v_local: np.ndarray) -> np.ndarray:
        # human local: [left, down, forward]
        left, down, forward = v_local

        # robot: [forward, left, up]
        return np.array([
            self.map_x_from_z * forward,
            self.map_y_from_x * left,
            self.map_z_from_y * down,
        ], dtype=float)

    def build_human_features(self, pts: Dict[str, np.ndarray]):
        origin, x_left, y_down, z_forward = self.build_torso_frame(pts)

        local = {}
        for name, p in pts.items():
            local[name] = self.to_torso_local(p, origin, x_left, y_down, z_forward)

        features = {}

        for side in ('left', 'right'):
            shoulder = local[f'{side}_shoulder']
            elbow = local[f'{side}_elbow']
            wrist = local[f'{side}_wrist']

            upper_vec = elbow - shoulder
            fore_vec = wrist - elbow

            upper_dir = normalize(upper_vec)
            fore_dir = normalize(fore_vec)

            # Правило yaw в исходных MediaPipe-координатах:
            # y направлена вниз, значит wrist_y > mid_y означает "кисть ниже".
            shoulder_raw = pts[f'{side}_shoulder']
            elbow_raw = pts[f'{side}_elbow']
            wrist_raw = pts[f'{side}_wrist']

            mid_y = 0.5 * (shoulder_raw[1] + elbow_raw[1])
            diff = wrist_raw[1] - mid_y

            features[side] = {
                'shoulder_local': shoulder,
                'elbow_local': elbow,
                'wrist_local': wrist,
                'upper_dir': upper_dir,
                'fore_dir': fore_dir,
                'wrist_mid_diff': float(diff),
            }

        return features

    def on_control(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd in ('calibrate', 'c'):
            if self.last_features is None:
                self.get_logger().warn('Cannot calibrate: no valid filtered MediaPipe pose yet')
                return

            if self.robot_q_current is None:
                self.get_logger().warn('Cannot calibrate: no /lowstate yet')
                return

            self.human_zero = self.last_features
            self.robot_q_zero = dict(self.robot_q_current)
            self.q_cmd = dict(self.robot_q_current)
            self.capture_robot_geometry_zero()

            for side in ('left', 'right'):
                self.filtered_targets[side]['elbow'] = None
                self.filtered_targets[side]['distal'] = None
                self.yaw_flip_state[side] = False

            self.active = True

            self.get_logger().info('IK calibration saved')
            self.get_logger().info('Stable retargeting is active')

        elif cmd in ('reset', 'r'):
            self.active = False
            self.human_zero = None

            if self.robot_q_zero is not None:
                self.q_cmd = dict(self.robot_q_zero)

            self.get_logger().info('IK reset. Press C to calibrate again')

    def update_yaw_state(self, side: str, wrist_mid_diff: float):
        state = self.yaw_flip_state[side]

        if not state and wrist_mid_diff > self.yaw_hysteresis:
            state = True
        elif state and wrist_mid_diff < -self.yaw_hysteresis:
            state = False

        self.yaw_flip_state[side] = state
        return state

    def yaw_target(self, side: str, features_side) -> float:
        joints = SIDE_JOINTS[side]
        yaw_joint = joints['yaw']

        base = self.robot_q_zero[yaw_joint]

        flip = self.update_yaw_state(side, features_side['wrist_mid_diff'])

        if flip:
            sign = self.left_yaw_flip_sign if side == 'left' else self.right_yaw_flip_sign
            q = base + sign * self.yaw_flip_angle
        else:
            q = base

        return self.clamp_joint(yaw_joint, q)

    def build_targets_for_side(self, side: str, features_side):
        shoulder0 = self.robot_zero['shoulder_pos'][side]
        elbow0 = self.robot_zero['elbow_pos'][side]
        distal0 = self.robot_zero['distal_pos'][side]

        upper_len = self.robot_zero['upper_len'][side]
        fore_len = self.robot_zero['fore_len'][side]

        # Направления человека в момент калибровки.
        zero_side = self.human_zero[side]

        upper_now_robot = normalize(self.human_local_to_robot_vec(features_side['upper_dir']))
        upper_zero_robot = normalize(self.human_local_to_robot_vec(zero_side['upper_dir']))

        fore_now_robot = normalize(self.human_local_to_robot_vec(features_side['fore_dir']))
        fore_zero_robot = normalize(self.human_local_to_robot_vec(zero_side['fore_dir']))

        # Базовые направления самой H1-модели в момент калибровки.
        h1_upper_zero_dir = normalize(elbow0 - shoulder0)
        h1_fore_zero_dir = normalize(distal0 - elbow0)

        # Главное исправление:
        # используем не абсолютное направление руки человека,
        # а изменение относительно калибровочной позы.
        upper_delta = upper_now_robot - upper_zero_robot
        fore_delta = fore_now_robot - fore_zero_robot

        target_upper_dir = normalize(
            h1_upper_zero_dir + self.motion_scale * self.upper_dir_scale * upper_delta,
            fallback=h1_upper_zero_dir,
        )

        target_fore_dir = normalize(
            h1_fore_zero_dir + self.motion_scale * self.fore_dir_scale * fore_delta,
            fallback=h1_fore_zero_dir,
        )

        target_elbow = shoulder0 + upper_len * target_upper_dir
        target_distal = target_elbow + fore_len * target_fore_dir

        # Сглаживаем цели.
        if self.filtered_targets[side]['elbow'] is None:
            self.filtered_targets[side]['elbow'] = target_elbow
        else:
            self.filtered_targets[side]['elbow'] = (
                self.filtered_targets[side]['elbow'] * (1.0 - self.target_alpha)
                + target_elbow * self.target_alpha
            )

        if self.filtered_targets[side]['distal'] is None:
            self.filtered_targets[side]['distal'] = target_distal
        else:
            self.filtered_targets[side]['distal'] = (
                self.filtered_targets[side]['distal'] * (1.0 - self.target_alpha)
                + target_distal * self.target_alpha
            )

        return self.filtered_targets[side]['elbow'], self.filtered_targets[side]['distal']

    def solve_shoulder_for_elbow(
        self,
        side: str,
        target_elbow_pos: np.ndarray,
        seed: Dict[str, float],
        yaw_q: float,
        iterations: int = 10,
    ) -> Dict[str, float]:

        joints = SIDE_JOINTS[side]
        pitch_joint = joints['pitch']
        roll_joint = joints['roll']
        yaw_joint = joints['yaw']
        elbow_joint = joints['elbow']

        q = dict(seed)
        q[yaw_joint] = yaw_q
        q[elbow_joint] = self.robot_q_zero[elbow_joint]

        elbow_jid = self.joint_id[elbow_joint]
        solve_joints = [pitch_joint, roll_joint]

        for _ in range(iterations):
            self.update_model_q(q)
            mujoco.mj_forward(self.model, self.data)

            current_elbow = np.array(self.data.xanchor[elbow_jid], dtype=float)
            err = target_elbow_pos - current_elbow

            if safe_norm(err) < 1e-4:
                break

            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)

            mujoco.mj_jac(
                self.model,
                self.data,
                jacp,
                jacr,
                current_elbow,
                self.body_id[(side, 'distal')],
            )

            cols = [self.joint_dofadr[j] for j in solve_joints]
            J = jacp[:, cols]

            damping = 1e-2
            A = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)

            step = 0.35

            for joint, dqi in zip(solve_joints, dq):
                q[joint] = self.clamp_joint(joint, q[joint] + step * float(dqi))

        return {
            pitch_joint: q[pitch_joint],
            roll_joint: q[roll_joint],
            yaw_joint: q[yaw_joint],
        }

    def solve_elbow_for_distal(
        self,
        side: str,
        target_distal_pos: np.ndarray,
        seed: Dict[str, float],
        iterations: int = 8,
    ) -> float:

        joints = SIDE_JOINTS[side]
        elbow_joint = joints['elbow']
        distal_bid = self.body_id[(side, 'distal')]

        q = dict(seed)
        elbow_sign = self.left_elbow_sign if side == 'left' else self.right_elbow_sign

        for _ in range(iterations):
            self.update_model_q(q)
            mujoco.mj_forward(self.model, self.data)

            current_distal = np.array(self.data.xpos[distal_bid], dtype=float)
            err = target_distal_pos - current_distal

            if safe_norm(err) < 1e-4:
                break

            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)

            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, distal_bid)

            col = self.joint_dofadr[elbow_joint]
            J = jacp[:, col]

            denom = float(np.dot(J, J) + 1e-3)
            dq = float(np.dot(J, err) / denom)

            step = 0.35
            q[elbow_joint] = self.clamp_joint(
                elbow_joint,
                q[elbow_joint] + elbow_sign * step * dq,
            )

        return q[elbow_joint]

    def build_lowcmd(self, q_cmd: Dict[str, float]) -> LowCmd:
        msg = LowCmd()

        for joint, q in q_cmd.items():
            if joint not in ARM_ACTUATOR_INDEX:
                continue

            idx = ARM_ACTUATOR_INDEX[joint]
            mc = msg.motor_cmd[idx]

            mc.mode = 1
            mc.q = float(q)
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self.kp)
            mc.kd = float(self.kd)

        return msg

    def on_pose(self, msg: PoseLandmarks3D):
        features = self.parse_and_filter_pose(msg)
        if features is None:
            return

        self.last_features = features

        if not self.active:
            return
        if self.human_zero is None or self.robot_q_zero is None:
            return

        q_target = dict(self.q_cmd)

        for side in ('left', 'right'):
            target_elbow, target_distal = self.build_targets_for_side(side, features[side])
            yaw_q_target = self.yaw_target(side, features[side])

            shoulder_solution = self.solve_shoulder_for_elbow(
                side=side,
                target_elbow_pos=target_elbow,
                seed=q_target,
                yaw_q=yaw_q_target,
            )

            for joint, target in shoulder_solution.items():
                q_target[joint] = target

            elbow_target = self.solve_elbow_for_distal(
                side=side,
                target_distal_pos=target_distal,
                seed=q_target,
            )

            q_target[SIDE_JOINTS[side]['elbow']] = elbow_target

        # Финальное сглаживание и rate limit.
        q_next = dict(self.q_cmd)

        for joint, target in q_target.items():
            q_next[joint] = self.smooth_joint(joint, target, self.q_cmd[joint])

        self.q_cmd = q_next

        self.counter += 1

        if self.counter % 20 == 0:
            print()
            print('===== H1 MUJOCO STABLE RETARGETING =====')
            for side in ('right', 'left'):
                f = features[side]
                print(
                    f'{side}: wrist_mid_diff={f["wrist_mid_diff"]:+.4f} '
                    f'yaw_flip={self.yaw_flip_state[side]}'
                )
                for key in ('pitch', 'roll', 'yaw', 'elbow'):
                    joint = SIDE_JOINTS[side][key]
                    print(f'  {joint:35s} q={self.q_cmd[joint]:+.4f}')

        if self.dry_run or not self.publish_enabled:
            return

        self.pub.publish(self.build_lowcmd(self.q_cmd))


def main():
    rclpy.init()
    node = H1MujocoIKAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
