#!/usr/bin/env python3
import math
from enum import IntEnum
from typing import Dict, List

import numpy as np


class H1MotorIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2

    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5

    kWaistYaw = 6

    kLeftHipYaw = 7
    kRightHipYaw = 8

    kNotUsedJoint = 9

    kLeftAnkle = 10
    kRightAnkle = 11

    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19


UPPER_BODY_JOINTS: List[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

JOINT_TO_MOTOR: Dict[str, int] = {
    "left_shoulder_pitch_joint": int(H1MotorIndex.kLeftShoulderPitch),
    "left_shoulder_roll_joint": int(H1MotorIndex.kLeftShoulderRoll),
    "left_shoulder_yaw_joint": int(H1MotorIndex.kLeftShoulderYaw),
    "left_elbow_joint": int(H1MotorIndex.kLeftElbow),

    "right_shoulder_pitch_joint": int(H1MotorIndex.kRightShoulderPitch),
    "right_shoulder_roll_joint": int(H1MotorIndex.kRightShoulderRoll),
    "right_shoulder_yaw_joint": int(H1MotorIndex.kRightShoulderYaw),
    "right_elbow_joint": int(H1MotorIndex.kRightElbow),

    "left_shoulder_pitch": int(H1MotorIndex.kLeftShoulderPitch),
    "left_shoulder_roll": int(H1MotorIndex.kLeftShoulderRoll),
    "left_shoulder_yaw": int(H1MotorIndex.kLeftShoulderYaw),
    "left_elbow": int(H1MotorIndex.kLeftElbow),

    "right_shoulder_pitch": int(H1MotorIndex.kRightShoulderPitch),
    "right_shoulder_roll": int(H1MotorIndex.kRightShoulderRoll),
    "right_shoulder_yaw": int(H1MotorIndex.kRightShoulderYaw),
    "right_elbow": int(H1MotorIndex.kRightElbow),
}

ARM_MOTOR_IDS = [
    int(H1MotorIndex.kRightShoulderPitch),
    int(H1MotorIndex.kRightShoulderRoll),
    int(H1MotorIndex.kRightShoulderYaw),
    int(H1MotorIndex.kRightElbow),
    int(H1MotorIndex.kLeftShoulderPitch),
    int(H1MotorIndex.kLeftShoulderRoll),
    int(H1MotorIndex.kLeftShoulderYaw),
    int(H1MotorIndex.kLeftElbow),
]

# Лимиты взяты в порядке UPPER_BODY_JOINTS.
# Они согласованы с текущей рабочей симуляционной реализацией.
# ============================================================
# SAFE WORKSPACE LIMITS FOR REAL H1 UPPER BODY
# Порядок:
# left_pitch, left_roll, left_yaw, left_elbow,
# right_pitch, right_roll, right_yaw, right_elbow
#
# Это рабочий безопасный коридор, а не полные механические лимиты.
# Он шире предыдущего, чтобы робот мог поднимать руки,
# но всё ещё не даёт рукам уходить внутрь корпуса.
# ============================================================

LOWER_8 = np.array([
    -1.80,   0.00,  -2.20,   0.95,
    -1.80,  -1.95,  -2.20,   0.95,
], dtype=float)

UPPER_8 = np.array([
     1.80,   1.95,   2.20,   2.45,
     1.80,   0.00,   2.20,   2.45,
], dtype=float)

TPOSE_8 = np.array([
    0.00,  1.45, -1.30, 1.57,
    0.00, -1.45,  1.30, 1.57,
], dtype=float)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def finite_array(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))


def clamp8(q: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, LOWER_8), UPPER_8)


def rate_limit8(target: np.ndarray, prev: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    return prev + np.clip(target - prev, -max_step, max_step)


def parse_q8_from_upper_body_msg(msg) -> np.ndarray:
    """
    Возвращает q8 в едином порядке:
    [
      left_shoulder_pitch_joint,
      left_shoulder_roll_joint,
      left_shoulder_yaw_joint,
      left_elbow_joint,
      right_shoulder_pitch_joint,
      right_shoulder_roll_joint,
      right_shoulder_yaw_joint,
      right_elbow_joint,
    ]

    Поддерживает два формата:
    1) *_joint names в правильном порядке;
    2) retarget из симуляции: right_* затем left_* без суффикса _joint.
    """
    if len(msg.position) < 8:
        raise ValueError("UpperBodyCommand.position has less than 8 elements")

    if len(msg.joint_names) == len(msg.position) and len(msg.joint_names) > 0:
        by_name = {str(n): float(v) for n, v in zip(msg.joint_names, msg.position)}

        aliases = {
            "left_shoulder_pitch_joint": ["left_shoulder_pitch_joint", "left_shoulder_pitch"],
            "left_shoulder_roll_joint": ["left_shoulder_roll_joint", "left_shoulder_roll"],
            "left_shoulder_yaw_joint": ["left_shoulder_yaw_joint", "left_shoulder_yaw"],
            "left_elbow_joint": ["left_elbow_joint", "left_elbow"],

            "right_shoulder_pitch_joint": ["right_shoulder_pitch_joint", "right_shoulder_pitch"],
            "right_shoulder_roll_joint": ["right_shoulder_roll_joint", "right_shoulder_roll"],
            "right_shoulder_yaw_joint": ["right_shoulder_yaw_joint", "right_shoulder_yaw"],
            "right_elbow_joint": ["right_elbow_joint", "right_elbow"],
        }

        out = []
        missing = []

        for canonical in UPPER_BODY_JOINTS:
            value = None
            for a in aliases[canonical]:
                if a in by_name:
                    value = by_name[a]
                    break
            if value is None:
                missing.append(canonical)
            else:
                out.append(value)

        if missing:
            raise ValueError(f"UpperBodyCommand missing joints/aliases: {missing}; got={list(by_name.keys())}")

        return np.array(out, dtype=float)

    # Без имен считаем, что сообщение уже в canonical left-first порядке.
    return np.array(list(msg.position[:8]), dtype=float)


def q8_to_motor_targets(q8: np.ndarray, motor_q: np.ndarray) -> np.ndarray:
    out = np.array(motor_q, dtype=float).copy()

    out[int(H1MotorIndex.kLeftShoulderPitch)] = q8[0]
    out[int(H1MotorIndex.kLeftShoulderRoll)] = q8[1]
    out[int(H1MotorIndex.kLeftShoulderYaw)] = q8[2]
    out[int(H1MotorIndex.kLeftElbow)] = q8[3]

    out[int(H1MotorIndex.kRightShoulderPitch)] = q8[4]
    out[int(H1MotorIndex.kRightShoulderRoll)] = q8[5]
    out[int(H1MotorIndex.kRightShoulderYaw)] = q8[6]
    out[int(H1MotorIndex.kRightElbow)] = q8[7]

    return out


def motor_arm_q_to_q8(motor_q: np.ndarray) -> np.ndarray:
    return np.array([
        motor_q[int(H1MotorIndex.kLeftShoulderPitch)],
        motor_q[int(H1MotorIndex.kLeftShoulderRoll)],
        motor_q[int(H1MotorIndex.kLeftShoulderYaw)],
        motor_q[int(H1MotorIndex.kLeftElbow)],
        motor_q[int(H1MotorIndex.kRightShoulderPitch)],
        motor_q[int(H1MotorIndex.kRightShoulderRoll)],
        motor_q[int(H1MotorIndex.kRightShoulderYaw)],
        motor_q[int(H1MotorIndex.kRightElbow)],
    ], dtype=float)


def default_step_limits(control_hz: float) -> np.ndarray:
    dt = 1.0 / max(1.0, float(control_hz))

    # Средняя скорость между быстрым симуляционным вариантом
    # и последним слишком медленным вариантом.
    #
    # При 100 Hz:
    # normal = 0.021 rad/step -> 2.1 rad/s
    # yaw    = 0.024 rad/step -> 2.4 rad/s
    # elbow  = 0.026 rad/step -> 2.6 rad/s
    normal = 2.10 * dt
    yaw = 2.40 * dt
    elbow = 2.60 * dt

    return np.array([
        normal, normal, yaw, elbow,
        normal, normal, yaw, elbow,
    ], dtype=float)



def apply_self_collision_guard(q8: np.ndarray) -> np.ndarray:
    """
    Программный guard от самостолкновений.

    Важно:
    - здесь НЕ пересчитывается IK;
    - здесь НЕ меняется логика FABRIK/segment yaw;
    - здесь только отсекаются опасные команды перед rt/lowcmd.

    Порядок q:
    left_pitch, left_roll, left_yaw, left_elbow,
    right_pitch, right_roll, right_yaw, right_elbow.
    """
    q = clamp8(np.array(q8, dtype=float).copy())

    left_roll = q[1]
    right_roll = q[5]

    # Базовая защита от ухода рук внутрь корпуса.
    q[1] = max(q[1], 0.00)    # left_roll не должен уходить через корпус
    q[5] = min(q[5], 0.00)    # right_roll не должен уходить через корпус

    # Если рука почти прижата к корпусу, ограничиваем yaw сильнее,
    # чтобы локоть не разворачивался внутрь робота.
    # Когда рука уже отведена наружу, yaw снова получает широкий диапазон.
    if left_roll < 0.20:
        q[2] = clamp(q[2], -1.60, 0.90)

    if right_roll > -0.20:
        q[6] = clamp(q[6], -0.90, 1.60)

    # Локоть: оставляем широкий, но безопасный коридор.
    q[3] = clamp(q[3], 0.95, 2.45)
    q[7] = clamp(q[7], 0.95, 2.45)

    return clamp8(q)


def format_q8(q8: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x): .3f}" for x in q8) + "]"
