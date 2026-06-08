#!/usr/bin/env bash
set -e

source ~/diploma/workspace/upper_body_teleop_home_full/scripts/robot_h1/h1_robot_env.sh

export INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"
export SDK_CMD_TOPIC="${SDK_CMD_TOPIC:-rt/lowcmd}"
export SDK_STATE_TOPIC="${SDK_STATE_TOPIC:-rt/lowstate}"

# Как в официальном H1 low-level example: control_dt = 0.01
export CONTROL_HZ="${CONTROL_HZ:-100.0}"

# Официальные параметры для weak motors H1: плечи и локти
# ============================================================
# PD-РЕГУЛЯТОРЫ РЕАЛЬНОГО РОБОТА
# ============================================================
#
# НЕ МЕНЯТЬ для реального H1.
# Используются официальные параметры weak motors из H1 SDK2 example.
#
# Эксперименты с kp/kd допускаются только в симуляции.
# ============================================================
export KP_ARM="${KP_ARM:-60.0}"
export KD_ARM="${KD_ARM:-1.5}"

export COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-0.50}"

unset TEST_MODE
unset STEP_SCALE

python3 -m upper_body_teleop_runtime.h1_sdk2py_upper_body_sender
