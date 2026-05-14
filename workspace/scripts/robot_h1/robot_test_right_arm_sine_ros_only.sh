#!/usr/bin/env bash
set -eo pipefail

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-0}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"

OUTPUT_TOPIC="${OUTPUT_TOPIC:-/arm_sdk}"
LOWSTATE_TOPIC="${LOWSTATE_TOPIC:-/lowstate}"

DURATION_SEC="${DURATION_SEC:-12.0}"
AMPLITUDE_RAD="${AMPLITUDE_RAD:-0.10}"
PERIOD_SEC="${PERIOD_SEC:-4.0}"
RATE_HZ="${RATE_HZ:-100.0}"

KP_ARM="${KP_ARM:-35.0}"
KD_ARM="${KD_ARM:-2.0}"

RIGHT_SHOULDER_PITCH_ID="${RIGHT_SHOULDER_PITCH_ID:-12}"

echo "============================================================"
echo " H1 ROS-ONLY RIGHT ARM SINE TEST"
echo "============================================================"
echo "ROS_DOMAIN_ID:             $ROS_DOMAIN_ID_VALUE"
echo "ROS_LOCALHOST_ONLY:        $ROS_LOCALHOST_ONLY_VALUE"
echo "OUTPUT_TOPIC:              $OUTPUT_TOPIC"
echo "LOWSTATE_TOPIC:            $LOWSTATE_TOPIC"
echo "DURATION_SEC:              $DURATION_SEC"
echo "AMPLITUDE_RAD:             $AMPLITUDE_RAD"
echo "PERIOD_SEC:                $PERIOD_SEC"
echo "RATE_HZ:                   $RATE_HZ"
echo "KP_ARM:                    $KP_ARM"
echo "KD_ARM:                    $KD_ARM"
echo "RIGHT_SHOULDER_PITCH_ID:   $RIGHT_SHOULDER_PITCH_ID"
echo "============================================================"
echo
echo "Это отправляет реальные команды в $OUTPUT_TOPIC."
read -r -p "Type YES to start: " CONFIRM
if [ "$CONFIRM" != "YES" ]; then
  echo "Cancelled."
  exit 1
fi

export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"
export ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE"

if [ -f /opt/ros/foxy/setup.bash ]; then
  source /opt/ros/foxy/setup.bash
fi

if [ -f /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash ]; then
  source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
else
  echo "ERROR: Unitree ROS2 workspace not found: /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash"
  exit 1
fi


/usr/bin/python3 - <<PY
import math
import os
import time

import rclpy
from rclpy.node import Node

from unitree_go.msg import LowCmd, LowState


OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "$OUTPUT_TOPIC")
LOWSTATE_TOPIC = os.environ.get("LOWSTATE_TOPIC", "$LOWSTATE_TOPIC")

DURATION_SEC = float(os.environ.get("DURATION_SEC", "$DURATION_SEC"))
AMPLITUDE_RAD = float(os.environ.get("AMPLITUDE_RAD", "$AMPLITUDE_RAD"))
PERIOD_SEC = float(os.environ.get("PERIOD_SEC", "$PERIOD_SEC"))
RATE_HZ = float(os.environ.get("RATE_HZ", "$RATE_HZ"))

KP_ARM = float(os.environ.get("KP_ARM", "$KP_ARM"))
KD_ARM = float(os.environ.get("KD_ARM", "$KD_ARM"))
RIGHT_SHOULDER_PITCH_ID = int(os.environ.get("RIGHT_SHOULDER_PITCH_ID", "$RIGHT_SHOULDER_PITCH_ID"))

ARM_IDS = [12, 13, 14, 15, 16, 17, 18, 19]


class TestNode(Node):
    def __init__(self):
        super().__init__("h1_ros_only_right_arm_sine_test")
        self.pub = self.create_publisher(LowCmd, OUTPUT_TOPIC, 10)
        self.sub = self.create_subscription(LowState, LOWSTATE_TOPIC, self.on_lowstate, 10)
        self.lowstate = None
        self.base_q = None
        self.start_time = None
        self.last_print = 0.0

        self.timer = self.create_timer(1.0 / RATE_HZ, self.on_timer)

        self.get_logger().info(f"Waiting for {LOWSTATE_TOPIC}...")

    def on_lowstate(self, msg):
        self.lowstate = msg
        if self.base_q is None:
            self.base_q = [float(msg.motor_state[i].q) for i in range(20)]
            self.start_time = time.time()
            self.get_logger().info("lowstate OK")
            self.get_logger().info("Initial arm q: " + str([round(self.base_q[i], 4) for i in ARM_IDS]))

    def make_msg(self, target_q):
        msg = LowCmd()
        try:
            msg.head[0] = 0xFE
            msg.head[1] = 0xEF
        except Exception:
            pass

        if hasattr(msg, "level_flag"):
            msg.level_flag = 0xFF
        if hasattr(msg, "gpio"):
            msg.gpio = 0

        for i in ARM_IDS:
            msg.motor_cmd[i].mode = 0x01
            msg.motor_cmd[i].q = float(target_q[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = KP_ARM
            msg.motor_cmd[i].kd = KD_ARM

        return msg

    def on_timer(self):
        if self.base_q is None:
            return

        t = time.time() - self.start_time
        target_q = list(self.base_q)

        if t <= DURATION_SEC:
            target_q[RIGHT_SHOULDER_PITCH_ID] = (
                self.base_q[RIGHT_SHOULDER_PITCH_ID]
                + AMPLITUDE_RAD * math.sin(2.0 * math.pi * t / PERIOD_SEC)
            )
        else:
            target_q[RIGHT_SHOULDER_PITCH_ID] = self.base_q[RIGHT_SHOULDER_PITCH_ID]

        self.pub.publish(self.make_msg(target_q))

        if t - self.last_print > 0.5:
            cur = None
            if self.lowstate is not None:
                cur = self.lowstate.motor_state[RIGHT_SHOULDER_PITCH_ID].q
            self.get_logger().info(
                f"t={t:.2f} target={target_q[RIGHT_SHOULDER_PITCH_ID]:.4f} current={cur}"
            )
            self.last_print = t

        if t > DURATION_SEC + 2.0:
            self.get_logger().info("Done.")
            raise SystemExit


rclpy.init()
node = TestNode()

try:
    rclpy.spin(node)
except SystemExit:
    pass
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
