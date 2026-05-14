#!/usr/bin/python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from unitree_go.msg import LowCmd, LowState


H1_NUM_MOTORS = 20

# H1 indexing from Unitree H1 / unitree_go LowCmd:
# 0..11 lower body + waist/not-used/ankles
# 12..15 right arm
# 16..19 left arm
RIGHT_SHOULDER_PITCH = 12
RIGHT_SHOULDER_ROLL = 13
RIGHT_SHOULDER_YAW = 14
RIGHT_ELBOW = 15

LEFT_SHOULDER_PITCH = 16
LEFT_SHOULDER_ROLL = 17
LEFT_SHOULDER_YAW = 18
LEFT_ELBOW = 19

ARM_IDS = [
    RIGHT_SHOULDER_PITCH,
    RIGHT_SHOULDER_ROLL,
    RIGHT_SHOULDER_YAW,
    RIGHT_ELBOW,
    LEFT_SHOULDER_PITCH,
    LEFT_SHOULDER_ROLL,
    LEFT_SHOULDER_YAW,
    LEFT_ELBOW,
]

WEAK_IDS = set([
    10, 11,  # ankles
    RIGHT_SHOULDER_PITCH,
    RIGHT_SHOULDER_ROLL,
    RIGHT_SHOULDER_YAW,
    RIGHT_ELBOW,
    LEFT_SHOULDER_PITCH,
    LEFT_SHOULDER_ROLL,
    LEFT_SHOULDER_YAW,
    LEFT_ELBOW,
])


class H1ArmSdkFullLowcmdTest(Node):
    def __init__(self):
        super().__init__("h1_arm_sdk_full_lowcmd_test")

        self.declare_parameter("output_topic", "/arm_sdk")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("rate_hz", 250.0)
        self.declare_parameter("duration_sec", 12.0)
        self.declare_parameter("amplitude_rad", 0.10)
        self.declare_parameter("period_sec", 4.0)
        self.declare_parameter("kp_arm", 60.0)
        self.declare_parameter("kd_arm", 3.0)
        self.declare_parameter("kp_body", 300.0)
        self.declare_parameter("kd_body", 5.0)
        self.declare_parameter("test_motor_id", RIGHT_SHOULDER_PITCH)

        self.output_topic = self.get_parameter("output_topic").value
        self.lowstate_topic = self.get_parameter("lowstate_topic").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.amplitude_rad = float(self.get_parameter("amplitude_rad").value)
        self.period_sec = float(self.get_parameter("period_sec").value)
        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)
        self.kp_body = float(self.get_parameter("kp_body").value)
        self.kd_body = float(self.get_parameter("kd_body").value)
        self.test_motor_id = int(self.get_parameter("test_motor_id").value)

        # Unitree robot DDS endpoints often appear as bare DDS endpoints.
        # Keep QoS permissive for discovery.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.pub = self.create_publisher(LowCmd, self.output_topic, qos)
        self.sub = self.create_subscription(LowState, self.lowstate_topic, self.on_lowstate, qos)

        self.lowstate = None
        self.base_q = None
        self.start_time = None
        self.last_print = 0.0

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("H1 ARM SDK FULL LOWCMD TEST")
        self.get_logger().info(f"output_topic:   {self.output_topic}")
        self.get_logger().info(f"lowstate_topic: {self.lowstate_topic}")
        self.get_logger().info(f"rate_hz:        {self.rate_hz}")
        self.get_logger().info(f"duration_sec:   {self.duration_sec}")
        self.get_logger().info(f"amplitude_rad:  {self.amplitude_rad}")
        self.get_logger().info(f"period_sec:     {self.period_sec}")
        self.get_logger().info(f"kp_arm:         {self.kp_arm}")
        self.get_logger().info(f"kd_arm:         {self.kd_arm}")
        self.get_logger().info(f"test_motor_id:  {self.test_motor_id}")
        self.get_logger().info("============================================================")
        self.get_logger().info(f"Waiting for {self.lowstate_topic}...")

    def on_lowstate(self, msg):
        self.lowstate = msg

        if self.base_q is None:
            self.base_q = [float(msg.motor_state[i].q) for i in range(H1_NUM_MOTORS)]
            self.start_time = time.time()

            self.get_logger().info("lowstate OK")
            self.get_logger().info(
                "Initial arm q: "
                + str([round(self.base_q[i], 4) for i in ARM_IDS])
            )

    def make_cmd(self, target_q):
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

        for i in range(H1_NUM_MOTORS):
            c = msg.motor_cmd[i]

            # Unitree H1 style:
            # weak/arm motors in servo mode 0x01,
            # stronger body motors in lock mode 0x0A.
            if i in WEAK_IDS:
                c.mode = 0x01
                c.kp = self.kp_arm if i in ARM_IDS else 140.0
                c.kd = self.kd_arm if i in ARM_IDS else 3.0
            else:
                c.mode = 0x0A
                c.kp = self.kp_body
                c.kd = self.kd_body

            c.q = float(target_q[i])
            c.dq = 0.0
            c.tau = 0.0

        return msg

    def on_timer(self):
        if self.base_q is None or self.lowstate is None:
            return

        t = time.time() - self.start_time

        target_q = list(self.base_q)

        if t <= self.duration_sec:
            target_q[self.test_motor_id] = (
                self.base_q[self.test_motor_id]
                + self.amplitude_rad * math.sin(2.0 * math.pi * t / self.period_sec)
            )
        else:
            target_q[self.test_motor_id] = self.base_q[self.test_motor_id]

        self.pub.publish(self.make_cmd(target_q))

        if t - self.last_print > 0.5:
            current = float(self.lowstate.motor_state[self.test_motor_id].q)
            self.get_logger().info(
                f"t={t:.2f} target={target_q[self.test_motor_id]:.4f} current={current:.4f}"
            )
            self.last_print = t

        if t > self.duration_sec + 2.0:
            self.get_logger().info("Done.")
            raise SystemExit


def main():
    rclpy.init()
    node = H1ArmSdkFullLowcmdTest()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
