import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


JOINTS = [
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


LABELS = {
    'torso_joint': 'torso',
    'neck_joint': 'neck',
    'head_joint': 'head',

    'left_shoulder_pitch_joint': 'L sh pitch',
    'left_shoulder_roll_joint': 'L sh roll',
    'left_shoulder_yaw_joint': 'L sh yaw',
    'left_elbow_joint': 'L elbow',

    'right_shoulder_pitch_joint': 'R sh pitch',
    'right_shoulder_roll_joint': 'R sh roll',
    'right_shoulder_yaw_joint': 'R sh yaw',
    'right_elbow_joint': 'R elbow',

    'left_hip_joint': 'L hip',
    'left_knee_joint': 'L knee',
    'left_ankle_joint': 'L ankle',

    'right_hip_joint': 'R hip',
    'right_knee_joint': 'R knee',
    'right_ankle_joint': 'R ankle',
}


# Отладочные диапазоны. Они специально широкие.
# Если сустав ведёт себя опасно/странно — уменьшим диапазон.
LIMITS = {
    'torso_joint': (-1.0, 1.0),
    'neck_joint': (-1.0, 1.0),
    'head_joint': (-1.0, 1.0),

    'left_shoulder_pitch_joint': (-2.0, 2.0),
    'left_shoulder_roll_joint': (-2.0, 2.0),
    'left_shoulder_yaw_joint': (-2.0, 2.0),
    'left_elbow_joint': (-2.2, 2.2),

    'right_shoulder_pitch_joint': (-2.0, 2.0),
    'right_shoulder_roll_joint': (-2.0, 2.0),
    'right_shoulder_yaw_joint': (-2.0, 2.0),
    'right_elbow_joint': (-2.2, 2.2),

    'left_hip_joint': (-1.0, 1.0),
    'left_knee_joint': (-1.0, 1.0),
    'left_ankle_joint': (-1.0, 1.0),

    'right_hip_joint': (-1.0, 1.0),
    'right_knee_joint': (-1.0, 1.0),
    'right_ankle_joint': (-1.0, 1.0),
}


ZERO = {
    'torso_joint': 0.0,
    'neck_joint': 0.0,
    'head_joint': 0.0,

    'left_shoulder_pitch_joint': 0.0,
    'left_shoulder_roll_joint': 0.0,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 0.2,

    'right_shoulder_pitch_joint': 0.0,
    'right_shoulder_roll_joint': 0.0,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 0.2,

    'left_hip_joint': 0.0,
    'left_knee_joint': 0.0,
    'left_ankle_joint': 0.0,

    'right_hip_joint': 0.0,
    'right_knee_joint': 0.0,
    'right_ankle_joint': 0.0,
}


SLIDER_MAX = 1000


def value_to_slider(joint: str, value: float) -> int:
    lo, hi = LIMITS[joint]
    value = max(lo, min(hi, value))
    return int((value - lo) / (hi - lo) * SLIDER_MAX)


def slider_to_value(joint: str, slider: int) -> float:
    lo, hi = LIMITS[joint]
    return lo + (slider / SLIDER_MAX) * (hi - lo)


class JointSliderPanel(Node):
    def __init__(self):
        super().__init__('joint_slider_panel')

        self.window = 'Joint debug sliders'
        self.manual_enabled = False

        self.pub = self.create_publisher(JointState, '/debug/manual_joint_states', 10)
        self.enabled_pub = self.create_publisher(Bool, '/debug/manual_enabled', 10)

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 620, 820)

        cv2.createTrackbar('manual_mode 0/1', self.window, 0, 1, self.on_manual_changed)

        for joint in JOINTS:
            cv2.createTrackbar(
                LABELS[joint],
                self.window,
                value_to_slider(joint, ZERO[joint]),
                SLIDER_MAX,
                lambda value: None
            )

        self.timer = self.create_timer(0.05, self.tick)

        self.get_logger().info('Joint slider panel started.')
        self.get_logger().info('Set manual_mode=1 to override MediaPipe retargeting.')
        self.get_logger().info('Set manual_mode=0 to return to camera retargeting.')
        self.get_logger().info('Press R in slider window to reset sliders to neutral.')
        self.get_logger().info('Press Q in slider window to close slider window.')

    def on_manual_changed(self, value: int):
        self.manual_enabled = bool(value)
        self.publish_enabled()
        self.get_logger().info(f'manual_mode={int(self.manual_enabled)}')

    def publish_enabled(self):
        msg = Bool()
        msg.data = self.manual_enabled
        self.enabled_pub.publish(msg)

    def reset_sliders(self):
        for joint in JOINTS:
            cv2.setTrackbarPos(LABELS[joint], self.window, value_to_slider(joint, ZERO[joint]))
        self.get_logger().info('Joint sliders reset to neutral.')

    def read_joint_values(self):
        values = {}
        for joint in JOINTS:
            slider = cv2.getTrackbarPos(LABELS[joint], self.window)
            values[joint] = slider_to_value(joint, slider)
        return values

    def tick(self):
        self.publish_enabled()

        values = self.read_joint_values()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [values[j] for j in JOINTS]
        self.pub.publish(msg)

        canvas = 255 * cv2.UMat(220, 600, cv2.CV_8UC3).get()
        mode = 'MANUAL OVERRIDE ON' if self.manual_enabled else 'camera retargeting mode'
        color = (0, 170, 0) if self.manual_enabled else (0, 0, 180)

        cv2.putText(canvas, mode, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(canvas, 'manual_mode=1: sliders control RViz robot', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(canvas, 'manual_mode=0: MediaPipe retarget controls robot', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(canvas, 'R: reset sliders | Q: close this window', (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(canvas, 'Use this to discover real URDF joint axes.', (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        cv2.imshow(self.window, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('r'), ord('R'), 234, 202):
            self.reset_sliders()

        elif key in (ord('q'), ord('Q'), 233, 201):
            cv2.destroyWindow(self.window)
            self.get_logger().info('Joint slider window closed. Node continues publishing.')

    def destroy_node(self):
        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = JointSliderPanel()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
