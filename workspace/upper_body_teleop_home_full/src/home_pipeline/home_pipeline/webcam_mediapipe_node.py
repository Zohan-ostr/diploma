import math
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from upper_body_msgs.msg import PoseLandmarks3D

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = exc
else:
    MEDIAPIPE_IMPORT_ERROR = None


MP_NAMES = {
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    23: 'left_hip',
    24: 'right_hip',
}


class WebcamMediaPipeNode(Node):
    def __init__(self):
        super().__init__('webcam_mediapipe_node')

        self.camera_index = int(self.declare_parameter('camera_index', 0).value)
        self.preview = bool(self.declare_parameter('preview', True).value)
        self.width = int(self.declare_parameter('width', 640).value)
        self.height = int(self.declare_parameter('height', 480).value)
        self.fps = float(self.declare_parameter('fps', 30.0).value)

        if mp is None:
            raise RuntimeError(f'MediaPipe import failed: {MEDIAPIPE_IMPORT_ERROR}')

        self.pub = self.create_publisher(PoseLandmarks3D, '/pose/landmarks', 10)
        self.control_pub = self.create_publisher(String, '/teleop/control', 10)
        self.teleop_control_pub = self.create_publisher(String, '/teleop/control', 10)

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {self.camera_index}')

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.preview_enabled = self.preview

        self.calibration_countdown_active = False
        self.calibration_deadline_sec = 0.0
        self.calibration_duration_sec = 3.0

        self.last_key_time_sec = 0.0
        self.key_debounce_sec = 0.5

        self.timer = self.create_timer(1.0 / self.fps, self.tick)

        self.get_logger().info(
            f'Webcam MediaPipe node started: camera_index={self.camera_index}, preview={self.preview}'
        )
        self.get_logger().info(
            'Publishing pose_world_landmarks when available. C = countdown calibrate, R = reset, Q = close preview.'
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_control(self, command: str):
        msg = String()
        msg.data = command
        self.control_pub.publish(msg)
        self.get_logger().info(f'Published /teleop/control: {command}')

    def start_calibration_countdown(self):
        self.calibration_countdown_active = True
        self.calibration_deadline_sec = self.now_sec() + self.calibration_duration_sec
        self.get_logger().info('Calibration countdown started: 3 seconds')

    def cancel_calibration_countdown(self):
        if self.calibration_countdown_active:
            self.get_logger().info('Calibration countdown cancelled')
        self.calibration_countdown_active = False
        self.calibration_deadline_sec = 0.0

    def draw_countdown(self, frame):
        if not self.calibration_countdown_active:
            return

        remaining = self.calibration_deadline_sec - self.now_sec()

        if remaining <= 0.0:
            self.calibration_countdown_active = False
            self.publish_control('calibrate')

            cv2.putText(
                frame,
                'CALIBRATED',
                (170, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 255, 0),
                4
            )
            return

        shown_number = max(1, int(math.ceil(remaining)))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.putText(frame, 'Calibration in', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, str(shown_number), (290, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)
        cv2.putText(frame, 'Stand still in neutral pose', (80, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    def tick(self):
        ok, frame = self.cap.read()

        if not ok:
            self.get_logger().warn('Failed to read frame from camera', throttle_duration_sec=2.0)
            return

        raw_frame = frame.copy()

        # Не зеркалим кадр до MediaPipe, чтобы left/right не путались.
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        msg = PoseLandmarks3D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mediapipe_world'
        msg.valid = False

        preview_frame = raw_frame.copy()

        if result.pose_landmarks:
            names, xs, ys, zs, vis = [], [], [], [], []

            if result.pose_world_landmarks:
                landmarks_for_control = result.pose_world_landmarks.landmark
                frame_id = 'mediapipe_world'
                source_label = 'world3d'
            else:
                landmarks_for_control = result.pose_landmarks.landmark
                frame_id = 'mediapipe_image'
                source_label = 'image2d_fallback'

            msg.header.frame_id = frame_id

            for idx, name in MP_NAMES.items():
                lm = landmarks_for_control[idx]
                names.append(name)
                xs.append(float(lm.x))
                ys.append(float(lm.y))
                zs.append(float(lm.z))
                vis.append(float(getattr(lm, 'visibility', 1.0)))

            msg.names = names
            msg.x = xs
            msg.y = ys
            msg.z = zs
            msg.visibility = vis
            msg.valid = True
            self.pub.publish(msg)

            if self.preview_enabled:
                self.mp_draw.draw_landmarks(preview_frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                cv2.putText(
                    preview_frame,
                    f'Pose: valid [{source_label}] | C calibrate | R reset | Q close',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                try:
                    image_landmarks = result.pose_landmarks.landmark
                    lsh_v = image_landmarks[11].visibility
                    rsh_v = image_landmarks[12].visibility
                    lel_v = image_landmarks[13].visibility
                    rel_v = image_landmarks[14].visibility
                    lwr_v = image_landmarks[15].visibility
                    rwr_v = image_landmarks[16].visibility

                    cv2.putText(
                        preview_frame,
                        f'vis LS:{lsh_v:.2f} RS:{rsh_v:.2f} LE:{lel_v:.2f} RE:{rel_v:.2f} LW:{lwr_v:.2f} RW:{rwr_v:.2f}',
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2
                    )
                except Exception:
                    pass

        else:
            self.pub.publish(msg)

            if self.preview_enabled:
                cv2.putText(
                    preview_frame,
                    'Pose: no person | C calibrate | R reset | Q close',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

        if self.preview_enabled:
            self.draw_countdown(preview_frame)

            cv2.imshow('Home webcam + MediaPipe pose', preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('c'), ord('C')):
                self.publish_teleop_control('calibrate')
            elif key in (ord('r'), ord('R')):
                self.publish_teleop_control('reset')

            calibrate_keys = {ord('c'), ord('C'), 241, 209}
            reset_keys = {ord('r'), ord('R'), 234, 202}
            quit_keys = {ord('q'), ord('Q'), 233, 201}

            if key != 255:
                now = self.now_sec()

                if now - self.last_key_time_sec > self.key_debounce_sec:
                    self.last_key_time_sec = now
                    self.get_logger().info(f'Pressed key code: {key}')

                    if key in calibrate_keys:
                        self.start_calibration_countdown()

                    elif key in reset_keys:
                        self.cancel_calibration_countdown()
                        self.publish_control('reset')

                    elif key in quit_keys:
                        self.preview_enabled = False
                        self.cancel_calibration_countdown()
                        cv2.destroyWindow('Home webcam + MediaPipe pose')
                        self.get_logger().info('Preview window closed. ROS nodes continue running.')

    def destroy_node(self):
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

        super().destroy_node()



    def publish_teleop_control(self, command: str):
        msg = String()
        msg.data = command
        self.teleop_control_pub.publish(msg)
        try:
            self.get_logger().info(f"Published /teleop/control: {command}")
        except Exception:
            pass

def main():
    rclpy.init()
    node = WebcamMediaPipeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
