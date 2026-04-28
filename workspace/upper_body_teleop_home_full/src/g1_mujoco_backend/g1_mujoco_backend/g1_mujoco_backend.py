import os
import time
from typing import Dict

import rclpy
from rclpy.node import Node
from upper_body_msgs.msg import UpperBodyCommand

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

# The names below are present in Unitree G1 29DoF XML in unitree_mujoco.
# If a joint is absent in a different G1 XML, it is skipped with a warning.
SUPPORTED_UPPER_JOINTS = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
]

class G1MujocoBackend(Node):
    def __init__(self):
        super().__init__('g1_mujoco_backend')
        if mujoco is None:
            raise RuntimeError(f'MuJoCo import failed: {MUJOCO_IMPORT_ERROR}')

        model_xml = str(self.declare_parameter('model_xml', '').value).strip()
        if not model_xml:
            model_xml = os.environ.get('G1_MUJOCO_XML', '/opt/unitree_mujoco/unitree_robots/g1/g1_29dof.xml')
        self.model_xml = model_xml
        if not os.path.exists(self.model_xml):
            raise FileNotFoundError(
                f'G1 MuJoCo XML not found: {self.model_xml}. '
                'Check UNITREE_MUJOCO_ROOT or clone https://github.com/unitreerobotics/unitree_mujoco'
            )

        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_addr: Dict[str, int] = {}
        for name in SUPPORTED_UPPER_JOINTS:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            except Exception:
                jid = -1
            if jid < 0:
                self.get_logger().warn(f'Joint not found in MuJoCo model, skipping: {name}')
                continue
            self.joint_qpos_addr[name] = int(self.model.jnt_qposadr[jid])

        self.latest_cmd: Dict[str, float] = {}
        self.create_subscription(UpperBodyCommand, '/upper_body/command', self.on_cmd, 10)
        self.get_logger().info(f'Loaded MuJoCo model: {self.model_xml}')
        self.get_logger().info(f'Controllable G1 upper joints found: {list(self.joint_qpos_addr.keys())}')
        self.get_logger().info('MuJoCo backend applies upper-body qpos overlay only. Lower body is not commanded by this node.')

    def on_cmd(self, msg: UpperBodyCommand):
        if not msg.valid:
            return
        for name, pos in zip(msg.joint_names, msg.position):
            if name in self.joint_qpos_addr:
                self.latest_cmd[name] = float(pos)

    def run_viewer_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last = time.time()
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                for name, pos in self.latest_cmd.items():
                    self.data.qpos[self.joint_qpos_addr[name]] = pos
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                now = time.time()
                sleep_time = max(0.0, (1.0 / 60.0) - (now - last))
                time.sleep(sleep_time)
                last = now


def main():
    rclpy.init()
    node = G1MujocoBackend()
    try:
        node.run_viewer_loop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
