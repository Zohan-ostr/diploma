from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def robot_description():
    pkg = get_package_share_directory('humanoid_description_ros2')
    urdf = os.path.join(pkg, 'urdf', 'full_humanoid_upper_body.urdf')
    with open(urdf, 'r', encoding='utf-8') as f:
        return f.read()


def generate_launch_description():
    desc_pkg = get_package_share_directory('humanoid_description_ros2')
    rviz_cfg = os.path.join(desc_pkg, 'rviz', 'home.rviz')
    return LaunchDescription([
        Node(
            package='robot_state_publisher', executable='robot_state_publisher', output='screen',
            parameters=[{'robot_description': robot_description()}]
        ),
        Node(package='rviz2', executable='rviz2', output='screen', arguments=['-d', rviz_cfg]),
        Node(package='home_pipeline', executable='mock_pose_source', output='screen'),
        Node(package='home_pipeline', executable='retarget_node', output='screen', parameters=[{
            'smoothing_alpha': 0.25,
            'motion_scale': 1.0,
        }]),
    ])
