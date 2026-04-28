from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_index = LaunchConfiguration('camera_index')
    model_xml = LaunchConfiguration('model_xml')
    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('model_xml', default_value=''),
        Node(package='home_pipeline', executable='webcam_mediapipe_node', name='webcam_mediapipe_node', output='screen', parameters=[{'camera_index': camera_index, 'preview': True}]),
        Node(package='home_pipeline', executable='retarget_node', name='retarget_node', output='screen'),
        Node(package='g1_mujoco_backend', executable='g1_mujoco_backend', name='g1_mujoco_backend', output='screen', parameters=[{'model_xml': model_xml}]),
    ])
