from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    camera_id = LaunchConfiguration('camera_id')

    desc_pkg = get_package_share_directory('humanoid_description_ros2')
    urdf_path = os.path.join(desc_pkg, 'urdf', 'full_humanoid_upper_body.urdf')
    rviz_path = os.path.join(desc_pkg, 'rviz', 'home.rviz')

    with open(urdf_path, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='OpenCV camera index'
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'publish_frequency': 60.0,
            }],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_path],
        ),

        Node(
            package='home_pipeline',
            executable='webcam_mediapipe_node',
            name='webcam_mediapipe_node',
            output='screen',
            parameters=[{
                'camera_index': camera_id,
                'preview': True,
                'width': 640,
                'height': 480,
                'fps': 30.0,
            }],
        ),

        Node(
            package='home_pipeline',
            executable='retarget_node',
            name='retarget_node',
            output='screen',
        ),

        Node(
            package='home_pipeline',
            executable='joint_slider_panel',
            name='joint_slider_panel',
            output='screen',
        ),
    ])
