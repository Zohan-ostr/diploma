from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='home_pipeline',
            executable='realsense_mediapipe_node',
            name='realsense_mediapipe_node',
            output='screen',
            parameters=[{
                'width': 640,
                'height': 480,
                'fps': 30,
                'preview': True,
                'preview_mirror': True,
                'depth_window': 5,
                'min_depth_m': 0.15,
                'max_depth_m': 6.0,
            }],
        ),
    ])
