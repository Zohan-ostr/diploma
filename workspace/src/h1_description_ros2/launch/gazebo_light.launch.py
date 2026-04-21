from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('h1_description_ros2')
    urdf_file = os.path.join(pkg_share, 'urdf', 'h1_gazebo_light.urdf')

    with open(urdf_file, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    for decl in [
        '<?xml version="1.0" ?>',
        '<?xml version="1.0"?>',
        '<?xml version="1.0" encoding="utf-8"?>',
        '<?xml version="1.0" encoding="utf-8" ?>',
    ]:
        robot_description = robot_description.replace(decl, '')

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'unitree_h1_light',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '1.2'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        rsp,
        TimerAction(period=3.0, actions=[spawn]),
    ])
