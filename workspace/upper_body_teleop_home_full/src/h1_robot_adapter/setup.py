from setuptools import setup

package_name = 'h1_robot_adapter'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zohan',
    maintainer_email='zohan@example.com',
    description='Adapter from upper_body command to Unitree H1 LowCmd',
    license='MIT',
    entry_points={
        'console_scripts': [
            'h1_geometric_retarget_node = h1_robot_adapter.h1_geometric_retarget_node:main',
            'h1_sdk2py_upper_body_bridge = h1_robot_adapter.h1_sdk2py_upper_body_bridge:main',
            'h1_unitree_style_arm_controller = h1_robot_adapter.h1_unitree_style_arm_controller:main',
            'h1_arm_adapter = h1_robot_adapter.h1_arm_adapter:main',
            'h1_mujoco_ik_adapter = h1_robot_adapter.h1_mujoco_ik_adapter:main',
            'arm_sdk_to_lowcmd_bridge = h1_robot_adapter.arm_sdk_to_lowcmd_bridge:main',
        ],
    },
)
