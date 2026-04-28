from setuptools import setup
from glob import glob
import os

package_name = 'home_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zahar',
    maintainer_email='zahar@example.com',
    description='Home mode upper-body teleoperation pipeline.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mock_pose_source = home_pipeline.mock_pose_source:main',
            'webcam_mediapipe_node = home_pipeline.webcam_mediapipe_node:main',
            'retarget_node = home_pipeline.retarget_node:main',
        ],
    },
)
