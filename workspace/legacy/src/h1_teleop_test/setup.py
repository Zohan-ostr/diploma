from setuptools import setup

package_name = 'h1_teleop_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zahar',
    maintainer_email='zahar@example.com',
    description='Test teleoperation pipeline for Unitree H1 upper body',
    license='MIT',
    entry_points={
        'console_scripts': [
            'upper_body_cmd_pub = h1_teleop_test.upper_body_cmd_pub:main',
        ],
    },
)
