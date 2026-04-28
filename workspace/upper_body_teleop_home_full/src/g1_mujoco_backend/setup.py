from setuptools import setup
package_name = 'g1_mujoco_backend'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zahar',
    maintainer_email='zahar@example.com',
    description='Placeholder for future Jetson + Unitree G1 MuJoCo backend.',
    license='MIT',
    entry_points={'console_scripts': ['g1_mujoco_backend_stub = g1_mujoco_backend.stub:main']},
)
