cmake_minimum_required(VERSION 3.8)
project(h1_description_ros2)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
