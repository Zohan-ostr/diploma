cmake_minimum_required(VERSION 3.10)
project(sdk2_h1_upper_body_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UNITREE_SDK2_DIR "$ENV{HOME}/WS_OZ/unitree_sdk2")
set(UNITREE_ARCH "x86_64")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(upper_body_msgs REQUIRED)

include_directories(
  /usr/local/include
  ${UNITREE_SDK2_DIR}/thirdparty/include
  ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
)

add_executable(ros_upper_body_udp_forwarder ros_upper_body_udp_forwarder.cpp)
ament_target_dependencies(
  ros_upper_body_udp_forwarder
  rclcpp
  upper_body_msgs
)

add_executable(sdk2_h1_udp_lowcmd_sender sdk2_h1_udp_lowcmd_sender.cpp)
target_link_libraries(sdk2_h1_udp_lowcmd_sender
  /usr/local/lib/libunitree_sdk2.a
  ${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH}/libddscxx.so
  ${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH}/libddsc.so
  pthread
  dl
)

set_target_properties(sdk2_h1_udp_lowcmd_sender PROPERTIES
  BUILD_RPATH "${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH};/usr/local/lib"
)

install(TARGETS
  ros_upper_body_udp_forwarder
  sdk2_h1_udp_lowcmd_sender
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
