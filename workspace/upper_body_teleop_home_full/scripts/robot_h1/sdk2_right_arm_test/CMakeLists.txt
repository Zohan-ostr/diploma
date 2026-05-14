cmake_minimum_required(VERSION 3.10)
project(h1_sdk2_right_arm_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UNITREE_SDK2_DIR "$ENV{HOME}/WS_OZ/unitree_sdk2")
set(UNITREE_ARCH "x86_64")

include_directories(
  /usr/local/include
  ${UNITREE_SDK2_DIR}/thirdparty/include
  ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
)

add_executable(h1_sdk2_right_arm_test h1_sdk2_right_arm_test.cpp)

target_link_libraries(h1_sdk2_right_arm_test
  /usr/local/lib/libunitree_sdk2.a
  ${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH}/libddscxx.so
  ${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH}/libddsc.so
  pthread
  dl
)

set_target_properties(h1_sdk2_right_arm_test PROPERTIES
  BUILD_RPATH "${UNITREE_SDK2_DIR}/thirdparty/lib/${UNITREE_ARCH};/usr/local/lib"
)
