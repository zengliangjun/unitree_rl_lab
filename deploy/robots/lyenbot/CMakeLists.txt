cmake_minimum_required(VERSION 3.12)
project(lyenbot_ctrl)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

set(UNITREE_ROOT "${PROJECT_SOURCE_DIR}/../../../../unitree_sdk2/")

include_directories(
  /usr/include/eigen3
  ${UNITREE_ROOT}/include
  ${UNITREE_ROOT}/thirdparty/include
  ${UNITREE_ROOT}/thirdparty/include/ddscxx
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  ${UNITREE_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a
  ${UNITREE_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
  ${UNITREE_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
  rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(${PROJECT_NAME} main.cpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ROBOT_ID=6          # lyenbot
)
