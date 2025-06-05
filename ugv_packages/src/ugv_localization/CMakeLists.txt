cmake_minimum_required(VERSION 3.0.2)
project(ugv_localization)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#add_executable(robot_reconfigure_node nodes/robot_dynamic.cpp)
#target_link_libraries(robot_reconfigure_node
#  ${catkin_LIBRARIES}
#)

#add_dependencies(robot_reconfigure_node ${PROJECT_NAME}_gencfg)
