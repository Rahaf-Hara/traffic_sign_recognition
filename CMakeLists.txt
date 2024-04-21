cmake_minimum_required(VERSION 3.0.2)
project(traffic_sign_recognition)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()

###########
## Build ##
###########


include_directories(
  ${catkin_INCLUDE_DIRS}
)

