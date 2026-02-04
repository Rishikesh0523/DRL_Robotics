cmake_minimum_required(VERSION 3.5)

project(ros_gz_sim_demos)

find_package(ament_cmake REQUIRED)
find_package(urdf_tutorial REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY
    worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)


install(
  DIRECTORY meshes urdf launch worlds
  DESTINATION share/${PROJECT_NAME}/
)



ament_package()
