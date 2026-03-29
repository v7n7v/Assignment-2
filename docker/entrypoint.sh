#!/bin/bash
# Basic entrypoint for ROS / Colcon Docker containers

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "Sourced ROS 2 ${ROS_DISTRO}"

# Source the overlay workspace, if built
if [ -f /overlay_ws/install/setup.bash ]
then
  source /overlay_ws/install/setup.bash
  echo "Sourced DRL navigation overlay workspace"
fi

# Set TurtleBot3 environment
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:/overlay_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:/overlay_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/obstacle_plugin/lib
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}:/overlay_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models

# Execute the command passed into this entrypoint
exec "$@"
