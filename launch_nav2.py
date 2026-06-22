"""
Nav2 launch file for Genesis Simulator.

Starts the full Nav2 navigation stack alongside the Genesis turtlebot simulation.
Uses pre-built nav2_bringup launch files with a custom nav2_params.yaml.

Usage:
    # Terminal 1: Start Genesis sim
    cd ~/genesis_sim && python3 turtlebot_sim.py

    # Terminal 2: Start Nav2
    cd ~/genesis_sim && ros2 launch launch_nav2.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Paths
    genesis_sim_dir = os.path.dirname(os.path.abspath(__file__))
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    nav2_params_file = os.path.join(genesis_sim_dir, 'nav2_params.yaml')
    map_file = os.path.join(genesis_sim_dir, 'maps', 'empty_map.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation (Gazebo) clock if true'),

        # ── 1. Static TF: map → odom (identity) ──────────────────────
        # Since Genesis provides ground-truth odometry, we publish a
        # static identity transform from map to odom. This replaces AMCL.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ── 2. Map Server ─────────────────────────────────────────────
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'yaml_filename': map_file,
            }],
        ),

        # ── 3. Lifecycle manager for map_server ───────────────────────
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server'],
            }],
        ),

        # ── 4. Nav2 Navigation Stack ──────────────────────────────────
        # Uses nav2_bringup's navigation_launch.py which starts:
        #   - controller_server
        #   - planner_server
        #   - behavior_server
        #   - bt_navigator
        #   - lifecycle_manager_navigation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': nav2_params_file,
                'autostart': 'True',
            }.items(),
        ),
    ])
