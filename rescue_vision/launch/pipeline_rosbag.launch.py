from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    path = LaunchConfiguration('path')
    return LaunchDescription([
        DeclareLaunchArgument('path', default_value='', description='Path to rosbag'),
        ExecuteProcess(cmd=['ros2', 'bag', 'play', path], output='screen'),
        Node(package='rescue_vision', executable='rescue_vision_node', name='rescue_vision_node', output='screen'),
    ])
