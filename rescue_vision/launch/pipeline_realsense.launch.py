from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rescue_vision',
            executable='rescue_vision_node',
            name='rescue_vision_node',
            output='screen',
            parameters=['config/default_params.yaml'],
        ),
        Node(
            package='rescue_vision',
            executable='operator_visualization_node',
            name='operator_visualization_node',
            output='screen',
        ),
    ])
