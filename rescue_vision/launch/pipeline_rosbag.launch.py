#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_rosbag.launch.py — воспроизведение rosbag-сценария.

Соответствует Листингу А.7 ВКР (повторяемая отладка на ранее
записанных сценариях). Запускает три действия:
  1. ros2 bag play <bag_path> (ExecuteProcess)
  2. rescue_vision_node — основной конвейер
  3. operator_visualization_node — HUD оператора (опционально)

В rosbag должны присутствовать топики:
  /camera/color/image_raw    sensor_msgs/Image
  /camera/color/camera_info  sensor_msgs/CameraInfo (опционально)

Поддерживаемые параметры повторяют Листинг Е.3 ВКР:
  bag_path, model_path, enhance_mode, rate, loop
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, GroupAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rescue_vision')
    default_params_file = os.path.join(
        pkg_share, 'config', 'default_params.yaml',
    )

    declared_args = [
        DeclareLaunchArgument(
            'bag_path', default_value='',
            description='Path to rosbag2 directory (e.g. bags/test_episode)',
        ),
        DeclareLaunchArgument(
            'model_path', default_value='weights/best.pt',
            description='Path to YOLO11-pose .pt weights file',
        ),
        DeclareLaunchArgument(
            'enhance_mode', default_value='auto',
            description='off | night | fog | smoke | rain | auto',
        ),
        DeclareLaunchArgument(
            'rate', default_value='1.0',
            description='Playback rate multiplier (1.0 = real-time)',
        ),
        DeclareLaunchArgument(
            'loop', default_value='false',
            description='Loop playback (true | false)',
        ),
        DeclareLaunchArgument(
            'use_half_precision', default_value='true',
        ),
        DeclareLaunchArgument(
            'publish_debug_image', default_value='true',
            description='Publish /vision/debug/image (default true for debug)',
        ),
        DeclareLaunchArgument(
            'launch_operator_hud', default_value='true',
            description='Whether to start operator HUD (default true)',
        ),
        DeclareLaunchArgument(
            'params_file', default_value=default_params_file,
        ),
        DeclareLaunchArgument(
            'input_topic', default_value='/camera/color/image_raw',
        ),
    ]

    # --------------------------------------------------------------
    # ros2 bag play <bag_path> --rate <rate> [--loop]
    # --------------------------------------------------------------
    bag_play_cmd = [
        'ros2', 'bag', 'play',
        LaunchConfiguration('bag_path'),
        '--rate', LaunchConfiguration('rate'),
        '--clock',
    ]
    # Опциональный --loop (через PythonExpression)
    loop_args = ['--loop']
    rosbag_proc = ExecuteProcess(
        cmd=bag_play_cmd,
        output='screen',
        name='rosbag_player',
    )
    rosbag_proc_loop = ExecuteProcess(
        cmd=bag_play_cmd + loop_args,
        output='screen',
        name='rosbag_player_loop',
        condition=IfCondition(LaunchConfiguration('loop')),
    )

    rescue_vision_node = Node(
        package='rescue_vision',
        executable='rescue_vision_node',
        name='rescue_vision_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'model_path':         LaunchConfiguration('model_path'),
                'enhance_mode':       LaunchConfiguration('enhance_mode'),
                'use_half_precision': LaunchConfiguration('use_half_precision'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'input_topic':        LaunchConfiguration('input_topic'),
            },
        ],
    )

    operator_hud_node = Node(
        package='rescue_vision',
        executable='operator_visualization_node',
        name='operator_visualization_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_operator_hud')),
    )

    # Группа действий — для удобной композиции
    nodes_group = GroupAction([
        rosbag_proc,
        rosbag_proc_loop,
        rescue_vision_node,
        operator_hud_node,
    ])

    return LaunchDescription([*declared_args, nodes_group])
