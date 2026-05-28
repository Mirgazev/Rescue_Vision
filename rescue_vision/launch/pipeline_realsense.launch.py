#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_realsense.launch.py — штатный launch на роботе с RealSense D435i.

Соответствует Листингу А.6 ВКР. Запускает три узла:
  1. realsense2_camera (через IncludeLaunchDescription, опционально)
  2. rescue_vision_node — основной конвейер обработки
  3. operator_visualization_node — HUD оператора (опционально)

Также параллельно может быть запущен rosbag2 record для журналирования
входного видеопотока и всех исходящих топиков (см. Листинг Е.4).

Поддерживаемые параметры (Таблица 7 ВКР, Листинг Е.1):
  model_path, device, imgsz, conf_threshold, iou_threshold,
  enhance_mode, use_half_precision, tracker_config, publish_debug_image
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('rescue_vision')
    default_params_file = os.path.join(
        pkg_share, 'config', 'default_params.yaml',
    )

    # --------------------------------------------------------------
    # Аргументы запуска (соответствуют Листингу Е.1 ВКР)
    # --------------------------------------------------------------
    declared_args = [
        DeclareLaunchArgument(
            'model_path',
            default_value='weights/best.pt',
            description='Path to YOLO11-pose .pt weights file',
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda:0',
            description='Inference device: cuda:0 | cuda:1 | cpu',
        ),
        DeclareLaunchArgument(
            'imgsz', default_value='768',
            description='YOLO inference resolution (square)',
        ),
        DeclareLaunchArgument(
            'conf_threshold', default_value='0.25',
            description='Detection confidence threshold',
        ),
        DeclareLaunchArgument(
            'iou_threshold', default_value='0.5',
            description='IoU threshold for NMS',
        ),
        DeclareLaunchArgument(
            'enhance_mode', default_value='auto',
            description='off | night | fog | smoke | rain | auto',
        ),
        DeclareLaunchArgument(
            'use_half_precision', default_value='true',
            description='Enable FP16 inference on GPU',
        ),
        DeclareLaunchArgument(
            'tracker_config', default_value='bytetrack.yaml',
            description='ByteTrack config name (Ultralytics-resolved)',
        ),
        DeclareLaunchArgument(
            'publish_debug_image', default_value='false',
            description='Publish /vision/debug/image with annotated frame',
        ),
        DeclareLaunchArgument(
            'params_file', default_value=default_params_file,
            description='Path to YAML file with default node parameters',
        ),
        DeclareLaunchArgument(
            'launch_camera', default_value='true',
            description='Whether to start realsense2_camera_node',
        ),
        DeclareLaunchArgument(
            'launch_operator_hud', default_value='false',
            description='Whether to start operator_visualization_node',
        ),
        DeclareLaunchArgument(
            'input_topic', default_value='/camera/color/image_raw',
            description='Topic of source RGB image',
        ),
    ]

    # --------------------------------------------------------------
    # 1) Intel RealSense camera (опционально)
    # --------------------------------------------------------------
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py',
            ]),
        ]),
        condition=IfCondition(LaunchConfiguration('launch_camera')),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'rgb_camera.profile': '1280x720x30',
            'depth_module.profile': '848x480x30',
            'align_depth.enable': 'true',
        }.items(),
    )

    # --------------------------------------------------------------
    # 2) Основной узел rescue_vision_node
    # --------------------------------------------------------------
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
                'device':             LaunchConfiguration('device'),
                'imgsz':              LaunchConfiguration('imgsz'),
                'conf_threshold':     LaunchConfiguration('conf_threshold'),
                'iou_threshold':      LaunchConfiguration('iou_threshold'),
                'enhance_mode':       LaunchConfiguration('enhance_mode'),
                'use_half_precision': LaunchConfiguration('use_half_precision'),
                'tracker_config':     LaunchConfiguration('tracker_config'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'input_topic':        LaunchConfiguration('input_topic'),
            },
        ],
    )

    # --------------------------------------------------------------
    # 3) Operator HUD (опционально, по умолчанию выключен)
    # --------------------------------------------------------------
    operator_hud_node = Node(
        package='rescue_vision',
        executable='operator_visualization_node',
        name='operator_visualization_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_operator_hud')),
    )

    nodes_group = GroupAction([
        camera_launch,
        rescue_vision_node,
        operator_hud_node,
    ])

    return LaunchDescription([*declared_args, nodes_group])
