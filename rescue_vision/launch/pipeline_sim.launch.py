#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_sim.launch.py — ПРОЕКТНЫЙ ШАБЛОН для дальнейшего развития работы.

ВНИМАНИЕ: в рамках настоящей ВКР этот launch-файл НЕ использовался для
получения экспериментальных результатов. Все эксперименты главы 4 ВКР
выполнены через pipeline_realsense.launch.py (реальная камера D435i) и
pipeline_rosbag.launch.py (воспроизведение ранее записанных rosbag-
сценариев на реальных датасетах ExDark, RESIDE, COCO, MOT17 и
видеоматериале МЧС России).

Файл сохранён в репозитории в соответствии с Листингом А.1 ВКР (структура
пакета rescue_vision), где он упомянут как один из трёх launch-файлов
для разных сценариев запуска. Подраздел 3.8 ВКР явно указывает, что
"pipeline_sim.launch.py может использоваться при дальнейшем развитии
работы для проверки в симуляционной среде, но в экспериментальные
результаты настоящей ВКР не включается".

Назначение шаблона: подключение симулированного источника видеопотока
(например, Gazebo Ignition с моделью камеры или image_publisher_node
из тестового набора изображений) к тому же rescue_vision_node без
модификации самого узла. Сама симуляция не предоставляется и должна
быть подключена пользователем при необходимости.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rescue_vision')
    default_params_file = os.path.join(
        pkg_share, 'config', 'default_params.yaml',
    )

    declared_args = [
        DeclareLaunchArgument(
            'model_path', default_value='weights/best.pt',
        ),
        DeclareLaunchArgument(
            'enhance_mode', default_value='auto',
        ),
        DeclareLaunchArgument(
            'input_topic', default_value='/sim/camera/color/image_raw',
            description='Topic published by your simulation source',
        ),
        DeclareLaunchArgument(
            'use_half_precision', default_value='false',
            description='Default false for sim-style debug',
        ),
        DeclareLaunchArgument(
            'publish_debug_image', default_value='true',
        ),
        DeclareLaunchArgument(
            'params_file', default_value=default_params_file,
        ),
    ]

    info_action = LogInfo(
        msg=(
            '[rescue_vision] pipeline_sim.launch.py — это проектный шаблон. '
            'Подключите симулированный источник кадров к топику '
            '/sim/camera/color/image_raw (см. комментарии в файле).'
        ),
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

    return LaunchDescription([*declared_args, info_action, rescue_vision_node])
