#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescue_vision_node — главный узел подсистемы технического зрения.

Подписывается на видеопоток камеры, последовательно вызывает:
  1. SceneAnalyzer — классификатор условий видимости;
  2. RescueEnhancer — модуль адаптивного улучшения изображения;
  3. BaseDetector с YOLO11-pose v4 — детектор людей;
  4. ByteTrack — трекер с сохранением идентификаторов между кадрами.

Публикует результаты в четыре исходящих топика с дифференцированными
профилями QoS (см. таблицу QoS-профилей в разделе 3.7 пояснительной
записки).

Архитектура: SingleThreadedExecutor + MutuallyExclusiveCallbackGroup —
обоснование выбора монолитной структуры приведено в подразделе 3.6 ВКР.

Автор: Миргазев М. А., 2026.
"""

# === ИМПОРТЫ ===
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from cv_bridge import CvBridge

from rescue_vision_msgs.msg import SceneMode, PersonDetection

from .scene_analyzer import SceneAnalyzer
from .rescue_enhancer import RescueEnhancer
from .base_detector import BaseDetector


# === ОСНОВНОЙ КЛАСС ===
class RescueVisionNode(Node):
    """Главный узел подсистемы технического зрения."""

    def __init__(self):
        super().__init__('rescue_vision_node')

        # --- Параметры узла ---
        self.declare_parameter('weights_path', 'weights/best.pt')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 768)
        self.declare_parameter('enhance_mode', 'auto')
        self.declare_parameter('publish_debug_image', True)

        # ... (далее идёт твой код из Приложения А.4)

# === ТОЧКА ВХОДА ===
def main(args=None):
    rclpy.init(args=args)
    node = RescueVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
