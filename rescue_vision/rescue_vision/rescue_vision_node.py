#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescue_vision_node — основной узел подсистемы технического зрения.

Соответствует Листингу А.4 ВКР Миргазева М.А., 2026.

Монолитная архитектура (см. обоснование в подразделе 3.6 ВКР):
SingleThreadedExecutor + MutuallyExclusiveCallbackGroup, последовательный
конвейер SceneAnalyzer → RescueEnhancer → BaseDetector (YOLO11-pose +
ByteTrack). Все параметры Таблицы 7 ВКР параметризованы через
declare_parameter + add_on_set_parameters_callback для динамической
смены без перезапуска (Листинг Е.5).

QoS-профили (Таблица 13 ВКР):
  /scene/mode           - RELIABLE + TRANSIENT_LOCAL (depth 1)
  /detection/persons    - RELIABLE (depth 5)
  /detection/poses      - RELIABLE (depth 5)
  /vision/debug/image   - BEST_EFFORT (depth 1)
  /vision/performance   - RELIABLE (depth 5)
  Подписка /camera/...  - BEST_EFFORT (depth 1)
"""
import time
from typing import List

import numpy as np
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.parameter import Parameter
    from rcl_interfaces.msg import SetParametersResult
    from rclpy.qos import (
        QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
        QoSHistoryPolicy,
    )
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from diagnostic_msgs.msg import (
        DiagnosticArray, DiagnosticStatus, KeyValue,
    )
    try:
        from vision_msgs.msg import (
            Detection2D, Detection2DArray, BoundingBox2D,
            ObjectHypothesisWithPose,
        )
        HAS_VISION_MSGS = True
    except ImportError:
        HAS_VISION_MSGS = False
    from rescue_vision_msgs.msg import SceneMode, PersonDetection
except ImportError:
    rclpy = None
    Node = object
    HAS_VISION_MSGS = False

# Импорт реализаций из пакета (см. Приложения Б, В, Г ВКР)
try:
    from .scene_analyzer import SceneAnalyzer as ImprovedSceneAnalyzer
    from .rescue_enhancer import RescueEnhancer
    from .base_detector import BaseDetector
    from .enhancement_modes import EnhancementMode
    from .config import VisionConfig as Config
except ImportError:
    from rescue_vision.scene_analyzer import (  # type: ignore
        SceneAnalyzer as ImprovedSceneAnalyzer,
    )
    from rescue_vision.rescue_enhancer import RescueEnhancer  # type: ignore
    from rescue_vision.base_detector import BaseDetector  # type: ignore
    from rescue_vision.enhancement_modes import EnhancementMode  # type: ignore
    from rescue_vision.config import VisionConfig as Config  # type: ignore


# =====================================================================
# QoS-профили (см. Таблица 13 ВКР, подраздел 3.6)
# =====================================================================
if rclpy is not None:
    QOS_CAMERA = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    QOS_SCENE = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    QOS_DETECTION = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
    )
    QOS_PERFORMANCE = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
    )
    QOS_DEBUG = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )


# =====================================================================
# Основной узел
# =====================================================================
class RescueVisionNode(Node):
    """Основной узел подсистемы технического зрения."""

    # Сопоставление EnhancementMode -> uint8 для SceneMode.mode_id
    MODE_ID = {
        EnhancementMode.OFF: 0,
        EnhancementMode.NIGHT: 1,
        EnhancementMode.FOG: 2,
        EnhancementMode.SMOKE: 3,
        EnhancementMode.RAIN: 4,
    }

    def __init__(self):
        super().__init__('rescue_vision_node')
        self._declare_parameters()

        # --------------------------------------------------------------
        # Инициализация компонентов конвейера
        # --------------------------------------------------------------
        cfg = Config()
        cfg.model_path = self.get_parameter('model_path').value
        cfg.imgsz = int(self.get_parameter('imgsz').value)
        cfg.conf_threshold = float(self.get_parameter('conf_threshold').value)
        cfg.iou_threshold = float(self.get_parameter('iou_threshold').value)
        cfg.device = self.get_parameter('device').value

        self.scene_analyzer = ImprovedSceneAnalyzer()
        enhance_mode = EnhancementMode.from_string(
            self.get_parameter('enhance_mode').value,
        )
        self.enhancer = RescueEnhancer(enhance_mode, self.scene_analyzer)
        self.detector = BaseDetector(
            model_path=cfg.model_path,
            imgsz=cfg.imgsz,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            device=cfg.device,
            half_precision=bool(self.get_parameter('use_half_precision').value),
            tracker_config=self.get_parameter('tracker_config').value,
        )

        # CV bridge для конвертации sensor_msgs/Image <-> numpy
        self.bridge = CvBridge()

        # Callback group: одна MutuallyExclusive
        # (Ultralytics model.track несовместим с параллельными вызовами)
        self.cb_group = MutuallyExclusiveCallbackGroup()

        # --------------------------------------------------------------
        # Издатели и подписки
        # --------------------------------------------------------------
        input_topic = self.get_parameter('input_topic').value
        self.sub_image = self.create_subscription(
            Image, input_topic, self.image_callback,
            qos_profile=QOS_CAMERA, callback_group=self.cb_group,
        )
        self.scene_pub = self.create_publisher(
            SceneMode, '/scene/mode', qos_profile=QOS_SCENE,
        )
        self.person_pub = self.create_publisher(
            PersonDetection, '/detection/persons',
            qos_profile=QOS_DETECTION,
        )
        if HAS_VISION_MSGS:
            self.poses_pub = self.create_publisher(
                Detection2DArray, '/detection/poses',
                qos_profile=QOS_DETECTION,
            )
        else:
            self.poses_pub = None
        self.debug_image_pub = self.create_publisher(
            Image, '/vision/debug/image', qos_profile=QOS_DEBUG,
        )
        self.performance_pub = self.create_publisher(
            DiagnosticArray, '/vision/performance',
            qos_profile=QOS_PERFORMANCE,
        )

        # --------------------------------------------------------------
        # Динамическая смена параметров (см. Листинг Е.5 ВКР)
        # --------------------------------------------------------------
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Счётчики FPS
        self.frame_count = 0
        self.fps_last_time = time.time()
        self.last_fps = 0.0

        self.get_logger().info(
            "rescue_vision_node started: model=%s imgsz=%d conf=%.2f iou=%.2f "
            "enhance=%s half=%s tracker=%s",
            cfg.model_path, cfg.imgsz, cfg.conf_threshold, cfg.iou_threshold,
            enhance_mode.value,
            self.get_parameter('use_half_precision').value,
            self.get_parameter('tracker_config').value,
        )

    # ------------------------------------------------------------------
    # Объявление параметров (Таблица 7 ВКР)
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter('model_path', 'weights/best.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 768)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('enhance_mode', 'auto')
        self.declare_parameter('use_half_precision', True)
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('input_topic', '/camera/color/image_raw')

    # ------------------------------------------------------------------
    # Динамическая смена параметров (Листинг Е.5 ВКР)
    # ------------------------------------------------------------------
    def _on_set_parameters(self, params: List[Parameter]) -> SetParametersResult:
        for p in params:
            if p.name == 'enhance_mode':
                try:
                    self.enhancer.set_mode(EnhancementMode.from_string(p.value))
                    self.get_logger().info(
                        "enhance_mode -> %s", p.value,
                    )
                except ValueError as exc:
                    return SetParametersResult(
                        successful=False, reason=str(exc),
                    )
            elif p.name == 'conf_threshold':
                self.detector.conf = float(p.value)
                self.get_logger().info(
                    "conf_threshold -> %.3f", float(p.value),
                )
            elif p.name == 'iou_threshold':
                self.detector.iou = float(p.value)
                self.get_logger().info(
                    "iou_threshold -> %.3f", float(p.value),
                )
            elif p.name == 'publish_debug_image':
                self.get_logger().info(
                    "publish_debug_image -> %s", bool(p.value),
                )
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Главный callback
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image) -> None:
        # Этап 1: декодирование кадра
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().error(f'CvBridge failed: {exc}')
            return

        t_pipeline_start = time.time()

        # Этап 2: анализ сцены и выбор режима улучшения
        t_scene = time.time()
        scene_mode = self.scene_analyzer.analyze(frame)
        scene_time_ms = (time.time() - t_scene) * 1000

        # Этап 3: улучшение
        t_enh = time.time()
        enhanced = self.enhancer.enhance(frame)
        enhance_time_ms = (time.time() - t_enh) * 1000

        # Этап 4: детекция и трекинг
        t_inf = time.time()
        result = self.detector.process_frame(enhanced, persist=True)
        inference_time_ms = (time.time() - t_inf) * 1000

        # Этап 5: публикация
        self._publish_scene_mode(msg.header)
        self._publish_person_detections(msg.header, result['detections'])
        if HAS_VISION_MSGS and self.poses_pub is not None:
            self._publish_poses(msg.header, result['detections'])

        if self.get_parameter('publish_debug_image').value:
            self._publish_debug_image(
                msg.header, enhanced, result['detections'],
            )

        # Этап 6: FPS и диагностика
        pipeline_time_ms = (time.time() - t_pipeline_start) * 1000
        self._update_fps_counter()
        self._publish_performance(
            msg.header,
            scene_time_ms, enhance_time_ms,
            inference_time_ms, pipeline_time_ms,
            n_detections=len(result['detections']),
        )

    # ------------------------------------------------------------------
    # Публикаторы
    # ------------------------------------------------------------------
    def _publish_scene_mode(self, header) -> None:
        msg = SceneMode()
        msg.header = header
        mode = self.scene_analyzer.last_smoothed_mode
        msg.mode_id = self.MODE_ID.get(mode, 0)
        msg.mode_name = mode.name if hasattr(mode, 'name') else str(mode).upper()
        msg.confidence = 1.0
        f = self.scene_analyzer.last_features or {}
        msg.brightness = float(f.get('brightness', 0.0))
        msg.std_dev = float(f.get('std', 0.0))
        msg.laplacian_var = float(f.get('laplac_var', 0.0))
        msg.edge_density = float(f.get('edge_density', 0.0))
        self.scene_pub.publish(msg)

    def _publish_person_detections(self, header, detections) -> None:
        for det in detections:
            msg = PersonDetection()
            msg.header = header
            # Формат track-id 'track-N' для совместимости с Foxy
            msg.track_id = f'track-{det.track_id}' if det.track_id >= 0 else ''
            # Конверт XYWH (центр-форма) в XY-форму (левый верх + размеры)
            cx, cy, w, h = det.bbox_xywh
            msg.bbox_x = float(cx - w / 2)
            msg.bbox_y = float(cy - h / 2)
            msg.bbox_width = float(w)
            msg.bbox_height = float(h)
            msg.confidence = float(det.confidence)
            msg.class_name = 'person'
            if det.keypoints is not None:
                flat: List[float] = []
                for kp in det.keypoints:
                    flat.extend([float(kp[0]), float(kp[1]), float(kp[2])])
                msg.keypoints = flat
            else:
                msg.keypoints = []
            self.person_pub.publish(msg)

    def _publish_poses(self, header, detections) -> None:
        """Публикация в стандартный vision_msgs/Detection2DArray для совместимости
        с инструментами визуализации (rqt, RViz Detection2D plugin)."""
        arr = Detection2DArray()
        arr.header = header
        for det in detections:
            d = Detection2D()
            d.header = header
            bbox = BoundingBox2D()
            bbox.center.x = float(det.bbox_xywh[0])
            bbox.center.y = float(det.bbox_xywh[1])
            bbox.size_x = float(det.bbox_xywh[2])
            bbox.size_y = float(det.bbox_xywh[3])
            d.bbox = bbox
            h = ObjectHypothesisWithPose()
            try:
                h.hypothesis.class_id = 'person'
                h.hypothesis.score = float(det.confidence)
            except AttributeError:  # Foxy API
                h.id = 'person'
                h.score = float(det.confidence)
            d.results.append(h)
            arr.detections.append(d)
        self.poses_pub.publish(arr)

    def _publish_debug_image(self, header, frame, detections) -> None:
        vis = frame.copy()
        for det in detections:
            cx, cy, w, h = det.bbox_xywh
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID:{det.track_id} {det.confidence:.2f}'
            cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        mode_name = self.scene_analyzer.last_smoothed_mode.name
        cv2.putText(vis, f'mode: {mode_name}  fps: {self.last_fps:.1f}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            out_msg.header = header
            self.debug_image_pub.publish(out_msg)
        except Exception as exc:
            self.get_logger().warn(f'debug image publish failed: {exc}')

    def _publish_performance(self, header, scene_ms, enhance_ms,
                              inference_ms, pipeline_ms, n_detections):
        msg = DiagnosticArray()
        msg.header = header
        status = DiagnosticStatus()
        status.name = 'rescue_vision_node'
        status.message = (
            f'OK  fps={self.last_fps:.1f}  pipeline={pipeline_ms:.1f}ms'
        )
        status.level = DiagnosticStatus.OK
        status.values = [
            KeyValue(key='fps', value=f'{self.last_fps:.2f}'),
            KeyValue(key='scene_ms', value=f'{scene_ms:.2f}'),
            KeyValue(key='enhance_ms', value=f'{enhance_ms:.2f}'),
            KeyValue(key='inference_ms', value=f'{inference_ms:.2f}'),
            KeyValue(key='pipeline_ms', value=f'{pipeline_ms:.2f}'),
            KeyValue(key='n_detections', value=str(n_detections)),
            KeyValue(key='mode',
                     value=self.scene_analyzer.last_smoothed_mode.name),
        ]
        msg.status = [status]
        self.performance_pub.publish(msg)

    def _update_fps_counter(self) -> None:
        self.frame_count += 1
        now = time.time()
        if now - self.fps_last_time >= 1.0:
            self.last_fps = self.frame_count / (now - self.fps_last_time)
            self.frame_count = 0
            self.fps_last_time = now


# =====================================================================
# Entry point
# =====================================================================
def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError(
            "ROS2 rclpy is not available. Source /opt/ros/foxy/setup.bash",
        )
    rclpy.init(args=args)
    node = RescueVisionNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
