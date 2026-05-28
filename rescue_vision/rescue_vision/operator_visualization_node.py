#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
operator_visualization_node — опциональный узел визуализации для оператора.

Соответствует Листингу А.5 ВКР. Подписывается на /vision/debug/image,
/scene/mode и /vision/performance, выводит HUD-наложение (FPS, режим,
число детекций) и отображает поток в окне OpenCV. В штатном
эксплуатационном запуске на роботе данный узел не активируется, что
обеспечивает экономию вычислительных ресурсов на бортовой платформе.

Использование:
    ros2 run rescue_vision operator_visualization_node
"""
import sys

import cv2

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
        QoSHistoryPolicy,
    )
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from diagnostic_msgs.msg import DiagnosticArray
    from rescue_vision_msgs.msg import SceneMode
except ImportError:
    rclpy = None
    Node = object


if rclpy is not None:
    QOS_DEBUG = QoSProfile(
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


class OperatorVisualizationNode(Node):
    """HUD-окно оператора (опциональное)."""

    WINDOW_NAME = 'RescueVision operator HUD'

    def __init__(self):
        super().__init__('operator_visualization_node')
        self.bridge = CvBridge()
        self.current_mode = 'UNKNOWN'
        self.current_fps = 0.0
        self.current_n_dets = 0

        self.sub_image = self.create_subscription(
            Image, '/vision/debug/image',
            self.image_callback, qos_profile=QOS_DEBUG,
        )
        self.sub_scene = self.create_subscription(
            SceneMode, '/scene/mode',
            self.scene_callback, qos_profile=QOS_SCENE,
        )
        self.sub_perf = self.create_subscription(
            DiagnosticArray, '/vision/performance',
            self.perf_callback, 10,
        )

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.get_logger().info('operator_visualization_node started')

    def scene_callback(self, msg: SceneMode) -> None:
        self.current_mode = msg.mode_name or 'UNKNOWN'

    def perf_callback(self, msg: DiagnosticArray) -> None:
        if not msg.status:
            return
        for kv in msg.status[0].values:
            if kv.key == 'fps':
                try:
                    self.current_fps = float(kv.value)
                except ValueError:
                    pass
            elif kv.key == 'n_detections':
                try:
                    self.current_n_dets = int(kv.value)
                except ValueError:
                    pass

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().error(f'CvBridge failed: {exc}')
            return

        # HUD-наложение
        cv2.putText(frame, f'Mode: {self.current_mode}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'FPS:  {self.current_fps:.1f}',
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Dets: {self.current_n_dets}',
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('q pressed, shutting down')
            rclpy.shutdown()


def main(args=None) -> int:
    if rclpy is None:
        raise RuntimeError(
            "ROS2 rclpy is not available. "
            "Source /opt/ros/foxy/setup.bash",
        )
    rclpy.init(args=args)
    node = OperatorVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
