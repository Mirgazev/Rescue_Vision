#!/usr/bin/env python3
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Float32
    from cv_bridge import CvBridge
    from rescue_vision_msgs.msg import SceneMode, PersonDetection
except Exception:  # Allows importing this file outside ROS2 for static checks.
    rclpy = None
    Node = object

from .base_detector import BaseDetector
from .enhancement_modes import EnhancementMode
from .rescue_enhancer import RescueEnhancer
from .scene_analyzer import SceneAnalyzer


class RescueVisionNode(Node):
    """ROS2 integration node for adaptive enhancement, YOLO11-pose and ByteTrack."""

    def __init__(self):
        super().__init__("rescue_vision_node")
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.bridge = CvBridge()

        self.declare_parameter("model_path", "weights/yolo11_pose_v4.pt")
        self.declare_parameter("enhance_mode", "auto")
        self.declare_parameter("imgsz", 768)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("input_topic", "/camera/color/image_raw")

        self.scene_analyzer = SceneAnalyzer()
        self.enhancer = RescueEnhancer(
            EnhancementMode.from_string(self.get_parameter("enhance_mode").value),
            self.scene_analyzer,
        )
        self.detector = BaseDetector(
            self.get_parameter("model_path").value,
            imgsz=int(self.get_parameter("imgsz").value),
            conf=float(self.get_parameter("conf_threshold").value),
            device=self.get_parameter("device").value,
        )

        sensor_qos = QoSProfile(
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        transient_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(
            Image,
            self.get_parameter("input_topic").value,
            self.image_callback,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.scene_pub = self.create_publisher(SceneMode, "/scene/mode", transient_qos)
        self.detection_pub = self.create_publisher(
            PersonDetection, "/detection/persons", reliable_qos
        )
        self.debug_image_pub = self.create_publisher(
            Image, "/vision/debug/image", sensor_qos
        )
        self.performance_pub = self.create_publisher(
            Float32, "/vision/performance/fps", reliable_qos
        )
        self.status_pub = self.create_publisher(String, "/vision/status", transient_qos)

        self.frame_count = 0
        self.last_time = time.time()
        self.get_logger().info("rescue_vision_node started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        mode = self.scene_analyzer.analyze(frame)
        enhanced = self.enhancer.enhance(frame)
        result = self.detector.process_frame(enhanced)

        scene_msg = SceneMode()
        scene_msg.header = msg.header
        scene_msg.mode = mode.value
        scene_msg.confidence = 1.0
        for key, value in self.scene_analyzer.last_features.items():
            if hasattr(scene_msg, key):
                setattr(scene_msg, key, float(value))
        self.scene_pub.publish(scene_msg)

        for det in result["detections"]:
            det_msg = PersonDetection()
            det_msg.header = msg.header
            det_msg.track_id = int(det.track_id)
            det_msg.confidence = float(det.confidence)
            det_msg.bbox_center_x = float(det.bbox_xywh[0])
            det_msg.bbox_center_y = float(det.bbox_xywh[1])
            det_msg.bbox_width = float(det.bbox_xywh[2])
            det_msg.bbox_height = float(det.bbox_xywh[3])
            det_msg.scene_mode = mode.value
            if det.keypoints is not None:
                det_msg.keypoints_x = [float(k[0]) for k in det.keypoints]
                det_msg.keypoints_y = [float(k[1]) for k in det.keypoints]
                det_msg.keypoints_confidence = [float(k[2]) for k in det.keypoints]
            self.detection_pub.publish(det_msg)

        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            fps = self.frame_count / (now - self.last_time)
            fps_msg = Float32()
            fps_msg.data = float(fps)
            self.performance_pub.publish(fps_msg)
            self.frame_count = 0
            self.last_time = now

        self.debug_image_pub.publish(
            self.bridge.cv2_to_imgmsg(enhanced, encoding="bgr8")
        )


def main(args=None):
    if rclpy is None:
        raise RuntimeError("ROS2 rclpy is not available in this environment")
    rclpy.init(args=args)
    node = RescueVisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
