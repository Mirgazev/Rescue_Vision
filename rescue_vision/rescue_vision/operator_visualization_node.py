#!/usr/bin/env python3
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import cv2
except Exception:
    rclpy = None
    Node = object


class OperatorVisualizationNode(Node):
    """Simple operator display node for debug images."""

    def __init__(self):
        super().__init__('operator_visualization_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/vision/debug/image', self.callback, 10)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Rescue Vision Operator View', frame)
        cv2.waitKey(1)


def main(args=None):
    if rclpy is None:
        raise RuntimeError('ROS2 rclpy is not available')
    rclpy.init(args=args)
    node = OperatorVisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
