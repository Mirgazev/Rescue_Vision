# Architecture

Rescue Vision is a computer vision subsystem for a mobile rescue robot operating in degraded visibility conditions.

The pipeline consists of six main stages:

1. Frame acquisition
2. Scene analysis
3. Visibility mode classification
4. Adaptive image enhancement
5. Human detection with YOLO11-pose
6. Tracking and ROS2 publication

The subsystem does not control robot locomotion directly. It publishes perception results that can be used by higher-level navigation and operator interfaces.
