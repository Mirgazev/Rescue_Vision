# Architecture

Rescue Vision is a six-stage computer vision pipeline for rescue robotics:
frame acquisition, scene analysis, mode classification, adaptive enhancement, YOLO11-pose detection, ByteTrack tracking and ROS2 publication.

The perception subsystem is separated from the low-level Unitree H1 locomotion controller and does not publish joint commands or emergency control commands.
