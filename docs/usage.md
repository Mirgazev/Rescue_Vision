# Usage

Standalone video processing:

```bash
python -m rescue_vision.rescue_vision.main --source video.mp4 --model weights/yolo11_pose_v4.pt --enhance auto --show
```

ROS2:

```bash
ros2 launch rescue_vision pipeline_realsense.launch.py
ros2 launch rescue_vision pipeline_rosbag.launch.py path:=/path/to/rosbag
```
