# Installation

```bash
git clone https://github.com/Mirgazev/Rescue_Vision.git
cd Rescue_Vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ROS2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s ~/Rescue_Vision_Bachelor/rescue_vision .
ln -s ~/Rescue_Vision_Bachelor/rescue_vision_msgs .
cd ~/ros2_ws
colcon build
source install/setup.bash
```
