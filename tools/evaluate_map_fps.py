"""Placeholder utility for summarising mAP/FPS experiments.

Full datasets and trained weights are intentionally not included in the public repository.
"""

import argparse
import time
import cv2
from rescue_vision.rescue_vision.base_detector import BaseDetector
from rescue_vision.rescue_vision.enhancement_modes import EnhancementMode
from rescue_vision.rescue_vision.rescue_enhancer import RescueEnhancer

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--model", default="weights/yolo11_pose_v4.pt")
parser.add_argument("--enhance", default="off")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
detector = BaseDetector(args.model)
enhancer = RescueEnhancer(EnhancementMode.from_string(args.enhance))
frames = 0
start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    processed = enhancer.enhance(frame)
    detector.process_frame(processed)
    frames += 1
elapsed = max(time.time() - start, 1e-6)
print({"frames": frames, "fps": frames / elapsed})
