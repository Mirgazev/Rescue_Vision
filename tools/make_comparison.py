import argparse
import cv2
import numpy as np
from rescue_vision.rescue_vision.enhancement_modes import EnhancementMode
from rescue_vision.rescue_vision.rescue_enhancer import RescueEnhancer

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--enhance', default='auto')
parser.add_argument('--output', required=True)
args = parser.parse_args()

frame = cv2.imread(args.input)
if frame is None:
    raise FileNotFoundError(args.input)

enhancer = RescueEnhancer(EnhancementMode.from_string(args.enhance))
out = enhancer.enhance(frame)
comparison = np.hstack([frame, out])
cv2.imwrite(args.output, comparison)
