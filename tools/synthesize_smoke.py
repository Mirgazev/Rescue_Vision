import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--alpha", type=float, default=0.35)
args = parser.parse_args()

img = cv2.imread(args.input)
if img is None:
    raise FileNotFoundError(args.input)
noise = np.random.normal(128, 40, img.shape[:2]).astype(np.uint8)
smoke = cv2.GaussianBlur(noise, (0, 0), sigmaX=25)
smoke_bgr = cv2.cvtColor(smoke, cv2.COLOR_GRAY2BGR)
out = cv2.addWeighted(img, 1.0 - args.alpha, smoke_bgr, args.alpha, 0)
cv2.imwrite(args.output, out)
