import argparse

from .base_detector import BaseDetector, MP4Detector
from .enhancement_modes import EnhancementMode
from .rescue_enhancer import RescueEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Rescue Vision demo")
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--model", default="weights/yolo11_pose_v4.pt")
    parser.add_argument(
        "--enhance", default="auto", choices=[m.value for m in EnhancementMode]
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    enhancer = RescueEnhancer(EnhancementMode.from_string(args.enhance))
    detector = BaseDetector(
        args.model, imgsz=args.imgsz, conf=args.conf, device=args.device
    )
    runner = MP4Detector(detector, enhancer, save_path=args.save)
    runner.run(args.source, show=args.show)


if __name__ == "__main__":
    main()
