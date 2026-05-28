#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rescue_vision.main — Standalone-запуск pipeline без ROS 2.

Используется для:
  * быстрой проверки конвейера на MP4-файлах без поднятия ROS 2-workspace;
  * демонстрации работы алгоритмов на лабораторных занятиях;
  * подготовки демонстрационных видео для README.

Запуск:
    python3 -m rescue_vision.main \\
        --source path/to/video.mp4 \\
        --model weights/best.pt \\
        --enhance auto --show

или после установки пакета:
    rescue_vision_standalone --source video.mp4 --enhance auto --show
"""
import argparse
import sys
from pathlib import Path

from .base_detector import BaseDetector, MP4Detector
from .enhancement_modes import EnhancementMode
from .rescue_enhancer import RescueEnhancer
from .scene_analyzer import SceneAnalyzer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True,
                   help='Video path (mp4) or camera index (0, 1, ...)')
    p.add_argument('--model', default='weights/best.pt',
                   help='Path to YOLO11-pose .pt weights')
    p.add_argument('--enhance', default='auto',
                   choices=[m.name.lower() for m in EnhancementMode],
                   help='Enhancement mode')
    p.add_argument('--imgsz', type=int, default=768)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--half', action='store_true',
                   help='Enable FP16 inference')
    p.add_argument('--tracker', default='bytetrack.yaml')
    p.add_argument('--show', action='store_true',
                   help='Show OpenCV preview window')
    p.add_argument('--save', default=None,
                   help='Save output video to this path')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    mode = EnhancementMode.from_string(args.enhance)
    scene_analyzer = SceneAnalyzer()
    enhancer = RescueEnhancer(mode=mode, scene_analyzer=scene_analyzer)
    detector = BaseDetector(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half_precision=args.half,
        tracker_config=args.tracker,
    )

    runner = MP4Detector(detector, enhancer,
                         save_path=args.save, show=args.show)

    # Camera index или путь к видео
    source = args.source
    if source.isdigit():
        source_arg = int(source)
    else:
        source_arg = source
        if not Path(source).is_file() and not source.startswith(('rtsp://', 'http')):
            print(f'[ERROR] source not found: {source}', file=sys.stderr)
            return 1

    print(f'Starting Rescue Vision standalone on {source!r} '
          f'(mode={mode.name}, half={args.half})')
    stats = runner.run(source_arg, show=args.show)
    print(f'Done: {stats["frames"]} frames, '
          f'{stats["detections"]} detections, '
          f'{stats["fps"]:.1f} FPS, '
          f'{stats["elapsed_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
