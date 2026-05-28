#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_comparison.py — генератор split-view изображений для иллюстрации
работы режимов улучшения.

Использовался для подготовки Рисунка 10, 11, 12 ВКР (визуализация
работы режимов NIGHT, FOG, SMOKE на тестовых кадрах) и для README
репозитория. Поддерживает обработку одного изображения или каталога.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    from rescue_vision.rescue_vision.rescue_enhancer import RescueEnhancer
    from rescue_vision.rescue_vision.enhancement_modes import EnhancementMode
    from rescue_vision.rescue_vision.scene_analyzer import SceneAnalyzer
except ImportError:
    from rescue_vision.rescue_enhancer import RescueEnhancer  # type: ignore
    from rescue_vision.enhancement_modes import EnhancementMode  # type: ignore
    from rescue_vision.scene_analyzer import SceneAnalyzer  # type: ignore


def annotate(frame: np.ndarray, text: str,
             color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Добавить текстовую метку в верхний левый угол кадра."""
    overlay = frame.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(overlay, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)


def make_split_view(frame: np.ndarray, modes: List[str]) -> np.ndarray:
    """Сшить frame с его улучшенными версиями по списку режимов."""
    scene_analyzer = SceneAnalyzer()
    panels = [annotate(frame, 'ORIGINAL')]
    for mode_name in modes:
        try:
            mode = EnhancementMode.from_string(mode_name)
        except ValueError:
            print(f'[WARN] unknown mode: {mode_name}', file=sys.stderr)
            continue
        enh = RescueEnhancer(mode, scene_analyzer=scene_analyzer)
        processed = enh.enhance(frame.copy())
        if processed.shape != frame.shape:
            processed = cv2.resize(processed, (frame.shape[1], frame.shape[0]))
        label = mode_name.upper()
        if mode == EnhancementMode.AUTO:
            label = f'AUTO ({scene_analyzer.last_smoothed_mode.name})'
        panels.append(annotate(processed, label))
    return np.hstack(panels)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True,
                   help='Путь к изображению или каталогу с изображениями')
    p.add_argument('--output', required=True,
                   help='Путь к выходному файлу или каталогу')
    p.add_argument('--modes', nargs='+',
                   default=['off', 'night', 'fog', 'smoke'],
                   help='Режимы улучшения для отображения '
                        '(off / night / fog / smoke / rain / auto)')
    return p.parse_args()


def process_one(input_path: Path, output_path: Path, modes: List[str]) -> None:
    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f'[WARN] не удалось прочитать {input_path}', file=sys.stderr)
        return
    result = make_split_view(frame, modes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f'[OK] {input_path.name} -> {output_path}')


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if inp.is_file():
        process_one(inp, out, args.modes)
    elif inp.is_dir():
        if not out.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        for img in sorted(inp.glob('*.[jp][pn]g')):
            process_one(img, out / f'{img.stem}_comparison.jpg', args.modes)
    else:
        print(f'[ERROR] {inp} не существует', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
