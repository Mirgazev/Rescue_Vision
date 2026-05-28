#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_smoke.py — синтез дымовых сцен на базе COCO val2017.

Создаёт датасет SmokeTest (200 изображений), описанный в подразделе 4.2
ВКР. В исходном датасете D-Fire отсутствует класс person, что делает
невозможным прямую оценку детекции людей в задымлённой обстановке.
Поэтому в работе используется собственная процедура синтеза дыма:
выбираются 200 изображений из COCO val2017, содержащих класс person,
и к каждому применяется модель атмосферного рассеяния с шумом Перлина
для имитации неоднородной плотности дыма.

Модель:
    I(x) = J(x) * t(x) + A * (1 - t(x))

где J(x) — исходное чистое изображение, A — атмосферный свет
(светло-серый для дыма), t(x) — пространственно-неоднородная карта
пропускания, генерируемая через шум Перлина.

Результат сохраняется в две папки:
    output/clean/   — оригинальные изображения
    output/smoke/   — изображения с наложенным синтетическим дымом

Соответствующие YOLO-разметки копируются параллельно.

Пример запуска:
    python3 tools/synthesize_smoke.py \\
        --coco_images /datasets/COCO/val2017 \\
        --coco_labels /datasets/COCO/labels_yolo \\
        --output /datasets/SmokeTest \\
        --count 200 \\
        --seed 42
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# =====================================================================
# Генерация шума Перлина для неоднородной плотности дыма
# =====================================================================
def perlin_noise(shape: Tuple[int, int],
                 scale: int = 64,
                 octaves: int = 4,
                 persistence: float = 0.5) -> np.ndarray:
    """Простая реализация суммы 2D-шумов Перлина через гауссовое размытие.

    Returns:
        Массив [0..1] формы (H, W) с неоднородным шумом.
    """
    h, w = shape
    accumulator = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    current_scale = scale
    for _ in range(octaves):
        noise = np.random.rand(
            max(1, h // current_scale + 2),
            max(1, w // current_scale + 2),
        ).astype(np.float32)
        upsampled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        smoothed = cv2.GaussianBlur(upsampled, (0, 0),
                                    sigmaX=current_scale * 0.5,
                                    sigmaY=current_scale * 0.5)
        accumulator += amplitude * smoothed
        total_amplitude += amplitude
        amplitude *= persistence
        current_scale = max(1, current_scale // 2)
    accumulator = accumulator / max(total_amplitude, 1e-6)
    # Нормализация в [0..1]
    mn, mx = accumulator.min(), accumulator.max()
    if mx - mn > 1e-6:
        accumulator = (accumulator - mn) / (mx - mn)
    return accumulator


# =====================================================================
# Атмосферная модель синтеза дыма
# =====================================================================
def synthesize_smoke(image: np.ndarray,
                     t_min: float = 0.35,
                     t_max: float = 0.85,
                     atmospheric_light: Tuple[int, int, int] = (200, 200, 200),
                     ) -> np.ndarray:
    """Наложить синтетический дым по модели атмосферного рассеяния.

    Args:
        image: BGR-изображение uint8.
        t_min: минимальное пропускание (1.0 - полностью занавешено).
        t_max: максимальное пропускание (1.0 - без дыма).
        atmospheric_light: BGR-цвет атмосферного света (для дыма - серый).

    Returns:
        BGR-изображение uint8 с наложенным дымом.
    """
    h, w = image.shape[:2]
    # Шум Перлина задаёт неоднородную плотность
    noise = perlin_noise((h, w), scale=64, octaves=4, persistence=0.55)
    # Преобразуем в карту пропускания t(x)
    transmission = t_min + (t_max - t_min) * (1.0 - noise)
    transmission_3ch = np.stack([transmission] * 3, axis=-1)

    # Атмосферный свет как BGR-константа в форме (h, w, 3)
    a = np.array(atmospheric_light, dtype=np.float32).reshape(1, 1, 3)
    a = np.broadcast_to(a, (h, w, 3))

    # I(x) = J(x) * t(x) + A * (1 - t(x))
    j = image.astype(np.float32)
    i_smoke = j * transmission_3ch + a * (1.0 - transmission_3ch)
    return np.clip(i_smoke, 0, 255).astype(np.uint8)


# =====================================================================
# Фильтрация COCO по наличию класса person в YOLO-разметке
# =====================================================================
def find_person_images(images_dir: Path,
                       labels_dir: Path,
                       person_class: int = 0) -> List[Tuple[Path, Path]]:
    """Найти изображения COCO, содержащие хотя бы один person bbox."""
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(images_dir.rglob('*.jpg')):
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.is_file():
            continue
        with open(label_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0].isdigit():
                    if int(parts[0]) == person_class:
                        pairs.append((img_path, label_path))
                        break
    return pairs


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coco_images', required=True,
                   help='Каталог COCO val2017 с изображениями')
    p.add_argument('--coco_labels', required=True,
                   help='Каталог YOLO-разметок (labels/*.txt)')
    p.add_argument('--output', required=True,
                   help='Корневой каталог для SmokeTest')
    p.add_argument('--count', type=int, default=200,
                   help='Сколько изображений сгенерировать')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--t_min', type=float, default=0.35,
                   help='Минимальное пропускание t_min (плотный дым)')
    p.add_argument('--t_max', type=float, default=0.85,
                   help='Максимальное пропускание t_max (просветы)')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    images_dir = Path(args.coco_images)
    labels_dir = Path(args.coco_labels)
    output = Path(args.output)
    clean_out = output / 'clean'
    smoke_out = output / 'smoke'
    labels_out = output / 'labels'
    for d in (clean_out, smoke_out, labels_out):
        d.mkdir(parents=True, exist_ok=True)

    print(f'Сканируем COCO {images_dir} с person-разметками…')
    person_pairs = find_person_images(images_dir, labels_dir)
    print(f'Найдено {len(person_pairs)} person-изображений.')
    if len(person_pairs) < args.count:
        print(f'[WARN] требуется {args.count}, доступно {len(person_pairs)}')

    selected = random.sample(person_pairs, min(args.count, len(person_pairs)))

    for i, (img_path, label_path) in enumerate(selected, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        smoke = synthesize_smoke(frame,
                                 t_min=args.t_min, t_max=args.t_max)
        cv2.imwrite(str(clean_out / img_path.name), frame)
        cv2.imwrite(str(smoke_out / img_path.name), smoke)
        shutil.copy(label_path, labels_out / label_path.name)
        if i % 50 == 0 or i == len(selected):
            print(f'  [{i}/{len(selected)}] обработано')

    print(f'Готово. SmokeTest сохранён в {output}/')
    print(f'  clean:  {clean_out}')
    print(f'  smoke:  {smoke_out}')
    print(f'  labels: {labels_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
