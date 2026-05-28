#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_scene_thresholds.py — калибровка порогов классификатора сцены.

Реализует метод «середины безопасной зоны», описанный в Приложении Ж ВКР
(формулы 11-13). Именно этим методом получена точность AUTO-классификатора
90,5% (Таблица 19 ВКР), что превышает ~76% при наивном выборе порога по
одному классу.

Скрипт:
  1. Сканирует учебную выборку из пяти папок (OFF/NIGHT/FOG/SMOKE/RAIN).
  2. Извлекает 5 признаков из каждого кадра (ImprovedSceneAnalyzer).
  3. Калибрует 9 порогов методом «середины безопасной зоны».
  4. Прогоняет классификацию со старыми и новыми порогами.
  5. Строит матрицу ошибок 5x5 и сохраняет manifest.json.

Запуск:
    python3 scripts/calibrate_scene_thresholds.py \\
        --dataset /path/to/scene_dataset \\
        --output manifest.json
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import cv2

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    from rescue_vision.rescue_vision.scene_analyzer import SceneAnalyzer
    from rescue_vision.rescue_vision.enhancement_modes import EnhancementMode
except ImportError:
    from rescue_vision.scene_analyzer import SceneAnalyzer  # type: ignore
    from rescue_vision.enhancement_modes import EnhancementMode  # type: ignore

MODES = ['OFF', 'NIGHT', 'FOG', 'SMOKE', 'RAIN']


def collect_features(dataset_dir: Path,
                     analyzer: SceneAnalyzer) -> Dict[str, List[dict]]:
    """Извлечь признаки из всех кадров каждого класса."""
    feat_by_mode: Dict[str, List[dict]] = defaultdict(list)
    for mode in MODES:
        class_dir = dataset_dir / mode
        if not class_dir.is_dir():
            print(f'[WARN] нет папки {class_dir}')
            continue
        images = (list(class_dir.glob('*.jpg'))
                  + list(class_dir.glob('*.png'))
                  + list(class_dir.glob('*.jpeg')))
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            feats = analyzer.get_features(frame)
            if feats:
                feat_by_mode[mode].append(feats)
        print(f'  {mode}: {len(feat_by_mode[mode])} кадров')
    return feat_by_mode


def safe_zone_midpoint(values_a: List[float],
                       values_b: List[float]) -> float:
    """Метод «середины безопасной зоны» (формула 13 ВКР).

    Возвращает середину между p95(A) и p5(B), если безопасная зона есть.
    Иначе возвращает наивную оценку (среднее p95/p5).
    """
    if not values_a or not values_b:
        return None
    p95_a = float(np.percentile(values_a, 95))
    p5_b = float(np.percentile(values_b, 5))
    return round((p95_a + p5_b) / 2.0, 2)


def calibrate_thresholds(feat_by_mode: Dict[str, List[dict]]) -> dict:
    """Калибровка 9 порогов методом середины безопасной зоны."""
    t = {
        'night_brightness': 55,
        'fog_brightness_min': 150,
        'fog_std_max': 40,
        'fog_laplac_max': 100,
        'smoke_brightness_range': [60, 150],
        'smoke_std_max': 30,
        'smoke_laplac_max': 150,
        'rain_edge_density_min': 0.04,
        'rain_diagonal_ratio_min': 1.30,
    }

    def vals(mode, key):
        return [f[key] for f in feat_by_mode.get(mode, []) if key in f]

    # NIGHT: середина между верхом NIGHT (по brightness) и низом не-NIGHT
    night_b = vals('NIGHT', 'brightness')
    non_night_b = []
    for m in ['OFF', 'FOG', 'SMOKE', 'RAIN']:
        non_night_b.extend(vals(m, 'brightness'))
    mid = safe_zone_midpoint(night_b, non_night_b)
    if mid is not None:
        t['night_brightness'] = mid

    # FOG: brightness_min — середина между верхом не-FOG и низом FOG
    fog_b = vals('FOG', 'brightness')
    off_b = vals('OFF', 'brightness')
    if fog_b and off_b:
        fog_b5 = float(np.percentile(fog_b, 5))
        off_b95 = float(np.percentile(off_b, 95))
        if off_b95 < fog_b5:
            t['fog_brightness_min'] = round((off_b95 + fog_b5) / 2.0, 1)
    if fog_b:
        t['fog_std_max'] = round(float(np.percentile(vals('FOG', 'std'), 98)), 1)
        t['fog_laplac_max'] = round(
            float(np.percentile(vals('FOG', 'laplac_var'), 98)), 1)

    # SMOKE: диапазон brightness + std/laplac верхние перцентили
    smoke_b = vals('SMOKE', 'brightness')
    if smoke_b:
        t['smoke_brightness_range'] = [
            round(float(np.percentile(smoke_b, 5)), 1),
            round(float(np.percentile(smoke_b, 95)), 1),
        ]
        t['smoke_std_max'] = round(float(np.percentile(vals('SMOKE', 'std'), 98)), 1)
        t['smoke_laplac_max'] = round(
            float(np.percentile(vals('SMOKE', 'laplac_var'), 98)), 1)

    # RAIN: edge_density и diag_ratio нижние перцентили
    rain_ed = vals('RAIN', 'edge_density')
    rain_dr = vals('RAIN', 'diag_ratio')
    if rain_ed:
        t['rain_edge_density_min'] = round(float(np.percentile(rain_ed, 5)), 4)
    if rain_dr:
        t['rain_diagonal_ratio_min'] = round(float(np.percentile(rain_dr, 5)), 3)

    return t


def evaluate(feat_by_mode: Dict[str, List[dict]],
             thresholds: dict) -> tuple:
    """Прогнать классификацию и построить матрицу ошибок."""
    analyzer = SceneAnalyzer(thresholds=thresholds)
    confusion = {gt: defaultdict(int) for gt in MODES}
    total, correct = 0, 0
    for gt_mode, feats_list in feat_by_mode.items():
        for feats in feats_list:
            pred = analyzer._classify_from_features(feats)
            pred_name = pred.name
            confusion[gt_mode][pred_name] += 1
            total += 1
            if pred_name == gt_mode:
                correct += 1
    accuracy = correct / total if total else 0.0
    return accuracy, confusion


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True,
                   help='Каталог с папками OFF/NIGHT/FOG/SMOKE/RAIN')
    p.add_argument('--output', default='manifest.json')
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        sys.exit(f'[ERROR] нет каталога {dataset_dir}')

    analyzer = SceneAnalyzer()
    print('Извлечение признаков…')
    feat_by_mode = collect_features(dataset_dir, analyzer)

    # До калибровки
    acc_before, _ = evaluate(feat_by_mode, analyzer.DEFAULT_THRESHOLDS.copy())
    print(f'\nAccuracy с порогами по умолчанию: {acc_before*100:.1f}%')

    # Калибровка
    print('\nКалибровка методом середины безопасной зоны…')
    new_thresholds = calibrate_thresholds(feat_by_mode)
    acc_after, confusion = evaluate(feat_by_mode, new_thresholds)
    print(f'Accuracy после калибровки: {acc_after*100:.1f}%')

    # Матрица ошибок
    print('\nМатрица ошибок (строки — GT, столбцы — предсказание):')
    header = 'GT\\pred  ' + '  '.join(f'{m:>6}' for m in MODES)
    print(header)
    by_mode_acc = {}
    for gt in MODES:
        row = confusion[gt]
        n = sum(row.values())
        cells = '  '.join(f'{row.get(m, 0):>6}' for m in MODES)
        acc_class = row.get(gt, 0) / n * 100 if n else 0
        by_mode_acc[gt] = {'accuracy_pct': round(acc_class, 1), 'n': n}
        print(f'{gt:>7}  {cells}   ({acc_class:.1f}%)')

    # Сохранение manifest.json
    manifest = {
        'thresholds_used': new_thresholds,
        'overall_auto_accuracy_pct': round(acc_after * 100, 1),
        'accuracy_before_calibration_pct': round(acc_before * 100, 1),
        'by_mode': by_mode_acc,
    }
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f'\nСохранено: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
