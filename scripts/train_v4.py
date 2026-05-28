#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_v4.py — финальное дообучение модели v4 (enhance-aware safe).

Соответствует строке v4 (exdark_enhance_aware_safe2) Таблицы 20 ВКР.
Результат: mAP@0.5 = 0,8905 (ExDark) / 0,6749 (RESIDE) / 0,5646 (COCO).

Скрипт реализует три ключевых решения, отличающих v4 от отклонённой v3
(см. подраздел 4.3 ВКР):
  1. Старт с весов v2 (а не с базовой yolo11m.pt).
  2. Полностью отключены смешивающие аугментации (mosaic=0, mixup=0).
  3. RescueEnhancer применяется к 25% кадров через патч load_image;
     при отключённых mosaic/mixup каждый кадр обучения остаётся
     целостным распределением.

Запуск:
    python3 scripts/train_v4.py \\
        --data /path/to/data.yaml \\
        --init-weights weights/v2_best.pt \\
        --enhance-prob 0.25
"""
import os
import sys
import random
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def patch_dataset_with_enhancer(enhance_prob: float, repo_root: Path) -> bool:
    """Монки-патч YOLODataset.load_image для enhance-aware обучения.

    К каждому загружаемому кадру с вероятностью enhance_prob применяется
    RescueEnhancer в одном из трёх режимов (NIGHT/FOG/SMOKE), выбираемом
    через ImprovedSceneAnalyzer. ВАЖНО: патч работает корректно только при
    отключённых mosaic/mixup — иначе возникает distribution mismatch (v3).
    """
    sys.path.insert(0, str(repo_root))
    try:
        from rescue_vision.rescue_vision.rescue_enhancer import RescueEnhancer
        from rescue_vision.rescue_vision.enhancement_modes import EnhancementMode
        from rescue_vision.rescue_vision.scene_analyzer import SceneAnalyzer
    except ImportError:
        print('[WARN] rescue_vision не найден — обучение БЕЗ enhance-aware')
        return False

    try:
        from ultralytics.data.dataset import YOLODataset
    except ImportError:
        print('[WARN] не удалось импортировать YOLODataset — patch пропущен')
        return False

    scene_analyzer = SceneAnalyzer()
    enhancers = {
        EnhancementMode.NIGHT: RescueEnhancer(EnhancementMode.NIGHT, scene_analyzer),
        EnhancementMode.FOG:   RescueEnhancer(EnhancementMode.FOG, scene_analyzer),
        EnhancementMode.SMOKE: RescueEnhancer(EnhancementMode.SMOKE, scene_analyzer),
    }
    auto_enhancer = RescueEnhancer(EnhancementMode.AUTO, scene_analyzer)

    original_load_image = YOLODataset.load_image

    def patched_load_image(self, i, *args, **kwargs):
        im, hw_orig, hw_resized = original_load_image(self, i, *args, **kwargs)
        if im is not None and random.random() < enhance_prob:
            # Используем AUTO: ImprovedSceneAnalyzer сам решает режим для кадра.
            # OFF-кадры останутся без изменений.
            im = auto_enhancer.enhance(im)
        return im, hw_orig, hw_resized

    YOLODataset.load_image = patched_load_image
    print(f'[OK] enhance-aware patch активен (p={enhance_prob})')
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Путь к data.yaml')
    p.add_argument('--init-weights', default='weights/v2_best.pt',
                   help='Стартовые веса (по умолчанию веса v2)')
    p.add_argument('--enhance-prob', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--imgsz', type=int, default=768)
    p.add_argument('--project', default='runs_finetune_v4')
    p.add_argument('--name', default='exdark_enhance_aware_safe2')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    # Фикс фрагментации памяти (важно для 6 ГБ VRAM RTX 2060)
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)

    mem_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
              if torch.cuda.is_available() else 0)
    batch = 4 if mem_gb >= 8 else 2
    print(f'imgsz={args.imgsz}, batch={batch}, VRAM={mem_gb:.1f} GB')

    repo_root = Path(__file__).resolve().parent.parent
    patch_dataset_with_enhancer(args.enhance_prob, repo_root)

    # Стартуем с весов v2 (ключевое отличие от v3)
    if not Path(args.init_weights).is_file():
        print(f'[WARN] {args.init_weights} не найден — старт с yolo11m.pt')
        model = YOLO('yolo11m.pt')
    else:
        model = YOLO(args.init_weights)

    print('Запуск v4: enhance-aware safe fine-tuning…')
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        # --- Мягкая дообучка ---
        lr0=1e-4,                # в 10 раз меньше v2
        lrf=0.01,
        freeze=22,               # дообучаем только head + верхний neck
        warmup_epochs=0,         # старт с сошедшихся весов, прогрев не нужен
        # --- Смешивающие аугментации ОТКЛЮЧЕНЫ (ключевое отличие от v3) ---
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        # --- Геометрические отключены, цветовые минимальные ---
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.0, fliplr=0.5,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        # --- Прочее ---
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        cache=False,
        amp=True,
        single_cls=True,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        save_period=5,
        patience=25,
        val=True,
        plots=True,
    )

    print('\nФинальная валидация…')
    metrics = model.val()
    print(f'mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}')
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f'Лучшие веса: {best_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
