#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_map_fps.py — расчёт mAP@0.5, Precision, Recall, F1 и FPS.

Соответствует Листингу Д.1 ВКР. Расширенная версия с поддержкой шести
форматов аннотаций (ExDark, RESIDE, MOT17, Pascal VOC, COCO, YOLO) и
интеграцией модуля runtime-улучшения RescueEnhancer из пакета
rescue_vision.

Пример запуска:
    python3 tools/evaluate_map_fps.py \\
        --dataset exdark \\
        --root /path/to/ExDark \\
        --model weights/best.pt \\
        --imgsz 768 --conf 0.25 --iou 0.5 \\
        --enhance off --half \\
        --output experiments/results/exdark_v4_off.json
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from dataclasses import dataclass, asdict, field

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('evaluate_map_fps')

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    sys.exit(f"[ERROR] missing dependency: {exc}")


# =====================================================================
# КОНФИГУРАЦИЯ
# =====================================================================
@dataclass
class EvalConfig:
    dataset: str = 'exdark'         # exdark / mot17 / reside / voc / coco / yolo
    root: str = ''
    model_path: str = 'yolo11m-pose.pt'
    inference_size: int = 768
    device: str = 'cuda'
    half_precision: bool = False
    enhance_mode: str = 'off'       # off / night / fog / smoke / rain / auto
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    limit_images: Optional[int] = None
    warmup_iterations: int = 10
    output_json: Optional[str] = None
    person_class_id: int = 0


# =====================================================================
# ПАРСЕРЫ АННОТАЦИЙ
# =====================================================================
class AnnotationParser:
    """Парсеры для шести форматов разметок, поддерживаемых в работе."""

    @staticmethod
    def parse_yolo(txt: str, img_w: int, img_h: int,
                   person_class: int = 0) -> List[Tuple[float, float, float, float]]:
        boxes: List[Tuple[float, float, float, float]] = []
        if not os.path.exists(txt):
            return boxes
        with open(txt) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    if int(parts[0]) != person_class:
                        continue
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError:
                    continue
                boxes.append((
                    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                ))
        return boxes

    @staticmethod
    def parse_exdark_txt(txt: str) -> List[Tuple[float, float, float, float]]:
        """ExDark формат: `Person x y w h ...`."""
        boxes: List[Tuple[float, float, float, float]] = []
        if not os.path.exists(txt):
            return boxes
        keywords = {'people', 'person'}
        with open(txt, errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(('%', '#')):
                    continue
                parts = line.split()
                if len(parts) < 5 or parts[0].lower() not in keywords:
                    continue
                try:
                    x, y = float(parts[1]), float(parts[2])
                    w, h = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                if w > 0 and h > 0:
                    boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def parse_voc(xml_path: str,
                  target: str = 'person') -> List[Tuple[float, float, float, float]]:
        import xml.etree.ElementTree as ET
        if not os.path.exists(xml_path):
            return []
        boxes: List[Tuple[float, float, float, float]] = []
        try:
            tree = ET.parse(xml_path)
            for obj in tree.findall('object'):
                name = obj.find('name')
                if name is None or name.text.lower() != target.lower():
                    continue
                bb = obj.find('bndbox')
                if bb is None:
                    continue
                boxes.append((
                    float(bb.find('xmin').text), float(bb.find('ymin').text),
                    float(bb.find('xmax').text), float(bb.find('ymax').text),
                ))
        except Exception as exc:
            logger.warning("VOC parse %s: %s", xml_path, exc)
        return boxes

    @staticmethod
    def parse_coco(json_path: str,
                   target_class_id: int = 1) -> Dict[str, List[Tuple[float, float, float, float]]]:
        with open(json_path) as fh:
            coco = json.load(fh)
        img_by_id = {im['id']: im['file_name'] for im in coco['images']}
        result: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for ann in coco['annotations']:
            if ann['category_id'] != target_class_id:
                continue
            fn = img_by_id.get(ann['image_id'])
            if not fn:
                continue
            x, y, w, h = ann['bbox']
            result.setdefault(fn, []).append((x, y, x + w, y + h))
        return result

    @staticmethod
    def parse_mot17_gt(gt_txt: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
        result: Dict[int, List[Tuple[float, float, float, float]]] = {}
        if not os.path.exists(gt_txt):
            return result
        with open(gt_txt) as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                try:
                    frame, cls = int(parts[0]), int(parts[7])
                    vis = float(parts[8]) if len(parts) > 8 else 1.0
                except ValueError:
                    continue
                if cls != 1 or vis < 0.25:
                    continue
                try:
                    x, y = float(parts[2]), float(parts[3])
                    w, h = float(parts[4]), float(parts[5])
                except ValueError:
                    continue
                result.setdefault(frame, []).append((x, y, x + w, y + h))
        return result


# =====================================================================
# IoU и AP
# =====================================================================
def iou_xyxy(b1: Tuple[float, float, float, float],
             b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (b1[2] - b1[0]) * (b1[3] - b1[1])
        + (b2[2] - b2[0]) * (b2[3] - b2[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def match_predictions(pred_boxes: List[Tuple[float, float, float, float]],
                      pred_scores: List[float],
                      gt_boxes: List[Tuple[float, float, float, float]],
                      iou_thresh: float = 0.5) -> Tuple[List[Tuple[float, bool]], int]:
    """Сопоставление предсказаний с GT по убыванию confidence."""
    if not pred_boxes:
        return [], len(gt_boxes)
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    gt_used = [False] * len(gt_boxes)
    matches: List[Tuple[float, bool]] = []
    for idx in order:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            v = iou_xyxy(pred_boxes[idx], gb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and best_j >= 0:
            gt_used[best_j] = True
            matches.append((pred_scores[idx], True))
        else:
            matches.append((pred_scores[idx], False))
    return matches, len(gt_boxes)


def compute_ap(all_matches: List[Tuple[float, bool]],
               total_gt: int) -> Tuple[float, float, float]:
    """11-point AP (COCO-style 101 interpolation), P, R."""
    if not all_matches or total_gt == 0:
        return 0.0, 0.0, 0.0
    all_matches.sort(key=lambda x: -x[0])
    tp = np.cumsum([1 if m[1] else 0 for m in all_matches])
    fp = np.cumsum([0 if m[1] else 1 for m in all_matches])
    recall = tp / total_gt
    precision = tp / (tp + fp)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        if (recall >= t).any():
            p = precision[recall >= t].max()
        else:
            p = 0.0
        ap += p
    ap /= 101.0
    return float(ap), float(precision[-1]), float(recall[-1])


# =====================================================================
# ENHANCER
# =====================================================================
def load_enhancer(mode: str):
    """Загрузка RescueEnhancer из пакета rescue_vision."""
    if mode == 'off':
        return None
    # Сначала пробуем импортировать из локального пакета rescue_vision
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in [repo_root, repo_root / 'rescue_vision']:
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from rescue_vision.rescue_vision.rescue_enhancer import RescueEnhancer
        from rescue_vision.rescue_vision.enhancement_modes import EnhancementMode
    except ImportError:
        try:
            from rescue_vision.rescue_enhancer import RescueEnhancer
            from rescue_vision.enhancement_modes import EnhancementMode
        except ImportError:
            logger.warning("rescue_vision не найден — enhancement отключён")
            return None

    mode_map = {
        'night': EnhancementMode.NIGHT,
        'fog':   EnhancementMode.FOG,
        'smoke': EnhancementMode.SMOKE,
        'rain':  EnhancementMode.RAIN,
        'auto':  EnhancementMode.AUTO,
    }
    enum_mode = mode_map.get(mode.lower(), EnhancementMode.OFF)
    return RescueEnhancer(enum_mode)


# =====================================================================
# СБОР ПАР image -> gt_boxes ПО ДАТАСЕТАМ
# =====================================================================
def collect_exdark(root: Path) -> List[Tuple[Path, List[Tuple[float, float, float, float]]]]:
    pairs = []
    images_root = None
    annotations_root = None
    for candidate in [root, root / 'ExDark', root / 'images']:
        if (candidate / 'People').is_dir():
            images_root = candidate
            break
        if (candidate / 'Person').is_dir():
            images_root = candidate
            break
    for candidate in [root / 'ExDark_Annno', root / 'annotations',
                      root / 'ExDark_Anno', root]:
        if (candidate / 'People').is_dir() or (candidate / 'Person').is_dir():
            annotations_root = candidate
            break
    if images_root is None or annotations_root is None:
        logger.error("ExDark: не найдена структура People/Person")
        return pairs

    class_dir = 'People' if (images_root / 'People').is_dir() else 'Person'
    person_dir = images_root / class_dir
    ann_dir = annotations_root / class_dir
    images = sorted(
        list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.JPG'))
        + list(person_dir.glob('*.png')) + list(person_dir.glob('*.JPEG'))
    )
    for img_path in images:
        ann_candidates = [
            ann_dir / (img_path.name + '.txt'),
            ann_dir / (img_path.stem + '.txt'),
        ]
        ann_path = next((c for c in ann_candidates if c.exists()), None)
        if ann_path is None:
            continue
        boxes = AnnotationParser.parse_exdark_txt(str(ann_path))
        if boxes:
            pairs.append((img_path, boxes))
    return pairs


def collect_yolo(root: Path) -> List[Tuple[Path, List[Tuple[float, float, float, float]]]]:
    """YOLO-формат: images/*.jpg + labels/*.txt с относительными координатами."""
    pairs = []
    images_dir = root / 'images' if (root / 'images').is_dir() else root
    labels_dir = root / 'labels' if (root / 'labels').is_dir() else root
    for img_path in sorted(images_dir.rglob('*.jpg')):
        label_path = labels_dir / (img_path.stem + '.txt')
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = AnnotationParser.parse_yolo(str(label_path), w, h)
        if boxes:
            pairs.append((img_path, boxes))
    return pairs


def collect_mot17(root: Path) -> List[Tuple[Path, List[Tuple[float, float, float, float]]]]:
    pairs = []
    train_dir = root / 'train' if (root / 'train').is_dir() else root
    for seq in sorted(train_dir.iterdir()):
        if not seq.is_dir():
            continue
        img_dir = seq / 'img1'
        gt_file = seq / 'gt' / 'gt.txt'
        if not img_dir.is_dir() or not gt_file.is_file():
            continue
        gt_by_frame = AnnotationParser.parse_mot17_gt(str(gt_file))
        for img_path in sorted(img_dir.glob('*.jpg')):
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                continue
            boxes = gt_by_frame.get(frame_idx, [])
            if boxes:
                pairs.append((img_path, boxes))
    return pairs


def collect_voc(root: Path) -> List[Tuple[Path, List[Tuple[float, float, float, float]]]]:
    pairs = []
    img_dir = root / 'JPEGImages' if (root / 'JPEGImages').is_dir() else root / 'images'
    ann_dir = root / 'Annotations' if (root / 'Annotations').is_dir() else root / 'annotations'
    for img_path in sorted(img_dir.glob('*.jpg')):
        ann_path = ann_dir / (img_path.stem + '.xml')
        boxes = AnnotationParser.parse_voc(str(ann_path))
        if boxes:
            pairs.append((img_path, boxes))
    return pairs


def collect_coco(root: Path,
                 ann_file: str = 'annotations.json') -> List[Tuple[Path, List[Tuple[float, float, float, float]]]]:
    ann_path = root / ann_file
    if not ann_path.is_file():
        # Альтернатива: instances_val2017.json
        candidates = list(root.glob('**/instances_*.json'))
        if not candidates:
            return []
        ann_path = candidates[0]
    by_file = AnnotationParser.parse_coco(str(ann_path), target_class_id=1)
    pairs = []
    for img_dir_name in ['val2017', 'images', '']:
        img_dir = root / img_dir_name if img_dir_name else root
        if not img_dir.is_dir():
            continue
        for fn, boxes in by_file.items():
            p = img_dir / fn
            if p.is_file() and boxes:
                pairs.append((p, boxes))
        if pairs:
            break
    return pairs


def collect_pairs(cfg: EvalConfig) -> List[Tuple[Path, List[Tuple[float, float, float, float]]]]:
    root = Path(cfg.root)
    if not root.is_dir():
        sys.exit(f"[ERROR] dataset root does not exist: {root}")
    ds = cfg.dataset.lower()
    if ds == 'exdark':
        pairs = collect_exdark(root)
    elif ds in ('reside', 'yolo'):
        pairs = collect_yolo(root)
    elif ds == 'mot17':
        pairs = collect_mot17(root)
    elif ds == 'voc':
        pairs = collect_voc(root)
    elif ds == 'coco':
        pairs = collect_coco(root)
    else:
        sys.exit(f"[ERROR] unknown dataset: {cfg.dataset}")
    if cfg.limit_images:
        pairs = pairs[:cfg.limit_images]
    return pairs


# =====================================================================
# ОСНОВНОЙ ЦИКЛ
# =====================================================================
def run_evaluation(cfg: EvalConfig) -> Dict:
    pairs = collect_pairs(cfg)
    if not pairs:
        sys.exit("[ERROR] не найдено ни одной пары image+GT")
    logger.info("Загружено пар image+GT: %d", len(pairs))

    logger.info("Загрузка модели %s", cfg.model_path)
    model = YOLO(cfg.model_path)
    if cfg.device.startswith('cuda') and torch.cuda.is_available():
        model.to(cfg.device)

    enhancer = load_enhancer(cfg.enhance_mode)
    logger.info("Enhance mode: %s (active=%s)", cfg.enhance_mode, enhancer is not None)

    # Warm-up CUDA для стабильного измерения FPS
    if cfg.device.startswith('cuda') and torch.cuda.is_available():
        dummy = np.zeros((cfg.inference_size, cfg.inference_size, 3), dtype=np.uint8)
        for _ in range(cfg.warmup_iterations):
            _ = model.predict(dummy, imgsz=cfg.inference_size,
                              conf=cfg.conf_threshold, iou=cfg.iou_threshold,
                              half=cfg.half_precision, device=cfg.device,
                              classes=[cfg.person_class_id], verbose=False)
        torch.cuda.synchronize()

    all_matches: List[Tuple[float, bool]] = []
    total_gt = 0
    total_pred = 0
    enhance_times_ms: List[float] = []
    infer_times_ms: List[float] = []
    total_t_start = time.time()

    for i, (img_path, gt_boxes) in enumerate(pairs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Этап 1: улучшение
        if enhancer is not None:
            t_enh = time.time()
            enhanced = enhancer.enhance(frame)
            if cfg.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
            enhance_times_ms.append((time.time() - t_enh) * 1000)
        else:
            enhanced = frame
            enhance_times_ms.append(0.0)

        # Этап 2: инференс YOLO
        t_inf = time.time()
        results = model.predict(
            enhanced,
            imgsz=cfg.inference_size,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            half=cfg.half_precision,
            device=cfg.device,
            classes=[cfg.person_class_id],
            verbose=False,
        )
        if cfg.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_times_ms.append((time.time() - t_inf) * 1000)

        # Этап 3: парсинг предсказаний
        r = results[0]
        pred_boxes: List[Tuple[float, float, float, float]] = []
        pred_scores: List[float] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(xyxy, scores):
                pred_boxes.append(tuple(map(float, box)))
                pred_scores.append(float(score))
        total_pred += len(pred_boxes)

        # Этап 4: сопоставление
        matches, n_gt = match_predictions(
            pred_boxes, pred_scores, gt_boxes, iou_thresh=cfg.iou_threshold,
        )
        all_matches.extend(matches)
        total_gt += n_gt

        if (i + 1) % 100 == 0:
            mean_inf = float(np.mean(infer_times_ms[-100:]))
            logger.info("[%d/%d] inference avg %.1f ms", i + 1, len(pairs), mean_inf)

    total_elapsed = time.time() - total_t_start

    # Метрики
    ap, _, _ = compute_ap(list(all_matches), total_gt)
    tp = sum(1 for _, hit in all_matches if hit)
    fp = sum(1 for _, hit in all_matches if not hit)
    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fps = len(pairs) / total_elapsed if total_elapsed > 0 else 0.0

    summary = {
        'config': asdict(cfg),
        'metrics': {
            'map50': round(ap, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'fps': round(fps, 2),
        },
        'timings': {
            'avg_inference_ms': round(float(np.mean(infer_times_ms)), 2),
            'avg_enhance_ms': round(float(np.mean(enhance_times_ms)), 2),
            'total_elapsed_s': round(total_elapsed, 2),
        },
        'counts': {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'total_gt': int(total_gt), 'total_predictions': int(total_pred),
            'n_images': len(pairs),
        },
    }
    return summary


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True,
                   choices=['exdark', 'mot17', 'reside', 'voc', 'coco', 'yolo'])
    p.add_argument('--root', required=True, help='Корневой каталог датасета')
    p.add_argument('--model', dest='model_path', default='yolo11m-pose.pt')
    p.add_argument('--imgsz', dest='inference_size', type=int, default=768)
    p.add_argument('--device', default='cuda')
    p.add_argument('--half', action='store_true', dest='half_precision',
                   help='Включить FP16 на GPU')
    p.add_argument('--enhance', dest='enhance_mode', default='off',
                   choices=['off', 'night', 'fog', 'smoke', 'rain', 'auto'])
    p.add_argument('--conf', dest='conf_threshold', type=float, default=0.25)
    p.add_argument('--iou', dest='iou_threshold', type=float, default=0.5)
    p.add_argument('--limit', dest='limit_images', type=int, default=None)
    p.add_argument('--warmup', dest='warmup_iterations', type=int, default=10)
    p.add_argument('--output', dest='output_json', default=None)
    args = p.parse_args()
    return EvalConfig(**vars(args))


def main() -> int:
    cfg = parse_args()
    logger.info("Конфигурация: %s", asdict(cfg))
    summary = run_evaluation(cfg)

    metrics = summary['metrics']
    print()
    print('=' * 60)
    print(f'Dataset:    {cfg.dataset.upper()}  ({summary["counts"]["n_images"]} images)')
    print(f'Model:      {cfg.model_path}')
    print(f'Enhance:    {cfg.enhance_mode}  (imgsz={cfg.inference_size}, '
          f'conf={cfg.conf_threshold}, iou={cfg.iou_threshold}, '
          f'half={cfg.half_precision})')
    print('-' * 60)
    print(f'  mAP@0.5   = {metrics["map50"]:.4f}')
    print(f'  Precision = {metrics["precision"]:.4f}')
    print(f'  Recall    = {metrics["recall"]:.4f}')
    print(f'  F1-score  = {metrics["f1"]:.4f}')
    print(f'  FPS       = {metrics["fps"]:.2f}')
    print(f'  TP={summary["counts"]["tp"]}  FP={summary["counts"]["fp"]}  '
          f'FN={summary["counts"]["fn"]}  GT={summary["counts"]["total_gt"]}')
    print(f'  inference={summary["timings"]["avg_inference_ms"]:.1f} ms / '
          f'enhance={summary["timings"]["avg_enhance_ms"]:.1f} ms')
    print('=' * 60)

    if cfg.output_json:
        Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_json, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info("Результат сохранён: %s", cfg.output_json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
