"""BaseDetector — обёртка YOLO11-pose с интегрированным трекером ByteTrack.

Соответствует Листингу Г.1 (Приложение Г) ВКР Миргазева М.А., 2026.
Реализует прогрев модели на тензоре фиксированного размера и единый
метод process_frame для запуска инференса с трекером и сбором pose-
keypoints.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


@dataclass
class DetectionResult:
    """Результат детекции одного человека."""
    track_id: int
    confidence: float
    bbox_xywh: Tuple[float, float, float, float]  # центр + размер
    keypoints: Optional[np.ndarray]  # (17, 3) — x, y, visibility


# Стандартный скелет COCO 17 keypoints (см. подраздел 3.4 ВКР)
SKELETON_EDGES: List[Tuple[int, int]] = [
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class BaseDetector:
    """YOLO11-pose wrapper with optional ByteTrack tracking."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 768,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = 'cuda',
        half_precision: bool = False,
        tracker_config: str = 'bytetrack.yaml',
        target_class: int = 0,
        warmup_iterations: int = 3,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Run: "
                "pip install -r requirements.txt",
            )
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half_precision = half_precision
        self.tracker_config = tracker_config
        self.target_class = target_class
        self.warmup_iterations = warmup_iterations

        self.model: Any = None
        self.last_inference_ms = 0.0
        self.frame_count = 0

        self._load_model()

    # ------------------------------------------------------------------
    # Загрузка и прогрев модели
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Загрузка весов, перевод в FP16 (опционально) и прогрев на CUDA."""
        self.model = YOLO(self.model_path)
        if HAS_TORCH and self.device.startswith('cuda') and torch.cuda.is_available():
            self.model.to(self.device)
            try:
                self.model.fuse()
            except Exception:
                pass
            if self.half_precision:
                try:
                    self.model.model = self.model.model.half()
                except Exception:
                    pass
        self.warmup()

    def warmup(self) -> None:
        """Прогрев модели dummy-инференсами для стабилизации FPS-замеров."""
        if HAS_TORCH and self.device.startswith('cuda') and torch.cuda.is_available():
            dtype = torch.float16 if self.half_precision else torch.float32
            warmup_tensor = torch.zeros(
                1, 3, self.imgsz, self.imgsz, dtype=dtype,
            ).to(self.device)
            for _ in range(self.warmup_iterations):
                try:
                    _ = self.model(warmup_tensor, verbose=False)
                except Exception:
                    break
            torch.cuda.synchronize()
        else:
            dummy = np.zeros(
                (self.imgsz, self.imgsz, 3), dtype=np.uint8,
            )
            for _ in range(self.warmup_iterations):
                try:
                    _ = self.model.predict(
                        dummy, imgsz=self.imgsz,
                        conf=self.conf, iou=self.iou,
                        device=self.device, verbose=False,
                    )
                except Exception:
                    break

    # ------------------------------------------------------------------
    # Основной метод обработки кадра
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, persist: bool = True) -> Dict[str, Any]:
        """Полный цикл детекции+трекинга для одного кадра.

        Args:
            frame: BGR изображение (uint8).
            persist: сохранять состояние трекера между вызовами.

        Returns:
            dict с ключами 'detections' (List[DetectionResult]),
            'raw' (Ultralytics result objects), 'inference_ms' (float).
        """
        start = time.time()
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half_precision,
            persist=persist,
            tracker=self.tracker_config,
            classes=[self.target_class],
            verbose=False,
        )
        if HAS_TORCH and self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_inference_ms = (time.time() - start) * 1000
        self.frame_count += 1

        first_result = results[0] if results else None
        detections = self._parse_results(first_result)
        return {
            'detections': detections,
            'raw': results,
            'inference_ms': self.last_inference_ms,
        }

    # ------------------------------------------------------------------
    # Парсинг результатов Ultralytics в DetectionResult
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_results(result: Any) -> List[DetectionResult]:
        parsed: List[DetectionResult] = []
        if result is None or result.boxes is None:
            return parsed

        boxes = result.boxes
        if len(boxes) == 0:
            return parsed

        try:
            ids = (boxes.id.cpu().numpy().astype(int).tolist()
                   if boxes.id is not None
                   else [-1] * len(boxes))
        except Exception:
            ids = [-1] * len(boxes)

        try:
            confs = boxes.conf.cpu().numpy().tolist()
        except Exception:
            confs = [0.0] * len(boxes)

        try:
            xywh = boxes.xywh.cpu().numpy().tolist()
        except Exception:
            xywh = []

        kpts_array: Optional[np.ndarray] = None
        try:
            if getattr(result, 'keypoints', None) is not None and \
               result.keypoints is not None:
                kpts_array = result.keypoints.data.cpu().numpy()
        except Exception:
            kpts_array = None

        for i, box in enumerate(xywh):
            keypoints = None
            if kpts_array is not None and i < len(kpts_array):
                keypoints = kpts_array[i]
            parsed.append(DetectionResult(
                track_id=int(ids[i]),
                confidence=float(confs[i]),
                bbox_xywh=tuple(map(float, box)),
                keypoints=keypoints,
            ))
        return parsed


# =====================================================================
# Дополнительные источники видеоданных (см. подраздел 3.5 ВКР)
# =====================================================================
class MP4Detector:
    """Wrapper для обработки MP4-видеофайла кадр за кадром.

    Реализует одну из трёх специализаций интерфейса источника видеоданных,
    описанных в подразделе 3.5 ВКР. Аналогичные классы BagDetector и
    RealSenseDetector реализуются на уровне ROS 2-узла (см. Листинг А.4).
    """

    def __init__(
        self,
        detector: BaseDetector,
        enhancer=None,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        self.detector = detector
        self.enhancer = enhancer
        self.save_path = save_path
        self.show = show

    def run(self, source: str, show: bool = False) -> Dict[str, Any]:
        """Прогон видеопотока через конвейер. Возвращает базовую статистику."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")

        show_flag = show or self.show
        writer: Optional[cv2.VideoWriter] = None
        total_frames = 0
        total_detections = 0
        t_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.enhancer is not None:
                    enhanced = self.enhancer.enhance(frame)
                else:
                    enhanced = frame
                result = self.detector.process_frame(enhanced)
                vis = self._draw(enhanced, result['detections'])

                if self.save_path and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(
                        self.save_path, fourcc, fps,
                        (vis.shape[1], vis.shape[0]),
                    )
                if writer is not None:
                    writer.write(vis)
                if show_flag:
                    cv2.imshow('Rescue Vision', vis)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                total_frames += 1
                total_detections += len(result['detections'])
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        return {
            'frames': total_frames,
            'detections': total_detections,
            'fps': total_frames / elapsed if elapsed > 0 else 0.0,
            'elapsed_s': elapsed,
        }

    @staticmethod
    def _draw(frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        vis = frame.copy()
        for det in detections:
            cx, cy, w, h = det.bbox_xywh
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, f'ID:{det.track_id} {det.confidence:.2f}',
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            if det.keypoints is not None:
                for (kx, ky, kv) in det.keypoints:
                    if kv > 0.3:
                        cv2.circle(vis, (int(kx), int(ky)), 3,
                                   (255, 200, 0), -1)
                for a, b in SKELETON_EDGES:
                    if a < len(det.keypoints) and b < len(det.keypoints):
                        if det.keypoints[a][2] > 0.3 and det.keypoints[b][2] > 0.3:
                            pt_a = (int(det.keypoints[a][0]),
                                    int(det.keypoints[a][1]))
                            pt_b = (int(det.keypoints[b][0]),
                                    int(det.keypoints[b][1]))
                            cv2.line(vis, pt_a, pt_b, (255, 100, 0), 2)
        return vis
