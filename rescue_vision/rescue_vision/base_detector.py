import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class DetectionResult:
    track_id: int
    confidence: float
    bbox_xywh: Tuple[float, float, float, float]
    keypoints: Optional[np.ndarray]


class BaseDetector:
    """YOLO11-pose wrapper with optional ByteTrack tracking."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 768,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "cuda",
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            )
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(model_path)
        self.last_inference_ms = 0.0
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        except Exception:
            pass

    def process_frame(self, frame: np.ndarray, persist: bool = True) -> Dict[str, Any]:
        start = time.time()
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        self.last_inference_ms = (time.time() - start) * 1000
        detections = self._parse_results(results[0] if results else None)
        return {
            "detections": detections,
            "raw": results,
            "inference_ms": self.last_inference_ms,
        }

    def _parse_results(self, result: Any) -> List[DetectionResult]:
        parsed: List[DetectionResult] = []
        if result is None or result.boxes is None:
            return parsed

        boxes = result.boxes
        ids = (
            boxes.id.cpu().numpy().astype(int).tolist()
            if boxes.id is not None
            else [-1] * len(boxes)
        )
        confs = (
            boxes.conf.cpu().numpy().tolist()
            if boxes.conf is not None
            else [0.0] * len(boxes)
        )
        xywh = boxes.xywh.cpu().numpy().tolist() if boxes.xywh is not None else []
        kpts = None
        if (
            getattr(result, "keypoints", None) is not None
            and result.keypoints is not None
        ):
            try:
                kpts = result.keypoints.data.cpu().numpy()
            except Exception:
                kpts = None

        for i, box in enumerate(xywh):
            keypoints = kpts[i] if kpts is not None and i < len(kpts) else None
            parsed.append(
                DetectionResult(
                    ids[i], float(confs[i]), tuple(map(float, box)), keypoints
                )
            )
        return parsed


class MP4Detector:
    """Utility class for processing a video file frame by frame."""

    def __init__(
        self, detector: BaseDetector, enhancer=None, save_path: Optional[str] = None
    ):
        self.detector = detector
        self.enhancer = enhancer
        self.save_path = save_path

    def run(self, source: str, show: bool = False) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = self.enhancer.enhance(frame) if self.enhancer else frame
            result = self.detector.process_frame(processed)
            vis = processed.copy()
            for det in result["detections"]:
                x, y, w, h = det.bbox_xywh
                x1, y1, x2, y2 = (
                    int(x - w / 2),
                    int(y - h / 2),
                    int(x + w / 2),
                    int(y + h / 2),
                )
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"ID:{det.track_id} {det.confidence:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            if self.save_path and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self.save_path,
                    fourcc,
                    cap.get(cv2.CAP_PROP_FPS) or 25,
                    (vis.shape[1], vis.shape[0]),
                )
            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("Rescue Vision", vis)
                if cv2.waitKey(1) == 27:
                    break
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
