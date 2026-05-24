import time
from typing import Optional

import cv2
import numpy as np

from .enhancement_modes import EnhancementMode
from .scene_analyzer import SceneAnalyzer


def create_gamma_lut(gamma: float) -> np.ndarray:
    """Precompute look-up table for gamma correction."""
    inv = 1.0 / gamma
    return np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)


class RescueEnhancer:
    """Adaptive image enhancement module for rescue vision scenes."""

    def __init__(
        self,
        mode: EnhancementMode = EnhancementMode.AUTO,
        scene_analyzer: Optional[SceneAnalyzer] = None,
    ):
        self.current_mode = mode
        self.scene_analyzer = scene_analyzer or SceneAnalyzer()
        self.enhance_time_ms = 0.0
        self.clahe_night = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_fog = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.gamma_lut = create_gamma_lut(1.35)

    def set_mode(self, mode: EnhancementMode) -> None:
        self.current_mode = mode

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        mode = self.current_mode
        if mode == EnhancementMode.AUTO:
            mode = self.scene_analyzer.analyze(frame)

        if mode == EnhancementMode.OFF:
            self.enhance_time_ms = (time.time() - start_time) * 1000
            return frame.copy()
        if mode == EnhancementMode.NIGHT:
            result = self._enhance_night(frame)
        elif mode == EnhancementMode.FOG:
            result = self._enhance_fog(frame)
        elif mode == EnhancementMode.SMOKE:
            result = self._enhance_smoke(frame)
        elif mode == EnhancementMode.RAIN:
            result = self._enhance_rain(frame)
        else:
            result = frame.copy()

        self.enhance_time_ms = (time.time() - start_time) * 1000
        return result

    def _enhance_night(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_night.apply(l)
        corrected = cv2.LUT(l, self.gamma_lut)
        merged = cv2.merge((corrected, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _enhance_fog(self, frame: np.ndarray) -> np.ndarray:
        """Dark Channel Prior on a downsampled frame."""
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR
        )
        patch_size = 7
        min_channel = np.min(small, axis=2)
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)
        flat_dc = dark_channel.flatten()
        top_n = max(1, int(flat_dc.size * 0.001))
        idx = np.argpartition(flat_dc, -top_n)[-top_n:]
        atmospheric = float(np.mean(small.reshape(-1, 3)[idx]))
        atmospheric = max(160.0, min(atmospheric, 245.0))
        omega = 0.85
        transmission = 1.0 - omega * (dark_channel.astype(np.float32) / atmospheric)
        transmission = np.clip(transmission, 0.15, 1.0)
        transmission_full = cv2.resize(
            transmission, (w, h), interpolation=cv2.INTER_LINEAR
        )
        t_3ch = np.stack([transmission_full] * 3, axis=2)
        recovered = (frame.astype(np.float32) - atmospheric) / t_3ch + atmospheric
        result = np.clip(recovered, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_fog.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _enhance_smoke(self, frame: np.ndarray) -> np.ndarray:
        """Single-Scale Retinex with illumination estimation on a 4x smaller image."""
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA
        )
        img = small.astype(np.float32) + 1.0
        illumination = cv2.GaussianBlur(img, (0, 0), sigmaX=30, sigmaY=30) + 1.0
        retinex = np.log(img) - np.log(illumination)
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        retinex = retinex.astype(np.uint8)
        result = cv2.resize(retinex, (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _enhance_rain(self, frame: np.ndarray) -> np.ndarray:
        filtered = cv2.medianBlur(frame, 3)
        return cv2.bilateralFilter(filtered, d=5, sigmaColor=50, sigmaSpace=50)
