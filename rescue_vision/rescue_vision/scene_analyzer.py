"""SceneAnalyzer / ImprovedSceneAnalyzer — статистический классификатор сцены.

Соответствует Листингу Б.1 (Приложение Б) ВКР Миргазева М.А., 2026.

Класс вычисляет шесть статистических признаков кадра (brightness, std,
laplac_var, hist_peak_bin, edge_density, diag_ratio), применяет иерархическое
дерево решений с приоритетом NIGHT > FOG > SMOKE > RAIN > OFF и сглаживает
результат скользящим окном из 5 кадров с правилом "3 из 5".

Целевые количественные показатели работы (см. подраздел 4.5 ВКР):
  - точность 90,5 % на 911 кадрах ручной разметки (требование ТЗ ≥ 80 %);
  - матрица путаницы — Таблица 19 ВКР.
"""
from collections import Counter
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .enhancement_modes import EnhancementMode


class SceneAnalyzer:
    """Classifies scene degradation using interpretable image statistics."""

    DEFAULT_THRESHOLDS = dict(
        night_brightness=55,
        fog_brightness_min=150,
        fog_std_max=40,
        fog_laplac_max=100,
        smoke_brightness_range=(60, 150),
        smoke_std_max=30,
        smoke_laplac_max=150,
        rain_edge_density_min=0.04,
        rain_diagonal_ratio_min=1.3,
    )

    def __init__(
        self,
        thresholds: Optional[dict] = None,
        manifest_path: Optional[str] = None,
        history_len: int = 5,
        majority_vote: int = 3,
    ):
        if thresholds is not None:
            self.thresholds = thresholds
        elif manifest_path:
            self.thresholds = self._load_from_manifest(manifest_path)
        else:
            self.thresholds = self.DEFAULT_THRESHOLDS.copy()

        self._history: List[EnhancementMode] = []
        self._history_len = history_len
        self._majority_vote = majority_vote

        self.last_features: Dict[str, float] = {}
        self.last_raw_mode: EnhancementMode = EnhancementMode.OFF
        self.last_smoothed_mode: EnhancementMode = EnhancementMode.OFF

    # ------------------------------------------------------------------
    # Загрузка калибровочного manifest.json (опционально)
    # ------------------------------------------------------------------
    def _load_from_manifest(self, manifest_path: str) -> dict:
        thresholds = self.DEFAULT_THRESHOLDS.copy()
        if not HAS_YAML:
            return thresholds
        with open(manifest_path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        thresholds.update(data.get('thresholds', data))
        # Преобразование списка в кортеж для smoke_brightness_range
        if 'smoke_brightness_range' in thresholds and \
           isinstance(thresholds['smoke_brightness_range'], list):
            thresholds['smoke_brightness_range'] = tuple(
                thresholds['smoke_brightness_range'],
            )
        return thresholds

    # ------------------------------------------------------------------
    # Извлечение шести статистических признаков
    # ------------------------------------------------------------------
    def get_features(self, frame: np.ndarray) -> Dict[str, float]:
        if frame is None:
            return {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        brightness = float(np.mean(gray))
        std = float(np.std(gray))
        laplac_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_density = float(np.mean(magnitude > 50))
        diag_ratio = float(
            np.sum(np.abs(sobely)) / (np.sum(np.abs(sobelx)) + 1e-6)
        )

        return {
            'brightness':    brightness,
            'std':           std,
            'laplac_var':    laplac_var,
            'hist_peak_bin': int(np.argmax(hist)),
            'edge_density':  edge_density,
            'diag_ratio':    diag_ratio,
        }

    # ------------------------------------------------------------------
    # Иерархическое дерево решений (см. Таблица 2 ВКР)
    # ------------------------------------------------------------------
    def _classify_from_features(self, f: Dict[str, float]) -> EnhancementMode:
        """Decision tree: NIGHT > FOG > SMOKE > RAIN > OFF.

        Порядок важен: NIGHT проверяется первым (самый строгий критерий
        по яркости), FOG и SMOKE — следующие (низкоконтрастные с пересекающимися
        диапазонами), RAIN — отдельный путь по направленности градиентов,
        OFF — fallback.
        """
        t = self.thresholds

        if f['brightness'] < t['night_brightness']:
            return EnhancementMode.NIGHT

        if (f['brightness'] > t['fog_brightness_min']
                and f['std'] < t['fog_std_max']
                and f['laplac_var'] < t['fog_laplac_max']):
            return EnhancementMode.FOG

        sb_min, sb_max = t['smoke_brightness_range']
        if (sb_min < f['brightness'] < sb_max
                and f['std'] < t['smoke_std_max']
                and f['laplac_var'] < t['smoke_laplac_max']):
            return EnhancementMode.SMOKE

        if (f['edge_density'] > t['rain_edge_density_min']
                and f['diag_ratio'] > t['rain_diagonal_ratio_min']):
            return EnhancementMode.RAIN

        return EnhancementMode.OFF

    # ------------------------------------------------------------------
    # Главный метод с временным сглаживанием
    # ------------------------------------------------------------------
    def analyze(self, frame: np.ndarray) -> EnhancementMode:
        """Извлекает признаки, классифицирует, сглаживает по истории.

        Сглаживание окном из N кадров с правилом V из N
        (по умолчанию 3 из 5) описано в подразделе 2.4 ВКР как
        компромисс между быстротой отклика и устойчивостью.
        """
        if frame is None:
            return EnhancementMode.OFF

        features = self.get_features(frame)
        if not features:
            return EnhancementMode.OFF

        mode = self._classify_from_features(features)
        self.last_features = features
        self.last_raw_mode = mode

        self._history.append(mode)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        if len(self._history) >= self._majority_vote:
            counts = Counter(self._history)
            dominant, count = counts.most_common(1)[0]
            if count >= self._majority_vote:
                self.last_smoothed_mode = dominant
                return dominant

        self.last_smoothed_mode = mode
        return mode

    # ------------------------------------------------------------------
    # Сброс истории (полезно при смене источника видеопотока)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history.clear()
        self.last_features = {}
        self.last_raw_mode = EnhancementMode.OFF
        self.last_smoothed_mode = EnhancementMode.OFF


# =====================================================================
# Алиас для согласованности с Листингами А.4, Б.1 ВКР
# =====================================================================
# В тексте ВКР исторически используется имя ImprovedSceneAnalyzer
# (улучшенная версия начального классификатора, описанная в НИР).
# Этот алиас обеспечивает обратную совместимость с Листингами.
ImprovedSceneAnalyzer = SceneAnalyzer
