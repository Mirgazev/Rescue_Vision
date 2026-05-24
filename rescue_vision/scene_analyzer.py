# -*- coding: utf-8 -*-
"""
scene_analyzer — эвристический классификатор условий видимости.

Извлекает из BGR-кадра 5 статистических признаков:
  1. brightness     — средняя яркость
  2. std_dev        — стандартное отклонение яркости
  3. laplacian_var  — дисперсия Лапласиана (мера резкости)
  4. edge_density   — плотность краёв (от детектора Canny)
  5. diag_ratio     — диагональная асимметрия градиента

По дереву решающих правил выбирает один из 6 режимов:
  OFF / NIGHT / FOG / SMOKE / RAIN / AUTO

Решение сглаживается голосованием 3-из-5 по истории последних кадров —
это устраняет «дребезг» в пограничных случаях.

Полная блок-схема дерева решений приведена на рис. 13 ВКР.

Автор: Миргазев М. А., 2026.
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Dict


class SceneAnalyzer:
    """
    Классификатор условий видимости по 5 статистическим признакам кадра.

    Использование:
        >>> analyzer = SceneAnalyzer()
        >>> mode, features = analyzer.analyze(bgr_frame)
        >>> print(mode)  # 'NIGHT' | 'FOG' | 'SMOKE' | 'RAIN' | 'OFF'
    """

    def __init__(self,
                 history_size: int = 5,
                 vote_threshold: int = 3):
        """
        Args:
            history_size: размер окна голосования (по умолчанию 5 кадров)
            vote_threshold: минимальное число голосов за смену режима
        """
        self.history = deque(maxlen=history_size)
        self.vote_threshold = vote_threshold
        self.current_mode = 'OFF'

    def analyze(self, bgr_frame: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Главный метод — анализ одного кадра."""
        features = self._extract_features(bgr_frame)
        raw_mode = self._classify(features)
        self.history.append(raw_mode)
        self.current_mode = self._smooth_decision()
        return self.current_mode, features
