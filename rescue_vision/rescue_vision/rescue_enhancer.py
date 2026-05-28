"""RescueEnhancer — адаптивный модуль улучшения изображения.

Соответствует Листингу В.1 (Приложение В) ВКР Миргазева М.А., 2026.
Реализует пять режимов работы:

  OFF   — без обработки (полноценный рабочий режим, не отсутствие);
  NIGHT — CLAHE + гамма-коррекция + линейное растяжение
          (низкая освещённость, см. подраздел 2.3 ВКР);
  FOG   — Dark Channel Prior на 1/2 разрешении + CLAHE
          (туман, He K. et al., 2011, см. подраздел 2.1 ВКР);
  SMOKE — Single-Scale Retinex на 1/4 разрешении + CLAHE
          (дым, Jobson D. et al., 1997, см. подраздел 2.2 ВКР);
  RAIN  — медианный фильтр (5x5) + билатеральный фильтр
          (осадки, см. подраздел 2.4 ВКР).

Ключевая оптимизация: для FOG и SMOKE тяжёлые операции (Dark Channel,
Gaussian для Retinex) выполняются на уменьшенной копии кадра, а
результат масштабируется обратно. Это даёт прирост FPS ≈ 30-50 % без
значимой потери качества (см. Таблица 11 ВКР, FPS-колонка).
"""
import time
from typing import Optional

import cv2
import numpy as np

from .enhancement_modes import EnhancementMode
from .scene_analyzer import SceneAnalyzer


# =====================================================================
# Look-up table для гамма-коррекции (вычисляется один раз на старте)
# =====================================================================
def create_gamma_lut(gamma: float) -> np.ndarray:
    """Предвычисленная LUT для гамма-коррекции I_out = 255 * (I_in/255)^(1/gamma).

    Args:
        gamma: показатель степени.
            * gamma < 1.0 — осветляет тёмные области (для NIGHT, например 0.4);
            * gamma = 1.0 — тождественное преобразование;
            * gamma > 1.0 — затемняет светлые области.

    Returns:
        LUT формы (256,) типа uint8.
    """
    if gamma <= 0:
        raise ValueError(f'gamma must be positive, got {gamma}')
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.float32,
    )
    return np.clip(table, 0, 255).astype(np.uint8)


# =====================================================================
# RescueEnhancer — основной класс
# =====================================================================
class RescueEnhancer:
    """Адаптивный модуль улучшения изображения для пяти режимов."""

    def __init__(
        self,
        mode: EnhancementMode = EnhancementMode.AUTO,
        scene_analyzer: Optional[SceneAnalyzer] = None,
    ):
        self.current_mode = mode
        self.scene_analyzer = scene_analyzer or SceneAnalyzer()
        self.enhance_time_ms: float = 0.0
        self.last_active_mode: EnhancementMode = EnhancementMode.OFF

        # ----- CLAHE-экземпляры (создаются один раз) -----
        # NIGHT — агрессивный contrast limit для тёмных кадров
        self.clahe_night = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        # FOG — мягкий contrast limit после Dark Channel Prior
        self.clahe_fog = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        # SMOKE — мягкий contrast limit после Single-Scale Retinex
        self.clahe_smoke = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

        # ----- Предвычисленные LUT -----
        # NIGHT: gamma = 0.4 (агрессивное осветление тёмных областей).
        # Значение подобрано экспериментально на ExDark (см. подраздел
        # 4.8 ВКР, ablation gamma).
        self.gamma_lut_night = create_gamma_lut(0.4)

    # ------------------------------------------------------------------
    def set_mode(self, mode: EnhancementMode) -> None:
        """Динамическая смена режима (вызывается из ROS 2 callback)."""
        self.current_mode = mode

    # ------------------------------------------------------------------
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Главный метод: применить выбранный режим к кадру.

        Возвращает новый кадр (BGR uint8). Время обработки доступно через
        атрибут self.enhance_time_ms.
        """
        if frame is None or frame.size == 0:
            self.enhance_time_ms = 0.0
            return np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame

        start_time = time.time()
        mode = self.current_mode
        if mode == EnhancementMode.AUTO:
            mode = self.scene_analyzer.analyze(frame)

        self.last_active_mode = mode

        if mode == EnhancementMode.OFF:
            result = frame.copy()
        elif mode == EnhancementMode.NIGHT:
            result = self._enhance_night(frame)
        elif mode == EnhancementMode.FOG:
            result = self._enhance_fog(frame)
        elif mode == EnhancementMode.SMOKE:
            result = self._enhance_smoke(frame)
        elif mode == EnhancementMode.RAIN:
            result = self._enhance_rain(frame)
        else:
            result = frame.copy()

        self.enhance_time_ms = (time.time() - start_time) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Режим NIGHT: CLAHE + гамма-коррекция + линейное растяжение
    # ------------------------------------------------------------------
    def _enhance_night(self, frame: np.ndarray) -> np.ndarray:
        """Низкая освещённость: CLAHE + gamma + linear stretch."""
        # 1) CLAHE на L-канале LAB-пространства
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_night.apply(l)

        # 2) Гамма-коррекция (LUT, O(1) на пиксель)
        l = cv2.LUT(l, self.gamma_lut_night)

        # 3) Линейное растяжение динамического диапазона
        # Берём 2 и 98 процентили вместо min/max для устойчивости к шуму
        lo = np.percentile(l, 2)
        hi = np.percentile(l, 98)
        if hi - lo > 1e-3:
            l = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo),
                        0, 255).astype(np.uint8)

        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Режим FOG: Dark Channel Prior на 1/2 разрешении
    # ------------------------------------------------------------------
    def _enhance_fog(self, frame: np.ndarray,
                     patch_size: int = 7,
                     omega: float = 0.85,
                     t_min: float = 0.15) -> np.ndarray:
        """Dehazing по Dark Channel Prior (He, Sun, Tang, 2011).

        Модель атмосферного рассеяния:
            I(x) = J(x) * t(x) + A * (1 - t(x))

        Оптимизация: тяжёлые операции (минимум по каналам, эрозия,
        оценка A) выполняются на изображении 1/2 разрешения. Карта
        пропускания затем масштабируется обратно.
        """
        h, w = frame.shape[:2]

        # 1) Уменьшаем кадр для ускорения
        small = cv2.resize(
            frame, (max(1, w // 2), max(1, h // 2)),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2) Dark channel = минимум по 3 каналам + эрозия в окне patch_size
        min_channel = np.min(small, axis=2)
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)

        # 3) Атмосферный свет — среднее по top-0.1% самых ярких пикселей
        # dark_channel (соответствует наиболее туманным областям)
        flat_dc = dark_channel.flatten()
        top_n = max(1, int(flat_dc.size * 0.001))
        idx = np.argpartition(flat_dc, -top_n)[-top_n:]
        atmospheric = float(np.mean(small.reshape(-1, 3)[idx]))
        # Ограничиваем разумным диапазоном
        atmospheric = max(160.0, min(atmospheric, 245.0))

        # 4) Карта пропускания t(x) = 1 - omega * dark_channel / A
        transmission = 1.0 - omega * (dark_channel.astype(np.float32) / atmospheric)
        transmission = np.clip(transmission, t_min, 1.0)

        # 5) Возвращаем к полному разрешению
        transmission_full = cv2.resize(
            transmission, (w, h), interpolation=cv2.INTER_LINEAR,
        )
        t_3ch = np.stack([transmission_full] * 3, axis=2)

        # 6) Восстанавливаем J(x) = (I(x) - A) / max(t(x), t_min) + A
        recovered = (frame.astype(np.float32) - atmospheric) / t_3ch + atmospheric
        result = np.clip(recovered, 0, 255).astype(np.uint8)

        # 7) Лёгкий CLAHE для восстановления локального контраста
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_fog.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Режим SMOKE: Single-Scale Retinex на 1/4 разрешении
    # ------------------------------------------------------------------
    def _enhance_smoke(self, frame: np.ndarray,
                       sigma: float = 30.0) -> np.ndarray:
        """Single-Scale Retinex (Jobson, Rahman, Woodell, 1997).

        Алгоритм рассматривает изображение как I(x,y) = L(x,y) * R(x,y),
        где L — освещённость, R — отражательная способность. В лог-форме:
            log R(x,y) = log I(x,y) - log( G(x,y; sigma) * I(x,y) )

        Оптимизация: Gaussian blur выполняется на 1/4 разрешении, что даёт
        значимый прирост FPS без видимых артефактов.
        """
        h, w = frame.shape[:2]

        # 1) Уменьшаем кадр для тяжёлой свёртки
        small = cv2.resize(
            frame, (max(1, w // 4), max(1, h // 4)),
            interpolation=cv2.INTER_AREA,
        )

        # 2) Оценка освещённости через гауссов фильтр
        img = small.astype(np.float32) + 1.0
        illumination = cv2.GaussianBlur(
            img, (0, 0), sigmaX=sigma, sigmaY=sigma,
        ) + 1.0

        # 3) Retinex log-разность
        retinex = np.log(img) - np.log(illumination)
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        retinex = retinex.astype(np.uint8)

        # 4) Возврат к полному разрешению
        result = cv2.resize(retinex, (w, h), interpolation=cv2.INTER_LINEAR)

        # 5) Лёгкий CLAHE для финального улучшения локального контраста
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe_smoke.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Режим RAIN: медианный + билатеральный фильтр
    # ------------------------------------------------------------------
    def _enhance_rain(self, frame: np.ndarray) -> np.ndarray:
        """Удаление осадков: медианный + билатеральный фильтр.

        Медианный фильтр (5x5) удаляет тонкие направленные структуры
        капель, не размывая границы. Билатеральный фильтр сглаживает
        результат, сохраняя границы значимых объектов.
        """
        # Медианный с маленьким ядром — удаляет точечные структуры
        filtered = cv2.medianBlur(frame, 5)
        # Билатеральный — сохраняет границы
        return cv2.bilateralFilter(filtered, d=5, sigmaColor=50, sigmaSpace=50)
