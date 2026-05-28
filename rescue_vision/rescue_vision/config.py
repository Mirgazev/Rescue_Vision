"""VisionConfig — конфигурация подсистемы технического зрения.

Соответствует структуре Config из Листингов А.4, Г.1 ВКР Миргазева М.А., 2026.
Поля совпадают с параметрами Таблицы 7 (rescue_vision_node).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionConfig:
    """Полная конфигурация подсистемы (см. Таблица 7 ВКР)."""

    # Модель ------------------------------------------------------------
    model_path: str = 'weights/best.pt'
    device: str = 'cuda:0'
    imgsz: int = 768

    # Пороги детектирования --------------------------------------------
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5

    # Режим улучшения и трекинг ----------------------------------------
    enhance_mode: str = 'auto'         # off / night / fog / smoke / rain / auto
    use_half_precision: bool = True
    tracker_config: str = 'bytetrack.yaml'

    # Источник видеопотока --------------------------------------------
    source: str = '0'                  # для standalone-режима MP4Detector
    input_topic: str = '/camera/color/image_raw'

    # Debug / визуализация ---------------------------------------------
    publish_debug_image: bool = False
    show: bool = False
    save_debug_to: Optional[str] = None
    log_csv_path: Optional[str] = None

    # --- Алиасы для совместимости со старыми Листингами ---------------
    @property
    def weights_path(self) -> str:
        """Алиас для model_path (используется в части старых импортов)."""
        return self.model_path

    @weights_path.setter
    def weights_path(self, value: str) -> None:
        self.model_path = value

    @property
    def conf(self) -> float:
        return self.conf_threshold

    @conf.setter
    def conf(self, value: float) -> None:
        self.conf_threshold = float(value)


# =====================================================================
# Алиас Config для соответствия Листингам А.4, Г.1 ВКР
# =====================================================================
# В тексте ВКР исторически используется имя Config (без префикса Vision).
Config = VisionConfig
