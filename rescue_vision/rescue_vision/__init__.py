"""rescue_vision — подсистема технического зрения мобильного спасательного робота.

Бакалаврская ВКР Миргазева М.А., РТУ МИРЭА, 2026.

Доступные модули:
    * scene_analyzer       — ImprovedSceneAnalyzer (Приложение Б ВКР)
    * rescue_enhancer      — RescueEnhancer (Приложение В ВКР)
    * base_detector        — BaseDetector + MP4Detector (Приложение Г ВКР)
    * enhancement_modes    — EnhancementMode IntEnum
    * config               — VisionConfig dataclass
    * rescue_vision_node   — ROS2-узел (Приложение А, Листинг А.4)
    * operator_visualization_node — HUD оператора (Приложение А, Листинг А.5)
    * vision_pro_v5        — shim для совместимости с Листингами ВКР
"""
__version__ = '1.0.0'
__author__ = 'Mirgazev Marat Ayratovich'
__license__ = 'MIT'

# Удобный re-export для использования как `from rescue_vision import ...`
from .enhancement_modes import EnhancementMode  # noqa: F401
from .config import VisionConfig  # noqa: F401

# Тяжёлые модули (требуют cv2, torch, ultralytics) импортируются лениво,
# чтобы можно было импортировать сам пакет без полного стека зависимостей.

__all__ = ['EnhancementMode', 'VisionConfig', '__version__']
