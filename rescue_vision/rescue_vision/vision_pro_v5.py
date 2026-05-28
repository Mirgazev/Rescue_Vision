"""vision_pro_v5.py — shim-модуль для обратной совместимости с Листингами ВКР.

В тексте ВКР (Листинги А.4, Б.1, В.1, Г.1 Приложений А, Б, В, Г) импорт
выполняется в виде:

    from vision_pro_v5 import (
        ImprovedSceneAnalyzer, RescueEnhancer,
        BaseDetector, Config, EnhancementMode,
    )

Изначально все классы располагались в одном монолитном файле
vision_pro_v5.py. В репозитории они вынесены в отдельные модули пакета
rescue_vision для удобства поддержки и unit-тестирования. Этот файл
re-export-ит все необходимые символы, чтобы импорт из ВКР продолжал
работать без изменений.

Использовать в новом коде НЕ рекомендуется — этот модуль существует
только для воспроизводимости Листингов из текста ВКР.
"""
from .base_detector import BaseDetector, MP4Detector, DetectionResult  # noqa: F401
from .config import VisionConfig as Config  # noqa: F401
from .enhancement_modes import EnhancementMode  # noqa: F401
from .rescue_enhancer import RescueEnhancer, create_gamma_lut  # noqa: F401
from .scene_analyzer import (  # noqa: F401
    SceneAnalyzer,
    ImprovedSceneAnalyzer,
)

__all__ = [
    'BaseDetector',
    'MP4Detector',
    'DetectionResult',
    'Config',
    'EnhancementMode',
    'RescueEnhancer',
    'create_gamma_lut',
    'SceneAnalyzer',
    'ImprovedSceneAnalyzer',
]
