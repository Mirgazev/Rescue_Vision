"""
rescue_vision — подсистема технического зрения мобильного робота для
выполнения поисково-спасательных операций в условиях ухудшенной видимости.

Автор: Миргазев М. А., группа КРБО-03-22.
ВКР по направлению 15.03.06 «Мехатроника и робототехника», РТУ МИРЭА, 2026.
"""

__version__ = "1.0.0"
__author__ = "Marat Mirgazev"
__email__ = "your.email@example.com"

from .scene_analyzer import SceneAnalyzer
from .rescue_enhancer import RescueEnhancer
from .base_detector import BaseDetector

__all__ = [
    "SceneAnalyzer",
    "RescueEnhancer",
    "BaseDetector",
]
