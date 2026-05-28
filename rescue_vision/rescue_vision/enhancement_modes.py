"""EnhancementMode — режимы работы модуля улучшения изображения.

Пять основных режимов соответствуют главе 2 ВКР:
  OFF   — без улучшения (штатный рабочий режим)
  NIGHT — CLAHE + гамма-коррекция + линейное растяжение (низкая освещённость)
  FOG   — Dark Channel Prior (туман)
  SMOKE — Single-Scale Retinex (дым)
  RAIN  — медианный + билатеральный фильтр (осадки)

Шестой служебный режим AUTO означает динамический выбор одного из пяти
основных по решению ImprovedSceneAnalyzer на текущий кадр.

Числовые ID соответствуют полю mode_id в Листинге А.2 (SceneMode.msg).
"""
from enum import IntEnum
from typing import Iterable


class EnhancementMode(IntEnum):
    """Пять основных режимов + служебный AUTO."""

    OFF = 0
    NIGHT = 1
    FOG = 2
    SMOKE = 3
    RAIN = 4
    AUTO = 5

    @property
    def value_str(self) -> str:
        """Строковое представление в нижнем регистре (для CLI / YAML)."""
        return self.name.lower()

    @classmethod
    def from_string(cls, value: str) -> 'EnhancementMode':
        """Парсинг строкового представления.

        Поддерживается:
            'off', 'OFF', 'Off', 0, '0' — все вернут EnhancementMode.OFF

        Raises:
            ValueError: если строка не соответствует ни одному из режимов.
        """
        if isinstance(value, EnhancementMode):
            return value
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f'Unknown enhancement mode id: {value}')
        if not isinstance(value, str):
            raise ValueError(
                f'Expected str or int, got {type(value).__name__}: {value}',
            )
        v = value.strip().lower()
        if v.isdigit():
            return cls(int(v))
        for mode in cls:
            if mode.name.lower() == v:
                return mode
        valid: Iterable[str] = (m.name.lower() for m in cls)
        raise ValueError(
            f'Unknown enhancement mode: {value!r}. '
            f'Valid options: {", ".join(valid)}',
        )

    def __str__(self) -> str:
        return self.name
