from enum import Enum

class EnhancementMode(str, Enum):
    OFF = 'off'
    NIGHT = 'night'
    FOG = 'fog'
    SMOKE = 'smoke'
    RAIN = 'rain'
    AUTO = 'auto'

    @classmethod
    def from_string(cls, value: str) -> 'EnhancementMode':
        value = (value or 'off').lower().strip()
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f'Unknown enhancement mode: {value}')
