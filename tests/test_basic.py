#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Базовые unit-тесты для пакета rescue_vision.

Запуск:
    cd rescue_vision
    python3 -m pytest ../tests/ -v
или:
    python3 -m pytest tests/test_basic.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Добавляем пакет в путь
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / 'rescue_vision'))

from rescue_vision.enhancement_modes import EnhancementMode
from rescue_vision.scene_analyzer import SceneAnalyzer, ImprovedSceneAnalyzer
from rescue_vision.rescue_enhancer import RescueEnhancer
from rescue_vision.config import VisionConfig, Config


# =====================================================================
# EnhancementMode
# =====================================================================
class TestEnhancementMode:
    def test_mode_ids(self):
        """ID режимов должны соответствовать Листингу А.2 (SceneMode.mode_id)."""
        assert EnhancementMode.OFF == 0
        assert EnhancementMode.NIGHT == 1
        assert EnhancementMode.FOG == 2
        assert EnhancementMode.SMOKE == 3
        assert EnhancementMode.RAIN == 4
        assert EnhancementMode.AUTO == 5

    def test_from_string(self):
        assert EnhancementMode.from_string('off') == EnhancementMode.OFF
        assert EnhancementMode.from_string('NIGHT') == EnhancementMode.NIGHT
        assert EnhancementMode.from_string('Fog') == EnhancementMode.FOG
        assert EnhancementMode.from_string('auto') == EnhancementMode.AUTO

    def test_from_int(self):
        assert EnhancementMode.from_string(2) == EnhancementMode.FOG
        assert EnhancementMode.from_string('3') == EnhancementMode.SMOKE

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            EnhancementMode.from_string('unknown_mode')


# =====================================================================
# SceneAnalyzer
# =====================================================================
class TestSceneAnalyzer:
    def test_alias(self):
        """ImprovedSceneAnalyzer должен быть алиасом SceneAnalyzer (Листинг А.4)."""
        assert ImprovedSceneAnalyzer is SceneAnalyzer

    def test_features_keys(self):
        sa = SceneAnalyzer()
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        feats = sa.get_features(frame)
        assert set(feats.keys()) == {
            'brightness', 'std', 'laplac_var',
            'hist_peak_bin', 'edge_density', 'diag_ratio',
        }

    def test_dark_frame_is_night(self):
        """Очень тёмный кадр должен классифицироваться как NIGHT."""
        sa = SceneAnalyzer()
        dark = np.full((480, 640, 3), 20, dtype=np.uint8)
        assert sa.analyze(dark) == EnhancementMode.NIGHT

    def test_temporal_smoothing(self):
        """История из 5 кадров с правилом 3 из 5."""
        sa = SceneAnalyzer()
        dark = np.full((480, 640, 3), 20, dtype=np.uint8)
        for _ in range(5):
            result = sa.analyze(dark)
        assert result == EnhancementMode.NIGHT
        assert sa.last_smoothed_mode == EnhancementMode.NIGHT

    def test_empty_frame(self):
        sa = SceneAnalyzer()
        assert sa.analyze(None) == EnhancementMode.OFF


# =====================================================================
# RescueEnhancer
# =====================================================================
class TestRescueEnhancer:
    @pytest.mark.parametrize('mode', [
        EnhancementMode.OFF, EnhancementMode.NIGHT, EnhancementMode.FOG,
        EnhancementMode.SMOKE, EnhancementMode.RAIN,
    ])
    def test_enhance_preserves_shape(self, mode):
        """Все режимы должны сохранять размер кадра."""
        enh = RescueEnhancer(mode)
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        out = enh.enhance(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_off_is_passthrough(self):
        """OFF не должен менять содержимое (только копия)."""
        enh = RescueEnhancer(EnhancementMode.OFF)
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        out = enh.enhance(frame)
        assert np.array_equal(out, frame)

    def test_set_mode(self):
        enh = RescueEnhancer(EnhancementMode.OFF)
        enh.set_mode(EnhancementMode.NIGHT)
        assert enh.current_mode == EnhancementMode.NIGHT

    def test_timing_recorded(self):
        enh = RescueEnhancer(EnhancementMode.NIGHT)
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        enh.enhance(frame)
        assert enh.enhance_time_ms >= 0


# =====================================================================
# Config
# =====================================================================
class TestConfig:
    def test_alias(self):
        assert Config is VisionConfig

    def test_default_imgsz(self):
        assert VisionConfig().imgsz == 768

    def test_conf_alias(self):
        c = VisionConfig()
        c.conf = 0.3
        assert c.conf_threshold == 0.3


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
