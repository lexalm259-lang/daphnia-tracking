"""
Daphnia Tracking System
Автоматизированная система биоиндикации качества водных сред
"""

from .tracker import TrajectoryMerger
from .feature_extractor import (
    extract_basic_features,
    extract_extended_features,
    extract_subwindow_aggregates,
    extract_features_from_trajectory,
)
from .fractal_dimension import fractal_dimension

__version__ = "1.0.0"
__author__ = "Порошин Алексей Васильевич"

__all__ = [
    "TrajectoryMerger",
    "extract_basic_features",
    "extract_extended_features",
    "extract_subwindow_aggregates",
    "extract_features_from_trajectory",
    "fractal_dimension",
]