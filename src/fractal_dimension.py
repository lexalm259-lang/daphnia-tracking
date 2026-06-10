"""
Модуль вычисления фрактальной размерности методом счётных ящиков (box-counting).
"""

import numpy as np


def fractal_dimension(points: np.ndarray, max_splits: int = 3) -> float:
    """
    Вычисляет фрактальную размерность (клеточную) для набора 2D точек
    методом счётных ящиков (box-counting dimension).
    
    Параметры:
        points : np.ndarray
            Массив Nx2 координат точек (x, y).
        max_splits : int
            Максимальное количество разбиений сетки (по умолчанию 3).
    
    Возвращает:
        float
            Оценка фрактальной размерности (в диапазоне [1, 2] для 2D траекторий)
            или 1.0 при недостатке данных.
    """
    if len(points) < 10:
        return 1.0
    
    # Нормализация данных в диапазон [0, 1] x [0, 1]
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    
    points_norm = (points - min_vals) / ranges
    
    # Подготовка размеров ячеек (эпсилон)
    powers = np.arange(0, max_splits)
    scales = 2.0 ** -powers
    
    counts = []
    
    # Подсчёт занятых ячеек для каждого масштаба
    for scale in scales:
        bins = np.floor(points_norm / scale).astype(int)
        unique_bins = np.unique(bins, axis=0)
        counts.append(len(unique_bins))
    
    counts = np.array(counts)
    
    # Линейная регрессия в логарифмических координатах
    valid_mask = counts > 0
    if np.sum(valid_mask) < 2:
        return 1.0
    
    log_inv_scales = np.log(1.0 / scales[valid_mask])
    log_counts = np.log(counts[valid_mask])
    
    coeffs = np.polyfit(log_inv_scales, log_counts, 1)
    D = coeffs[0]
    
    return D