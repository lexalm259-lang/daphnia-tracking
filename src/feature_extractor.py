"""
Модуль извлечения поведенческих признаков из траекторий дафний.
Поддерживает различные временные окна и мультишкальную агрегацию.

Совместим с датасетами:
- dataset_5sec_7features.csv (7 признаков)
- dataset_30s15s_robust.csv (13 признаков)
- dataset_1min30s_robust.csv (13 признаков)
- dataset_5min_multiscale.csv (17 признаков)
- dataset_5min_multiscale1s_features.csv (17 признаков)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from typing import Dict, List, Optional, Tuple


def calculate_angles(x: np.ndarray, y: np.ndarray) -> List[float]:
    """
    Вычисление углов поворота между последовательными векторами скорости.
    
    Параметры:
        x, y : np.ndarray
            Координаты траектории.
    
    Возвращает:
        List[float]
            Список углов в градусах.
    """
    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angles.append(np.arccos(cos_a) * 180 / np.pi)
    return angles


def extract_basic_features(x: np.ndarray, y: np.ndarray, fps: int = 30) -> Dict:
    """
    Извлечение 7 базовых кинематических признаков.
    
    Параметры:
        x, y : np.ndarray
            Координаты траектории.
        fps : int
            Частота кадров (кадров в секунду).
    
    Возвращает:
        Dict
            Словарь с 7 базовыми признаками.
    """
    dx, dy = np.diff(x), np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    DT = 1.0 / fps
    speeds = dists / DT
    
    if len(speeds) == 0:
        return {
            'avg_speed': 0.0,
            'total_distance': 0.0,
            'straightness': 0.0,
            'median_speed': 0.0,
            'mean_turning_angle': 0.0,
            'angular_velocity': 0.0,
            'jump_frequency': 0.0
        }
    
    avg_speed = np.mean(speeds)
    total_distance = np.sum(dists)
    net_distance = euclidean([x[0], y[0]], [x[-1], y[-1]])
    straightness = net_distance / total_distance if total_distance > 1e-6 else 0.0
    median_speed = np.median(speeds)
    
    angles = calculate_angles(x, y)
    mean_turning_angle = np.mean(angles) if angles else 0.0
    angular_velocity = np.mean([a / DT for a in angles]) if angles else 0.0
    
    jump_threshold = 2.0 * median_speed
    jumps = np.sum(speeds > jump_threshold)
    jump_frequency = jumps / (len(speeds) * DT)
    
    return {
        'avg_speed': avg_speed,
        'total_distance': total_distance,
        'straightness': straightness,
        'median_speed': median_speed,
        'mean_turning_angle': mean_turning_angle,
        'angular_velocity': angular_velocity,
        'jump_frequency': jump_frequency
    }


def extract_extended_features(x: np.ndarray, y: np.ndarray, fps: int = 30) -> Dict:
    """
    Извлечение 11 признаков (7 базовых + 4 расширенных).
    
    Расширенные признаки:
    - trajectory_entropy: энтропия траектории
    - fractal_dimension: фрактальная размерность
    - sinuosity: извилистость
    - velocity_autocorrelation: автокорреляция скорости
    
    Параметры:
        x, y : np.ndarray
            Координаты траектории.
        fps : int
            Частота кадров.
    
    Возвращает:
        Dict
            Словарь с 11 признаками.
    """
    features = extract_basic_features(x, y, fps)
    
    dx, dy = np.diff(x), np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    speeds = dists / fps
    
    # Энтропия траектории
    H = 0.0
    if len(x) >= 10:
        hist, _, _ = np.histogram2d(x, y, bins=10)
        probs = hist.flatten()
        probs = probs / probs.sum()
        probs = probs[probs > 0]
        if len(probs) > 0:
            H = entropy(probs, base=2)
    
    # Фрактальная размерность
    from .fractal_dimension import fractal_dimension
    D_f = fractal_dimension(np.column_stack([x, y]))
    if np.isnan(D_f):
        D_f = 1.0
    
    # Извилистость (sinuosity)
    total_distance = np.sum(dists)
    net_distance = euclidean([x[0], y[0]], [x[-1], y[-1]])
    sinuosity = total_distance / net_distance if net_distance > 1e-6 else 0.0
    
    # Автокорреляция скорости
    vel_autocorr = 1.0
    if len(speeds) > 2 and np.std(speeds) > 1e-9:
        corr = np.corrcoef(speeds[:-1], speeds[1:])
        if not np.isnan(corr[0, 1]):
            vel_autocorr = corr[0, 1]
    
    features.update({
        'trajectory_entropy': H,
        'fractal_dimension': D_f,
        'sinuosity': sinuosity,
        'velocity_autocorrelation': vel_autocorr
    })
    
    return features


def extract_subwindow_aggregates(
    x: np.ndarray, 
    y: np.ndarray, 
    sub_frames: int, 
    fps: int = 30,
    suffix: str = "sub"  # "15s", "30s", "1min" и т.д.
) -> Dict:
    """
    Извлечение агрегированных статистик по подокнам.
    
    Параметры:
        x, y : np.ndarray
            Координаты траектории.
        sub_frames : int
            Размер подокна в кадрах.
        fps : int
            Частота кадров.
        suffix : str
            Суффикс для названий признаков (15s, 30s, 1min).
    
    Возвращает:
        Dict
            Словарь с агрегированными статистиками.
    """
    DT = 1.0 / fps
    speed_vals, angle_vals, jump_vals = [], [], []
    
    sub = 0
    while sub < len(x):
        se = sub + sub_frames
        xs, ys = x[sub:se], y[sub:se]
        
        if len(xs) < 10:
            sub += sub_frames
            continue
        
        # Скорость в подокне
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        ss = ds / DT
        speed_vals.append(np.mean(ss) if len(ss) > 0 else 0.0)
        
        # Углы в подокне
        angles = calculate_angles(xs, ys)
        angle_vals.append(np.mean(angles) if angles else 0.0)
        
        # Прыжки в подокне
        med_sp = np.median(ss) if len(ss) > 0 else 0.0
        jumps = np.sum(ss > 2.0 * med_sp) if len(ss) > 0 else 0
        jump_vals.append(jumps / (len(ss) * DT) if len(ss) > 0 else 0.0)
        
        sub += sub_frames
    
    # Агрегация статистик с ПРАВИЛЬНЫМИ названиями
    if len(speed_vals) >= 2:
        return {
            f'speed_{suffix}_mean': np.mean(speed_vals),
            f'speed_{suffix}_max': np.max(speed_vals),
            f'speed_{suffix}_std': np.std(speed_vals),
            f'angle_{suffix}_mean': np.mean(angle_vals),
            f'angle_{suffix}_std': np.std(angle_vals),
            f'jump_{suffix}_max': np.max(jump_vals)
        }
    else:
        return {
            f'speed_{suffix}_mean': 0.0,
            f'speed_{suffix}_max': 0.0,
            f'speed_{suffix}_std': 0.0,
            f'angle_{suffix}_mean': 0.0,
            f'angle_{suffix}_std': 0.0,
            f'jump_{suffix}_max': 0.0
        }


def extract_features_from_trajectory(
    track_df: pd.DataFrame,
    win_frames: int,
    sub_frames: Optional[int] = None,
    fps: int = 30,
    include_extended: bool = True,
    sub_suffix: str = "sub"  # Критически важно для совместимости!
) -> pd.DataFrame:
    """
    Извлечение признаков из траектории с разбиением на окна.
    
    Параметры:
        track_df : pd.DataFrame
            DataFrame с колонками [frame, x, y].
        win_frames : int
            Размер окна в кадрах.
        sub_frames : Optional[int]
            Размер подокна для агрегации (None если не нужна).
        fps : int
            Частота кадров.
        include_extended : bool
            Включать ли расширенные признаки (энтропия, фрактал).
        sub_suffix : str
            Суффикс для агрегированных признаков (15s, 30s, 1min).
    
    Возвращает:
        pd.DataFrame
            DataFrame с признаками для каждого окна.
    """
    track_df = track_df.sort_values('frame').reset_index(drop=True)
    n = len(track_df)
    results = []
    
    if n < 100:
        return pd.DataFrame(results)
    
    start = 0
    while start < n:
        end = start + win_frames
        win = track_df.iloc[start:end]
        
        if len(win) < win_frames // 2:
            start += win_frames
            break
        
        x, y = win['x'].values, win['y'].values
        
        # Извлечение признаков
        if include_extended:
            features = extract_extended_features(x, y, fps)
        else:
            features = extract_basic_features(x, y, fps)
        
        # Агрегаты по подокнам (если нужны)
        if sub_frames is not None:
            aggregates = extract_subwindow_aggregates(
                x, y, sub_frames, fps, suffix=sub_suffix
            )
            features.update(aggregates)
        
        # Метаданные
        features['window_start_frame'] = int(win['frame'].iloc[0])
        features['window_end_frame'] = int(win['frame'].iloc[-1])
        features['n_frames'] = len(win)
        
        results.append(features)
        start += win_frames
    
    return pd.DataFrame(results)


# ============================================================
# УДОБНЫЕ ФУНКЦИИ-ПОМОЩНИКИ ДЛЯ КАЖДОЙ КОНФИГУРАЦИИ
# ============================================================

def extract_features_5s(track_df: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Извлечение 7 признаков для 5-секундных окон.
    """
    win_frames = 5 * fps  # 150 кадров
    return extract_features_from_trajectory(
        track_df, 
        win_frames=win_frames,
        sub_frames=None,
        fps=fps,
        include_extended=False
    )


def extract_features_30s_15s(track_df: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Извлечение 13 признаков для 30-секундных окон с 15-сек агрегатами.
    """
    win_frames = 30 * fps  # 900 кадров
    sub_frames = 15 * fps  # 450 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=False,
        sub_suffix="15s"
    )


def extract_features_1min_30s(track_df: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Извлечение 13 признаков для 1-минутных окон с 30-сек агрегатами.
    """
    win_frames = 60 * fps  # 1800 кадров
    sub_frames = 30 * fps  # 900 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=False,
        sub_suffix="30s"
    )


def extract_features_5min_1min(track_df: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Извлечение 17 признаков для 5-минутных окон с 1-мин агрегатами.
    """
    win_frames = 5 * 60 * fps  # 9000 кадров
    sub_frames = 60 * fps  # 1800 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=True,
        sub_suffix="1min"
    )


def extract_features_5min_30s(track_df: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Извлечение 17 признаков для 5-минутных окон с 30-сек агрегатами.
    """
    win_frames = 5 * 60 * fps  # 9000 кадров
    sub_frames = 30 * fps  # 900 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=True,
        sub_suffix="30s"
    )