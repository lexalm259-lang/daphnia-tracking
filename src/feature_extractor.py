"""
Модуль извлечения поведенческих признаков из траекторий дафний.
Поддерживает различные временные окна и мультишкальную агрегацию.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from typing import Dict, List, Optional


def fractal_dimension(points: np.ndarray, max_splits: int = 4) -> float:
    """
    Вычисляет фрактальную размерность методом box-counting.
    
    Параметры:
        points : np.ndarray
            Массив Nx2 координат точек (x, y).
        max_splits : int
            Максимальное количество разбиений сетки.
    
    Возвращает:
        float
            Оценка фрактальной размерности.
    """
    if len(points) < 10:
        return 1.0
    
    # Нормализация данных в диапазон [0, 1] x [0, 1]
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    
    points_norm = (points - min_vals) / ranges
    
    # Подготовка размеров ячеек
    powers = np.arange(1, max_splits + 1)
    scales = 2.0 ** -powers
    
    counts = []
    for scale in scales:
        bins = np.floor(points_norm / scale).astype(int)
        unique_bins = np.unique(bins, axis=0)
        counts.append(len(unique_bins))
    
    counts = np.array(counts)
    
    # Линейная регрессия в логарифмических координатах
    valid_mask = counts > 1
    if np.sum(valid_mask) < 2:
        return 1.0
    
    log_inv_scales = np.log(1.0 / scales[valid_mask])
    log_counts = np.log(counts[valid_mask])
    
    coeffs = np.polyfit(log_inv_scales, log_counts, 1)
    return float(coeffs[0])


def calculate_angles(x: np.ndarray, y: np.ndarray) -> List[float]:
    """
    Вычисление углов поворота между последовательными векторами скорости.
    """
    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angles.append(np.arccos(cos_a) * 180.0 / np.pi)
    return angles


def extract_basic_features(x: np.ndarray, y: np.ndarray, fps: int = 30) -> Dict:
    """
    Извлечение 7 базовых кинематических признаков.
    """
    if len(x) < 2:
        return {
            'avg_speed': 0.0,
            'total_distance': 0.0,
            'straightness': 0.0,
            'median_speed': 0.0,
            'mean_turning_angle': 0.0,
            'angular_velocity': 0.0,
            'jump_frequency': 0.0
        }
    
    dx, dy = np.diff(x), np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    DT = 1.0 / fps
    speeds = dists / DT
    
    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    total_distance = float(np.sum(dists))
    net_distance = float(euclidean([x[0], y[0]], [x[-1], y[-1]]))
    straightness = net_distance / total_distance if total_distance > 1e-6 else 0.0
    median_speed = float(np.median(speeds)) if len(speeds) > 0 else 0.0
    
    angles = calculate_angles(x, y)
    mean_turning_angle = float(np.mean(angles)) if angles else 0.0
    angular_velocity = float(np.mean([a / DT for a in angles])) if angles else 0.0
    
    jump_threshold = 2.0 * median_speed
    jumps = int(np.sum(speeds > jump_threshold)) if len(speeds) > 0 else 0
    jump_frequency = jumps / (len(speeds) * DT) if len(speeds) > 0 else 0.0
    
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
    """
    features = extract_basic_features(x, y, fps)
    
    if len(x) < 2:
        features.update({
            'trajectory_entropy': 0.0,
            'fractal_dimension': 1.0,
            'sinuosity': 0.0,
            'velocity_autocorrelation': 1.0
        })
        return features
    
    dx, dy = np.diff(x), np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    DT = 1.0 / fps
    speeds = dists / DT
    
    # Энтропия траектории
    if len(x) >= 10:
        H, _, _ = np.histogram2d(x, y, bins=10)
        probs = H.flatten()
        probs = probs / probs.sum()
        probs = probs[probs > 0]
        traj_entropy = float(entropy(probs, base=2)) if len(probs) > 0 else 0.0
    else:
        traj_entropy = 0.0
    
    # Фрактальная размерность
    points = np.column_stack([x, y])
    fractal_dim = fractal_dimension(points, max_splits=4)
    if np.isnan(fractal_dim):
        fractal_dim = 1.0
    
    # Извилистость (sinuosity)
    total_distance = float(np.sum(dists))
    net_distance = float(euclidean([x[0], y[0]], [x[-1], y[-1]]))
    sinuosity = total_distance / net_distance if net_distance > 1e-6 else 0.0
    
    # Автокорреляция скорости
    if len(speeds) > 2 and np.std(speeds) > 1e-9:
        corr = np.corrcoef(speeds[:-1], speeds[1:])
        vel_autocorr = float(corr[0, 1])
        if np.isnan(vel_autocorr):
            vel_autocorr = 1.0
    else:
        vel_autocorr = 1.0
    
    features.update({
        'trajectory_entropy': traj_entropy,
        'fractal_dimension': fractal_dim,
        'sinuosity': sinuosity,
        'velocity_autocorrelation': vel_autocorr
    })
    
    return features


def extract_subwindow_aggregates(
    x: np.ndarray, 
    y: np.ndarray, 
    sub_frames: int, 
    fps: int = 30,
    suffix: str = "sub"
) -> Dict:
    """
    Извлечение агрегированных статистик по подокнам.
    """
    DT = 1.0 / fps
    
    if len(x) < 2:
        return {
            f'speed_{suffix}_mean': 0.0,
            f'speed_{suffix}_max': 0.0,
            f'speed_{suffix}_std': 0.0,
            f'angle_{suffix}_mean': 0.0,
            f'angle_{suffix}_std': 0.0,
            f'jump_{suffix}_max': 0.0
        }
    
    speed_vals, angle_vals, jump_vals = [], [], []
    
    sub = 0
    while sub < len(x):
        se = sub + sub_frames
        xs, ys = x[sub:se], y[sub:se]
        
        if len(xs) < 30:
            sub += sub_frames
            continue
        
        # Скорость в подокне
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        ss = ds / DT
        speed_vals.append(float(np.mean(ss)) if len(ss) > 0 else 0.0)
        
        # Углы в подокне
        angles = calculate_angles(xs, ys)
        angle_vals.append(float(np.mean(angles)) if angles else 0.0)
        
        # Прыжки в подокне
        med_sp = float(np.median(ss)) if len(ss) > 0 else 0.0
        jumps = int(np.sum(ss > 2.0 * med_sp)) if len(ss) > 0 else 0
        jump_vals.append(jumps / (len(ss) * DT) if len(ss) > 0 else 0.0)
        
        sub += sub_frames
    
    # Агрегация статистик
    if len(speed_vals) >= 2:
        return {
            f'speed_{suffix}_mean': float(np.mean(speed_vals)),
            f'speed_{suffix}_max': float(np.max(speed_vals)),
            f'speed_{suffix}_std': float(np.std(speed_vals)),
            f'angle_{suffix}_mean': float(np.mean(angle_vals)),
            f'angle_{suffix}_std': float(np.std(angle_vals)),
            f'jump_{suffix}_max': float(np.max(jump_vals))
        }
    elif len(speed_vals) == 1:
        return {
            f'speed_{suffix}_mean': speed_vals[0],
            f'speed_{suffix}_max': speed_vals[0],
            f'speed_{suffix}_std': 0.0,
            f'angle_{suffix}_mean': angle_vals[0] if angle_vals else 0.0,
            f'angle_{suffix}_std': 0.0,
            f'jump_{suffix}_max': jump_vals[0] if jump_vals else 0.0
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
    sub_suffix: str = "sub",
    overlap_fraction: float = 0.75
) -> pd.DataFrame:
    """
    Извлечение признаков из траектории с разбиением на СКОЛЬЗЯЩИЕ окна.
    """
    track_df = track_df.sort_values('frame').reset_index(drop=True)
    n = len(track_df)
    results = []
    
    if n < win_frames // 2:
        return pd.DataFrame(results)
    
    # Расчёт шага скользящего окна
    step = int(win_frames * (1 - overlap_fraction))
    if step < 1:
        step = 1
    
    start = 0
    while start < n:
        end = start + win_frames
        win = track_df.iloc[start:end]
        
        if len(win) < win_frames // 2:
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
        start += step
    
    return pd.DataFrame(results)


# ============================================================
# ФУНКЦИИ-ПОМОЩНИКИ ДЛЯ КАЖДОЙ КОНФИГУРАЦИИ
# ============================================================

def extract_features_5s(
    track_df: pd.DataFrame, 
    fps: int = 30,
    overlap: float = 0.75
) -> pd.DataFrame:
    """
    Извлечение 7 признаков для 5-секундных окон.
    СКОЛЬЗЯЩИЕ ОКНА с перекрытием 75%.
    """
    win_frames = 5 * fps  # 150 кадров
    return extract_features_from_trajectory(
        track_df, 
        win_frames=win_frames,
        sub_frames=None,
        fps=fps,
        include_extended=False,
        sub_suffix="5s",
        overlap_fraction=overlap
    )


def extract_features_30s_15s(
    track_df: pd.DataFrame, 
    fps: int = 30,
    overlap: float = 0.75
) -> pd.DataFrame:
    """
    Извлечение 13 признаков для 30-секундных окон с 15-сек агрегатами.
    СКОЛЬЗЯЩИЕ ОКНА с перекрытием 75%.
    """
    win_frames = 30 * fps  # 900 кадров
    sub_frames = 15 * fps  # 450 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=False,
        sub_suffix="15s",
        overlap_fraction=overlap
    )


def extract_features_1min_30s(
    track_df: pd.DataFrame, 
    fps: int = 30,
    overlap: float = 0.75
) -> pd.DataFrame:
    """
    Извлечение 13 признаков для 1-минутных окон с 30-сек агрегатами.
    СКОЛЬЗЯЩИЕ ОКНА с перекрытием 75%.
    """
    win_frames = 60 * fps  # 1800 кадров
    sub_frames = 30 * fps  # 900 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=False,
        sub_suffix="30s",
        overlap_fraction=overlap
    )


def extract_features_5min_30s(
    track_df: pd.DataFrame, 
    fps: int = 30,
    overlap: float = 0.75
) -> pd.DataFrame:
    """
    Извлечение 17 признаков для 5-минутных окон с 30-сек агрегатами.
    🔥 Включает сложные признаки: энтропия, фрактал, извилистость, автокорреляция.
    СКОЛЬЗЯЩИЕ ОКНА с перекрытием 75%.
    """
    win_frames = 5 * 60 * fps  # 9000 кадров
    sub_frames = 30 * fps  # 900 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=True,  # 🔥 ВКЛЮЧАЕМ СЛОЖНЫЕ ПРИЗНАКИ
        sub_suffix="30s",
        overlap_fraction=overlap
    )


def extract_features_5min_1min(
    track_df: pd.DataFrame, 
    fps: int = 30,
    overlap: float = 0.75
) -> pd.DataFrame:
    """
    Извлечение 17 признаков для 5-минутных окон с 1-мин агрегатами.
    🔥 Включает сложные признаки: энтропия, фрактал, извилистость, автокорреляция.
    СКОЛЬЗЯЩИЕ ОКНА с перекрытием 75%.
    """
    win_frames = 5 * 60 * fps  # 9000 кадров
    sub_frames = 60 * fps  # 1800 кадров
    return extract_features_from_trajectory(
        track_df,
        win_frames=win_frames,
        sub_frames=sub_frames,
        fps=fps,
        include_extended=True,  # 🔥 ВКЛЮЧАЕМ СЛОЖНЫЕ ПРИЗНАКИ
        sub_suffix="1min",
        overlap_fraction=overlap
    )


def get_feature_names(config: str) -> List[str]:
    """
    Возвращает список названий признаков для заданной конфигурации.
    """
    base_features = [
        'avg_speed', 'total_distance', 'straightness', 'median_speed',
        'mean_turning_angle', 'angular_velocity', 'jump_frequency'
    ]
    
    extended_features = [
        'trajectory_entropy', 'fractal_dimension', 'sinuosity', 'velocity_autocorrelation'
    ]
    
    if config == '5s':
        return base_features
    
    elif config == '30s_15s':
        return base_features + [
            'speed_15s_mean', 'speed_15s_max', 'speed_15s_std',
            'angle_15s_mean', 'angle_15s_std', 'jump_15s_max'
        ]
    
    elif config == '1min_30s':
        return base_features + [
            'speed_30s_mean', 'speed_30s_max', 'speed_30s_std',
            'angle_30s_mean', 'angle_30s_std', 'jump_30s_max'
        ]
    
    elif config == '5min_30s':
        return base_features + extended_features + [
            'speed_30s_mean', 'speed_30s_max', 'speed_30s_std',
            'angle_30s_mean', 'angle_30s_std', 'jump_30s_max'
        ]
    
    elif config == '5min_1min':
        return base_features + extended_features + [
            'speed_1min_mean', 'speed_1min_max', 'speed_1min_std',
            'angle_1min_mean', 'angle_1min_std', 'jump_1min_max'
        ]
    
    else:
        raise ValueError(f"Неизвестная конфигурация: {config}")


def get_n_features(config: str) -> int:
    """Возвращает количество признаков для заданной конфигурации."""
    return len(get_feature_names(config))
