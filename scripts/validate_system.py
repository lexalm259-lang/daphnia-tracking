"""
Практическая валидация системы биоиндикации.

Скрипт выполняет:
1. Загрузку координат треков из CSV
2. Извлечение признаков для всех 5 конфигураций окон
3. Применение обученных моделей Random Forest
4. Агрегацию результатов и итоговую классификацию
5. Сохранение детального отчёта

Поддерживаемые модели:
- окно 5 секунд.pkl (7 признаков)
- окно 30 сек(15 сек доп признаки).pkl (13 признаков)
- окно 1 минута(30 сек доп признаки).pkl (13 признаков)
- окно 5 минут(1 мин доп признаки).pkl (17 признаков)
- окно 5 минут(30 сек доп признаки).pkl (17 признаков)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import (
    extract_features_5s,
    extract_features_30s_15s,
    extract_features_1min_30s,
    extract_features_5min_1min,
    extract_features_5min_30s
)


# ============================================================
# 🔧 КОНФИГУРАЦИЯ
# ============================================================

MODEL_CONFIGS = {
    "5s": {
        "model_file": "окно 5 секунд.pkl",
        "feature_func": extract_features_5s,
        "feature_cols": [
            'avg_speed', 'total_distance', 'straightness', 'median_speed',
            'mean_turning_angle', 'angular_velocity', 'jump_frequency'
        ],
        "window_frames": 5 * 30,  # 150 кадров
        "description": "5-секундные окна (7 признаков)"
    },
    "30s_15s": {
        "model_file": "окно 30 сек(15 сек доп признаки).pkl",
        "feature_func": extract_features_30s_15s,
        "feature_cols": [
            'avg_speed', 'total_distance', 'straightness', 'median_speed',
            'mean_turning_angle', 'angular_velocity', 'jump_frequency',
            'speed_15s_mean', 'speed_15s_max', 'speed_15s_std',
            'angle_15s_mean', 'angle_15s_std', 'jump_15s_max'
        ],
        "window_frames": 30 * 30,  # 900 кадров
        "description": "30-сек окна с 15-сек агрегатами (13 признаков)"
    },
    "1min_30s": {
        "model_file": "окно 1 минута(30 сек доп признаки).pkl",
        "feature_func": extract_features_1min_30s,
        "feature_cols": [
            'avg_speed', 'total_distance', 'straightness', 'median_speed',
            'mean_turning_angle', 'angular_velocity', 'jump_frequency',
            'speed_30s_mean', 'speed_30s_max', 'speed_30s_std',
            'angle_30s_mean', 'angle_30s_std', 'jump_30s_max'
        ],
        "window_frames": 60 * 30,  # 1800 кадров
        "description": "1-мин окна с 30-сек агрегатами (13 признаков)"
    },
    "5min_1min": {
        "model_file": "окно 5 минут(1 мин доп признаки).pkl",
        "feature_func": extract_features_5min_1min,
        "feature_cols": [
            'avg_speed', 'total_distance', 'straightness', 'median_speed',
            'mean_turning_angle', 'angular_velocity', 'jump_frequency',
            'trajectory_entropy', 'fractal_dimension', 'sinuosity',
            'velocity_autocorrelation',
            'speed_1min_mean', 'speed_1min_max', 'speed_1min_std',
            'angle_1min_mean', 'angle_1min_std', 'jump_1min_max'
        ],
        "window_frames": 5 * 60 * 30,  # 9000 кадров
        "description": "5-мин окна с 1-мин агрегатами (17 признаков)"
    },
    "5min_30s": {
        "model_file": "окно 5 минут(30 сек доп признаки).pkl",
        "feature_func": extract_features_5min_30s,
        "feature_cols": [
            'avg_speed', 'total_distance', 'straightness', 'median_speed',
            'mean_turning_angle', 'angular_velocity', 'jump_frequency',
            'trajectory_entropy', 'fractal_dimension', 'sinuosity',
            'velocity_autocorrelation',
            'speed_30s_mean', 'speed_30s_max', 'speed_30s_std',
            'angle_30s_mean', 'angle_30s_std', 'jump_30s_max'
        ],
        "window_frames": 5 * 60 * 30,  # 9000 кадров
        "description": "5-мин окна с 30-сек агрегатами (17 признаков)"
    }
}

FPS = 30


# ============================================================
# 🔧 ФУНКЦИИ
# ============================================================

def load_coordinates(csv_path: str, max_duration_min: int = None) -> pd.DataFrame:
    """
    Загрузка и предобработка координат.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")
    
    print(f"📂 Загрузка координат: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Автоопределение колонок
    if df.columns[0] not in ['track_id', 'id']:
        df.columns = ['track_id', 'frame', 'x', 'y'][:len(df.columns)]
    
    # Преобразование типов
    df['track_id'] = pd.to_numeric(df['track_id'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()
    
    # Ограничение по времени
    if max_duration_min is not None:
        max_frame = max_duration_min * 60 * FPS
        df = df[df['frame'] <= max_frame].copy()
        print(f"⏱️  Ограничение: первые {max_duration_min} минут ({max_frame} кадров)")
    
    # Сортировка
    df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)
    
    print(f"✅ Загружено {len(df)} точек, {df['track_id'].nunique()} треков")
    return df


def load_models(models_dir: str) -> Dict:
    """
    Загрузка обученных моделей.
    """
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Папка с моделями не найдена: {models_dir}")
    
    print(f"\n📁 Загрузка моделей из: {models_dir}")
    
    for name, config in MODEL_CONFIGS.items():
        model_file = models_path / config["model_file"]
        if model_file.exists():
            models[name] = joblib.load(model_file)
            print(f"   ✅ {name:12} | {config['description']}")
        else:
            print(f"   ⚠️  {name:12} | модель не найдена: {config['model_file']}")
    
    if not models:
        raise FileNotFoundError("Ни одна модель не найдена!")
    
    return models


def extract_windows(group: pd.DataFrame, feature_func, window_frames: int, fps: int = 30) -> pd.DataFrame:
    """
    Извлечение признаков из окон одного трека.
    """
    if len(group) < 100:
        return pd.DataFrame()
    
    windows = []
    start = 0
    
    while start < len(group):
        end = start + window_frames
        win = group.iloc[start:end]
        
        # Пропускаем слишком короткие окна
        if len(win) < window_frames // 2:
            start += window_frames
            continue
        
        # Извлечение признаков
        try:
            feats = feature_func(win, fps=fps)
            if not feats.empty:
                windows.append(feats)
        except Exception as e:
            print(f"   ⚠️  Ошибка извлечения признаков: {e}")
        
        start += window_frames
    
    if windows:
        return pd.concat(windows, ignore_index=True)
    return pd.DataFrame()


def validate_tracks(df: pd.DataFrame, models: Dict, fps: int = 30) -> Tuple[pd.DataFrame, Dict]:
    """
    Валидация всех треков всеми моделями.
    """
    print("\n🔍 Запуск валидации...")
    
    all_results = []
    model_stats = {name: {'windows': 0, 'prob_sum': 0.0, 'predictions': []} 
                   for name in models}
    
    total_tracks = df['track_id'].nunique()
    track_num = 0
    
    for track_id, group in df.groupby('track_id'):
        track_num += 1
        if track_num % 10 == 0:
            print(f"   Обработано треков: {track_num}/{total_tracks}")
        
        group = group.sort_values('frame').reset_index(drop=True)
        
        for model_name, model in models.items():
            config = MODEL_CONFIGS[model_name]
            
            # Извлечение окон
            features_df = extract_windows(
                group, 
                config["feature_func"], 
                config["window_frames"], 
                fps
            )
            
            if features_df.empty:
                continue
            
            # Подготовка признаков
            feature_cols = config["feature_cols"]
            missing_cols = [col for col in feature_cols if col not in features_df.columns]
            if missing_cols:
                print(f"   ⚠️  Отсутствуют признаки для {model_name}: {missing_cols}")
                continue
            
            X = features_df[feature_cols].values
            
            # Предсказания
            try:
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]
                
                # Сохранение результатов
                for i, (pred, prob) in enumerate(zip(preds, probs)):
                    all_results.append({
                        'track_id': track_id,
                        'model': model_name,
                        'window': i,
                        'prediction': int(pred),
                        'probability': float(prob)
                    })
                
                # Статистика
                model_stats[model_name]['windows'] += len(probs)
                model_stats[model_name]['prob_sum'] += np.sum(probs)
                model_stats[model_name]['predictions'].extend(preds.tolist())
                
            except Exception as e:
                print(f"   ❌ Ошибка предсказания {model_name}: {e}")
                continue
    
    results_df = pd.DataFrame(all_results)
    return results_df, model_stats


def generate_report(results_df: pd.DataFrame, model_stats: Dict, output_path: str):
    """
    Генерация итогового отчёта.
    """
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ КЛАССИФИКАЦИЯ")
    print("="*60)
    
    if results_df.empty:
        print("❌ Нет результатов для анализа!")
        return
    
    # Общая статистика
    total_windows = len(results_df)
    all_probs = results_df['probability'].values
    overall_prob = np.mean(all_probs)
    final_class = "Медь (токсическое воздействие)" if overall_prob > 0.5 else "Чистая вода"
    
    print(f"🔍 Обработано треков: {results_df['track_id'].nunique()}")
    print(f"📦 Всего окон проанализировано: {total_windows}")
    print(f"🎯 Средняя уверенность (probability): {overall_prob:.4f}")
    print(f"✅ Итоговый класс: {final_class}")
    
    # Разбивка по моделям
    print("\n📈 Разбивка по моделям:")
    print("-" * 60)
    print(f"{'Модель':<15} | {'Окон':>6} | {'Avg Prob':>10} | {'Класс':<10}")
    print("-" * 60)
    
    for model_name, stats in model_stats.items():
        if stats['windows'] > 0:
            avg_prob = stats['prob_sum'] / stats['windows']
            pred_class = "Медь" if avg_prob > 0.5 else "Чистая"
            print(f"{model_name:<15} | {stats['windows']:>6} | {avg_prob:>10.4f} | {pred_class:<10}")
    
    print("="*60 + "\n")
    
    # Сохранение результатов
    if output_path:
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 Детальные результаты сохранены: {output_path}")


# ============================================================
# 🚀 ОСНОВНОЙ ПРОЦЕСС
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Практическая валидация системы биоиндикации на новых данных'
    )
    
    parser.add_argument(
        '--input', type=str, required=True,
        help='Путь к CSV файлу с координатами (track_id, frame, x, y)'
    )
    parser.add_argument(
        '--models', type=str, default='models/rf',
        help='Папка с обученными моделями (по умолчанию: models/rf)'
    )
    parser.add_argument(
        '--output', type=str, default='results_validation.csv',
        help='Путь к выходному CSV файлу с результатами'
    )
    parser.add_argument(
        '--max-duration', type=int, default=18,
        help='Максимальная длительность анализа в минутах (по умолчанию: 18)'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Частота кадров (по умолчанию: 30)'
    )
    
    args = parser.parse_args()
    
    # Обновление FPS в конфигурации
    global FPS
    FPS = args.fps
    
    # 1. Загрузка координат
    try:
        df = load_coordinates(args.input, args.max_duration)
    except Exception as e:
        print(f"❌ Ошибка загрузки координат: {e}")
        return
    
    # 2. Загрузка моделей
    try:
        models = load_models(args.models)
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return
    
    # 3. Валидация
    results_df, model_stats = validate_tracks(df, models, FPS)
    
    # 4. Генерация отчёта
    generate_report(results_df, model_stats, args.output)
    
    print("\n✅ Валидация завершена!")


if __name__ == "__main__":
    main()