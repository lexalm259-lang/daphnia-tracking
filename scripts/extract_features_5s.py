"""
Извлечение признаков с окном 5 секунд (7 базовых признаков).

Признаки:
- avg_speed — средняя скорость
- total_distance — общее пройденное расстояние
- straightness — прямолинейность траектории
- median_speed — медианная скорость
- mean_turning_angle — средний угол поворота
- angular_velocity — угловая скорость
- jump_frequency — частота прыжков

Результат сохраняется в CSV файл для обучения модели Random Forest.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Добавляем корень проекта в путь для импорта из src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import extract_features_5s


# ============================================================
# 🔧 ФУНКЦИИ ОБРАБОТКИ
# ============================================================

def load_coordinates_file(filepath: str) -> pd.DataFrame:
    """
    Загрузка CSV-файла с координатами траекторий.
    
    Ожидаемые колонки: track_id, frame, x, y
    
    Параметры:
        filepath : str
            Путь к CSV файлу.
    
    Возвращает:
        pd.DataFrame
            DataFrame с координатами.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Автоопределение колонок
    if df.columns[0] not in ['track_id', 'id']:
        df.columns = ['track_id', 'frame', 'x', 'y'][:len(df.columns)]
    
    # Преобразование типов
    df['track_id'] = pd.to_numeric(df['track_id'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Удаление строк с NaN
    df = df.dropna()
    
    return df


def process_file(filepath: str, label: int, fps: int = 30) -> pd.DataFrame:
    """
    Обработка одного файла с координатами.
    
    Параметры:
        filepath : str
            Путь к CSV файлу с координатами.
        label : int
            Метка класса (0 = чистая вода, 1 = токсичная среда).
        fps : int
            Частота кадров (по умолчанию 30).
    
    Возвращает:
        pd.DataFrame
            DataFrame с извлечёнными признаками.
    """
    print(f"📄 {Path(filepath).name} [Label: {label}]")
    
    df = load_coordinates_file(filepath)
    
    all_features = []
    
    for tid, group in df.groupby('track_id'):
        # Извлечение признаков для текущего трека
        feats = extract_features_5s(group, fps=fps)
        
        if not feats.empty:
            # Добавляем метаданные
            feats['track_id'] = tid
            feats['label'] = label
            all_features.append(feats)
    
    if all_features:
        result = pd.concat(all_features, ignore_index=True)
        # Переупорядочиваем колонки
        cols = ['track_id', 'label', 'window_start_frame', 'window_end_frame', 'n_frames']
        feature_cols = [c for c in result.columns if c not in cols]
        result = result[cols + feature_cols]
        return result
    
    return pd.DataFrame()


def load_config(config_path: str) -> list:
    """
    Загрузка конфигурации из JSON файла.
    
    Формат JSON:
    [
        {"path": "path/to/file1.csv", "label": 0},
        {"path": "path/to/file2.csv", "label": 1}
    ]
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


# ============================================================
# 🚀 ОСНОВНОЙ ПРОЦЕСС
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Извлечение 7 базовых признаков из 5-секундных окон траекторий дафний'
    )
    
    # Вариант 1: через JSON-конфиг
    parser.add_argument(
        '--config', type=str,
        help='Путь к JSON файлу с конфигурацией (список файлов и меток)'
    )
    
    # Вариант 2: через аргументы командной строки
    parser.add_argument(
        '--input', type=str, action='append',
        help='Путь к CSV файлу с координатами (можно указывать несколько раз)'
    )
    parser.add_argument(
        '--label', type=int, action='append', choices=[0, 1],
        help='Метка для соответствующего файла (0=чистая, 1=токсичная). Можно указывать несколько раз.'
    )
    
    # Общие параметры
    parser.add_argument(
        '--output', type=str, default='dataset_5sec_7features.csv',
        help='Путь к выходному CSV файлу (по умолчанию: dataset_5sec_7features.csv)'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Частота кадров видео (по умолчанию: 30)'
    )
    
    args = parser.parse_args()
    
    # Формирование списка файлов для обработки
    file_settings = []
    
    if args.config:
        # Загрузка из JSON
        print(f"📋 Загрузка конфигурации: {args.config}")
        file_settings = load_config(args.config)
    elif args.input and args.label:
        # Из аргументов командной строки
        if len(args.input) != len(args.label):
            print("❌ Количество файлов (--input) и меток (--label) должно совпадать!")
            return
        file_settings = [
            {'path': path, 'label': label}
            for path, label in zip(args.input, args.label)
        ]
    else:
        print("❌ Укажите либо --config, либо --input и --label")
        print("\nПримеры использования:")
        print("  python scripts/extract_features_5s.py --config config.json --output dataset.csv")
        print("  python scripts/extract_features_5s.py \\")
        print("      --input file1.csv --label 0 \\")
        print("      --input file2.csv --label 1 \\")
        print("      --output dataset.csv")
        return
    
    if not file_settings:
        print("❌ Нет файлов для обработки!")
        return
    
    print(f"\n🔍 Извлечение признаков (окно 5 сек, 7 признаков)...")
    print(f"⚙️  FPS: {args.fps}")
    print(f"📁 Файлов для обработки: {len(file_settings)}\n")
    
    # Обработка всех файлов
    all_data = []
    
    for cfg in file_settings:
        filepath = cfg['path']
        label = cfg['label']
        
        if not Path(filepath).exists():
            print(f"❌ Файл не найден: {filepath}")
            continue
        
        try:
            df = process_file(filepath, label, fps=args.fps)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"❌ Ошибка при обработке {filepath}: {e}")
            continue
    
    # Сохранение результатов
    if not all_data:
        print("\n❌ Нет данных для сохранения!")
        return
    
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.sort_values(
        ['label', 'track_id', 'window_start_frame']
    ).reset_index(drop=True)
    
    # Создаём директорию для выходного файла, если нужно
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Итоговый отчёт
    print(f"\n{'='*60}")
    print(f"✅ Готово! Датасет сохранён: {output_path}")
    print(f"{'='*60}")
    print(f"📊 Строк: {len(df_final)}")
    print(f"🏷️  Классы: {df_final['label'].value_counts().sort_index().to_dict()}")
    print(f"🔢 Треков: {df_final['track_id'].nunique()}")
    print(f"📐 Окно: 5 секунд ({5 * args.fps} кадров)")
    print(f"\n📋 Признаки (7 шт.):")
    meta = ['track_id', 'label', 'window_start_frame', 'window_end_frame', 'n_frames']
    feature_cols = [c for c in df_final.columns if c not in meta]
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()