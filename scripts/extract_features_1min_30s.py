"""
Извлечение признаков с окном 1 минута и 30-секундной агрегацией (13 признаков).
🔥 ИСПОЛЬЗУЕТ СКОЛЬЗЯЩИЕ ОКНА С ПЕРЕКРЫТИЕМ 75%

Признаки:
Базовые (7):
- avg_speed — средняя скорость
- total_distance — общее пройденное расстояние
- straightness — прямолинейность траектории
- median_speed — медианная скорость
- mean_turning_angle — средний угол поворота
- angular_velocity — угловая скорость
- jump_frequency — частота прыжков

Агрегаты по 30-секундным подокнам (6):
- speed_30s_mean/max/std — статистики скорости
- angle_30s_mean/std — статистики углов
- jump_30s_max — максимальная частота прыжков

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

from src.feature_extractor import extract_features_1min_30s


# ============================================================
# 🔧 ФУНКЦИИ ОБРАБОТКИ
# ============================================================

def load_coordinates_file(filepath: str) -> pd.DataFrame:
    """
    Загрузка CSV-файла с координатами траекторий.
    
    Ожидаемые колонки: track_id, frame, x, y
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


def process_file(
    filepath: str, 
    label: int, 
    fps: int = 30,
    overlap: float = 0.75,
    start_frame: int = 0
) -> pd.DataFrame:
    """
    Обработка одного файла с координатами.
    """
    print(f"📄 {Path(filepath).name} [Label: {label}]")
    
    df = load_coordinates_file(filepath)
    
    # 🔥 ФИЛЬТРАЦИЯ ПО ВРЕМЕНИ (для меди - с 30 минуты)
    if start_frame > 0:
        original_frames = len(df)
        df = df[df['frame'] >= start_frame].copy()
        filtered_frames = len(df)
        print(f"   ⏰ Пропущено первых кадров: {original_frames - filtered_frames} "
              f"({start_frame/(fps*60):.1f} мин)")
    
    all_features = []
    
    for tid, group in df.groupby('track_id'):
        # 🔥 ИЗВЛЕЧЕНИЕ С УЧЁТОМ СКОЛЬЗЯЩИХ ОКОН
        feats = extract_features_1min_30s(group, fps=fps, overlap=overlap)
        
        if not feats.empty:
            # Добавляем метаданные
            feats['track_id'] = tid
            feats['label'] = label
            all_features.append(feats)
    
    if all_features:
        result = pd.concat(all_features, ignore_index=True)
        # 🔥 ПРАВИЛЬНЫЙ ПОРЯДОК КОЛОНОК (как в датасетах)
        cols = ['track_id', 'label', 'window_start_frame', 'window_end_frame', 'n_frames']
        feature_cols = [c for c in result.columns if c not in cols]
        result = result[cols + feature_cols]
        return result
    
    return pd.DataFrame()


def load_config(config_path: str) -> list:
    """
    Загрузка конфигурации из JSON файла.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


# ============================================================
# 🚀 ОСНОВНОЙ ПРОЦЕСС
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Извлечение 13 признаков из 1-минутных окон с 30-сек агрегатами'
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
    parser.add_argument(
        '--start-frame', type=int, action='append', default=None,
        help='Начальный кадр для каждого файла (для фильтрации). Можно указывать несколько раз.'
    )
    
    # Общие параметры
    parser.add_argument(
        '--output', type=str, default='dataset_1min30s_robust.csv',
        help='Путь к выходному CSV файлу (по умолчанию: dataset_1min30s_robust.csv)'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Частота кадров видео (по умолчанию: 30)'
    )
    parser.add_argument(
        '--overlap', type=float, default=0.75,
        help='🔥 Перекрытие скользящих окон (по умолчанию: 0.75 = 75%%)'
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
        
        # Обработка start_frame
        start_frames = args.start_frame if args.start_frame else [0] * len(args.input)
        if len(start_frames) != len(args.input):
            start_frames = [0] * len(args.input)
        
        file_settings = [
            {'path': path, 'label': label, 'start_frame': sf}
            for path, label, sf in zip(args.input, args.label, start_frames)
        ]
    else:
        print("❌ Укажите либо --config, либо --input и --label")
        print("\nПримеры использования:")
        print("  python scripts/extract_features_1min_30s.py --config config.json --output dataset.csv")
        print("  python scripts/extract_features_1min_30s.py \\")
        print("      --input file1.csv --label 0 --start-frame 0 \\")
        print("      --input file2.csv --label 1 --start-frame 54000 \\")
        print("      --overlap 0.75 \\")
        print("      --output dataset.csv")
        return
    
    if not file_settings:
        print("❌ Нет файлов для обработки!")
        return
    
    print(f"\n🔍 Извлечение признаков (окно 1 мин, подокно 30 сек, 13 признаков)...")
    print(f"⚙️  FPS: {args.fps}")
    print(f"🔥 Перекрытие окон: {args.overlap*100:.0f}%")
    print(f"📁 Файлов для обработки: {len(file_settings)}\n")
    
    # Обработка всех файлов
    all_data = []
    
    for cfg in file_settings:
        filepath = cfg['path']
        label = cfg['label']
        start_frame = cfg.get('start_frame', 0)
        
        if not Path(filepath).exists():
            print(f"❌ Файл не найден: {filepath}")
            continue
        
        try:
            df = process_file(
                filepath, label, 
                fps=args.fps,
                overlap=args.overlap,
                start_frame=start_frame
            )
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"❌ Ошибка при обработке {filepath}: {e}")
            import traceback
            traceback.print_exc()
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
    print(f"📐 Окно: 1 минута ({60 * args.fps} кадров)")
    print(f"📐 Подокно: 30 секунд ({30 * args.fps} кадров)")
    print(f"🔥 Перекрытие окон: {args.overlap*100:.0f}%")
    print(f"\n📋 Признаки (13 шт.):")
    meta = ['track_id', 'label', 'window_start_frame', 'window_end_frame', 'n_frames']
    feature_cols = [c for c in df_final.columns if c not in meta]
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
