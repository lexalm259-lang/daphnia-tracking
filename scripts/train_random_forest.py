"""
Обучение модели Random Forest для классификации состояния водной среды.

Скрипт выполняет:
1. Загрузку датасета с признаками
2. Разделение на train/test (80/20) со стратификацией
3. Поиск оптимальных параметров через GridSearchCV (5-fold CV)
4. Оценку на тестовой выборке (Accuracy, ROC-AUC, F1)
5. Построение матрицы ошибок
6. Сохранение обученной модели в models/rf/
7. Анализ важности признаков

Поддерживаемые датасеты:
- dataset_5sec_7features.csv (7 признаков)
- dataset_30s15s_robust.csv (13 признаков)
- dataset_1min30s_robust.csv (13 признаков)
- dataset_5min_multiscale.csv (17 признаков)
- dataset_5min_multiscale1s_features.csv (17 признаков)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 🔧 КОНФИГУРАЦИЯ
# ============================================================

# Сетка параметров для GridSearchCV (как в оригинальном коде)
PARAM_GRID = {
    'n_estimators': [50, 100, 200],        # Количество деревьев
    'max_depth': [5, 10, 15, None],        # Максимальная глубина
    'min_samples_leaf': [1, 3, 5],         # Мин. семплов в листе
    'class_weight': ['balanced'],          # Балансировка классов
    'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.8, 5, 7]
}

# Служебные колонки (не используются для обучения)
META_COLS = ['track_id', 'window_start_frame', 'window_end_frame', 'n_frames', 'label']


# ============================================================
# 🔧 ФУНКЦИИ
# ============================================================

def load_dataset(csv_path: str) -> tuple:
    """
    Загрузка датасета и разделение на признаки/метки.
    
    Возвращает:
        X : np.ndarray — матрица признаков
        y : np.ndarray — вектор меток
        feature_cols : list — названия признаков
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Датасет не найден: {csv_path}")
    
    print(f"📂 Загрузка датасета: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Определяем колонки признаков
    feature_cols = [col for col in df.columns if col not in META_COLS]
    
    print(f"✅ Загружено {len(df)} строк")
    print(f"🔢 Признаков: {len(feature_cols)}")
    print(f"📋 Признаки: {feature_cols}")
    print(f"🏷️  Распределение классов: {df['label'].value_counts().sort_index().to_dict()}")
    
    X = df[feature_cols].values
    y = df['label'].values
    
    return X, y, feature_cols


def train_model(X_train: np.ndarray, y_train: np.ndarray, verbose: int = 1):
    """
    Обучение модели с GridSearchCV.
    
    Возвращает:
        best_model — лучшая обученная модель
        grid_search — объект GridSearchCV с результатами
    """
    print("\n🔍 Запуск поиска оптимальных параметров (GridSearchCV)...")
    print(f"   Сетка: {len(PARAM_GRID['n_estimators'])} × {len(PARAM_GRID['max_depth'])} × "
          f"{len(PARAM_GRID['min_samples_leaf'])} × {len(PARAM_GRID['max_features'])} = "
          f"{len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_leaf']) * len(PARAM_GRID['max_features'])} комбинаций")
    
    # Стратегия кросс-валидации
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Инициализация модели
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        cv=cv_strategy,
        scoring='f1',          # Оптимизация по F1-score
        n_jobs=-1,             # Все ядра CPU
        verbose=verbose
    )
    
    # Обучение
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, feature_cols: list):
    """
    Оценка модели на тестовой выборке.
    """
    print("\n📊 Оценка на тестовых данных:")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n📈 Отчёт классификации:")
    print(classification_report(y_test, y_pred, target_names=['Чистая (0)', 'Медь (1)']))
    
    print(f"🎯 Accuracy:  {acc:.4f}")
    print(f"🎯 F1-Score:  {f1:.4f}")
    print(f"🎯 ROC-AUC:   {roc_auc:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Чистая', 'Медь'],
                yticklabels=['Чистая', 'Медь'])
    plt.xlabel('Предсказано')
    plt.ylabel('Фактически')
    plt.title('Матрица ошибок (Test Set)')
    plt.tight_layout()
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def save_model(model, save_path: str, feature_cols: list, metrics: dict, grid_search):
    """
    Сохранение модели и отчёта.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохранение модели
    joblib.dump(model, save_path)
    print(f"\n💾 Модель сохранена: {save_path}")
    
    # Сохранение отчёта
    report_path = save_path.with_suffix('.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЁТ ОБ ОБУЧЕНИИ МОДЕЛИ RANDOM FOREST\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"🔧 Лучшие параметры:\n")
        for k, v in grid_search.best_params_.items():
            f.write(f"   {k}: {v}\n")
        
        f.write(f"\n📈 Лучший F1-Score на валидации: {grid_search.best_score_:.4f}\n")
        f.write(f"\n🎯 Метрики на тесте:\n")
        f.write(f"   Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"   F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"   ROC-AUC:  {metrics['roc_auc']:.4f}\n")
        
        f.write(f"\n🔝 Важность признаков:\n")
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        for col, imp in importances.sort_values(ascending=False).items():
            f.write(f"   {col:30s} {imp:.4f}\n")
    
    print(f"📝 Отчёт сохранён: {report_path}")
    
    # Сохранение матрицы ошибок
    cm_path = save_path.with_suffix('.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"🖼️  Матрица ошибок сохранена: {cm_path}")
    plt.close()


def print_feature_importance(model, feature_cols: list, top_n: int = 10):
    """
    Вывод важности признаков.
    """
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    
    print(f"\n🔝 Топ-{top_n} самых важных признаков:")
    for i, (col, imp) in enumerate(importances.head(top_n).items(), 1):
        print(f"   {i:2d}. {col:30s} {imp:.4f}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    importances.head(top_n).plot(kind='barh', color='steelblue')
    plt.xlabel('Важность (Gini importance)')
    plt.ylabel('Признак')
    plt.title(f'Топ-{top_n} важных признаков')
    plt.tight_layout()
    plt.show()


# ============================================================
# 🚀 ОСНОВНОЙ ПРОЦЕСС
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Обучение модели Random Forest для классификации водной среды'
    )
    
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Путь к CSV файлу с датасетом'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Путь для сохранения модели (по умолчанию: автоматически)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Доля тестовой выборки (по умолчанию: 0.2)'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Случайное состояние (по умолчанию: 42)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Не показывать графики (для headless-серверов)'
    )
    
    args = parser.parse_args()
    
    if args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
    
    # Проверка входного файла
    if not Path(args.dataset).exists():
        print(f"❌ Датасет не найден: {args.dataset}")
        return
    
    # Соответствие датасетов и имён моделей (как у вас в папке)
    MODEL_NAMES = {
        'dataset_5sec_7features': 'окно 5 секунд',
        'dataset_30s15s_robust': 'окно 30 сек(15 сек доп признаки)',
        'dataset_1min30s_robust': 'окно 1 минута(30 сек доп признаки)',
        'dataset_5min_multiscale1s_features': 'окно 5 минут(1 мин доп признаки)',
        'dataset_5min_multiscale': 'окно 5 минут(30 сек доп признаки)',
    }
    
    # Определяем путь для сохранения модели
    if args.output:
        model_path = args.output
    else:
        dataset_name = Path(args.dataset).stem
        # Используем русское имя если есть в словаре
        if dataset_name in MODEL_NAMES:
            model_name = MODEL_NAMES[dataset_name]
        else:
            model_name = dataset_name
        
        model_path = str(PROJECT_ROOT / 'models' / 'rf' / f'{model_name}.pkl')
    
    # 1. Загрузка данных
    try:
        X, y, feature_cols = load_dataset(args.dataset)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    
    print(f"\n✂️  Разделение: Train={len(X_train)}, Test={len(X_test)}")
    print(f"   Train метки: {np.bincount(y_train)}")
    print(f"   Test метки:  {np.bincount(y_test)}")
    
    # 3. Обучение
    best_model, grid_search = train_model(X_train, y_train, verbose=1)
    
    print(f"\n✅ Лучшие параметры: {grid_search.best_params_}")
    print(f"📈 Лучший F1-Score (CV): {grid_search.best_score_:.4f}")
    
    # 4. Оценка
    metrics = evaluate_model(best_model, X_test, y_test, feature_cols)
    
    # 5. Важность признаков
    print_feature_importance(best_model, feature_cols, top_n=10)
    
    # 6. Сохранение
    save_model(best_model, model_path, feature_cols, metrics, grid_search)
    
    print(f"\n{'='*60}")
    print(f"✅ Обучение завершено!")
    print(f"{'='*60}\n")