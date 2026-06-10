# Daphnia Tracking System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Автоматизированная система биоиндикации качества водных сред на основе анализа поведения *Daphnia spp.* методами компьютерного зрения и машинного обучения.

## 📋 Описание

Проект реализует полный конвейер обработки видео для экспресс-оценки токсичности водных сред:

1. **Детекция** дафний с помощью нейронной сети YOLO26n
2. **Многоцелевой трекинг** с механизмом восстановления идентичности
3. **Извлечение** 11 поведенческих биомаркеров
4. **Классификация** состояния среды (Random Forest)

### ⚡ Преимущества

- **Быстро**: анализ за 5-15 минут вместо 24-96 часов
- **Точно**: Accuracy = 0.916, ROC-AUC = 0.947
- **Интерпретируемо**: выявление ключевых биомаркеров токсичности
- **Интегрально**: оценка комплексного воздействия смесей веществ

## 🚀 Установка

```bash
git clone https://github.com/your_username/daphnia-tracking.git
cd daphnia-tracking
pip install -r requirements.txt
📖 Использование
python scripts/extract_coordinates.py \
    --video path/to/video.mp4 \
    --model models/yolo_daphnia_best.pt \
    --output results/coords.csv
2. Извлечение признаков
# Для 5-секундных окон
python scripts/extract_features_5s.py --coords results/coords.csv --output datasets/features_5s.csv

# Для 30-секундных окон
python scripts/extract_features_30s_15s.py --coords results/coords.csv --output datasets/features_30s.csv

# Для 1-минутных окон
python scripts/extract_features_1min_30s.py --coords results/coords.csv --output datasets/features_1min.csv

# Для 5-минутных окон
python scripts/extract_features_5min.py --coords results/coords.csv --output datasets/features_5min.csv
3. Обучение моделей
python scripts/train_random_forest.py \
    --dataset datasets/features_30s.csv \
    --output models/rf_30s_15s.pkl
4. Практическая валидация
python scripts/validate_system.py \
    --coords results/coords.csv \
    --models_dir models/ \
    --output results/validation.csv
📊 Датасеты
Поддерживаются различные временные окна анализа:
Окно     Подокно       Признаков    Выборка(N)       Accuracy       ROC-AUC
5 мин     1 мин           17           105            0.952          0.990
1 мин     30 сек          13           537            0.907          0.956
30 сек    15 сек          11           1074           0.916          0.947
5 сек       -             7            6429           0.888          0.948
Оптимальная конфигурация: 30-секундные окна с 15-секундной агрегацией.
🔬 Поведенческие биомаркеры
Базовые признаки (7):
avg_speed — средняя скорость
total_distance — общее расстояние
straightness — прямолинейность траектории
median_speed — медианная скорость
mean_turning_angle — средний угол поворота
angular_velocity — угловая скорость
jump_frequency — частота прыжков
Расширенные признаки (4):
trajectory_entropy — энтропия траектории
fractal_dimension — фрактальная размерность
sinuosity — извилистость
velocity_autocorrelation — автокорреляция скорости
Агрегированные статистики (6):
speed_sub_mean/max/std — статистики скорости по подокнам
angle_sub_mean/std — статистики углов по подокнам
jump_sub_max — максимальная частота прыжков
🧪 Требования
Python 3.8+
NumPy >= 1.21.0
Pandas >= 1.3.0
SciPy >= 1.7.0
scikit-learn >= 1.0.0
OpenCV >= 4.5.0
Ultralytics YOLO >= 8.0.0
📄 Лицензия
MIT License — см. файл LICENSE
👨‍ Автор
Порошин Алексей Васильевич
Вятский государственный университет
Направление: 02.03.01 Математика и компьютерные науки
Научный руководитель: к.ф.-м.н., доцент Чупраков Дмитрий Вячеславович
