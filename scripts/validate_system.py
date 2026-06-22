"""
ПОЛНЫЙ ПАЙПЛАЙН: Видео → YOLO + Трекинг → Признаки → 5 моделей Random Forest
АВТОМАТИЧЕСКИЙ АНАЛИЗ ВСЕГО ВИДЕО С ВЫВОДОМ КАЖДЫЕ 5 МИНУТ
🔥 УМНАЯ ИТОГОВАЯ КЛАССИФИКАЦИЯ ПО ПОСЛЕДНИМ N СЕГМЕНТАМ
АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ ГРАФИКОВ И НУМЕРАЦИЯ ОПЫТОВ
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import joblib
from pathlib import Path
from collections import defaultdict, deque
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from tqdm import tqdm
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ============================================================
# ⚙️ НАСТРОЙКИ (ПУТИ)
# ============================================================
VIDEO_PATH = r"path/to/video.mp4"  # Измените на свой путь
YOLO_PATH  = r"path/to/yolo_model.pt"  # Измените на свой путь
MODELS_DIR = r"path/to/models"  # Измените на свой путь
OUT_DIR    = "./Итоговые результаты"
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 30
EXPECTED_COUNT = 3
OVERLAP_FRACTION = 0.75
CLASSIFICATION_THRESHOLD = 0.7

# 🔧 Период вывода результатов
REPORT_INTERVAL_MIN = 5  # Вывод каждые 5 минут

# 🔥 НОВОЕ: Умная итоговая классификация
FINAL_EVALUATION_SEGMENTS = 3  # Учитываем последние 3 сегмента для итогового решения

# 🔧 Конфигурация 5 моделей
MODEL_CONFIGS = {
    "5s":              (5*FPS,       None,       "окно 5 секунд без подокон.pkl"),
    "30s_15s":         (30*FPS,      15*FPS,     "окно 30 секунд подокно 15 секунд.pkl"),
    "1min_30s":        (1*60*FPS,    30*FPS,     "окно 1 минута подокно 30 секунд.pkl"),
    "5min_30s_slide":  (5*60*FPS,    30*FPS,     "окно 5 минут 30 секунд подокно(скользящие окна).pkl"),
    "5min_1min_slide": (5*60*FPS,    60*FPS,     "окно 5 минут 1 минута подокно(скользящие окна).pkl"),
}

FEATURE_SETS = {
    "5s": [
        'avg_speed', 'total_distance', 'straightness', 'median_speed',
        'mean_turning_angle', 'angular_velocity', 'jump_frequency'
    ],
    "30s_15s": [
        'avg_speed', 'total_distance', 'straightness', 'median_speed',
        'mean_turning_angle', 'angular_velocity', 'jump_frequency',
        'speed_15s_mean', 'speed_15s_max', 'speed_15s_std',
        'angle_15s_mean', 'angle_15s_std', 'jump_15s_max'
    ],
    "1min_30s": [
        'avg_speed', 'total_distance', 'straightness', 'median_speed',
        'mean_turning_angle', 'angular_velocity', 'jump_frequency',
        'speed_30s_mean', 'speed_30s_max', 'speed_30s_std',
        'angle_30s_mean', 'angle_30s_std', 'jump_30s_max'
    ],
    "5min_30s_slide": [
        'avg_speed', 'total_distance', 'straightness', 'median_speed',
        'mean_turning_angle', 'angular_velocity', 'jump_frequency',
        'trajectory_entropy', 'fractal_dimension', 'sinuosity', 'velocity_autocorrelation',
        'speed_30s_mean', 'speed_30s_max', 'speed_30s_std',
        'angle_30s_mean', 'angle_30s_std', 'jump_30s_max'
    ],
    "5min_1min_slide": [
        'avg_speed', 'total_distance', 'straightness', 'median_speed',
        'mean_turning_angle', 'angular_velocity', 'jump_frequency',
        'trajectory_entropy', 'fractal_dimension', 'sinuosity', 'velocity_autocorrelation',
        'speed_1min_mean', 'speed_1min_max', 'speed_1min_std',
        'angle_1min_mean', 'angle_1min_std', 'jump_1min_max'
    ],
}

# ============================================================
# 📁 УПРАВЛЕНИЕ ПАПКАМИ И ФАЙЛАМИ
# ============================================================
def get_next_experiment_number(out_dir):
    existing_folders = [f for f in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, f))]
    experiment_numbers = []
    for folder in existing_folders:
        if folder.startswith("Опыт_"):
            try:
                num = int(folder.split("_")[1])
                experiment_numbers.append(num)
            except (IndexError, ValueError):
                continue
    return max(experiment_numbers) + 1 if experiment_numbers else 1


def create_analysis_folder(video_path, duration_min):
    experiment_num = get_next_experiment_number(OUT_DIR)
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Опыт_{experiment_num}_{video_name}_0-{duration_min:.0f}min_{timestamp}"
    folder_path = os.path.join(OUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, experiment_num


def save_metadata(folder_path, video_path, duration_min, video_info, 
                  overall_prob, final_class, experiment_num, periodic_results):
    metadata = {
        "experiment_number": experiment_num,
        "video_file": os.path.basename(video_path),
        "video_path": video_path,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_min": duration_min,
        "video_info": {
            "fps": video_info['fps'],
            "resolution": f"{video_info['width']}x{video_info['height']}",
            "total_duration_min": video_info['duration_min']
        },
        "results": {
            "overall_probability": float(overall_prob),
            "classification": final_class,
            "threshold": CLASSIFICATION_THRESHOLD,
            "final_evaluation_segments": FINAL_EVALUATION_SEGMENTS
        },
        "periodic_results": periodic_results,
        "parameters": {
            "overlap_fraction": OVERLAP_FRACTION,
            "expected_daphnia_count": EXPECTED_COUNT,
            "fps": FPS,
            "report_interval_min": REPORT_INTERVAL_MIN
        }
    }
    metadata_path = os.path.join(folder_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata_path


# ============================================================
# 📊 ГЕНЕРАЦИЯ ГРАФИКОВ
# ============================================================
def plot_analysis_results(results_df, track_model_probs, analysis_folder, 
                         duration_min, experiment_num):
    print("\n📊 Генерация графиков...")
    
    # График 1: Временные ряды
    fig, axes = plt.subplots(len(track_model_probs), 1, figsize=(14, 4*len(track_model_probs)))
    if len(track_model_probs) == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(MODEL_CONFIGS)))
    
    for idx, (track_id, models_data) in enumerate(sorted(track_model_probs.items())):
        ax = axes[idx]
        for m_idx, (model_name, probs) in enumerate(models_data.items()):
            if probs:
                windows = range(len(probs))
                ax.plot(windows, probs, label=model_name, color=colors[m_idx], 
                       linewidth=2, alpha=0.7)
                anomaly_windows = [w for w, p in enumerate(probs) if p > CLASSIFICATION_THRESHOLD]
                if anomaly_windows:
                    ax.scatter(anomaly_windows, [probs[w] for w in anomaly_windows], 
                             color='red', s=50, zorder=5, alpha=0.8)
        
        ax.axhline(y=CLASSIFICATION_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, label=f'Порог ({CLASSIFICATION_THRESHOLD})')
        ax.set_xlabel('Номер окна')
        ax.set_ylabel('Вероятность токсичности')
        ax.set_title(f'Особь {track_id} - Временной ряд вероятностей')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, f'Опыт_{experiment_num}_временные_ряды.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Временные ряды сохранены")
    
    # График 2: Heatmap
    track_ids = sorted(track_model_probs.keys())
    model_names = list(MODEL_CONFIGS.keys())
    
    heatmap_data = []
    for track_id in track_ids:
        row = []
        for model in model_names:
            probs = track_model_probs[track_id].get(model, [])
            row.append(np.median(probs) if probs else 0)
        heatmap_data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
               xticklabels=model_names, yticklabels=[f'Особь {tid}' for tid in track_ids],
               vmin=0, vmax=1, cbar_kws={'label': 'Медианная вероятность токсичности'})
    plt.title(f'Опыт {experiment_num} - Heatmap медианных вероятностей')
    plt.xlabel('Модель')
    plt.ylabel('Особь')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, f'Опыт_{experiment_num}_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Heatmap сохранён")
    
    # График 3: Box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data, box_labels = [], []
    for model in model_names:
        all_probs = []
        for track_id in track_ids:
            all_probs.extend(track_model_probs[track_id].get(model, []))
        if all_probs:
            box_data.append(all_probs)
            box_labels.append(model)
    
    ax.boxplot(box_data, labels=box_labels, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red', linewidth=2))
    ax.axhline(y=CLASSIFICATION_THRESHOLD, color='red', linestyle='--', 
               linewidth=2, label=f'Порог ({CLASSIFICATION_THRESHOLD})')
    ax.set_xlabel('Модель')
    ax.set_ylabel('Вероятность токсичности')
    ax.set_title(f'Опыт {experiment_num} - Распределение вероятностей по моделям')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, f'Опыт_{experiment_num}_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Box plot сохранён")
    
    # График 4: Сравнение среднего и медианы
    fig, ax = plt.subplots(figsize=(12, 6))
    models_list, means_list, medians_list = [], [], []
    for model in model_names:
        all_probs = []
        for track_id in track_ids:
            all_probs.extend(track_model_probs[track_id].get(model, []))
        if all_probs:
            models_list.append(model)
            means_list.append(np.mean(all_probs))
            medians_list.append(np.median(all_probs))
    
    x = np.arange(len(models_list))
    width = 0.35
    bars1 = ax.bar(x - width/2, means_list, width, label='Среднее', color='steelblue')
    bars2 = ax.bar(x + width/2, medians_list, width, label='Медиана', color='orange')
    ax.axhline(y=CLASSIFICATION_THRESHOLD, color='red', linestyle='--', 
               linewidth=2, label=f'Порог ({CLASSIFICATION_THRESHOLD})')
    ax.set_xlabel('Модель')
    ax.set_ylabel('Вероятность токсичности')
    ax.set_title(f'Опыт {experiment_num} - Сравнение среднего и медианы')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, f'Опыт_{experiment_num}_сравнение.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Сравнение моделей сохранено")
    
    print(f"✅ Все графики сохранены в: {analysis_folder}")


# ============================================================
# 🎬 ИНФОРМАЦИЯ О ВИДЕО (БЕЗ ИНТЕРАКТИВА)
# ============================================================
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    duration_min = duration_sec / 60
    
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_sec': duration_sec,
        'duration_min': duration_min
    }


# ============================================================
# 🔧 ТРЕКЕР (TrajectoryMerger)
# ============================================================
class TrajectoryMerger:
    def __init__(self, max_disappeared=60, max_distance=250, expected_count=EXPECTED_COUNT):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.expected_count = expected_count
        self.active_tracks = {}
        self.lost_tracks = {}
        self.next_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=30))

    def predict_next_position_tangent(self, history):
        n = len(history)
        if n < 2:
            return None
        if n == 2:
            p1, p2 = history[-2], history[-1]
            return (p2[0] + (p2[0] - p1[0]), p2[1] + (p2[1] - p1[1]))
        recent = history[-min(5, n):]
        velocities = [(recent[i][0] - recent[i-1][0], recent[i][1] - recent[i-1][1]) for i in range(1, len(recent))]
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        return (recent[-1][0] + avg_vx, recent[-1][1] + avg_vy)

    def update(self, detections):
        centers = [((d[0]+d[2])/2, (d[1]+d[3])/2, d) for d in detections]
        if not self.active_tracks:
            for cx, cy, det in centers:
                tid = self.next_id
                self.next_id += 1
                self.active_tracks[tid] = {'center': (cx, cy), 'detection': det, 'disappeared': 0}
                self.track_history[tid].append((cx, cy))
            return self.active_tracks

        active_ids = list(self.active_tracks.keys())
        cost = np.zeros((len(active_ids), len(centers)))
        for i, tid in enumerate(active_ids):
            pred = self.predict_next_position_tangent(list(self.track_history[tid]))
            pos = pred if pred else self.active_tracks[tid]['center']
            for j, (cx, cy, _) in enumerate(centers):
                cost[i, j] = euclidean(pos, (cx, cy))

        matched_tracks, matched_dets = set(), set()
        if len(active_ids) and len(centers):
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= self.max_distance:
                    tid = active_ids[i]
                    cx, cy, det = centers[j]
                    self.active_tracks[tid].update({'center': (cx, cy), 'detection': det, 'disappeared': 0})
                    self.track_history[tid].append((cx, cy))
                    matched_tracks.add(tid)
                    matched_dets.add(j)

        for tid in active_ids:
            if tid not in matched_tracks:
                self.active_tracks[tid]['disappeared'] += 1
                if self.active_tracks[tid]['disappeared'] >= self.max_disappeared:
                    self.lost_tracks[tid] = {
                        'last_center': self.active_tracks[tid]['center'],
                        'history': list(self.track_history[tid]),
                        'frames_lost': 0
                    }
                    del self.active_tracks[tid]

        for j, (cx, cy, det) in enumerate(centers):
            if j not in matched_dets:
                current_known = len(self.active_tracks) + len(self.lost_tracks)
                rec_id = self._recover_lost((cx, cy))
                if rec_id is not None:
                    self.active_tracks[rec_id] = {'center': (cx, cy), 'detection': det, 'disappeared': 0}
                    self.track_history[rec_id].append((cx, cy))
                    del self.lost_tracks[rec_id]
                elif current_known < self.expected_count:
                    tid = self.next_id
                    self.next_id += 1
                    self.active_tracks[tid] = {'center': (cx, cy), 'detection': det, 'disappeared': 0}
                    self.track_history[tid].append((cx, cy))

        for lid in list(self.lost_tracks.keys()):
            self.lost_tracks[lid]['frames_lost'] += 1
            if self.lost_tracks[lid]['frames_lost'] > 150:
                del self.lost_tracks[lid]
        return self.active_tracks

    def _recover_lost(self, center):
        if not self.lost_tracks:
            return None
        sorted_lost = sorted(self.lost_tracks.items(), key=lambda x: x[1]['frames_lost'])
        best_id, best_dist = None, float('inf')
        for lid, data in sorted_lost:
            hist = data['history']
            pred = self.predict_next_position_tangent(hist) if len(hist) >= 2 else None
            dist = euclidean(center, pred) if pred else euclidean(center, data['last_center'])
            if dist <= self.max_distance * 2.5 and dist < best_dist:
                best_dist, best_id = dist, lid
        return best_id


# ============================================================
# 📐 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ============================================================
def extract_features(win_df, cfg_name, fps=30):
    x, y = win_df['x'].values, win_df['y'].values
    dt = 1.0 / fps
    dx, dy = np.diff(x), np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    speeds = dists / dt

    avg_sp = np.mean(speeds) if len(speeds) else 0.0
    tot_dist = np.sum(dists)
    net_dist = euclidean([x[0], y[0]], [x[-1], y[-1]])
    straight = net_dist / tot_dist if tot_dist > 1e-6 else 0.0
    med_sp = np.median(speeds) if len(speeds) else 0.0

    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            angles.append(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)) * 180 / np.pi)

    m_ang = np.mean(angles) if angles else 0.0
    a_vel = np.mean([a / dt for a in angles]) if angles else 0.0
    j_freq = np.sum(speeds > 2.0 * med_sp) / (len(speeds) * dt) if len(speeds) else 0.0

    feats = {
        'avg_speed': avg_sp, 'total_distance': tot_dist, 'straightness': straight,
        'median_speed': med_sp, 'mean_turning_angle': m_ang,
        'angular_velocity': a_vel, 'jump_frequency': j_freq
    }

    if cfg_name.startswith("5min"):
        H = 0.0
        if len(x) >= 10:
            hist, _, _ = np.histogram2d(x, y, bins=10)
            probs = hist.flatten()
            probs = probs / probs.sum()
            probs = probs[probs > 0]
            H = entropy(probs, base=2) if len(probs) > 0 else 0.0

        D_f = 1.0
        if len(x) >= 20:
            pts = np.column_stack([x, y])
            mins, maxs = pts.min(0), pts.max(0)
            rng = maxs - mins
            rng[rng == 0] = 1
            pts_n = (pts - mins) / rng
            scales = 2.0 ** -np.arange(1, 5)
            counts = [len(np.unique(np.floor(pts_n / s).astype(int), axis=0)) for s in scales]
            valid = np.array(counts) > 1
            if np.sum(valid) >= 2:
                D_f = np.polyfit(np.log(1.0 / scales[valid]), np.log(np.array(counts)[valid]), 1)[0]

        sinuosity = tot_dist / net_dist if net_dist > 1e-6 else 0.0
        vel_autocorr = 1.0
        if len(speeds) > 2 and np.std(speeds) > 1e-9:
            c = np.corrcoef(speeds[:-1], speeds[1:])
            vel_autocorr = c[0, 1] if not np.isnan(c[0, 1]) else 1.0

        feats.update({
            'trajectory_entropy': H, 'fractal_dimension': D_f,
            'sinuosity': sinuosity, 'velocity_autocorrelation': vel_autocorr
        })

    sub_frames = MODEL_CONFIGS[cfg_name][1]
    if sub_frames is not None:
        suffix = "1min" if sub_frames == 60 * FPS else ("15s" if sub_frames == 15 * FPS else "30s")
        agg = {}
        for base, key in [('speed', 'avg_speed'), ('angle', 'mean_turning_angle'), ('jump', 'jump_frequency')]:
            vals = []
            start = 0
            while start < len(win_df):
                sub = win_df.iloc[start:start + sub_frames]
                if len(sub) < 10:
                    start += sub_frames
                    continue
                sx, sy = sub['x'].values, sub['y'].values
                sd = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
                ss = sd / dt
                sm = np.median(ss) if len(ss) else 0.0

                if key == 'avg_speed':
                    vals.append(np.mean(ss) if len(ss) else 0.0)
                elif key == 'mean_turning_angle':
                    sang = []
                    for j in range(1, len(sx) - 1):
                        v1 = np.array([sx[j] - sx[j-1], sy[j] - sy[j-1]])
                        v2 = np.array([sx[j+1] - sx[j], sy[j+1] - sy[j]])
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-9 and n2 > 1e-9:
                            sang.append(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)) * 180 / np.pi)
                    vals.append(np.mean(sang) if sang else 0.0)
                else:
                    vals.append(np.sum(ss > 2.0 * sm) / (len(ss) * dt) if len(ss) else 0.0)
                start += sub_frames

            if len(vals) >= 2:
                agg.update({
                    f'{base}_{suffix}_mean': np.mean(vals),
                    f'{base}_{suffix}_max': np.max(vals),
                    f'{base}_{suffix}_std': np.std(vals)
                })
            else:
                agg.update({
                    f'{base}_{suffix}_mean': feats[key],
                    f'{base}_{suffix}_max': feats[key],
                    f'{base}_{suffix}_std': 0.0
                })
        feats.update(agg)

    return feats


# ============================================================
# 🎥 ШАГ 1: YOLO + ТРЕКИНГ (ВСЁ ВИДЕО)
# ============================================================
def run_tracking_full_video(video_path, yolo_path, fps=30):
    """Запускает YOLO + трекинг на ВСЁМ видео без вопросов."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Не удалось открыть видео")

    w, h, vid_fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    total_frames = int(cap.get(7))
    vid_fps = vid_fps if vid_fps > 0 else fps

    yolo_model = YOLO(yolo_path)
    merger = TrajectoryMerger(max_disappeared=60, max_distance=250, expected_count=EXPECTED_COUNT)

    trajectories = defaultdict(list)

    for fr in tqdm(range(total_frames), unit="кадр", desc="🎥 Обработка видео"):
        ret, frame = cap.read()
        if not ret:
            break

        res = yolo_model(frame, verbose=False, conf=0.7)
        dets = []
        if res[0].boxes is not None:
            for box in res[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                dets.append([x1, y1, x2, y2])

        tracks = merger.update(dets)
        for tid, data in tracks.items():
            trajectories[tid].append((fr, data['center'][0], data['center'][1]))

    cap.release()

    long_tracks = {tid: traj for tid, traj in trajectories.items() if len(traj) >= 200}
    return long_tracks, vid_fps


# ============================================================
# 🤖 ШАГ 2: КЛАССИФИКАЦИЯ ПО 5-МИНУТНЫМ СЕГМЕНТАМ
# ============================================================
def run_classification_by_segments(trajectories, fps, models_dir, duration_min):
    """
    Разбивает видео на 5-минутные сегменты и для каждого применяет все 5 моделей.
    Выводит результат для каждого сегмента.
    """
    models = {}
    for name, (_, _, fname) in MODEL_CONFIGS.items():
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)
            print(f"   ✅ Загружена модель: {name}")
        else:
            print(f"   ⚠️  Модель не найдена: {fname}")

    if not models:
        raise FileNotFoundError("❌ Ни одна модель не найдена!")

    # Собираем все данные
    all_rows = []
    for tid, traj in trajectories.items():
        for frame, x, y in traj:
            all_rows.append({'track_id': tid, 'frame': frame, 'x': x, 'y': y})
    df = pd.DataFrame(all_rows)

    if df.empty:
        raise ValueError("❌ Нет данных для анализа!")

    # 🔥 РАЗБИЕНИЕ НА 5-МИНУТНЫЕ СЕГМЕНТЫ
    segment_frames = REPORT_INTERVAL_MIN * 60 * fps
    total_frames = int(duration_min * 60 * fps)
    
    periodic_results = []
    all_results = []
    all_track_model_probs = defaultdict(lambda: defaultdict(list))
    all_model_stats = {n: {'cnt': 0, 'prob': 0.0, 'preds': []} for n in models}

    segment_start = 0
    segment_num = 0
    
    while segment_start < total_frames:
        segment_end = min(segment_start + segment_frames, total_frames)
        segment_min_start = segment_start / (fps * 60)
        segment_min_end = segment_end / (fps * 60)
        segment_num += 1
        
        # Фильтруем данные для этого сегмента
        segment_df = df[(df['frame'] >= segment_start) & (df['frame'] < segment_end)]
        
        if len(segment_df) < 100:
            segment_start += segment_frames
            continue
        
        # Классификация для этого сегмента
        segment_track_probs = defaultdict(lambda: defaultdict(list))
        segment_model_stats = {n: {'cnt': 0, 'prob': 0.0, 'preds': []} for n in models}
        
        for track_id, group in segment_df.groupby('track_id'):
            group = group.sort_values('frame').reset_index(drop=True)
            if len(group) < 100:
                continue
            
            for m_name, (win_frames, _, _) in MODEL_CONFIGS.items():
                if m_name not in models:
                    continue
                
                windows = []
                start = 0
                step = int(win_frames * (1 - OVERLAP_FRACTION))
                if step < 1:
                    step = 1
                
                while start < len(group):
                    end = start + win_frames
                    w = group.iloc[start:end]
                    if len(w) < win_frames // 2:
                        break
                    windows.append(extract_features(w, m_name, fps))
                    start += step
                
                if not windows:
                    continue
                
                feat_cols = FEATURE_SETS[m_name]
                X = pd.DataFrame(windows)[feat_cols].values
                X = np.nan_to_num(X, nan=0.0)
                
                preds = models[m_name].predict(X)
                probs = models[m_name].predict_proba(X)[:, 1]
                
                segment_track_probs[track_id][m_name].extend(probs.tolist())
                all_track_model_probs[track_id][m_name].extend(probs.tolist())
                
                segment_model_stats[m_name]['cnt'] += len(probs)
                segment_model_stats[m_name]['prob'] += np.sum(probs)
                segment_model_stats[m_name]['preds'].extend(preds.tolist())
                
                all_model_stats[m_name]['cnt'] += len(probs)
                all_model_stats[m_name]['prob'] += np.sum(probs)
                all_model_stats[m_name]['preds'].extend(preds.tolist())
                
                for i, (p, pr) in enumerate(zip(preds, probs)):
                    res = {
                        'track_id': int(track_id),
                        'model': m_name,
                        'window': i,
                        'segment': segment_num,
                        'pred': int(p),
                        'prob': float(pr)
                    }
                    all_results.append(res)
        
        # 🔥 ВЫВОД РЕЗУЛЬТАТОВ ДЛЯ ЭТОГО СЕГМЕНТА
        segment_report = print_segment_results(
            segment_track_probs, segment_num, segment_min_start, segment_min_end
        )
        periodic_results.append(segment_report)
        
        segment_start += segment_frames
    
    return all_results, all_model_stats, all_track_model_probs, periodic_results


def print_segment_results(track_model_probs, segment_num, min_start, min_end):
    """Выводит результаты для одного 5-минутного сегмента."""
    print("\n" + "="*90)
    print(f"📊 СЕГМЕНТ №{segment_num}: {min_start:.1f} - {min_end:.1f} мин")
    print("="*90)
    
    # Рассчитываем медианные вероятности для каждой особи
    track_median_probs = {}
    for track_id in sorted(track_model_probs.keys()):
        row_probs = []
        for m_name in MODEL_CONFIGS.keys():
            probs = track_model_probs[track_id].get(m_name, [])
            if probs:
                median_prob = np.median(probs)
                row_probs.append(median_prob)
        
        if row_probs:
            track_median_probs[track_id] = np.median(row_probs)
    
    # Общая вероятность (медиана по всем особям)
    if track_median_probs:
        overall_prob = np.median(list(track_median_probs.values()))
    else:
        overall_prob = 0.0
    
    final_class = "🔴 МЕДЬ (токсичность)" if overall_prob > CLASSIFICATION_THRESHOLD else "🟢 ЧИСТАЯ ВОДА"
    
    print(f"\n🔹 Обработано особей: {len(track_median_probs)}")
    print(f"🔹 Итоговая вероятность: {overall_prob:.3f}")
    print(f"🔹 Класс: {final_class}")
    
    # Детальная информация по моделям
    print(f"\n📈 Результаты по моделям:")
    print("-"*90)
    print(f"{'Модель':<20} | {'Вероятность':>12} | {'Медиана':>10} | {'Статус':>15}")
    print("-"*90)
    
    model_details = {}
    for m_name in MODEL_CONFIGS.keys():
        all_probs = []
        for track_id in track_model_probs.keys():
            all_probs.extend(track_model_probs[track_id].get(m_name, []))
        
        if all_probs:
            avg_prob = np.mean(all_probs)
            median_prob = np.median(all_probs)
            status = "🔴 Токсичность" if median_prob > CLASSIFICATION_THRESHOLD else "🟢 Чистая"
            print(f"{m_name:<20} | {avg_prob:>12.3f} | {median_prob:>10.3f} | {status:>15}")
            model_details[m_name] = {'avg': float(avg_prob), 'median': float(median_prob)}
    
    # Детальная информация по особям
    print(f"\n📊 Результаты по особям:")
    print("-"*90)
    print(f"{'Особь':<10} | {'Вероятность':>12} | {'Статус':>15}")
    print("-"*90)
    
    track_details = {}
    for track_id, prob in sorted(track_median_probs.items()):
        status = "🔴 Токсичность" if prob > CLASSIFICATION_THRESHOLD else "🟢 Чистая"
        print(f"{track_id:<10} | {prob:>12.3f} | {status:>15}")
        track_details[int(track_id)] = float(prob)
    
    print("="*90)
    
    # Формат для ВКР
    print(f"\n📝 ФОРМАТ ДЛЯ ВКР (копирование):")
    print(f"На {min_end:.1f} минуте анализа (сегмент {min_start:.1f}-{min_end:.1f} мин):")
    print(f"  - Обработано особей: {len(track_median_probs)}")
    print(f"  - Итоговая вероятность токсичности: {overall_prob:.3f}")
    print(f"  - Классификация: {'МЕДЬ (токсичность)' if overall_prob > CLASSIFICATION_THRESHOLD else 'ЧИСТАЯ ВОДА'}")
    print(f"  - Статус: {'🔴 Загрязнение обнаружено' if overall_prob > CLASSIFICATION_THRESHOLD else '🟢 Вода чистая'}")
    
    return {
        'segment': int(segment_num),
        'min_start': float(min_start),
        'min_end': float(min_end),
        'overall_prob': float(overall_prob),
        'final_class': str(final_class),
        'n_tracks': int(len(track_median_probs)),
        'track_details': track_details,
        'model_details': model_details
    }


# ============================================================
# 🧠 УМНАЯ ИТОГОВАЯ КЛАССИФИКАЦИЯ
# ============================================================
def smart_final_classification(periodic_results):
    """
    Умная итоговая классификация по последним N сегментам.
    Это позволяет системе быстро реагировать на изменения среды.
    """
    if not periodic_results:
        return 0.0, "🟢 ЧИСТАЯ ВОДА"
    
    # 🔥 Берём последние N сегментов для итоговой оценки
    n_last = min(FINAL_EVALUATION_SEGMENTS, len(periodic_results))
    recent_segments = periodic_results[-n_last:]
    
    recent_probs = [s['overall_prob'] for s in recent_segments]
    final_prob = np.mean(recent_probs)
    
    print(f"\n🧠 УМНАЯ ИТОГОВАЯ КЛАССИФИКАЦИЯ:")
    print(f"   Учитываем последние {n_last} сегментов:")
    for i, seg in enumerate(recent_segments):
        print(f"     Сегмент {seg['segment']}: {seg['min_start']:.1f}-{seg['min_end']:.1f} мин → {seg['overall_prob']:.3f}")
    print(f"   Среднее: {final_prob:.3f}")
    
    final_class = "🔴 МЕДЬ (токсическое воздействие)" if final_prob > CLASSIFICATION_THRESHOLD else "🟢 ЧИСТАЯ ВОДА"
    
    return final_prob, final_class


# ============================================================
# 🚀 ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================
def run_pipeline(video_path=None, yolo_path=None, models_dir=None, out_dir=None):
    """Запускает полный пайплайн анализа ВСЕГО видео автоматически."""

    video_path = video_path or VIDEO_PATH
    yolo_path = yolo_path or YOLO_PATH
    models_dir = models_dir or MODELS_DIR
    out_dir = out_dir or OUT_DIR

    print("="*80)
    print(f"🎬 АВТОМАТИЧЕСКИЙ АНАЛИЗ ВСЕГО ВИДЕО")
    print(f"   Порог классификации: {CLASSIFICATION_THRESHOLD}")
    print(f"   Вывод результатов каждые {REPORT_INTERVAL_MIN} минут")
    print(f"   🧠 Умная итоговая классификация по последним {FINAL_EVALUATION_SEGMENTS} сегментам")
    print("="*80)
    print(f"📹 Видео: {video_path}")
    print(f"🤖 YOLO: {yolo_path}")
    print(f"🌲 Модели: {models_dir}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Видео не найдено: {video_path}")
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"❌ Модель YOLO не найдена: {yolo_path}")

    video_info = get_video_info(video_path)
    duration_min = video_info['duration_min']
    
    print(f"\n📐 Разрешение: {video_info['width']}x{video_info['height']}")
    print(f"🎞️  FPS: {video_info['fps']:.1f}")
    print(f"🖼️  Всего кадров: {video_info['total_frames']}")
    print(f"⏱️  Длительность: {duration_min:.2f} мин")
    print(f"📊 Количество сегментов: {int(np.ceil(duration_min / REPORT_INTERVAL_MIN))}")

    # 🔥 СОЗДАНИЕ УНИКАЛЬНОЙ ПАПКИ С НОМЕРОМ ОПЫТА
    analysis_folder, experiment_num = create_analysis_folder(video_path, duration_min)
    print(f"\n📁 Опыт №{experiment_num}")
    print(f"📁 Результаты будут сохранены в: {analysis_folder}")

    print("\n" + "="*80)
    print("🎥 ШАГ 1: YOLO + Трекинг (ВСЁ ВИДЕО)")
    print("="*80)
    trajectories, fps = run_tracking_full_video(video_path, yolo_path, FPS)

    if not trajectories:
        print("❌ Треки не найдены. Проверьте видео и модель YOLO.")
        return

    print(f"✅ Трекинг завершён. Найдено {len(trajectories)} треков.")
    print(f"📈 Длинных треков (≥200 кадров): {len(trajectories)}")

    coords_rows = []
    for tid, traj in trajectories.items():
        for frame, x, y in traj:
            coords_rows.append({'track_id': tid, 'frame': frame, 'x': x, 'y': y})
    coords_df = pd.DataFrame(coords_rows)
    coords_path = os.path.join(analysis_folder, f"coordinates_0-{duration_min:.0f}min.csv")
    coords_df.to_csv(coords_path, index=False, encoding='utf-8-sig')
    print(f"💾 Координаты сохранены: {coords_path}")

    print("\n" + "="*80)
    print("🤖 ШАГ 2: Классификация по 5-минутным сегментам")
    print("="*80)
    
    results, model_stats, track_model_probs, periodic_results = run_classification_by_segments(
        trajectories, fps, models_dir, duration_min
    )

    if not track_model_probs:
        print("❌ Не удалось получить предсказания. Проверьте модели.")
        return

    # 🔥 ШАГ 3: УМНАЯ ИТОГОВАЯ КЛАССИФИКАЦИЯ
    print("\n" + "="*90)
    print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМУ ВИДЕО")
    print("="*90)
    
    # 🔥 Используем умную классификацию по последним N сегментам
    overall_prob, final_class = smart_final_classification(periodic_results)
    
    print(f"\n📍 Анализ видео: 0 - {duration_min:.2f} мин")
    print(f"🔍 Обработано треков: {len(trajectories)}")
    print(f"🎯 ИТОГОВАЯ вероятность (по последним {FINAL_EVALUATION_SEGMENTS} сегментам): {overall_prob:.3f}")
    print(f"✅ ИТОГОВЫЙ КЛАСС: {final_class}")
    
    print("\n📈 Разбивка по моделям (по МЕДИАНЕ за всё видео):")
    print("-"*90)
    print(f"{'Модель':<20} | {'Окон':>6} | {'Среднее':>10} | {'Медиана':>10} | {'Чистая':>7} | {'Медь':>7} | Класс")
    print("-"*90)
    
    for m_name in MODEL_CONFIGS.keys():
        if m_name in model_stats and model_stats[m_name]['cnt'] > 0:
            st = model_stats[m_name]
            avg = st['prob'] / st['cnt']
            
            all_model_probs = []
            for track_id in track_model_probs:
                all_model_probs.extend(track_model_probs[track_id].get(m_name, []))
            
            median = np.median(all_model_probs) if all_model_probs else 0.0
            
            clean_cnt = sum(1 for p in st['preds'] if p == 0)
            copper_cnt = sum(1 for p in st['preds'] if p == 1)
            
            cls = '🔴 Медь' if median > CLASSIFICATION_THRESHOLD else '🟢 Чистая'
            print(f"{m_name:<20} | {st['cnt']:>6} | {avg:>10.3f} | {median:>10.3f} | {clean_cnt:>7} | {copper_cnt:>7} | {cls}")
    
    print("="*90)

    # 🔥 СВОДНАЯ ТАБЛИЦА ПО СЕГМЕНТАМ (для ВКР)
    print("\n" + "="*90)
    print("📋 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ПО СЕГМЕНТАМ (ДЛЯ ВКР)")
    print("="*90)
    print(f"{'Сегмент':<10} | {'Время (мин)':<15} | {'Вероятность':>12} | {'Класс':<25} | {'Особей':>7}")
    print("-"*90)
    
    for pr in periodic_results:
        time_range = f"{pr['min_start']:.1f} - {pr['min_end']:.1f}"
        class_short = "МЕДЬ (токсичность)" if "МЕДЬ" in pr['final_class'] else "ЧИСТАЯ ВОДА"
        print(f"№{pr['segment']:<9} | {time_range:<15} | {pr['overall_prob']:>12.3f} | {class_short:<25} | {pr['n_tracks']:>7}")
    
    print("="*90)

    # 💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    res_df = pd.DataFrame(results)
    res_path = os.path.join(analysis_folder, f"prediction_0-{duration_min:.0f}min.csv")
    res_df.to_csv(res_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Детальные результаты сохранены: {res_path}")

    # 🔥 ГРАФИКИ
    plot_analysis_results(res_df, track_model_probs, analysis_folder, 
                         duration_min, experiment_num)

    # 💾 МЕТАДАННЫЕ
    metadata_path = save_metadata(analysis_folder, video_path, duration_min, 
                                   video_info, overall_prob, final_class, experiment_num,
                                   periodic_results)
    print(f"💾 Метаданные сохранены: {metadata_path}")

    # 💾 СВОДНЫЙ ОТЧЁТ
    summary_path = os.path.join(analysis_folder, f"SUMMARY_Опыт_{experiment_num}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"📊 СВОДНЫЙ ОТЧЁТ - ОПЫТ №{experiment_num}\n")
        f.write("="*80 + "\n\n")
        f.write(f"📹 Видео: {os.path.basename(video_path)}\n")
        f.write(f"⏱️  Длительность видео: {duration_min:.2f} мин\n")
        f.write(f"🔍 Обработано треков: {len(trajectories)}\n\n")
        f.write(f"🎯 ИТОГОВАЯ ВЕРОЯТНОСТЬ: {overall_prob:.3f}\n")
        f.write(f"✅ КЛАССИФИКАЦИЯ: {final_class}\n\n")
        f.write("-"*80 + "\n")
        f.write(f"🧠 УМНАЯ КЛАССИФИКАЦИЯ (последние {FINAL_EVALUATION_SEGMENTS} сегмента):\n")
        f.write("-"*80 + "\n")
        recent_segments = periodic_results[-FINAL_EVALUATION_SEGMENTS:]
        for seg in recent_segments:
            f.write(f"  Сегмент {seg['segment']}: {seg['min_start']:.1f}-{seg['min_end']:.1f} мин → {seg['overall_prob']:.3f}\n")
        f.write(f"  Среднее: {overall_prob:.3f}\n\n")
        f.write("-"*80 + "\n")
        f.write("📋 РЕЗУЛЬТАТЫ ПО СЕГМЕНТАМ (каждые 5 минут):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Сегмент':<10} | {'Время (мин)':<15} | {'Вероятность':>12} | {'Класс':<25} | {'Особей':>7}\n")
        f.write("-"*80 + "\n")
        for pr in periodic_results:
            time_range = f"{pr['min_start']:.1f} - {pr['min_end']:.1f}"
            class_short = "МЕДЬ" if "МЕДЬ" in pr['final_class'] else "ЧИСТАЯ"
            f.write(f"№{pr['segment']:<9} | {time_range:<15} | {pr['overall_prob']:>12.3f} | {class_short:<25} | {pr['n_tracks']:>7}\n")
        f.write("-"*80 + "\n\n")
        f.write("-"*80 + "\n")
        f.write("📈 РЕЗУЛЬТАТЫ ПО МОДЕЛЯМ:\n")
        f.write("-"*80 + "\n")
        for m_name in MODEL_CONFIGS.keys():
            if m_name in model_stats and model_stats[m_name]['cnt'] > 0:
                st = model_stats[m_name]
                avg = st['prob'] / st['cnt']
                all_model_probs = []
                for track_id in track_model_probs:
                    all_model_probs.extend(track_model_probs[track_id].get(m_name, []))
                median = np.median(all_model_probs) if all_model_probs else 0.0
                cls = 'МЕДЬ' if median > CLASSIFICATION_THRESHOLD else 'ЧИСТАЯ'
                f.write(f"{m_name:<20} | Окно: {st['cnt']:>4} | Медиана: {median:.3f} | {cls}\n")
        f.write("="*80 + "\n")
        f.write(f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"💾 Сводный отчёт сохранён: {summary_path}")

    print("\n" + "="*80)
    if overall_prob > 0.8:
        print("🔴 ВЕРДИКТ: СРЕДА ЗАГРЯЗНЕНА (высокая уверенность)")
    elif overall_prob > CLASSIFICATION_THRESHOLD:
        print("⚠️ ВЕРДИКТ: СРЕДА ЗАГРЯЗНЕНА (средняя уверенность)")
    elif overall_prob > 0.5:
        print("⚠️ ВЕРДИКТ: НЕОПРЕДЕЛЁННОЕ СОСТОЯНИЕ")
    else:
        print("🟢 ВЕРДИКТ: СРЕДА ЧИСТАЯ (высокая уверенность)")
    print("="*80)
    print(f"\n📁 ВСЕ РЕЗУЛЬТАТЫ ОПЫТА №{experiment_num} СОХРАНЕНЫ В:")
    print(f"   {analysis_folder}")
    print("="*80)


# ============================================================
# 🚀 ЗАПУСК
# ============================================================
if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n❌ Анализ прерван пользователем.")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
