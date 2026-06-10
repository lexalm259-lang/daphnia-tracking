"""
Трекер дафний — Извлечение координат из видео
Многоцелевой трекинг с механизмом восстановления идентичности
и ограничением на количество особей (expected_count = 3).
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

# ИМПОРТ КЛАССА ИЗ МОДУЛЯ
from src.tracker import TrajectoryMerger


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение координат дафний из видео с помощью YOLO и трекинга'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Путь к видеофайлу')
    parser.add_argument('--model', type=str, required=True,
                       help='Путь к модели YOLO (best.pt)')
    parser.add_argument('--output', type=str, default='.',
                       help='Папка для сохранения результатов')
    parser.add_argument('--conf', type=float, default=0.7,
                       help='Порог уверенности детекции (по умолчанию: 0.7)')
    parser.add_argument('--max-distance', type=int, default=200,
                       help='Максимальное расстояние для ассоциации (пиксели)')
    parser.add_argument('--max-disappeared', type=int, default=60,
                       help='Максимальное кадров отсутствия трека')
    parser.add_argument('--expected-count', type=int, default=3,
                       help='Ожидаемое количество особей в кадре')
    
    args = parser.parse_args()
    
    # Проверка существования файлов
    if not Path(args.video).exists():
        print(f"❌ Видео не найдено: {args.video}")
        return
    
    if not Path(args.model).exists():
        print(f" Модель не найдена: {args.model}")
        return
    
    print(f"🎥 Видео: {Path(args.video).name}")
    print(f" Модель: {Path(args.model).name}")
    print(f"🔢 Ожидаемое количество особей: {args.expected_count}")
    print(f"⚙️  Параметры: max_dist={args.max_distance}, max_lost={args.max_disappeared}\n")
    
    # Инициализация
    model = YOLO(args.model)
    merger = TrajectoryMerger(
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance,
        expected_count=args.expected_count
    )
    
    cap = cv2.VideoCapture(args.video)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    total_frames = int(cap.get(7))
    
    # Пути для сохранения
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(args.video).stem
    out_video = out_dir / f"{video_name}_tracked.mp4"
    out_csv = out_dir / f"{video_name}_coords.csv"
    out_img = out_dir / f"{video_name}_trajectories.png"
    
    writer = cv2.VideoWriter(
        str(out_video), 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, (w, h)
    )
    
    trajectories = defaultdict(list)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    print(f"⏳ Обработка {total_frames} кадров...\n")
    
    # Основной цикл обработки
    for frame_idx in tqdm(range(total_frames), unit="кадр"):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False, conf=args.conf)
        
        dets = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                dets.append([x1, y1, x2, y2, conf, 0])
        
        tracks = merger.update(dets)
        
        for tid, data in tracks.items():
            cx, cy = data['center']
            trajectories[tid].append((frame_idx, cx, cy))
            col = colors[tid % len(colors)]
            x1, y1, x2, y2 = map(int, data['detection'][:4])
            conf_val = data['detection'][4]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"ID:{tid} ({conf_val:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), col, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        info_text = f"Active: {len(tracks)} | Lost: {len(merger.lost_tracks)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
    
    cap.release()
    writer.release()
    
    # Сохранение координат
    with open(out_csv, 'w') as f:
        f.write("track_id,frame,x,y\n")
        for tid, traj in trajectories.items():
            for fr, x, y in traj:
                f.write(f"{tid},{fr},{x:.2f},{y:.2f}\n")
    
    # Визуализация траекторий
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    valid_count = 0
    
    for tid, traj in trajectories.items():
        if len(traj) < 20:
            continue
        valid_count += 1
        col = colors[tid % len(colors)]
        pts = [(int(x), int(y)) for _, x, y in traj]
        
        if len(pts) > 1:
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], col, 2)
            cv2.putText(img, f"ID:{tid}", (pts[-1][0]+5, pts[-1][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    
    cv2.putText(img, "Daphnia Tracking - Known Count Method", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.imwrite(str(out_img), img)
    
    print(f"\n✅ Готово!")
    print(f"📊 Траекторий (длина > 20): {valid_count}")
    print(f"\n Результаты:")
    print(f"    Видео: {out_video.name}")
    print(f"   📄 Координаты: {out_csv.name}")
    print(f"   🖼️ Траектории: {out_img.name}")


if __name__ == "__main__":
    main()