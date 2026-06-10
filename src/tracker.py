"""
Модуль многоцелевого трекинга с механизмом восстановления идентичности.
"""

import numpy as np
from collections import defaultdict, deque
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment


class TrajectoryMerger:
    """
    Класс многоцелевого трекинга с механизмом восстановления идентичности
    и ограничением на количество отслеживаемых объектов.
    """
    
    def __init__(self, max_disappeared=60, max_distance=200, expected_count=3):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.expected_count = expected_count
        self.active_tracks = {}
        self.lost_tracks = {}
        self.next_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=30))
    
    def predict_next_position_tangent(self, history):
        """Прогнозирование следующей позиции на основе усреднённой скорости."""
        n = len(history)
        if n < 2:
            return None
        
        if n == 2:
            p1, p2 = history[-2], history[-1]
            return (p2[0] + (p2[0] - p1[0]), p2[1] + (p2[1] - p1[1]))
        
        recent = history[-min(5, n):]
        velocities = [
            (recent[i][0] - recent[i-1][0], recent[i][1] - recent[i-1][1])
            for i in range(1, len(recent))
        ]
        
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        return (recent[-1][0] + avg_vx, recent[-1][1] + avg_vy)
    
    def update(self, detections):
        """Обновление треков на основе новых детекций."""
        centers = [((d[0]+d[2])/2, (d[1]+d[3])/2, d) for d in detections]
        
        if not self.active_tracks:
            for cx, cy, det in centers:
                tid = self.next_id
                self.next_id += 1
                self.active_tracks[tid] = {
                    'center': (cx, cy),
                    'detection': det,
                    'disappeared': 0
                }
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
                    self.active_tracks[tid].update({
                        'center': (cx, cy),
                        'detection': det,
                        'disappeared': 0
                    })
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
                
                if current_known >= self.expected_count:
                    rec_id = self._recover_lost((cx, cy))
                    if rec_id is not None:
                        self.active_tracks[rec_id] = {
                            'center': (cx, cy),
                            'detection': det,
                            'disappeared': 0
                        }
                        self.track_history[rec_id].append((cx, cy))
                        del self.lost_tracks[rec_id]
                else:
                    rec_id = self._recover_lost((cx, cy))
                    if rec_id is not None:
                        self.active_tracks[rec_id] = {
                            'center': (cx, cy),
                            'detection': det,
                            'disappeared': 0
                        }
                        self.track_history[rec_id].append((cx, cy))
                        del self.lost_tracks[rec_id]
                    else:
                        tid = self.next_id
                        self.next_id += 1
                        self.active_tracks[tid] = {
                            'center': (cx, cy),
                            'detection': det,
                            'disappeared': 0
                        }
                        self.track_history[tid].append((cx, cy))
        
        for lid in list(self.lost_tracks.keys()):
            self.lost_tracks[lid]['frames_lost'] += 1
            if self.lost_tracks[lid]['frames_lost'] > 120:
                del self.lost_tracks[lid]
        
        return self.active_tracks
    
    def _recover_lost(self, center):
        """Восстановление потерянного трека по новой детекции."""
        if not self.lost_tracks:
            return None
        
        sorted_lost = sorted(self.lost_tracks.items(), 
                           key=lambda x: x[1]['frames_lost'])
        
        best_id = None
        best_dist = float('inf')
        max_rec_dist = self.max_distance * 1.5
        
        for lid, data in sorted_lost:
            hist = data['history']
            if len(hist) >= 2:
                pred = self.predict_next_position_tangent(hist)
                dist = euclidean(center, pred) if pred else euclidean(center, data['last_center'])
            else:
                dist = euclidean(center, data['last_center'])
            
            if dist <= max_rec_dist and dist < best_dist:
                best_dist = dist
                best_id = lid
        
        return best_id