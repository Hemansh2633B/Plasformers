"""Multi-object tracking utilities with ByteTrack-style association."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .engine.inference import Detection


@dataclass
class Track:
    """A tracked object trajectory."""

    track_id: int
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    age: int = 0
    hits: int = 1
    trajectory: list[tuple[float, float]] = field(default_factory=list)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


class ByteTrackTracker:
    """Lightweight ByteTrack-style IoU tracker."""

    def __init__(self, high_thresh: float = 0.5, match_thresh: float = 0.3, max_age: int = 30) -> None:
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        active = [det for det in detections if det.score >= self.high_thresh]
        assigned: set[int] = set()
        for track in self.tracks:
            best_idx = -1
            best_iou = 0.0
            for idx, det in enumerate(active):
                if idx in assigned or det.class_id != track.class_id:
                    continue
                iou = box_iou(track.box, det.box)
                if iou > best_iou:
                    best_idx = idx
                    best_iou = iou
            if best_idx >= 0 and best_iou >= self.match_thresh:
                det = active[best_idx]
                track.box = det.box
                track.score = det.score
                track.age = 0
                track.hits += 1
                track.trajectory.append(((det.box[0] + det.box[2]) * 0.5, (det.box[1] + det.box[3]) * 0.5))
                assigned.add(best_idx)
            else:
                track.age += 1
        for idx, det in enumerate(active):
            if idx in assigned:
                continue
            center = ((det.box[0] + det.box[2]) * 0.5, (det.box[1] + det.box[3]) * 0.5)
            self.tracks.append(
                Track(self.next_id, det.box, det.score, det.class_id, trajectory=[center])
            )
            self.next_id += 1
        self.tracks = [track for track in self.tracks if track.age <= self.max_age]
        return list(self.tracks)


class ZoneCounter:
    """Count trajectories entering polygonal zones using bounding-box centers."""

    def __init__(self, polygon: Sequence[tuple[float, float]]) -> None:
        self.polygon = tuple(polygon)
        self.counted: set[int] = set()

    def update(self, tracks: Iterable[Track]) -> int:
        for track in tracks:
            if track.track_id in self.counted or not track.trajectory:
                continue
            if self._inside(track.trajectory[-1]):
                self.counted.add(track.track_id)
        return len(self.counted)

    def _inside(self, point: tuple[float, float]) -> bool:
        x, y = point
        inside = False
        j = len(self.polygon) - 1
        for i, pi in enumerate(self.polygon):
            pj = self.polygon[j]
            if ((pi[1] > y) != (pj[1] > y)) and (
                x < (pj[0] - pi[0]) * (y - pi[1]) / max(pj[1] - pi[1], 1e-6) + pi[0]
            ):
                inside = not inside
            j = i
        return inside


TRACKER_REGISTRY = {
    "bytetrack": ByteTrackTracker,
    "bot-sort": ByteTrackTracker,
    "deepsort": ByteTrackTracker,
}
