"""Geometry and temporal logic for matching vehicle boxes to parking slots."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VehicleDetection:
    box: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class ParkingMatch:
    detection: VehicleDetection
    space_id: str
    overlap: float
    anchor: tuple[int, int]


def is_plausible_vehicle_box(
    detection,
    frame_shape,
    min_area_ratio=0.0005,
    max_area_ratio=0.20,
):
    """Reject tiny noise and implausibly large merged detections."""
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = detection.box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = frame_width * frame_height

    if frame_area == 0 or x2 <= x1 or y2 <= y1:
        return False

    area_ratio = box_area / frame_area
    return min_area_ratio <= area_ratio <= max_area_ratio


def vehicle_footprint(box, lower_box_fraction=0.45):
    """Return the lower portion of a YOLO box, where a vehicle meets a slot."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    footprint_top = y2 - (height * lower_box_fraction)

    return np.array(
        [
            [x1, footprint_top],
            [x2, footprint_top],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def match_detections_to_spaces(
    detections,
    parking_spaces,
    frame_shape,
    min_overlap=0.20,
    strong_overlap=0.55,
):
    """
    Match each plausible detection only to its best parking space.

    Overlap is intersection area divided by the lower vehicle-footprint area.
    The lower-center anchor normally must also be inside the space. A very
    strong overlap is accepted without the anchor to tolerate imperfect boxes.
    If duplicate detections choose one space, only the strongest one is kept.
    """
    polygons = {
        space_id: cv2.convexHull(np.asarray(points, dtype=np.float32))
        for space_id, points in parking_spaces.items()
    }
    candidates = []

    for detection in detections:
        if not is_plausible_vehicle_box(detection, frame_shape):
            continue

        x1, y1, x2, y2 = detection.box
        footprint = vehicle_footprint(detection.box)
        footprint_area = cv2.contourArea(footprint)
        if footprint_area <= 0:
            continue

        anchor = (
            int(round((x1 + x2) / 2)),
            int(round(y1 + (0.75 * (y2 - y1)))),
        )

        best_space_id = None
        best_overlap = 0.0
        best_anchor_inside = False

        for space_id, polygon in polygons.items():
            intersection_area, _ = cv2.intersectConvexConvex(
                footprint,
                polygon,
            )
            overlap = float(intersection_area) / footprint_area
            if overlap > best_overlap:
                best_space_id = space_id
                best_overlap = overlap
                best_anchor_inside = (
                    cv2.pointPolygonTest(
                        polygon,
                        (float(anchor[0]), float(anchor[1])),
                        False,
                    )
                    >= 0
                )

        if best_space_id is None or best_overlap < min_overlap:
            continue

        if not best_anchor_inside and best_overlap < strong_overlap:
            continue

        candidates.append(
            ParkingMatch(
                detection=detection,
                space_id=best_space_id,
                overlap=best_overlap,
                anchor=anchor,
            )
        )

    # One slot can contain only one vehicle. This also removes duplicate YOLO
    # boxes that survived non-maximum suppression.
    best_by_space = {}
    for match in candidates:
        previous = best_by_space.get(match.space_id)
        match_quality = (match.overlap, match.detection.confidence)
        previous_quality = (
            (previous.overlap, previous.detection.confidence)
            if previous is not None
            else (-1.0, -1.0)
        )
        if match_quality > previous_quality:
            best_by_space[match.space_id] = match

    return list(best_by_space.values())


class OccupancyTracker:
    """Smooth intermittent detections without making vacant slots stay red."""

    def __init__(
        self,
        space_ids,
        window_size=12,
        occupied_min_hits=3,
        available_after_misses=30,
    ):
        if occupied_min_hits > window_size:
            raise ValueError("occupied_min_hits cannot exceed window_size")

        self.window_size = window_size
        self.occupied_min_hits = occupied_min_hits
        self.available_after_misses = available_after_misses
        self.status = {space_id: "AVAILABLE" for space_id in space_ids}
        self._history = {
            space_id: deque(maxlen=window_size) for space_id in space_ids
        }
        self._misses = {space_id: 0 for space_id in space_ids}

    def update(self, detected_space_ids):
        detected_space_ids = set(detected_space_ids)

        for space_id in self.status:
            detected = space_id in detected_space_ids
            self._history[space_id].append(detected)

            if detected:
                self._misses[space_id] = 0
            else:
                self._misses[space_id] += 1

            if self._misses[space_id] >= self.available_after_misses:
                self.status[space_id] = "AVAILABLE"
            elif sum(self._history[space_id]) >= self.occupied_min_hits:
                self.status[space_id] = "OCCUPIED"

        return self.status.copy()
