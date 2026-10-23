import unittest

from parking_logic import (
    OccupancyTracker,
    VehicleDetection,
    is_plausible_vehicle_box,
    match_detections_to_spaces,
)


class ParkingGeometryTests(unittest.TestCase):
    def setUp(self):
        self.spaces = {
            "A1": [[10, 10], [90, 10], [90, 90], [10, 90]],
            "A2": [[110, 10], [190, 10], [190, 90], [110, 90]],
        }
        self.frame_shape = (200, 200, 3)

    def detection(self, box, confidence=0.8):
        return VehicleDetection(box, confidence, 2, "car")

    def test_matches_vehicle_to_only_its_best_space(self):
        matches = match_detections_to_spaces(
            [self.detection((20, 20, 85, 85))],
            self.spaces,
            self.frame_shape,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].space_id, "A1")
        self.assertGreater(matches[0].overlap, 0.9)

    def test_vehicle_outside_spaces_is_not_matched(self):
        matches = match_detections_to_spaces(
            [self.detection((10, 110, 80, 180))],
            self.spaces,
            self.frame_shape,
        )

        self.assertEqual(matches, [])

    def test_giant_false_box_is_rejected(self):
        detection = self.detection((0, 0, 200, 200), confidence=0.2)

        self.assertFalse(
            is_plausible_vehicle_box(detection, self.frame_shape)
        )
        self.assertEqual(
            match_detections_to_spaces(
                [detection],
                self.spaces,
                self.frame_shape,
            ),
            [],
        )

    def test_duplicate_boxes_keep_one_match_per_space(self):
        matches = match_detections_to_spaces(
            [
                self.detection((20, 20, 85, 85), 0.6),
                self.detection((18, 18, 88, 88), 0.9),
            ],
            self.spaces,
            self.frame_shape,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].space_id, "A1")


class OccupancyTrackerTests(unittest.TestCase):
    def test_intermittent_hits_can_mark_occupied(self):
        tracker = OccupancyTracker(
            ["A1"],
            window_size=5,
            occupied_min_hits=3,
            available_after_misses=3,
        )

        tracker.update({"A1"})
        tracker.update(set())
        tracker.update({"A1"})
        status = tracker.update({"A1"})

        self.assertEqual(status["A1"], "OCCUPIED")

    def test_consecutive_misses_mark_available(self):
        tracker = OccupancyTracker(
            ["A1"],
            window_size=3,
            occupied_min_hits=1,
            available_after_misses=3,
        )
        tracker.update({"A1"})

        tracker.update(set())
        tracker.update(set())
        status = tracker.update(set())

        self.assertEqual(status["A1"], "AVAILABLE")


if __name__ == "__main__":
    unittest.main()
