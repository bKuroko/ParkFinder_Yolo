import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from parking_logic import (
    OccupancyTracker,
    VehicleDetection,
    is_plausible_vehicle_box,
)


PROJECT_DIR = Path(__file__).resolve().parent

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

DEFAULT_CONFIDENCE = 0.20
DEFAULT_IMAGE_SIZE = 640
NMS_IOU_THRESHOLD = 0.50

# Occupancy smoothing
OCCUPANCY_WINDOW = 12
OCCUPIED_MIN_HITS = 3
AVAILABLE_AFTER_MISSES = 30

# Colors
AVAILABLE_COLOR = (0, 210, 0)
OCCUPIED_COLOR = (0, 0, 255)
UNMATCHED_COLOR = (255, 120, 0)
CENTER_COLOR = (0, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ParkFinder parking slot detection."
    )

    parser.add_argument(
        "--source",
        default="videos/parking.mp4",
        help="Video file, camera index, or stream URL.",
    )

    parser.add_argument(
        "--model",
        default="runs/detect/train/weights/best.pt",
        help="Custom trained YOLO model.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Minimum YOLO confidence.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="YOLO image size.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a video window.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. 0 means full video.",
    )

    parser.add_argument(
        "--output",
        help="Optional annotated output video.",
    )

    return parser.parse_args()


def resolve_project_path(value):
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


def resolve_source(value):
    if str(value).isdigit():
        return int(value)

    local_path = resolve_project_path(value)

    if local_path.exists():
        return str(local_path)

    return value


def load_parking_spaces():
    spaces_path = (
        PROJECT_DIR
        / "data"
        / "parking_spaces.json"
    )

    if not spaces_path.exists():
        raise FileNotFoundError(
            f"Parking spaces file not found: {spaces_path}"
        )

    with spaces_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        spaces = json.load(file)

    if not spaces:
        raise ValueError(
            "No parking spaces found."
        )

    return spaces


def get_car_class_ids(model):
    """
    Find the class ID named 'car'
    from the custom model.
    """

    car_ids = []

    for class_id, class_name in model.names.items():
        if class_name.lower() == "car":
            car_ids.append(class_id)

    print("Model classes:", model.names)
    print("Car class IDs:", car_ids)

    if not car_ids:
        raise RuntimeError(
            "This model does not contain a 'car' class."
        )

    return car_ids


def yolo_detections(result, model):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        class_id = int(box.cls[0])

        detection = VehicleDetection(
            box=(x1, y1, x2, y2),
            confidence=float(box.conf[0]),
            class_id=class_id,
            class_name=model.names[class_id],
        )

        detections.append(detection)

    return detections


def get_vehicle_center(detection):
    """
    Return center of car bounding box.
    """

    x1, y1, x2, y2 = detection.box

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return center_x, center_y


def find_space_for_detection(
    detection,
    parking_spaces
):
    """
    A parking slot is matched only when
    the CENTER of the detected car is
    inside the parking polygon.
    """

    center = get_vehicle_center(detection)

    matched_spaces = []

    for space_id, points in parking_spaces.items():

        polygon = np.asarray(
            points,
            dtype=np.int32
        )

        result = cv2.pointPolygonTest(
            polygon,
            center,
            False
        )

        # >= 0 means inside or on the border
        if result >= 0:
            matched_spaces.append(
                (space_id, polygon)
            )

    if not matched_spaces:
        return None

    if len(matched_spaces) == 1:
        return matched_spaces[0][0]

    # If polygons overlap,
    # use the closest polygon center.
    car_x, car_y = center

    best_space = None
    best_distance = float("inf")

    for space_id, polygon in matched_spaces:

        moments = cv2.moments(polygon)

        if moments["m00"] != 0:
            slot_x = int(
                moments["m10"] / moments["m00"]
            )

            slot_y = int(
                moments["m01"] / moments["m00"]
            )

        else:
            x, y, w, h = cv2.boundingRect(
                polygon
            )

            slot_x = x + w // 2
            slot_y = y + h // 2

        distance = (
            (car_x - slot_x) ** 2
            +
            (car_y - slot_y) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_space = space_id

    return best_space


def assign_cars_to_spaces(
    detections,
    parking_spaces
):
    assignments = []
    occupied_spaces = set()

    for detection in detections:

        center = get_vehicle_center(
            detection
        )

        space_id = find_space_for_detection(
            detection,
            parking_spaces
        )

        if space_id is not None:
            occupied_spaces.add(space_id)

        assignments.append(
            (
                detection,
                space_id,
                center
            )
        )

    return assignments, occupied_spaces


def draw_label(
    frame,
    text,
    origin,
    color,
    scale=0.55
):
    x, y = origin

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    x = max(
        0,
        min(
            x,
            frame.shape[1]
            - text_width
            - 8
        )
    )

    text_y = max(
        text_height + 6,
        y
    )

    cv2.rectangle(
        frame,
        (
            x,
            text_y
            - text_height
            - 6
        ),
        (
            x
            + text_width
            + 6,
            text_y
            + baseline
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x + 3,
            text_y - 3
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_parking_spaces(
    frame,
    parking_spaces,
    parking_status
):
    tint = frame.copy()

    for space_id, points in parking_spaces.items():

        polygon = np.asarray(
            points,
            dtype=np.int32
        )

        if (
            parking_status[space_id]
            == "OCCUPIED"
        ):
            color = OCCUPIED_COLOR

        else:
            color = AVAILABLE_COLOR

        cv2.fillPoly(
            tint,
            [polygon],
            color
        )

    cv2.addWeighted(
        tint,
        0.10,
        frame,
        0.90,
        0,
        frame
    )

    for space_id, points in parking_spaces.items():

        polygon = np.asarray(
            points,
            dtype=np.int32
        )

        status = parking_status[
            space_id
        ]

        if status == "OCCUPIED":
            color = OCCUPIED_COLOR
        else:
            color = AVAILABLE_COLOR

        cv2.polylines(
            frame,
            [polygon],
            True,
            color,
            3,
            cv2.LINE_AA,
        )

        x, y = points[0]

        draw_label(
            frame,
            f"{space_id} {status}",
            (
                x,
                max(24, y - 8)
            ),
            color,
        )


def draw_vehicle_boxes(
    frame,
    assignments,
    parking_status
):
    for (
        detection,
        space_id,
        center
    ) in assignments:

        x1, y1, x2, y2 = (
            detection.box
        )

        if space_id is not None:

            if (
                parking_status[space_id]
                == "OCCUPIED"
            ):
                color = OCCUPIED_COLOR

            else:
                color = UNMATCHED_COLOR

            label = (
                f"{space_id} "
                f"car "
                f"{detection.confidence:.2f}"
            )

            thickness = 3

        else:
            color = UNMATCHED_COLOR

            label = (
                f"car "
                f"{detection.confidence:.2f}"
            )

            thickness = 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        # Yellow center dot
        cv2.circle(
            frame,
            center,
            7,
            CENTER_COLOR,
            -1,
        )

        draw_label(
            frame,
            label,
            (
                x1,
                max(22, y1 - 5)
            ),
            color,
        )


def draw_status_panel(
    frame,
    parking_status,
    processing_fps
):
    available_count = sum(
        status == "AVAILABLE"
        for status
        in parking_status.values()
    )

    total = len(parking_status)

    occupied_count = (
        total - available_count
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (370, 165),
        (0, 0, 0),
        -1,
    )

    rows = [
        (
            "ParkFinder",
            (255, 255, 255),
            0.85
        ),
        (
            f"Available: {available_count}",
            AVAILABLE_COLOR,
            0.65
        ),
        (
            f"Occupied: {occupied_count}",
            OCCUPIED_COLOR,
            0.65
        ),
        (
            f"Total: {total}",
            (255, 255, 255),
            0.65
        ),
        (
            f"Custom Car YOLO | "
            f"{processing_fps:.1f} FPS",
            (200, 200, 200),
            0.50
        ),
    ]

    y_positions = [
        40,
        70,
        98,
        126,
        153
    ]

    for (
        text,
        color,
        scale
    ), y in zip(
        rows,
        y_positions
    ):

        cv2.putText(
            frame,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )


def open_output_writer(
    output_path,
    source_fps
):
    if not output_path:
        return None

    output_path = resolve_project_path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(output_path),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        (
            source_fps
            if source_fps > 0
            else 30.0
        ),

        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    return writer


def run(args):

    parking_spaces = (
        load_parking_spaces()
    )

    print(
        f"Loaded "
        f"{len(parking_spaces)} "
        f"parking spaces."
    )

    tracker = OccupancyTracker(
        parking_spaces.keys(),
        window_size=OCCUPANCY_WINDOW,
        occupied_min_hits=OCCUPIED_MIN_HITS,
        available_after_misses=AVAILABLE_AFTER_MISSES,
    )

    model_path = resolve_project_path(
        args.model
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path}"
        )

    print(
        f"Loading model: "
        f"{model_path}"
    )

    model = YOLO(
        str(model_path)
    )

    car_class_ids = (
        get_car_class_ids(
            model
        )
    )

    source = resolve_source(
        args.source
    )

    cap = cv2.VideoCapture(
        source
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: "
            f"{args.source}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    writer = open_output_writer(
        args.output,
        source_fps
    )

    frame_count = 0

    run_started = (
        time.perf_counter()
    )

    print(
        "ParkFinder started with custom car model."
    )

    print(
        "Occupancy rule: "
        "car CENTER must be inside parking ROI."
    )

    print(
        f"Confidence: "
        f"{args.confidence}"
    )

    if not args.headless:
        print(
            "Press Q to stop."
        )

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if (
                not success
                or frame is None
            ):
                print(
                    "Video finished."
                )
                break

            frame = cv2.resize(
                frame,
                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                )
            )

            result = model.predict(
                frame,
                classes=car_class_ids,
                conf=args.confidence,
                iou=NMS_IOU_THRESHOLD,
                imgsz=args.imgsz,
                agnostic_nms=True,
                max_det=50,
                verbose=False,
            )[0]

            detections = (
                yolo_detections(
                    result,
                    model
                )
            )

            plausible_detections = [
                detection
                for detection
                in detections
                if is_plausible_vehicle_box(
                    detection,
                    frame.shape
                )
            ]

            (
                assignments,
                occupied_spaces
            ) = assign_cars_to_spaces(
                plausible_detections,
                parking_spaces
            )

            parking_status = (
                tracker.update(
                    occupied_spaces
                )
            )

            frame_count += 1

            elapsed = max(
                time.perf_counter()
                - run_started,
                0.001
            )

            processing_fps = (
                frame_count
                / elapsed
            )

            draw_parking_spaces(
                frame,
                parking_spaces,
                parking_status
            )

            draw_vehicle_boxes(
                frame,
                assignments,
                parking_status
            )

            draw_status_panel(
                frame,
                parking_status,
                processing_fps
            )

            if writer is not None:
                writer.write(
                    frame
                )

            if not args.headless:

                cv2.imshow(
                    "ParkFinder - Parking Slot Detection",
                    frame
                )

                if (
                    cv2.waitKey(1)
                    & 0xFF
                    == ord("q")
                ):
                    print(
                        "Stopped."
                    )
                    break

            if (
                args.max_frames
                and frame_count
                >= args.max_frames
            ):
                print(
                    f"Stopped after "
                    f"{frame_count} frames."
                )
                break

    finally:

        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    print(
        f"Processed "
        f"{frame_count} frames."
    )

    print(
        "Final parking status:"
    )

    print(
        tracker.status
    )

    return tracker.status


if __name__ == "__main__":
    run(parse_args())