"""Run the same parking detector as main.py against the live CCTV stream."""

import os
from types import SimpleNamespace


# Prefer reliable RTSP/TCP transport. This must be set before importing cv2
# through main.py.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from main import DEFAULT_CONFIDENCE, DEFAULT_IMAGE_SIZE, run  # noqa: E402


CAMERA_URL = "rtsp://192.168.1.39/live/ch00_0"


if __name__ == "__main__":
    run(
        SimpleNamespace(
            source=CAMERA_URL,
            model="yolov8s.pt",
            confidence=DEFAULT_CONFIDENCE,
            imgsz=DEFAULT_IMAGE_SIZE,
            headless=False,
            max_frames=0,
            output=None,
        )
    )
