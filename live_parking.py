"""Run the same parking detector as main.py against the live CCTV stream."""

import os
from types import SimpleNamespace


# Prefer reliable RTSP/TCP transport. This must be set before importing cv2
# through main.py.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from main import DEFAULT_CONFIDENCE, DEFAULT_IMAGE_SIZE, run  # noqa: E402


CAMERA_URL = "http://172.20.78.67:8080/video"


if __name__ == "__main__":
    run(
        SimpleNamespace(
            source=CAMERA_URL,
            model="yolov8s.pt",
            confidence=DEFAULT_CONFIDENCE,
            imgsz=DEFAULT_IMAGE_SIZE,
            headless=False,
            # Sync detected occupancy to Appwrite. Set this to True for
            # camera-only operation without database writes.
            offline=False,
            max_frames=0,
            output=None,
        )
    )
