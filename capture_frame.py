import cv2
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

camera_url = "rtsp://192.168.1.39/live/ch00_0"

print("Connecting to camera...")

cap = cv2.VideoCapture(
    camera_url,
    cv2.CAP_FFMPEG
)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

frame = None

# Try several times
for attempt in range(30):

    success, temp_frame = cap.read()

    if success and temp_frame is not None:
        frame = temp_frame
        print("Frame captured!")
        break

    print(f"Waiting for camera... {attempt + 1}")
    time.sleep(0.2)

cap.release()

if frame is None:
    print("ERROR: Could not capture a frame.")
    exit()

cv2.imwrite(
    "images/parking_reference.jpg",
    frame
)

print(
    "Saved: images/parking_reference.jpg"
)

cv2.imshow(
    "Captured CCTV Frame",
    frame
)

cv2.waitKey(0)
cv2.destroyAllWindows()