from ultralytics import YOLO
import cv2
import os

# Prefer RTSP over TCP instead of UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

model = YOLO("yolov8n.pt")

camera_url = "rtsp://192.168.1.39/live/ch00_0"

cap = cv2.VideoCapture(
    camera_url,
    cv2.CAP_FFMPEG
)

# Reduce buffering/latency where supported
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Could not connect to CCTV.")
    exit()

print("CCTV connected!")
print("Starting YOLO detection...")

while True:

    success, frame = cap.read()

    # Don't immediately terminate because of one bad frame
    if not success or frame is None:
        print("Warning: dropped CCTV frame.")
        continue

    results = model(
        frame,
        classes=[2, 3, 5, 7],
        conf=0.35,
        verbose=False
    )

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Vehicle {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "ParkFinder - Live YOLO CCTV",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()