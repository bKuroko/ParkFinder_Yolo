import cv2

camera_url = "rtsp://192.168.1.39/live/ch00_0"

print("Connecting to camera...")

cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("ERROR: Could not open camera stream.")
    exit()

print("Camera connected successfully!")

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("ParkFinder - Live CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()