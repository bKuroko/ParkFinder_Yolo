import cv2
import os

video_path = "videos/parking.mp4"
output_folder = "dataset_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_number = 0
saved_number = 0

# Save 1 frame every 15 frames
frame_interval = 15

while True:
    success, frame = cap.read()

    if not success:
        break

    if frame_number % frame_interval == 0:
        filename = os.path.join(
            output_folder,
            f"frame_{saved_number:04d}.jpg"
        )

        cv2.imwrite(filename, frame)
        saved_number += 1

    frame_number += 1

cap.release()

print(f"Done. Saved {saved_number} frames.")