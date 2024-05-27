from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

# Load the YOLOv8 model
model = YOLO('models/best.pt')

# Open the video file
video_path = "input_videos/video_pedestrians.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
frame_start = 500
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_count > frame_start:

        model.track(frame, persist=True, show=True, conf=0.01)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()