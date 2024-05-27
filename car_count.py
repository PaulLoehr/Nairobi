from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

# Load the YOLOv8 model
model = YOLO('models/yolov8l.pt')

# Open the video file
video_path = "input_videos/video2_short.mp4"
cap = cv2.VideoCapture(video_path)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

crop_h = 450
video_writer = cv2.VideoWriter("output_videos/count_crop.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h - crop_h))

region_points = [(0, 450), (w/1.9, 450), (w/1.9, 470), (0, 470)]
classes_to_count = [2, 3, 5, 7, ]  # car, motorcycle, bus, truck

counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = frame[crop_h:, :]

    tracks = model.track(frame, persist=True, show=False,
                         classes=classes_to_count)

    annotated_frame = counter.start_counting(frame, tracks)
    video_writer.write(annotated_frame)


cap.release()
video_writer.release()
cv2.destroyAllWindows()