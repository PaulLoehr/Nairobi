from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import numpy as np


# Load the YOLOv8 model
model = YOLO('models/yolov8x.pt')

# Open the video file
video_path = "input_videos/video2_short.mp4"
cap = cv2.VideoCapture(video_path)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

crop_h = 450
video_writer = cv2.VideoWriter("output_videos/count_both_directions_framerate.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h - crop_h))

left_region_points = [(0, 350), (int(w/1.9), 350), (int(w/1.9), 450), (0, 450)]
right_region_points = [(int(w/1.7), 450), (w, 450), (w, 550), (int(w/1.7), 550)]

classes_to_count = [2, 3, 5, 7]  # 2car, 3motorcycle, 5bus, 7truck

left_counter = object_counter.ObjectCounter()
left_counter.set_args(view_img=False,
                      view_in_counts=False,
                      view_out_counts=False,
                      reg_pts=left_region_points,
                      classes_names=model.names)

rigth_counter = object_counter.ObjectCounter()
rigth_counter.set_args(view_img=False,
                       view_in_counts=False,
                       view_out_counts=False,
                       reg_pts=right_region_points,
                       classes_names=model.names,
                       draw_tracks=True)

frame_count = 0
while cap.isOpened():
    
    success, frame = cap.read()
    if not success:
        break


    if frame_count % 3 == 0:

        frame = frame[crop_h:, :]

        tracks = model.track(frame, persist=True, show=False,
                            classes=classes_to_count)

        frame = left_counter.start_counting(frame, tracks)
        frame = rigth_counter.start_counting(frame, tracks)

        left_counts = left_counter.in_counts
        right_counts = rigth_counter.in_counts

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)
        left_text = f'Left Counts: {left_counts}'
        left_text_size = cv2.getTextSize(left_text, font, font_scale, font_thickness)[0]

        image_width = frame.shape[1]
        right_text = f'Right Counts: {right_counts}'
        right_text_size = cv2.getTextSize(right_text, font, font_scale, font_thickness)[0]
        right_text_position = (image_width - right_text_size[0] - 10, 30)
    
        cv2.putText(frame, left_text, (10, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, right_text, right_text_position, font, font_scale, font_color, font_thickness)
        cv2.polylines(frame, [np.array(left_region_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [np.array(right_region_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, str(frame_count), (int(w/2), 30), font, font_scale, font_color, font_thickness)
        cv2.imshow('Result', frame)

        video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()