import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import object_counter

def main():
    process_video('input_videos/video_pedestrians.mp4')

def process_video(path):
    cap = cv2.VideoCapture(path)

    # Video-Informationen erhalten
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Parameter für das Abschneiden definieren
    #top_crop = 450
    #bottom_crop = 400
    #left_crop = 500
    #right_crop = 400
    # Neue Breite und Höhe nach dem Beschneiden berechnen
    #new_width = frame_width - left_crop - right_crop
    #new_height = frame_height - top_crop - bottom_crop

    # Video-Writer initialisieren
    video_writer = cv2.VideoWriter("output_videos/main2.mp4",
                                   cv2.VideoWriter_fourcc(*'mp4v')
                                   ,fps,
                                   (frame_width, frame_height)) # neue größe könnte hier eingetragen werden
    
    vehicle_classes_to_count = [2, 3, 5, 7]  # 2car, 3motorcycle, 5bus, 7truck
    left_region_points = [(0, 800), (int(frame_width/1.9), 800), (int(frame_width/1.9), 900), (0, 900)]
    right_region_points = [(int(frame_width/1.7), 900), (frame_width, 900), (frame_width, 1000), (int(frame_width/1.7), 1000)]

    pedestrian_classes_to_count = [0] # 0 person
    pedestrian_region_points = [(int(frame_width/1.7), 550), (int(frame_width - 650), 550), (int(frame_width) - 200, 850), (int(frame_width/1.7), 850)]
    pedestrian_region_points2 = [(int(frame_width/2.6), 550), (int(frame_width/2), 550), (int(frame_width/2), 850),(int(frame_width/6), 850)]


    vehicle_tracker = Tracker('models/yolov8n.pt', vehicle_classes_to_count)
    left_counter = Counter(left_region_points, vehicle_tracker)
    right_counter = Counter(right_region_points, vehicle_tracker)

    pedestrian_tracker = Tracker('models/yolov8x.pt', pedestrian_classes_to_count)
    pedestrian_counter = Counter(pedestrian_region_points, pedestrian_tracker)
    pedestrian_counter2 = Counter(pedestrian_region_points2, pedestrian_tracker)


    frame_count = 0
    while cap.isOpened(): 
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 3 == 0:
            # neue größe könnte hier eingetragen werden
            #frame = frame[top_crop:-bottom_crop, left_crop:-right_crop]
            #frame = np.ascontiguousarray(frame)
            vehicle_tracks = vehicle_tracker.get_tracks(frame)
            pedestrian_tracks = pedestrian_tracker.get_tracks(frame)

            frame = left_counter.count(frame, vehicle_tracks)
            frame = right_counter.count(frame, vehicle_tracks)
            frame = pedestrian_counter.count(frame, pedestrian_tracks)
            frame = pedestrian_counter2.count(frame, pedestrian_tracks)


            frame = left_counter.draw(frame)
            frame = right_counter.draw(frame)
            frame = pedestrian_counter.draw(frame)
            frame = pedestrian_counter2.draw(frame)

            cv2.imshow('Video', frame)
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

class Tracker:
    def __init__(self, model, classes):
        self.model = YOLO(model)
        self.classes = classes

    def get_tracks(self, frame):
        tracks = self.model.track(frame, 
                                  persist=True, 
                                  show=False,
                                  classes=self.classes)
        return tracks


class Counter:
    def __init__(self, region_points, tracker):
        self.tracker = tracker
        self.region_points = region_points
        self.counter = object_counter.ObjectCounter()
        self.counter.set_args(view_img=False,
                            view_in_counts=False,
                            view_out_counts=False,
                            reg_pts=self.region_points,
                            draw_tracks=True,
                            classes_names=self.tracker.model.names)
        
    def count(self, frame, tracks):
        frame = self.counter.start_counting(frame, tracks)
        return frame
    
    def draw(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2
        counts = self.counter.in_counts
        text = f'Counts: {counts}'
        position = (self.region_points[0][0], self.region_points[0][1]+30)

        frame = cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        frame = cv2.polylines(frame, [np.array(self.region_points)], isClosed=True, color=(0, 255, 0), thickness=2)

        return frame

if __name__ == "__main__":
    main()