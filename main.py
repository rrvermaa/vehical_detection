from ultralytics import YOLO
import torch
import cv2 as cv
import numpy as np

model = YOLO('./model/best.pt')  
video_path = "nn.mp4"
video = cv.VideoCapture(video_path)

fps = video.get(cv.CAP_PROP_FPS)
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

output_path = 'new.mp4'

fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Change codec if needed
output_video = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
input_size = (820, 640)

while video.isOpened():

    success, frame = video.read()

    
    frame = cv.resize(frame, input_size)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    h, w, _ = frame.shape

    # frame = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
    if success:
        # Initialize variables for nearest object tracking
        nearest_distance = float('inf')
        nearest_box = None
        nearest_center = None
        nearest_trackID = None

        results_track = model.track(frame, persist=True )

        if (results_track[0].boxes.id is None):
            continue

        else:            
            bboxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
            tracking_ids = results_track[0].track_id.cpu().numpy().astype(int)  # Update this line
            class_ids = results_track[0].boxes.cls.cpu().numpy().astype(int)
        # ...

            centre_coord = (int(w / 2), int(h / 2))
            cv.circle(frame, centre_coord, 5, (0, 0, 255), cv.FILLED)
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                cls_id = class_ids[i]
                track_id = tracking_ids[i]
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv.circle(frame, (cx, cy), 3, (20, 255, 160), cv.FILLED)
                cv.line(frame, centre_coord, (cx, cy), (0, 125, 255), 1)

                distance = ((cx - w / 2)  **2 + (cy - h / 2)** 2) ** 0.5
                print(f'Distance of id no #{track_id} from centre is {distance}')

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_box = bbox
                    nearest_center = (cx, cy)
                    nearest_trackID = track_id

            print("\n", nearest_distance, nearest_trackID, "\n")
            if nearest_box is not None:
                x1, y1, x2, y2 = nearest_box
                # Normalize the coordinates
                norm_cx = (nearest_center[0] - frame_width / 2) / frame_width
                norm_cy = (nearest_center[1] - frame_height / 2) / frame_height

                norm_cx = norm_cx - 0.5
                norm_cy = norm_cy - 0.5

                print(f"(X,Y) ({norm_cx:.2f}, {norm_cy:.2f})")

                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(frame, f" X {norm_cx:.2f}, Y {norm_cy:.2f}",
                           (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv.imwrite('frame.png', frame)
            output_video.write(frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        break

video.release()
output_video.release()

cv.destroyAllWindows()