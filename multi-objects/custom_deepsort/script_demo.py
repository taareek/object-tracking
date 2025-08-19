import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from deep_sort.deep_sort import DeepSort
from utils import COLORS_10, draw_bboxes, xyxy_to_xywh, draw_bboxes_trajectory


# Demo video paths
video_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/run_5.mp4"

# Check for GPU
use_cuda = torch.cuda.is_available()

# Load YOLOv8 model and DeepSort tracker
yolo_v8 = YOLO("yolov8n.pt")

# Path for the DeepSort checkpoint
deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'
deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)

# Frame size settings
height, width = 480, 720

# Initialize video capture
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise IOError("Error opening video file!")

# Initialize an empty dictionary to store the trajectory of tracked objects
trajectories = {}
track_labels_dict = {}  # A dictionary to store labels associated with track IDs
num_frame = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("No more frames or failed to grab a frame!")
        break

    num_frame += 1
    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height))
    ori_im = resized_frame.copy()

    # Run YOLOv8 inference
    results = yolo_v8(resized_frame)

    # Initialize lists for tracking detections
    all_bboxes = []       # Stores detections in [xc, yc, w, h] format
    all_confidences = []  # Confidence scores
    track_labels = []     # Track labels for the detections

    # Process each result from YOLOv8
    for result in results:
        if result.boxes is not None and len(result.boxes.data):
            detections = result.boxes.data.cpu().numpy()
            boxes = detections[:, :4]
            confidences = detections[:, 4]
            class_ids = detections[:, 5]
            
            for bbox, conf, cls in zip(boxes, confidences, class_ids):
                label = result.names[int(cls)]
                if label in ["car", "person", "horse", "dog"]:
                    # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                    bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                    all_bboxes.append(bbox_xcycwh)
                    all_confidences.append(conf)
                    track_labels.append(label)

    # Store initial labels for the first frame
    if num_frame == 1:
        initial_labels = track_labels.copy()

    # If there are detections, update DeepSort with the full batch
    if len(all_bboxes) > 0:
        all_bboxes = np.array(all_bboxes).reshape(-1, 4)  # Ensure shape is (N, 4)
        all_confidences = np.array(all_confidences)
        outputs = deepsort.update(all_bboxes, all_confidences, resized_frame)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]  # Bounding boxes: [x1, y1, x2, y2]
            identities = outputs[:, -1]  # IDs of the tracked objects

            # Update trajectories and associate labels with tracking IDs
            for i, box in enumerate(bbox_xyxy):
                track_id = int(identities[i])
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if track_id not in trajectories:
                    trajectories[track_id] = []

                if track_id not in track_labels_dict:
                    # Assign the label associated with this track ID
                    track_labels_dict[track_id] = track_labels[i]

                # Append the new center point
                trajectories[track_id].append((center_x, center_y))

            # Draw bounding boxes, trajectories, and labels
            ori_im = draw_bboxes_trajectory(resized_frame, bbox_xyxy, identities, 
                                            trajectories=trajectories, 
                                             # Pass label dictionary
                                            offset=(0, 0))

    # Show the annotated frame
    cv2.imshow('video', ori_im)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

# Cleanup
video.release()
cv2.destroyAllWindows()

# Summary stats
print(f"Total frames processed: {num_frame}")
print(f"Tracked Label mapping: {track_labels_dict}")
