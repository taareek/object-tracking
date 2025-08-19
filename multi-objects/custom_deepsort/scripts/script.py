import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from deep_sort.deep_sort import DeepSort
from utils import COLORS_10, draw_bboxes, xyxy_to_xywh, draw_bboxes_trajectory
from tracker_utils import *

# demo video files path
run5_path = "C:/Users/AI/Desktop/videos/run_5.mp4"
sports5_path = "C:/Users/AI/Desktop/videos/sports_5.mp4"


# check for gpu
use_cuda = torch.cuda.is_available()

#### LOAD OBJECT DETECTOR AND TRACKER #####

# Object detector
yolo_v8 = YOLO("yolov8n.pt")

# defining video path
video_path = sports5_path
# video_path = run5_path

# defining custom height and width of the frame 
height = 1080
width = 1920
# height = 480
# width = 720


# Initialize video capture
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise IOError("Error opening video file!")

# getting the frame per second value
fps = video.get(cv2.CAP_PROP_FPS)

# defining root path to save the resulted video, cropped images and trajetory
root_path =  os.path.splitext(os.path.basename(video_path))[0]
output_video_path = root_path + '/'+ 'output.mp4'
trajectory_file_path = os.path.join(root_path, 'trajectories.json')

# creating the directory
os.makedirs(root_path, exist_ok=True)

# Create a VideoWriter object to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# Initialize an empty dictionary to store the trajectory of tracked objects
trajectories = {}
num_frame = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("No more frames or failed to grab a frame!")
        break

    num_frame += 1
    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height))
    # storing a copyt of the current frame 
    ori_im = resized_frame.copy()
    # defining the list of the objectes that we want to track
    obj_to_track = ['car', 'person', 'dog']

    # defining lists to store all bboxes, confidences 
    # all_bboxes = []
    # all_confidences = []

    # getting the detection results 
    detection_results = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=obj_to_track)
    # if detection_results['all_bboxes'] is not None:
    #     for i in range(len(detection_results['all_bboxes'])):
    #         all_bboxes.append(detection_results['all_bboxes'][i])
    #         all_confidences.append(detection_results['all_confidences'][i])

    # store the labels of the first frame 
    if num_frame == 1:
        initial_labels = detection_results['track_labels'].copy()
        initial_labels_idx = [k+1 for k in range(len(initial_labels))]
   
    # making a dictionary to map the tracked id's with their labels 
    track_label_maps = {}

    # If there are detections, update DeepSort with the full batch
    if len(detection_results['all_bboxes']) > 0:
        # apply deepsort tracking 
        # outputs = apply_deepsort(all_bboxes= all_bboxes, all_confidences= all_confidences, frame= resized_frame)
        outputs = apply_deepsort(all_bboxes= detection_results['all_bboxes'], all_confidences= detection_results['all_confidences'], frame= resized_frame)
        print(f"Frame number: {num_frame}")
        
        if len(outputs) > 0:
            # DeepSort outputs: first 4 columns are [x1, y1, x2, y2], last column is identity or tracked number.
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
           
            # filter the initial label and their index list 
            filtered_initial_labels = [label for idx, label in zip(initial_labels_idx, initial_labels) if idx in identities]
            filtered_initial_labels_idx = [idx for idx in initial_labels_idx if idx in identities]
            print(f"Filtered Index: {filtered_initial_labels_idx}")
            print(f"Filterd Index labels: {filtered_initial_labels}")
            # now we will crop the tracked objects and plot that 
            ann_img = resized_frame.copy()

            if len(filtered_initial_labels_idx) ==0:
                sp_obj_name = None

            for k, id in enumerate(identities):
                tracked_box = bbox_xyxy[k]

                if id in filtered_initial_labels_idx:
                    label_idx = filtered_initial_labels_idx.index(id)
                    sp_obj_name = filtered_initial_labels[label_idx]

                if id not in initial_labels_idx:
                    # print(f"Track not found for ID {id}!!")
                    initial_labels_idx.append(id)
                    # this last_labels logic is vague, it should be investigated
                    last_labels = detection_results['track_labels'][-1]
                    initial_labels.append(last_labels)

                # track id 
                tr_id = id
                # we can store these into a dictionary for further usage
                t_label = initial_labels[k]

                # plotting the tracked box 
                track_label_maps[id] = t_label

                # crop and save 
                if sp_obj_name is None:
                    pass
                else:
                    crop_n_save_tracked_object(tracked_box= tracked_box, annot_image= ann_img, 
                                           path= root_path, object_name= sp_obj_name, track_id= tr_id, 
                                           frame_no= num_frame)

            # **Update trajectories dictionary**
            for i, box in enumerate(bbox_xyxy):
                track_id = int(identities[i])
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # If this ID is not in trajectories, initialize it
                if track_id not in trajectories:
                    trajectories[track_id] = []

                # Append the new center point
                trajectories[track_id].append((center_x, center_y))


            ori_im = draw_bboxes(resized_frame, bbox_xyxy, identities)
            # ori_im = draw_bboxes_trajectory(resized_frame,  bbox_xyxy, identities, trajectories= trajectories, offset=(0,0))

    # Show the annotated frame
    cv2.imshow('video', ori_im)
    # Write the annotated frame to the output video
    out_video.write(ori_im)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

video.release()

cv2.destroyAllWindows()

print(f"Total frame: {num_frame}")

# Save the trajectory data to the JSON file
with open(trajectory_file_path, 'w') as json_file:
    json.dump(trajectories, json_file, indent=4)

print(f"Trajectory data saved to {trajectory_file_path}")