# This script aims to achieve class specific multiple object tracking 

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from deep_sort.deep_sort import DeepSort
# from utils import COLORS_10, draw_bboxes, xyxy_to_xywh, draw_bboxes_trajectory
from tracker_utils import *

# demo video files path
run3_path = "C:/Users/AI/Desktop/videos/run_3.mp4"
run5_path = "C:/Users/AI/Desktop/videos/run_5.mp4"
sports5_path = "C:/Users/AI/Desktop/videos/sports_5.mp4"



# check for gpu
use_cuda = torch.cuda.is_available()

#### LOAD OBJECT DETECTOR AND TRACKER #####

# Object detector
yolo_v8 = YOLO("yolov8n.pt")

# path of the person re-ID model
deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'

# taking class-specific deepsort algorithm  
deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)
car_deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)
person_deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)
dog_deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)

# getting from ip camera
rtsp_cam = 'rtsp://admin:admin123@192.168.1.108/live'

# defining video path
# video_path = sports5_path
video_path = run5_path
# video_path = rtsp_cam

# defining custom height and width of the frame (here, 1080p resolution)
height = 1080
width = 1920
# height = 600
# width = 800
vid_resolutiuon = (1920, 1080)

# Initialize video capture
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise IOError("Error opening video file!")

# getting the frame per second value
fps = video.get(cv2.CAP_PROP_FPS)

# defining root path to save the resulted video, cropped images and trajetory
root_path =  os.path.splitext(os.path.basename(video_path))[0]
root_path = root_path + '_v2'
output_video_path = root_path + '/'+ 'output.mp4'
person_trajectory_file_path = os.path.join(root_path, 'person_trajectories.json')
dog_trajectory_file_path = os.path.join(root_path, 'dog_trajectories.json')

# creating the directory
os.makedirs(root_path, exist_ok=True)

# Create a VideoWriter object to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))



# Initialize an empty dictionary to store the trajectory of tracked objects
person_trajectories = {}
dog_trajectories = {}
num_frame = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("No more frames or failed to grab a frame!")
        break

    num_frame += 1
    # Resize the frame
    # resized_frame = cv2.resize(frame, (width, height))
    resized_frame = cv2.resize(frame, vid_resolutiuon)
    # storing a copyt of the current frame 
    ori_im = resized_frame.copy()
    
    # defining the list of the objectes that we want to track
    obj_to_track = ['car', 'person', 'dog']

    # getting the detection results 
    # detection_results = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=obj_to_track)
    person_detection = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=['person'])
    dog_detection = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=['dog'])
    car_detection = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=['car'])
   

    # store the labels of the first frame 
    if num_frame == 1:
        person_initial_labels = person_detection['track_labels'].copy()
        person_initial_labels_idx = [k+1 for k in range(len(person_initial_labels))]

        dog_initial_labels = dog_detection['track_labels'].copy()
        dog_initial_labels_idx = [k+1 for k in range(len(dog_initial_labels))]

        car_initial_labels = car_detection['track_labels'].copy()
        car_initial_labels_idx = [k+1 for k in range(len(car_initial_labels))]

    # apply class-specific deepsort tracking 
    # person
    # person_tracked_frame, person_bbox_xyxy, person_identities = track_object(object_name='Person', detections= person_detection, tracker= person_deepsort, 
    #                                     initial_labels_idx= person_initial_labels_idx, frame= resized_frame,
    #                                     num_frame= num_frame, annot_frame= ori_im, save_img= True, path= root_path)

    person_tracked_frame, person_bbox_xyxy, person_identities = track_object(object_name='Person', detections= person_detection, tracker= person_deepsort, 
                                        initial_labels_idx= person_initial_labels_idx, frame= resized_frame,
                                        num_frame= num_frame, annot_frame= ori_im)
    
    # save the tracked object trajectories
    # draw_object_trajectory(object_trajectories= person_trajectories, tracked_bbox_xyxy= person_bbox_xyxy, 
    #                        tracked_identities= person_identities, trajectory_file_path= person_trajectory_file_path)
    
    # dog
    # dog_tracked_frame, dog_bbox_xyxy, dog_identities = track_object(object_name='Dog', detections= dog_detection, tracker= dog_deepsort, 
    #                                     initial_labels_idx= dog_initial_labels_idx, frame= resized_frame,
    #                                     num_frame= num_frame, annot_frame= ori_im, save_img= True, path= root_path)
    
    dog_tracked_frame, dog_bbox_xyxy, dog_identities = track_object(object_name='Dog', detections= dog_detection, tracker= dog_deepsort, 
                                        initial_labels_idx= dog_initial_labels_idx, frame= resized_frame,
                                        num_frame= num_frame, annot_frame= ori_im)
    # save the tracked trajectories
    # draw_object_trajectory(object_trajectories= dog_trajectories, tracked_bbox_xyxy= dog_bbox_xyxy, 
    #                        tracked_identities= dog_identities, trajectory_file_path= dog_trajectory_file_path)    

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
