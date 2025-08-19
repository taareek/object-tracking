import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from deep_sort.deep_sort import DeepSort
from utils import COLORS_10, draw_bboxes, xyxy_to_xywh, draw_bboxes_trajectory


# demo video files path
run5_path = "C:/Users/AI/Desktop/videos/run_5.mp4"


# check for gpu
use_cuda = torch.cuda.is_available()

# Load YOLOv8 model and DeepSort tracker
yolo_v8 = YOLO("yolov8n.pt")

# path of the person re-ID model
deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'

# taking the deepsort algorithms 
deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)

# defining video path
video_path = run5_path

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
    ori_im = resized_frame.copy()
    # ori_im_f = resized_frame.copy()

    # Run YOLOv8 inference
    results = yolo_v8(resized_frame)

    # Accumulate detections for this frame
    all_bboxes = []       # Will store detections in [xc, yc, w, h] format
    all_confidences = []  # Confidence scores
    all_labels = []

    # Process each result from YOLOv8
    for result in results:
        if result.boxes is not None and len(result.boxes.data):
            # Get detections as an array: [x1, y1, x2, y2, confidence, class_id]
            detections = result.boxes.data.cpu().numpy()
            boxes = detections[:, :4]
            confidences = detections[:, 4]
            class_ids = detections[:, 5]
            # here we are getting all the detections within this specfic frame
            # print(f"Boxes: {boxes}\n Confidences: {confidences}\n Class Id's: {class_ids}")
            track_labels = []
            # Loop through detections and filter by label
            for bbox, conf, cls in zip(boxes, confidences, class_ids):
                label = result.names[int(cls)]
                if label in ["car", "person", "horse", "dog"]:
                    # this detected bbox need to be stored based on the tracking ID
                    ann_img = resized_frame.copy()
                    # getting the bounding box coordinates and croping it from the original frame 
                    # x_min, y_min, x_max, y_max = map(int, bbox) 
                    # cropped_img = ann_img[y_min:y_max, x_min:x_max]
                    # usually the cropped image will be in poor quality,
                    # therefore, we need to apply an interpolation to enhance its quality 
                    # resized_cropped = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_CUBIC)
                    # plt the image 
                    # plt.imshow(cropped_img[:,:,::-1])
                    # plt.imshow(resized_cropped[:,:,::-1])
                    # plt.show()
                    # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                    # Ensure the result is a flat array of shape (4,)
                    bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                    all_bboxes.append(bbox_xcycwh)
                    all_confidences.append(conf)
                    track_labels.append(label)

    # store the labels of the first frame 
    if num_frame == 1:
        initial_labels = track_labels.copy()
        initial_labels_idx = [k+1 for k in range(len(initial_labels))]

    # making a dictionary to map the tracked id's with their labels 
    track_label_maps = {}

    # print tracked labels 
    print(f"Initial Labels: {initial_labels}")
    print(f"Initial Labels index: {initial_labels_idx}")
    # print(f"Labels accross consecutive frames: {track_labels}")

    # If there are detections, update DeepSort with the full batch
    if len(all_bboxes) > 0:
        # debug-> get insights about total detections 
        print(f"Frame number: {num_frame}")
        # print(f"Total boxes: {len(all_bboxes)}")
        # print(f"Total confidences: {len(all_confidences)}")
        # print(f"Total labels: {len(track_labels)}")
        all_bboxes = np.array(all_bboxes).reshape(-1, 4)  # Ensure shape is (N, 4)
        all_confidences = np.array(all_confidences)
        outputs = deepsort.update(all_bboxes, all_confidences, resized_frame)
        # debug-> output 
        # print(f"The output: {outputs}")
        if len(outputs) > 0:
            # DeepSort outputs: first 4 columns are [x1, y1, x2, y2], last column is identity or tracked number.
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            print(f"Identities: {identities}")
            # filter the initial label and their index list 
            filtered_initial_labels = [label for idx, label in zip(initial_labels_idx, initial_labels) if idx in identities]
            filtered_initial_labels_idx = [idx for idx in initial_labels_idx if idx in identities]

            print(f"Filtered Index: {filtered_initial_labels_idx}")
            print(f"Filterd Index labels: {filtered_initial_labels}")
            # now we will crop the tracked objects and plot that 
            ann_img = resized_frame.copy()
            for k, id in enumerate(identities):
                # print(f"The ID: {id}")
                # tr_id = int(identities[k])
                t_box = bbox_xyxy[k]

                if id in filtered_initial_labels_idx:
                    label_idx = filtered_initial_labels_idx.index(id)
                    print(f"Label index: {label_idx}")
                    obj_name = filtered_initial_labels[label_idx]
                    print(f"Object Name: {obj_name}")

                if id not in initial_labels_idx:
                    # print(f"Track not found for ID {id}!!")
                    initial_labels_idx.append(id)
                    # print(f"Initial label indexes after: {initial_labels_idx}")
                    # this last_labels logic is vague, it should be investigated
                    last_labels = track_labels[-1]
                    initial_labels.append(last_labels)

                # track id 
                tr_id = id
                # we can store these into a dictionary for further usage
                t_label = initial_labels[k]
                track_label_maps[id] = t_label
                # plotting the tracked box 
                x_min, y_min, x_max, y_max = map(int, t_box) 
                cropped_img = ann_img[y_min:y_max, x_min:x_max]
                c_h, c_w, _ = cropped_img.shape
                resized_cropped = cv2.resize(cropped_img, (c_w, c_h), interpolation=cv2.INTER_CUBIC)
                # plt.imshow(resized_cropped[:,:,::-1])
                # # plt.text(10, 20, "Tracked ID: "+ str(tr_id), fontsize=12, color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
                # plt.title("Tracked ID: "+str(tr_id) + "  Label: "+ t_label, fontsize=14, fontweight="bold", color="blue")
                # plt.show()
                # Get the label for the current track ID
                # track_label = track_labels[tr_id]

                # creating folder for objects 
                object_folder = os.path.join(root_path, obj_name)
                os.makedirs(object_folder, exist_ok=True)

                # creating specific track folder 
                track_folder = os.path.join(object_folder, str(tr_id))
                os.makedirs(track_folder, exist_ok=True)

                # now we will save the specific tracked object's images to their specific folder 
                img_name = f"frame_{num_frame:04d}.jpg"
                img_path = os.path.join(track_folder, img_name)
                cv2.imwrite(img_path, resized_cropped)
                print(f"Saved cropped image for Track ID {tr_id}, Object {obj_name} at {img_path}")

                # print(f"Fron new-> Track ID: {label_idx+1}, Label: {obj_name}, Bounding box: {t_box}")
                # print(f"Track ID: {id}, Label: {t_label}, Bounding box: {t_box}")
            # print(f"Tracked mapping: \n{track_label_maps}")

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

# print(f"Total identities found: {identities}")
# print(f"Trajectories: {trajectories}")
print(f"Total frame: {num_frame}")
# print(f"Total boxes: {len(all_bboxes)}")
# print(f"Total confidemnces: {len(all_confidences)}")
# print(f"Total labels: {len(all_labels)}")

# Save the trajectory data to the JSON file
with open(trajectory_file_path, 'w') as json_file:
    json.dump(trajectories, json_file, indent=4)

print(f"Trajectory data saved to {trajectory_file_path}")