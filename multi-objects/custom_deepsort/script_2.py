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
cycle_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/single_cv/videos/byc.mp4"
video2_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/single_cv/videos/walk_cycle.mp4"
video3_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/single_cv/videos/walk_dog.mp4"
video5_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/single_cv/videos/demo.mp4"

run1_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/run_1.mp4"
run2_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/run_2.mp4"
run3_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/run_3.mp4"
run4_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/run_4.mp4"
run5_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/run_5.mp4"

sports1_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_1.mp4"
sports2_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_2.mp4"
sports3_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/single_cv/videos/football.mp4"
sports4_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_4.mp4"
sports5_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_5.mp4"
sports6_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_6.mp4"
sports7_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_7.mp4"
sports8_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/sports_8.mp4"


boxing_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/boxing_1.mp4"
car1_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/car_1.mp4"
car2_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/car_2.mp4"
car3_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/car_3.mp4"
car4_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/car_4.mp4"
horse_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/r_videos/horse_1.mp4"

scl_kids1_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids1.mp4"
scl_kids2_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids2.mp4"
scl_kids3_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids3.mp4"
scl_kids4_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids4.mp4"
scl_kids5_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids5.mp4"
scl_kids6_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids6.mp4"
scl_kids7_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/school_kids/scl_kids7.mp4"


# check for gpu
use_cuda = torch.cuda.is_available()

# Load YOLOv8 model and DeepSort tracker
yolo_v8 = YOLO("yolov8n.pt")

# path of the person re-ID model
deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'

# taking the deepsort algorithms 
deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)

# defining video path
video_path = car4_path 

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
root_path = root_path + '_v2'
output_video_path = root_path + '/'+ 'output.mp4'
car_trajectory_file_path = os.path.join(root_path, 'car_trajectories.json')

person_trajectory_file_path = os.path.join(root_path, 'person_trajectories.json')
dog_trajectory_file_path = os.path.join(root_path, 'dog_trajectories.json')
horse_trajectory_file_path = os.path.join(root_path, 'horse_trajectories.json')

# creating the directory
os.makedirs(root_path, exist_ok=True)

# Create a VideoWriter object to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# Initialize an empty dictionary to store the trajectory of tracked objects
car_trajectories = {}
person_trajectories = {}
dog_trajectories = {}
horse_trajectories = {}
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
    # car
    car_labels = []
    car_confidences = []
    car_boxes = []
    # person 
    person_boxes = []
    person_labels = []
    person_confidences=[]
    # dog 
    dog_boxes = []
    dog_labels = []
    dog_confidences = []
    # horse 
    horse_boxes = []
    horse_labels = []
    horse_confidences = []


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
            car_track_labels = []
            person_track_labels = []
            dog_track_labels = []
            horse_track_labels = []
            # Loop through detections and filter by label
            for bbox, conf, cls in zip(boxes, confidences, class_ids):
                label = result.names[int(cls)]
                ann_img = resized_frame.copy()
                if label in ['car', 'person', 'horse', 'dog']:
                    if label == "car": 
                        # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                        bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                        car_boxes.append(bbox_xcycwh)
                        car_confidences.append(conf)
                        car_track_labels.append(label)
                    if label == "person":
                        print(f"Person loading.........")
                        # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                        bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                        person_boxes.append(bbox_xcycwh)
                        person_confidences.append(conf)
                        person_track_labels.append(label)
                    if label == "horse":
                        # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                        bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                        horse_boxes.append(bbox_xcycwh)
                        horse_confidences.append(conf)
                        horse_track_labels.append(label)
                    if label == "dog":
                        print(f"Dog loading...........")
                        # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                        bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                        dog_boxes.append(bbox_xcycwh)
                        dog_confidences.append(conf)
                        dog_track_labels.append(label)
              

    # store the labels of the first frame 
    if num_frame == 1:
        # car
        initial_car_labels = car_track_labels.copy()
        initial_car_labels_idx = [k+1 for k in range(len(initial_car_labels))]
        # person
        initial_person_labels = person_track_labels.copy()
        initial_person_labels_idx = [k+1 for k in range(len(initial_person_labels))]
        # dog
        initial_dog_labels = dog_track_labels.copy()
        initial_dog_labels_idx = [k+1 for k in range(len(initial_dog_labels))]
        # horse
        initial_horse_labels = horse_track_labels.copy()
        initial_horse_labels_idx = [k+1 for k in range(len(initial_horse_labels))]

    # debug
    print(f"Intitial car labels: {initial_car_labels}")
    print(f"Initial car labels indexes: {initial_car_labels_idx}")

    print(f"Intitial person labels: {initial_person_labels}")
    print(f"Initial person labels indexes: {initial_person_labels_idx}")

    print(f"Intitial dog labels: {initial_dog_labels}")
    print(f"Initial dog labels indexes: {initial_dog_labels_idx}")

    print(f"Intitial horse labels: {initial_horse_labels}")
    print(f"Initial horse labels indexes: {initial_horse_labels_idx}")

    print(f"Total car boxes: {len(car_boxes)}")
    print(f"Total person boxes: {len(person_boxes)}")
    print(f"Total dog boxes: {len(dog_boxes)}")
    print(f"Total horse boxes: {len(horse_boxes)}")

    # making a dictionary to map the tracked id's with their labels 
    car_track_label_maps = {}
    person_track_label_maps = {}
    dog_track_label_maps = {}
    horse_track_label_maps = {}

    # print tracked labels 
    # print(f"Initial Labels: {initial_labels}")
    # print(f"Initial Labels index: {initial_labels_idx}")
    # print(f"Labels accross consecutive frames: {track_labels}")

    # If there are detections, update DeepSort with the full batch
    ########## CAR #############
    if len(car_boxes) > 0:
        # debug-> get insights about total detections 
        print(f"Frame number: {num_frame}")
        # print(f"Total boxes: {len(all_bboxes)}")
        # print(f"Total confidences: {len(all_confidences)}")
        # print(f"Total labels: {len(track_labels)}")
        car_boxes = np.array(car_boxes).reshape(-1, 4)  # Ensure shape is (N, 4)
        car_confidences = np.array(car_confidences)
        car_outputs = deepsort.update(car_boxes, car_confidences, resized_frame)
        # debug-> output 
        print(f"The output: {car_outputs}")
        if len(car_outputs) > 0:
            # DeepSort outputs: first 4 columns are [x1, y1, x2, y2], last column is identity or tracked number.
            bbox_xyxy = car_outputs[:, :4]
            car_identities = car_outputs[:, -1]
            print(f"Identities: {car_identities}")
            # filter the initial label and their index list 
            filtered_car_initial_labels = [label for idx, label in zip(initial_car_labels_idx, initial_car_labels) if idx in car_identities]
            filtered_car_initial_labels_idx = [idx for idx in initial_car_labels_idx if idx in car_identities]

            print(f"Filtered Index: {filtered_car_initial_labels_idx}")
            print(f"Filterd Index labels: {filtered_car_initial_labels}")
            # now we will crop the tracked objects and plot that 
            ann_img = resized_frame.copy()
            for k, id in enumerate(car_identities):
                # print(f"The ID: {id}")
                # tr_id = int(identities[k])
                t_box = bbox_xyxy[k]

                if id in filtered_car_initial_labels_idx:
                    label_idx = filtered_car_initial_labels_idx.index(id)
                    # print(f"Label index: {label_idx}")
                    obj_name = filtered_car_initial_labels[label_idx]

                if id not in initial_car_labels_idx:
                    # print(f"Track not found for ID {id}!!")
                    initial_car_labels_idx.append(id)
                    # print(f"Initial label indexes after: {initial_labels_idx}")
                    # this last_labels logic is vague, it should be investigated
                    last_labels = car_track_labels[-1]
                    initial_car_labels.append(last_labels)

                # track id 
                tr_id = id
                # we can store these into a dictionary for further usage
                t_label = initial_car_labels[k]
                car_track_label_maps[id] = t_label
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
                track_id = int(car_identities[i])
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # If this ID is not in trajectories, initialize it
                if track_id not in car_trajectories:
                    car_trajectories[track_id] = []

                # Append the new center point
                car_trajectories[track_id].append((center_x, center_y))


            ori_im = draw_bboxes(resized_frame, bbox_xyxy, car_identities)
            # ori_im = draw_bboxes_trajectory(resized_frame,  bbox_xyxy, identities, trajectories= trajectories, offset=(0,0))
    
    # ######### DOG ##########
    # If there are detections, update DeepSort with the full batch
    if len(dog_boxes) > 0:
        # debug-> get insights about total detections 
        print(f"Frame number: {num_frame}")
        # print(f"Total boxes: {len(all_bboxes)}")
        # print(f"Total confidences: {len(all_confidences)}")
        # print(f"Total labels: {len(track_labels)}")
        dog_boxes = np.array(dog_boxes).reshape(-1, 4)  # Ensure shape is (N, 4)
        dog_confidences = np.array(dog_confidences)
        dog_outputs = deepsort.update(dog_boxes, dog_confidences, resized_frame)
        # debug-> output 
        # print(f"The output: {outputs}")
        if len(dog_outputs) > 0:
            # DeepSort outputs: first 4 columns are [x1, y1, x2, y2], last column is identity or tracked number.
            bbox_xyxy = dog_outputs[:, :4]
            dog_identities = dog_outputs[:, -1]
            print(f"Dog Identities: {dog_identities}")
            # filter the initial label and their index list 
            filtered_dog_initial_labels = [label for idx, label in zip(initial_dog_labels_idx, initial_dog_labels) if idx in dog_identities]
            filtered_dog_initial_labels_idx = [idx for idx in initial_dog_labels_idx if idx in dog_identities]

            print(f"Filtered Dog Index: {filtered_dog_initial_labels_idx}")
            print(f"Filterd Dog Index labels: {filtered_dog_initial_labels}")
            # now we will crop the tracked objects and plot that 
            ann_img = resized_frame.copy()
            for k, id in enumerate(dog_identities):
                # print(f"The ID: {id}")
                # tr_id = int(identities[k])
                t_box = bbox_xyxy[k]

                if id in filtered_dog_initial_labels_idx:
                    label_idx = filtered_dog_initial_labels_idx.index(id)
                    # print(f"Label index: {label_idx}")
                    obj_name = filtered_dog_initial_labels[label_idx]

                if id not in initial_dog_labels_idx:
                    # print(f"Track not found for ID {id}!!")
                    initial_dog_labels_idx.append(id)
                    # print(f"Initial label indexes after: {initial_labels_idx}")
                    # this last_labels logic is vague, it should be investigated
                    last_labels = dog_track_labels[-1]
                    initial_dog_labels.append(last_labels)

                # track id 
                tr_id = id
                # we can store these into a dictionary for further usage
                t_label = initial_dog_labels[k]
                dog_track_label_maps[id] = t_label
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
                track_id = int(dog_identities[i])
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # If this ID is not in trajectories, initialize it
                if track_id not in dog_trajectories:
                    dog_trajectories[track_id] = []

                # Append the new center point
                dog_trajectories[track_id].append((center_x, center_y))


            ori_im = draw_bboxes(resized_frame, bbox_xyxy, dog_identities)
            # ori_im = draw_bboxes_trajectory(resized_frame,  bbox_xyxy, identities, trajectories= trajectories, offset=(0,0))

    # ######### PERSON ###########
    # If there are detections, update DeepSort with the full batch
    if len(person_boxes) > 0:
        # debug-> get insights about total detections 
        print(f"Frame number: {num_frame}")
        # print(f"Total boxes: {len(all_bboxes)}")
        # print(f"Total confidences: {len(all_confidences)}")
        # print(f"Total labels: {len(track_labels)}")
        person_boxes = np.array(person_boxes).reshape(-1, 4)  # Ensure shape is (N, 4)
        person_confidences = np.array(person_confidences)
        person_outputs = deepsort.update(person_boxes, person_confidences, resized_frame)
        # debug-> output 
        # print(f"The output: {outputs}")
        if len(person_outputs) > 0:
            # DeepSort outputs: first 4 columns are [x1, y1, x2, y2], last column is identity or tracked number.
            bbox_xyxy = person_outputs[:, :4]
            person_identities = person_outputs[:, -1]
            print(f"Identities: {person_identities}")
            # filter the initial label and their index list 
            filtered_person_initial_labels = [label for idx, label in zip(initial_person_labels_idx, initial_person_labels) if idx in person_identities]
            filtered_person_initial_labels_idx = [idx for idx in initial_person_labels_idx if idx in person_identities]

            print(f"Filtered Index: {filtered_person_initial_labels_idx}")
            print(f"Filterd Index labels: {filtered_person_initial_labels}")
            # now we will crop the tracked objects and plot that 
            ann_img = resized_frame.copy()
            for k, id in enumerate(person_identities):
                # print(f"The ID: {id}")
                # tr_id = int(identities[k])
                t_box = bbox_xyxy[k]

                if id in filtered_person_initial_labels_idx:
                    label_idx = filtered_person_initial_labels_idx.index(id)
                    print(f"Label index: {label_idx}")
                    # print(f"Total indexes: {len}")
                    obj_name = filtered_person_initial_labels[label_idx]


                if id not in initial_person_labels_idx:
                    # print(f"Track not found for ID {id}!!")
                    initial_person_labels_idx.append(id)
                    # print(f"Initial label indexes after: {initial_labels_idx}")
                    # this last_labels logic is vague, it should be investigated
                    last_labels = person_track_labels[-1]
                    initial_person_labels.append(last_labels)

                # track id 
                tr_id = id
                # we can store these into a dictionary for further usage
                t_label = initial_person_labels[k]
                person_track_label_maps[id] = t_label
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
                track_id = int(person_identities[i])
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # If this ID is not in trajectories, initialize it
                if track_id not in person_trajectories:
                    person_trajectories[track_id] = []

                # Append the new center point
                person_trajectories[track_id].append((center_x, center_y))


            ori_im = draw_bboxes(resized_frame, bbox_xyxy, person_identities)
            # ori_im = draw_bboxes_trajectory(resized_frame,  bbox_xyxy, identities, trajectories= trajectories, offset=(0,0))

    # ######### Horse #############
    # If there are detections, update DeepSort with the full batch
    if len(horse_boxes) > 0:
        # debug-> get insights about total detections 
        print(f"Frame number: {num_frame}")
        # print(f"Total boxes: {len(all_bboxes)}")
        # print(f"Total confidences: {len(all_confidences)}")
        # print(f"Total labels: {len(track_labels)}")
        horse_boxes = np.array(horse_boxes).reshape(-1, 4)  # Ensure shape is (N, 4)
        horse_confidences = np.array(horse_confidences)
        horse_outputs = deepsort.update(horse_boxes, horse_confidences, resized_frame)
        # debug-> output 
        # print(f"The output: {outputs}")
        if len(horse_outputs) > 0:
            # DeepSort outputs: first 4 columns are [x1, y1, x2, y2], last column is identity or tracked number.
            bbox_xyxy = horse_outputs[:, :4]
            horse_identities = horse_outputs[:, -1]
            print(f"Horse Identities: {horse_identities}")
            # filter the initial label and their index list 
            filtered_horse_initial_labels = [label for idx, label in zip(initial_horse_labels_idx, initial_horse_labels) if idx in horse_identities]
            filtered_horse_initial_labels_idx = [idx for idx in initial_horse_labels_idx if idx in horse_identities]

            print(f"Filtered Index: {filtered_horse_initial_labels_idx}")
            print(f"Filterd Index labels: {filtered_horse_initial_labels}")
            # now we will crop the tracked objects and plot that 
            ann_img = resized_frame.copy()
            for k, id in enumerate(horse_identities):
                # print(f"The ID: {id}")
                # tr_id = int(identities[k])
                t_box = bbox_xyxy[k]

                if id in filtered_horse_initial_labels_idx:
                    label_idx = filtered_horse_initial_labels_idx.index(id)
                    # print(f"Label index: {label_idx}")
                    obj_name = filtered_horse_initial_labels[label_idx]

                if id not in initial_horse_labels_idx:
                    # print(f"Track not found for ID {id}!!")
                    initial_horse_labels_idx.append(id)
                    # print(f"Initial label indexes after: {initial_labels_idx}")
                    # this last_labels logic is vague, it should be investigated
                    last_labels = horse_track_labels[-1]
                    initial_horse_labels.append(last_labels)

                # track id 
                tr_id = id
                # we can store these into a dictionary for further usage
                t_label = initial_horse_labels[k]
                horse_track_label_maps[id] = t_label
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
                track_id = int(horse_identities[i])
                x1, y1, x2, y2 = map(int, box)  # Convert to int
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # If this ID is not in trajectories, initialize it
                if track_id not in horse_trajectories:
                    horse_trajectories[track_id] = []

                # Append the new center point
                horse_trajectories[track_id].append((center_x, center_y))


            ori_im = draw_bboxes(resized_frame, bbox_xyxy, horse_identities)
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
with open(car_trajectory_file_path, 'w') as json_file:
    json.dump(car_trajectories, json_file, indent=4)
print(f"Trajectory data saved to {car_trajectory_file_path}")

# person
with open(person_trajectory_file_path, 'w') as json_file:
    json.dump(person_trajectories, json_file, indent=4)
print(f"Trajectory data saved to {person_trajectory_file_path}")

# dog
with open(dog_trajectory_file_path, 'w') as json_file:
    json.dump(dog_trajectories, json_file, indent=4)
print(f"Trajectory data saved to {dog_trajectory_file_path}")

# horse
with open(horse_trajectory_file_path, 'w') as json_file:
    json.dump(horse_trajectories, json_file, indent=4)
print(f"Trajectory data saved to {horse_trajectory_file_path}")