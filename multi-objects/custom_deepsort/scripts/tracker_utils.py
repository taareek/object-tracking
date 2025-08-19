import os
import cv2
import torch
import numpy as np
import json
from utils import xyxy_to_xywh, draw_bboxes
from deep_sort.deep_sort import DeepSort


# check for gpu
use_cuda = torch.cuda.is_available()

# path of the person re-ID model
deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'

# taking the deepsort algorithms 
deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)


# function to get object detection results 
def get_detection_results(detector, frame, obj_to_track):
    """
    This function will take `object detection` model and `frame` as input and return detection results
    
    Parameters
    ----------
    model: object detection model
        YOLO, Faster R-CNN, SSD etc
    frame: numpy array
        static image or frame of a video
    obj_to_track: list of strings 
        It takes a list of strings that specifies which object we want to track 

    Returns
    -------
    detection results
        A dictionary that contains all detected boxes, confidence scores and their labels 
    """
    # Run YOLOv8 inference
    results = detector(frame)

    # Accumulate detections for this frame
    all_bboxes = []       # Will store detections in [xc, yc, w, h] format
    all_confidences = []  # Confidence scores
    track_labels = []    # store the labels

    # Process each result from object detector
    for result in results:
        if result.boxes is not None and len(result.boxes.data):
            # Get detections as an array: [x1, y1, x2, y2, confidence, class_id]
            detections = result.boxes.data.cpu().numpy()
            boxes = detections[:, :4]
            confidences = detections[:, 4]
            class_ids = detections[:, 5]
            # Loop through detections and filter by label
            for bbox, conf, cls in zip(boxes, confidences, class_ids):
                label = result.names[int(cls)]
                if label in obj_to_track:
                    # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                    # As, DeepSORT takes input as [xc, yc, w, h] format
                    bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                    all_bboxes.append(bbox_xcycwh)
                    all_confidences.append(conf)
                    track_labels.append(label)
    
    # storing the detection results in a dictionary 
    det_results = {
        'all_bboxes': all_bboxes,
        'all_confidences': all_confidences,
        'track_labels': track_labels
    }
    return det_results

# function to apply deepsort 
def apply_deepsort(all_bboxes, all_confidences, frame):
    """
    This function will take all detected boxes and their confidence scores 
    and return the tracked outputs 

    Parameters
    ----------
    all_boxes: list 
        A list of numpy arrays that contains all the dected boxes 
    all_confidences: list
        A list of cofidence scores 
    frame: numpy array 
        Current frame that contains the objects 
    Returns
    -------
    tracked_output: list
        A list with 5 elemets where first four contains the bbox and last one track ID 
    """
    all_bboxes = np.array(all_bboxes).reshape(-1, 4)  # Ensure shape is (N, 4)
    all_confidences = np.array(all_confidences)
    outputs = deepsort.update(all_bboxes, all_confidences, frame)
    return outputs

# function to track object 
def track_object(object_name, detections, tracker, initial_labels_idx, frame, num_frame, annot_frame, save_img=False, path=None):
    """
    This function takes all the detections and return tracked outputs

    Parameters
    ---------
    object_name: list 
        A list with single string that takes the object to track
    detections: list
        List of specific objects obtained from the object detector
    tracker: object
        Object tracker instance for specific object
    initial_label_idx: list
        Initial label index predicted by the object detector
    frame: ndarray
        Video frame or image
    num_frame: int 
        number of current frame
    annot_frame: ndarray
        copy of original frame for annotation -> drawing bounding boxes and labels 
    save_img: bool
        True if we want to save tracked objects, default false
    path: dir
        Root directory, where we want to store the tracked object images 
    
    Return
    -----
    annotated_frame: ndarray
        Video frame that annotated bounding boxes and labels for a specific object  
    tracked_bbox: list
        A list of bounding boxes of tracked objects
    tracked_ids: list
        A list of tracked objects identities label
    """
    annotated_frame = None
    bbox_xyxy = []
    identities = []
    if len(detections['all_bboxes']) > 0:
        all_bboxes = np.array(detections['all_bboxes']).reshape(-1, 4)  # Ensure shape is (N, 4)
        all_confidences = np.array(detections['all_confidences'])
        outputs = tracker.update(all_bboxes, all_confidences, frame)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            # filter out the those indexes in initial label indexes that also appears in identities 
            filtered_initial_labels_idx = [idx for idx in initial_labels_idx if idx in identities]
            # defining the object name 
            obj_name = None if len(filtered_initial_labels_idx) == 0 else object_name

            for k, id in enumerate(identities):
                tracked_box = bbox_xyxy[k]
                # appending new tracked objects in the initial indexes
                if id not in initial_labels_idx:
                    initial_labels_idx.append(id)
                # crop and save 
                if obj_name is not None and save_img==True:
                    crop_n_save_tracked_object(tracked_box= tracked_box, annot_image= frame, 
                                           path= path, object_name= obj_name, track_id= id, 
                                           frame_no= num_frame)
            annotated_frame = draw_bboxes(annot_frame, obj_name, bbox_xyxy, identities)
    return annotated_frame, bbox_xyxy, identities

# function to crop and save tracked images
def crop_n_save_tracked_object(tracked_box, annot_image, path, object_name, track_id, frame_no):
    """
    This function aims to crop the tracked object from the frame and save it in a structured folder hierarchy.
    
    Parameters
    ----------
    tracked_box: [x1, y1, x2, y2]
        Bounding box that contains the location of the tracked object.
    annot_image: numpy array
        Duplicate of the original frame to crop.
    path: str
        Root directory where the object folders will be saved.
    object_name: str
        Name of the object to create a folder for.
    track_id: int
        Unique track identifier for the object.
    frame_no: int
        Frame number to use as part of the filename for saving images.

    Returns
    -------
    None
        Saves the cropped images to specific folders.
    """
    # Crop the image based on the bounding box
    x_min, y_min, x_max, y_max = map(int, tracked_box)
    cropped_img = annot_image[y_min:y_max, x_min:x_max]
    c_h, c_w, _ = cropped_img.shape
   
    # checking the cropped image is valid or not and resize  
    if cropped_img is None or cropped_img.size == 0:
        resized_cropped = None
    else:
        resized_cropped = cv2.resize(cropped_img, (c_w, c_h), interpolation=cv2.INTER_CUBIC)
    
    # Resize the cropped image (optional step depending on your use case)
    # resized_cropped = cv2.resize(cropped_img, (c_w, c_h), interpolation=cv2.INTER_CUBIC)

    # Create object folder if it doesn't exist
    object_folder = os.path.join(path, object_name)
    os.makedirs(object_folder, exist_ok=True)  # If it exists, it does nothing

    # Create track folder inside the object folder if it doesn't exist
    track_folder = os.path.join(object_folder, str(track_id))
    os.makedirs(track_folder, exist_ok=True)  # If it exists, it does nothing

    # Save the cropped image to the track folder
    img_name = f"frame_{frame_no:04d}.jpg"  # Formatting the frame number for consistent naming
    img_path = os.path.join(track_folder, img_name)
    if resized_cropped is not None:
        cv2.imwrite(img_path, resized_cropped)  # Save the cropped image
        print(f"Saved cropped image for Track ID {track_id}, Object {object_name} at {img_path}")

# function to draw trajectory
def draw_object_trajectory(object_trajectories, tracked_bbox_xyxy, tracked_identities, trajectory_file_path):
    """
    This function aims to store the tracked object trajectories as a json file to the desired path 

    Parameters
    ---------
    objet_trajectories: dict
        A dictionary that contains the relative positions of object accross frames 
    tracked_bbox_xyxy: list
        A list contains the tracked objects 
    tracked_identities: list
        Id's of tracked objects 
    trajectory_file_path: path
        Path to save the trajectories 
    
    Return 
    -----
    None
    """
    if tracked_bbox_xyxy is None:
        pass
    else:
        for i, box in enumerate(tracked_bbox_xyxy):
            track_id = int(tracked_identities[i])
            x1, y1, x2, y2 = map(int, box)  # Convert to int
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # If this ID is not in trajectories, initialize it
            if track_id not in object_trajectories:
                object_trajectories[track_id] = []

            # Append the new center point
            object_trajectories[track_id].append((center_x, center_y))
    
    # save the trajectories to the desired path
    with open(trajectory_file_path, 'w') as json_file:
        json.dump(object_trajectories, json_file, indent=4)
    
    print(f"Trajectory data saved to {trajectory_file_path}")