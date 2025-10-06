import os
import cv2
import torch
import numpy as np
import json
from utils.common_utils import xyxy_to_xywh, draw_bboxes, draw_bboxes_trajectory
from deep_sort.deep_sort import DeepSort


# check for gpu
use_cuda = torch.cuda.is_available()

# path of the person re-ID model
# deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'

# taking the deepsort algorithms 
# deepsort = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)


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
    all_ori_bboxs = []
    all_converted_bboxes = []       # Will store detections in [xc, yc, w, h] format
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
                    # store the original predicted bboxes
                    all_ori_bboxs.append(bbox)
                    # Convert the bbox from [x1, y1, x2, y2] to [xc, yc, w, h]
                    # As, DeepSORT takes input as [xc, yc, w, h] format
                    bbox_xcycwh = np.array(xyxy_to_xywh(bbox)).flatten()
                    all_converted_bboxes.append(bbox_xcycwh)
                    all_confidences.append(conf)
                    track_labels.append(label)
    
    # storing the detection results in a dictionary 
    det_results = {
        'all_ori_boxes': all_ori_bboxs,
        'all_bboxes': all_converted_bboxes,
        'all_confidences': all_confidences,
        'track_labels': track_labels
    }
    return det_results

# function to apply deepsort 
# def apply_deepsort(all_bboxes, all_confidences, frame):
#     """
#     This function will take all detected boxes and their confidence scores 
#     and return the tracked outputs 

#     Parameters
#     ----------
#     all_boxes: list 
#         A list of numpy arrays that contains all the dected boxes 
#     all_confidences: list
#         A list of cofidence scores 
#     frame: numpy array 
#         Current frame that contains the objects 
#     Returns
#     -------
#     tracked_output: list
#         A list with 5 elemets where first four contains the bbox and last one track ID 
#     """
#     all_bboxes = np.array(all_bboxes).reshape(-1, 4)  # Ensure shape is (N, 4)
#     all_confidences = np.array(all_confidences)
#     outputs = deepsort.update(all_bboxes, all_confidences, frame)
#     return outputs

# function to track object 
def track_object(object_name, detections, tracker, initial_labels_idx, frame, num_frame, annot_frame, draw_trajectory= False, annot_trajectory= False, obj_trjactory= None, save_img=False, path=None):
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
    draw_trajectory: bool
        Flag to get object trajectories or not 
    annot_trajectory: bool
        Flag to determine annotate tracked object  
    object_trajectories: dict 
        Dictionary to store the object trajectories 
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
    object_trajectory: dict
        A dictionary that contains the relative positions of the tracked object
    """
    annotated_frame = None
    bbox_xyxy = []
    identities = []
    object_trajectories = {}
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
            if draw_trajectory:
                object_trajectories = draw_object_trajectory(object_trajectories= obj_trjactory, tracked_bbox_xyxy= bbox_xyxy, tracked_identities= identities, 
                                    trajectory_file_path= None)
                if annot_trajectory:
                    annot_frame = draw_bboxes_trajectory(annot_frame, obj_name, bbox_xyxy, identities=identities, trajectories=object_trajectories)
                annotated_frame = draw_bboxes(annot_frame, obj_name, bbox_xyxy, identities)
            else:
                annotated_frame = draw_bboxes(annot_frame, obj_name, bbox_xyxy, identities)
    return annotated_frame, bbox_xyxy, identities, object_trajectories

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


# fucntion to count single
def count_single_object(num_frame, obj, identities, counter_set, count_data):
    """
    The aim of this function is to count the single-class tracked objects over the time 
    
    Parameters
    ----------
    num_frame: int 
        Number of the current frame 
    obj: string 
        Name of the object we are tracking
    identities: list 
        List of tracked ID's in a single frame 
    counter_set = set
        Global set that store all the tracked ID's
    count_data: dict
        A dictionary that contains the total counting report 
    
    Return
    ------
    count_data: dict
        Full counting history during the tracking time
    """
    # update the counter set based on objects in current tracked frame 
    counter_set.update(identities)

    # Dynamically update count_data for the tracked object
    count_data['frame number'] = num_frame
    if obj not in count_data:
        count_data[f'total_{obj}'] = 0
        count_data[f'in_frame_{obj}'] = 0

    count_data[f'total_{obj}'] = len(counter_set)
    count_data[f'in_frame_{obj}'] = len(identities)

    return count_data

# fucntion to count multiple objects 
def count_multiple_objects(num_frame, obj, multi_counter_set, identities, counter_set, obj_counter, count_data):
    """
    The aim of this function is to count the multi-class tracked objects over the time 
    
    Parameters
    ----------
    num_frame: int 
        Number of the current frame 
    obj: string 
        Name of the object we are tracking
    multi_counter_set: dict 
        Gloabal dictionry that stores the counting values of multi-class objects
    identities: list 
        List of tracked ID's in a single frame 
    counter_set = set
        class specific counting set 
    obj_counter: string
        Key of multi_counter_set that determines the number of class-specific object in a single frame
    count_data: dict
        A dictionary that contains the total counting report 
    
    Return
    ------
    count_data: dict
        Full counting history during the tracking time 
    """

    # count the current tracked objects for a specific class 
    multi_counter_set[obj_counter] = len(identities)

    # update the counter set for that specific class 
    multi_counter_set[counter_set].update(identities)

    # Dynamically update count_data for the tracked object
    count_data['frame number'] = num_frame
    if obj not in count_data:
        count_data[f'total_{obj}'] = 0
        count_data[f'in_frame_{obj}'] = 0

    # count_data[f'total_{obj}'] = len(counter_set)
    count_data[f'total_{obj}'] = len(multi_counter_set[counter_set])
    count_data[f'in_frame_{obj}'] = multi_counter_set[obj_counter]

    return count_data

# function to annotate counting results 
def annote_count_results(frame, count_data):
    """
    This function aims to take the current frame and counting results to represent real-time counting

    Parameter
    --------
    frame: ndarray
        current video frame, which is a n-dimensional numpy array 
    count_data: dict
        A dictionary that contains the counting results 

    Return
    ----- 
    frame: ndarray
        Annotated frame with counting results
    """
    # Set the initial position for the text (top-left corner)
    position = (10, 30) 
    # Choose font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Font scale
    font_scale = 0.7
    # Color of the text (BGR format)
    color = (0, 255, 0)  # Green color
    # Thickness of the text
    thickness = 2
    # Line type
    line_type = cv2.LINE_AA

    # annote the counting information 
    if count_data:
        for key, value in count_data.items():
            text = f"{key}: {value}"
            # Write the text on the image
            cv2.putText(frame, text, position, font, font_scale, color, thickness, line_type)         
            # Move the position down for the next line of text
            position = (position[0], position[1] + 30)  # Move 30 pixels down
    return frame


# Define the line for visual representation (Vertical line)
def draw_vertical_line(frame, frame_width, frame_hight):
    start_point_x = frame_width // 2
    start_point_y = 0  # height 
    end_point_x = start_point_x
    end_point_y = frame_hight 
    cv2.line(frame, (start_point_x, start_point_y), (end_point_x, end_point_y), (0, 255, 0), 2)  # Green vertical line at 'line_x'

# function to draw horizontal line 
def draw_horizontal_line(frame, frame_width, frame_height):
    start_point_x = 0
    start_point_y = frame_height // 2
    end_point_x = frame_width
    end_point_y = frame_height // 2
    cv2.line(frame, (start_point_x, start_point_y), (end_point_x, end_point_y), (0, 255, 0), 2)

# function to draw relative line based on frame 
def draw_relative_line(frame, frame_width, frame_height, start_x_ratio=None, start_y_ratio=None, end_x_ratio=None, end_y_ratio=None):
    start_point_x = int(frame_width * start_x_ratio)
    start_point_y = int(frame_height * start_y_ratio)
    end_point_x = int(frame_width * end_x_ratio)
    end_point_y = int(frame_height * end_y_ratio)
    cv2.line(frame, (start_point_x, start_point_y), (end_point_x, end_point_y), (0, 255, 0), 2)

# function to get the relative height of the line based on the drawn line 
def calculate_line_height(frame_width, frame_height, start_x_ratio=None, is_straight= True, start_y_ratio=None, end_x_ratio=None, end_y_ratio=None):
    """
    Parameters
    ---------

    start_x_ratio: 1/4
    start_y_ratio:3/8
    end_x_ratio:
    end_y_ratio: 5/8
    """
    start_point_x = int(frame_width * (start_x_ratio))
    start_point_y = int(frame_height * (start_y_ratio))
    end_point_x = frame_width * end_x_ratio
    end_point_y = int(frame_height * (end_y_ratio))
    # If we would like to use the slope and intercept to determine the crossing 
    if is_straight:
        return start_point_y
    else:
        # Compute the slope (m) and intercept (b) of the line equation y = mx + b
        m = (end_point_y - start_point_y) / (end_point_x - start_point_x)  # Slope
        b = start_point_y - m * start_point_x  # Intercept 
        return m, b
    # return start_point_x, start_point_y, end_point_x, end_point_y


# function to draw trajectory
def draw_object_trajectory(object_trajectories, tracked_bbox_xyxy, tracked_identities, trajectory_file_path=None):
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
    if trajectory_file_path:
        with open(trajectory_file_path, 'w') as json_file:
            json.dump(object_trajectories, json_file, indent=4)
        
        print(f"Trajectory data saved to {trajectory_file_path}")
    return object_trajectories

# funtion to determine a point on which side of the line 
def side_of_line(px, py, x1, y1, x2, y2):
    """
    Determines which side of the line a point (px, py) is on.
    Parameter
    --------
    px, px: (coord)
        given point which we want to check
    x1, y1: (coord)
        position of starting point of the line 
    x2, y2: (coord)
        position of end point of the line 
    Returns:
    - Negative value if below the line
    - Positive value if above the line
    - Zero if exactly on the line
    """
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


def check_crossing(obj_trajectories, crossing_dict, is_straight=True, line_height=None, line_slope=None, line_intercept=None):
    # for straight reference line 
    if is_straight:
        for obj_id, positions in obj_trajectories.items():
            for i in range(1, len(positions)):
                prev_x, prev_y = positions[i-1]
                curr_x, curr_y = positions[i]
                # if moving down 
                if prev_y < line_height and curr_y >= line_height:
                    crossing_dict['incoming'].append(obj_id)
                    print(f"Object {obj_id} crossed the line UPWARD -> Incoming..")
                # elif moving up
                elif prev_y > line_height and curr_y <= line_height:
                    crossing_dict['outgoing'].append(obj_id)
                    print(f"Object {obj_id} crossed the line DOWNWARD -> Outgoing ..")
    else:
        for obj_id, positions in obj_trajectories.items():
            for i in range(1, len(positions)):
                prev_x, prev_y = positions[i - 1]
                curr_x, curr_y = positions[i]

                # Get the y-values of the crossing line at prev_x and curr_x
                # m-> slope, b-> intercept
                expected_prev_y = line_slope * prev_x + line_intercept
                expected_curr_y = line_slope * curr_x + line_intercept

                # Check if the object moved from one side of the line to the other
                if prev_y < expected_prev_y and curr_y >= expected_curr_y:
                    print(f"Object {obj_id} crossed the line UPWARD -> Incoming..")
                    crossing_dict['incoming'].append(obj_id)
                elif prev_y > expected_prev_y and curr_y <= expected_curr_y:
                    print(f"Object {obj_id} crossed the line DOWNWARD -> Outgoing ..")
                    crossing_dict['outgoing'].append(obj_id)
    return crossing_dict

def check_multiple_crossing(obj_name, obj_trajectories, crossover_dict, is_straight=True, line_height=None, line_slope=None, line_intercept=None):
    # Now process the crossings
    if is_straight:
        for obj_id, positions in obj_trajectories.items():
            for i in range(1, len(positions)):
                prev_x, prev_y = positions[i - 1]
                curr_x, curr_y = positions[i]

                # Check if the object crossed the line
                if prev_y < line_height and curr_y >= line_height:
                    print(f"Object {obj_id} crossed the line UPWARD -> Incoming {obj_name}")
                    crossover_dict[f'incoming_{obj_name}'].append(obj_id)
                elif prev_y > line_height and curr_y <= line_height:
                    print(f"Object {obj_id} crossed the line DOWNWARD -> Outgoing {obj_name}")
                    crossover_dict[f'outgoing_{obj_name}'].append(obj_id)
    else:
        for obj_id, positions in obj_trajectories.items():
            for i in range(1, len(positions)):
                prev_x, prev_y = positions[i - 1]
                curr_x, curr_y = positions[i]

                # Get the y-values of the crossing line at prev_x and curr_x
                expected_prev_y = line_slope * prev_x + line_intercept
                expected_curr_y = line_slope * curr_x + line_intercept

                # Check if the object crossed the line
                if prev_y < expected_prev_y and curr_y >= expected_curr_y:
                    print(f"Object {obj_id} crossed the line UPWARD -> Incoming {obj_name}")
                    crossover_dict[f'incoming_{obj_name}'].append(obj_id)
                elif prev_y > expected_prev_y and curr_y <= expected_curr_y:
                    print(f"Object {obj_id} crossed the line DOWNWARD -> Outgoing {obj_name}")
                    # crossover_dict[f'outgoing_{obj_name}'].append(obj_id)
                    crossover_dict[f'outgoing_{obj_name}'].append(obj_id)

    return crossover_dict


# function to annotate crossing results 
def annotate_crossing_results(frame, count_data):
    """
    Annotates the current frame with counting results in the top-right corner.

    Parameters
    ----------
    frame: ndarray
        Current video frame (n-dimensional numpy array).
    count_data: dict
        A dictionary containing the counting results.

    Returns
    -------
    frame: ndarray
        Annotated frame with counting results.
    """
    # Choose font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)  # Green color
    thickness = 2
    line_type = cv2.LINE_AA

    # Get frame dimensions
    (h, w, _) = frame.shape

    # Set initial position near the top-right corner
    margin = 10  # Padding from the edges
    y_position = margin + 20  # Start 20 pixels from the top

    # Annotate the counting information
    if count_data:
        for key, value in count_data.items():
            if isinstance(value, (list, set, tuple)):  # Check if value is iterable
                text = f"{key}: {len(set(value))}"  # Count unique values
            else:
                text = f"{key}: {value}"

            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

            # Calculate X position to align text to the right
            x_position = w - text_width - margin  

            # Write the text on the frame
            cv2.putText(frame, text, (x_position, y_position), font, font_scale, color, thickness, line_type)

            # Move the Y position down for the next line
            y_position += text_height + 10  # Add spacing between lines

    return frame