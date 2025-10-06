import os
import torch
import cv2
import argparse
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from utils.tracker_utils import *
from utils.common_utils import *

# path of the person re-ID model
deepsort_checkpoint = 'deep_sort/deep/checkpoint/ckpt.t7'
# Object detector
yolo_v8 = YOLO("yolov8n.pt")

parser = argparse.ArgumentParser()

# check for gpu
use_cuda = torch.cuda.is_available()

def parse_resolution(resolution_str):
    """Convert a comma-separated string like '1920,1080' to a tuple (1920, 1080)"""
    width, height = map(int, resolution_str.split(','))
    return width, height

# Define a function to load the video
def load_video(video_path, resolution=None):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Error opening video file!")
    
    # Set the resolution if provided
    if resolution:
        width, height = resolution
        video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    return video, fps

# Define a function to save results
def save_tracked_video(annot_frame, out_video):
    out_video.write(annot_frame)

# function to process video
def process_video(video_path, resolution, obj_to_track, tracker, single_tracker=False, get_trajectory= False, annote_trajectory=False, is_straight_line= False, save_images=False, save_video=False):
    video, fps = load_video(video_path, resolution)
    
    if save_images:
        root_path = os.path.splitext(os.path.basename(video_path))[0] 
        os.makedirs(root_path, exist_ok=True)
        output_video_path = os.path.join(root_path, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
    else:
        root_path = None

    num_frame = 0
    # store object trajectory 
    obj_trajetory = {}
    multi_obj_trajectory = {}
    # flag to ensure about storing object trajectories or not 
    trajetory_flag = get_trajectory
    # defining line is straight or not 
    is_straight = is_straight_line

    # single crossing
    single_crossing = {'incoming':[], 'outgoing':[]}
    multi_crossover_dict = {}
    crossing_reports = None

    # strore counting 
    count_data = {}
    counter_set = set()
    multi_counter_sets = {}
    
    # defining multiple counter sets for multiple object tracking
    if not single_tracker:
        for obj in obj_to_track:
            counter_set = obj + '_counter_set'
            obj_counter = obj + '_counter'
            # obj_traject = obj + '_trajectories'
            if counter_set not in multi_counter_sets:
                multi_counter_sets[counter_set] = set()
                multi_counter_sets[obj_counter] = 0
            # for multiple cross-over
            if f'incoming_{obj}' not in multi_crossover_dict:
                multi_crossover_dict[f'incoming_{obj}'] = []
            if f'outgoing_{obj}' not in multi_crossover_dict:
                multi_crossover_dict[f'outgoing_{obj}'] = []
            

    while True:
        ret, frame = video.read()
        if not ret:
            print("No more frames or failed to grab a frame!")
            break

        num_frame += 1
        resized_frame = cv2.resize(frame, resolution)
        ori_im = resized_frame.copy()
        # getting the frame width and frame height  
        f_width, f_height = resolution

        #### Single Object Tracking ####
        # Get detections
        if single_tracker:
            detections = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=obj_to_track)
            # store the labels of the first frame 
            if num_frame == 1:
                initial_labels = detections['track_labels'].copy()
                initial_labels_idx = [k+1 for k in range(len(initial_labels))]
            
            #### Object Crossover -> count incoming and outgoing object ####
            # object_positions will contain the object trajectories if draw_trajectory=True
            if trajetory_flag:
                tracked_frame, bbox_xyxy, identities, obj_positions = track_object(object_name=obj_to_track[0], detections= detections, tracker= tracker, 
                                        initial_labels_idx= initial_labels_idx, frame= resized_frame, num_frame= num_frame, annot_frame= ori_im,
                                        draw_trajectory= True, obj_trjactory= obj_trajetory, save_img= save_images, path= root_path)
            # draw the tracked box 
            if trajetory_flag and annote_trajectory:
                tracked_frame, _, _, _ = track_object(object_name=obj_to_track[0], detections= detections, tracker= tracker, 
                                        initial_labels_idx= initial_labels_idx, frame= resized_frame, num_frame= num_frame, annot_frame= ori_im,
                                        draw_trajectory= True, annot_trajectory= annote_trajectory, obj_trjactory= obj_trajetory)
                
                # reference line can be straight or non-straight
                if is_straight:
                    crossing_reports = None
                    line_height = calculate_line_height(frame_width= f_width, frame_height= f_height, start_x_ratio=0.9/4, is_straight= True, 
                                                        start_y_ratio=2.7/8, end_x_ratio=7/8, end_y_ratio=1/3)
                    crossing_reports = check_crossing(obj_trajectories=obj_positions, crossing_dict=single_crossing, is_straight=True, line_height=line_height)
                   
                else:
                    line_slope, line_intercept = calculate_line_height(frame_width= f_width, frame_height= f_height, start_x_ratio=0.9/4, is_straight= False, 
                                                        start_y_ratio=2.7/8, end_x_ratio=7/8, end_y_ratio=1/3)
                    crossing_reports = check_crossing(obj_trajectories=obj_positions, crossing_dict=single_crossing, is_straight=False, line_height=None,
                                                      line_slope=line_slope, line_intercept=line_intercept)
            else:
                tracked_frame, bbox_xyxy, identities, _ = track_object(object_name=obj_to_track[0], detections= detections, tracker= tracker, initial_labels_idx= initial_labels_idx, 
                                                                                    frame= resized_frame, num_frame= num_frame, annot_frame= ori_im,
                                                                                    )
    
            #### Count Total Object #### 
            count_data = count_single_object(num_frame= num_frame, obj= obj_to_track[0], identities=identities, 
                                             counter_set=counter_set, count_data=count_data)
        #### Multiple Object Tracking ####
        else:
            for i, obj in enumerate(obj_to_track):
                detections= obj + '_detections'
                counter_set = obj + '_counter_set'
                obj_counter = obj + '_counter'
                obj_traject = f"{obj}_trajectories"

                detections = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=obj)
               
                if num_frame == 1:
                    initial_labels = obj+'_initial_labels'
                    initial_labels = detections['track_labels'].copy()
                    initial_labels_idx = [k+1 for k in range(len(initial_labels))]

                tracked_results = {}

                #### Multiple objects crossover ####             
                if trajetory_flag:
                    tr_frmae, tr_bbox, tr_ids, _ = track_object(object_name=obj, detections= detections, tracker= tracker[i], 
                                            initial_labels_idx= initial_labels_idx, frame= resized_frame, num_frame= num_frame, 
                                            annot_frame= ori_im, draw_trajectory= True, obj_trjactory= obj_trajetory, 
                                            save_img= save_images, path= root_path)

                    # storing object-specific trajectory 
                    if obj_traject not in multi_obj_trajectory:
                        multi_obj_trajectory[obj_traject] = {}  
                    multi_obj_trajectory[obj_traject] = draw_object_trajectory(object_trajectories=multi_obj_trajectory[obj_traject], tracked_bbox_xyxy=tr_bbox, tracked_identities=tr_ids)
                    
                    # if reference line is straight
                    if is_straight:
                        line_height = calculate_line_height(frame_width= f_width, frame_height= f_height, start_x_ratio=0.9/4, is_straight= True, 
                                                        start_y_ratio=2.7/8, end_x_ratio=7/8, end_y_ratio=1/3)
                        crossing_reports = check_multiple_crossing(obj_name=obj, obj_trajectories=multi_obj_trajectory[obj_traject], crossover_dict=multi_crossover_dict, is_straight=True, line_height=line_height)
                    # if non-straight reference line
                    else:
                        line_slope, line_intercept = calculate_line_height(frame_width= f_width, frame_height= f_height, start_x_ratio=0.9/4, is_straight= False, 
                                                        start_y_ratio=2.7/8, end_x_ratio=7/8, end_y_ratio=1/3)      
                        crossing_reports = check_multiple_crossing(obj_name=obj, obj_trajectories=multi_obj_trajectory[obj_traject], crossover_dict=multi_crossover_dict, is_straight=False, line_height=None,
                                                      line_slope=line_slope, line_intercept=line_intercept)
                else:
                    tracked_results[obj] = track_object(object_name=obj, detections= detections, tracker= tracker[i], initial_labels_idx= initial_labels_idx, 
                                                        frame= resized_frame, num_frame= num_frame, annot_frame= ori_im,
                                                                                        )

                tracked_results[obj] = track_object(object_name=obj, detections= detections, 
                                                        tracker= tracker[i], initial_labels_idx= initial_labels_idx, frame= resized_frame,
                                                        num_frame= num_frame, annot_frame= ori_im, save_img= save_images, path= root_path)
                
                # get specific object values 
                tracked_frame = tracked_results[obj][0]
                bbox_xyxy = tracked_results[obj][1]
                identities = tracked_results[obj][2]
                positions = tracked_results[obj][3]

                # count number of objects 
                count_data = count_multiple_objects(num_frame=num_frame, obj=obj, multi_counter_set=multi_counter_sets,
                                                    identities=identities, counter_set=counter_set, obj_counter=obj_counter, 
                                                    count_data=count_data)

        # annotate the current frame with counting results 
        ori_im = annote_count_results(frame=ori_im, count_data=count_data)
        if get_trajectory:
            # drawing line to count incoming and outgoing objects 
            draw_relative_line(ori_im, f_width, f_height, start_x_ratio=0.9/4, start_y_ratio=2.7/8, 
                           end_x_ratio=7/8, end_y_ratio=1/3)
            ori_im = annotate_crossing_results(frame=ori_im, count_data= crossing_reports)

        # Show annotated frames
        cv2.imshow('video', ori_im)
        
        # Save video frames
        if save_video:
            save_tracked_video(ori_im, out_video)

        # using frame rate of CPU power
        key = cv2.waitKey(1)
        if key == 27:  # Press ESC to exit
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"Total frame: {num_frame}")
    # out_video.release()



def main(args):
    video_path =  args.video_path
    vid_resolution = tuple(args.video_resolution)
    obj_to_track = args.obj_to_track.split()
    save_tracked_img = args.save_tracked_img
    save_tracked_video = args.save_tracked_video
    get_trajectory = args.get_trajectory
    annote_trajectory = args.annote_trajectory
    is_straight_line = args.is_straight_line
    
    single_tracker = False
    tracker_list = []
    trajectori_list = []
    

    if len(obj_to_track) == 1:
        tracker = obj_to_track[0] + '_tracker'
        tracker = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)
        single_tracker = True
        trajectory = obj_to_track[0] + '_trajectories'
        trajectory = {}
        # call the process function 
        process_video(video_path, vid_resolution, obj_to_track, tracker, single_tracker=single_tracker, get_trajectory=get_trajectory, 
                       annote_trajectory= annote_trajectory, is_straight_line=is_straight_line, save_images=save_tracked_img, 
                       save_video=save_tracked_video)
    else:
        for obj in obj_to_track:
            tracker = obj + '_tracker'
            tracker = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)
            tracker_list.append(tracker)
            tr = obj + '_trajectories'
            tr = {}
            trajectori_list.append(tr)
        # call the process function 
        process_video(video_path, vid_resolution, obj_to_track, tracker=tracker_list, single_tracker=single_tracker, get_trajectory=get_trajectory,
                      is_straight_line=is_straight_line, save_images=save_tracked_img, save_video=save_tracked_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiple Object Tracking Script")

    # adding arguements 
    parser.add_argument('--video-path', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--video-resolution', type=parse_resolution, default=(1920,1080), help="Resolution of the video as (width,height)")
    parser.add_argument('--obj-to-track', type=str, required=True, help="List of objecets name that you want to track")
    # Add argument with action='store_true' for boolean flag, if we want to store images, then we will pass this arguments
    # If we don't want to save the images then we will not pass this arg, by default it will remail False
    parser.add_argument('--save-tracked-img', action='store_true', help="Save tracked images")
    parser.add_argument('--save-tracked-video', action='store_true', help="Save tracked video")

    # flag to determine to get the trajectory or not 
    parser.add_argument('--get-trajectory', action='store_true', help="Draw the movement trajectory of objects and get as json")
    parser.add_argument('--annote-trajectory', action='store_true', help="Draw bounding boxes over tracked objects")
    # flag to determine the crossing line is straight or non-straight
    parser.add_argument('--is-straight-line', action= 'store_true', help="Define straight or non-straight line for crossing")

    # defining arguement 
    args = parser.parse_args()
    main(args)
