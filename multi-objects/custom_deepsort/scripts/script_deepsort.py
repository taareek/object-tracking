import os
import torch
import cv2
import argparse
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from tracker_utils import *

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
def process_video(video_path, resolution, obj_to_track, tracker, single_tracker=False, save_images=False, save_video=False):
    video, fps = load_video(video_path, resolution)
    
    if save_images:
        root_path = os.path.splitext(os.path.basename(video_path))[0] + '_v2'
        os.makedirs(root_path, exist_ok=True)
        output_video_path = os.path.join(root_path, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
    else:
        root_path = None

    num_frame = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("No more frames or failed to grab a frame!")
            break

        num_frame += 1
        resized_frame = cv2.resize(frame, resolution)
        ori_im = resized_frame.copy()

        # Get detections
        if single_tracker:
            detections = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=obj_to_track)
            # store the labels of the first frame 
            if num_frame == 1:
                initial_labels = detections['track_labels'].copy()
                initial_labels_idx = [k+1 for k in range(len(initial_labels))]
            
            tracked_frame, bbox_xyxy, identities = track_object(object_name=obj_to_track[0], detections= detections, tracker= tracker, 
                                        initial_labels_idx= initial_labels_idx, frame= resized_frame,
                                        num_frame= num_frame, annot_frame= ori_im, save_img= save_images, path= root_path)
        else:
            for i, obj in enumerate(obj_to_track):
                detections= obj + '_detections'
                detections = get_detection_results(detector= yolo_v8, frame= resized_frame, obj_to_track=obj)
                if num_frame == 1:
                    initial_labels = obj+'_initial_labels'
                    initial_labels = detections['track_labels'].copy()
                    initial_labels_idx = [k+1 for k in range(len(initial_labels))]

                tracked_results = {}
                tracked_results[obj] = track_object(object_name=obj, detections= detections, 
                                                        tracker= tracker[i], initial_labels_idx= initial_labels_idx, frame= resized_frame,
                                                        num_frame= num_frame, annot_frame= ori_im, save_img= save_images, path= root_path)
                
                # get specific object values 
                tracked_frame = tracked_results[obj][0]
                bbox_xyxy = tracked_results[obj][1]
                identities = tracked_results[obj][2]


        # Show annotated frames
        cv2.imshow('video', ori_im)
        
        # Save video frames
        if save_video:
            save_tracked_video(ori_im, out_video)

        # getting the actual frame rate from the input video 
        # wait_time = int(1000 / fps)  # wait time in milliseconds
        # key = cv2.waitKey(wait_time)

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
        process_video(video_path, vid_resolution, obj_to_track, tracker, single_tracker=single_tracker, 
                      save_images=save_tracked_img, save_video=save_tracked_video)
    else:
        for obj in obj_to_track:
            tracker = obj + '_tracker'
            tracker = DeepSort(deepsort_checkpoint, use_cuda=use_cuda)
            tracker_list.append(tracker)
            tr = obj + '_trajectories'
            tr = {}
            trajectori_list.append(tr)
        # call the process function 
        process_video(video_path, vid_resolution, obj_to_track, tracker=tracker_list, single_tracker=single_tracker, 
                    save_images=save_tracked_img, save_video=save_tracked_video)


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

    # defining arguement 
    args = parser.parse_args()
    main(args)