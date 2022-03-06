import cv2
import yaml
import time
import tensorflow as tf
import numpy as np

from modules.depth_estimator import DepthEstimator
from utils.custom_video_capture import CustomVideoCapture
from modules.aruco_detector.aruco_detector import findArucoMarkers, arucoIndex
from modules.pose_estimator import PoseEstimator

with open("config.yaml") as file:
    config = yaml.full_load(file)

cap = CustomVideoCapture(0)

if config["depth_estimation"]["enable"]:
    depth_estimator = DepthEstimator()

if config["pose_estimation"]["enable"]:
    pose_estimator = PoseEstimator(use_poseviz=config["pose_estimation"]["poseviz"])

time.sleep(2)
frame_number = 0
tic = 0
while True:
    if config['print_fps']:
        fps = 1 / (time.time() - tic)
        tic = time.time()
        print("FPS: {}".format(fps))

    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find aruco markers
    arucoFound = findArucoMarkers(frame, frame_number, config['aruco_size'])

    # Estimate depth
    if config["depth_estimation"]["enable"]:
        disparity_map, scaled_disparity_map = depth_estimator.predict(rgb_frame)

    # Pose estimation
    if config["pose_estimation"]["enable"]:
        pred_poses = pose_estimator.predict(rgb_frame)
        if config["pose_estimation"]["draw_2d"]:
            frame = pose_estimator.draw2D(pred_poses, frame)

        chest_points = pose_estimator.get_chest(pred_poses)
        for human_point in chest_points:
            frame = cv2.circle(frame, human_point, radius=8, color=(0, 0, 255), thickness=-1)



    if config["pose_estimation"]["enable"] and config["depth_estimation"]["enable"]:
        if len(arucoFound[0]) != 0:
            bbox = arucoFound[0][0][0]
            aruco_dist = arucoFound[2][0]
            aruco_point = np.mean(arucoFound[0][0][0], 0)
            aruco_point = (round(aruco_point[0]), round(aruco_point[1]))
            depth_aruco = disparity_map[aruco_point[1], aruco_point[0]]

            for i, human_point in enumerate(chest_points):
                print("human point", human_point)
                human_point_x = min(frame.shape[1] - 1, max(0, human_point[0]))
                human_point_y = min(frame.shape[0] - 1, max(0, human_point[1]))
                depth_human = disparity_map[human_point_y, human_point_x]
                human_dist = aruco_dist * depth_human / depth_aruco
                cv2.putText(frame, "Dist to human {}: {:.2f}".format(i, human_dist), (frame.shape[1] - 300, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                            (255, 255, 0), 2)


    #Draw Aruco
    if len(arucoFound[0]) != 0:
        for i, (bbox, id, aruco_dist) in enumerate(zip(arucoFound[0], arucoFound[1], arucoFound[2])):
            frame = arucoIndex(bbox, id, frame)
            cv2.putText(frame, "Dist to aruco {}: {:.2f}".format(id[0], aruco_dist), (10, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                        (255, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if config["depth_estimation"]["enable"] and config["depth_estimation"]["show_depth_window"]:
        cv2.imshow("Depth estimation", scaled_disparity_map)
    
    key = cv2.waitKey(1)
    if not ret or key == ord('q'):
        break
    
    frame_number += 1

cap.release()
cv2.destroyAllWindows()