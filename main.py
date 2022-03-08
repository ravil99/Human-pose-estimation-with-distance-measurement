import cv2
import yaml
import time
import tensorflow as tf
import numpy as np
import logging
import os

from utils.vector_math import angle_between, distance_beetween2objects
from utils.custom_video_capture import CustomVideoCapture
from modules.depth_estimator import DepthEstimator
from modules.aruco_detector.aruco_detector import findArucoMarkers, arucoIndex, drawArUco
from modules.pose_estimator import PoseEstimator

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = "dist_stats.log",
                    filemode = "a",
                    format = Log_Format, 
                    level = 20)

logger = logging.getLogger()
logger.info("New experiment!!!!!!!!!")
logger.info("------------------------")
logger.info("------------------------")
cur_dir = os.path.dirname(os.path.realpath(__file__))

with open("config.yaml") as file:
    config = yaml.full_load(file)

cap = CustomVideoCapture(0)

if config["depth_estimation"]["enable"]:
    depth_estimator = DepthEstimator()

if config["pose_estimation"]["enable"]:
    pose_estimator = PoseEstimator(
        use_poseviz=config["pose_estimation"]["poseviz"])

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
    log = {}

    # Find aruco markers
    arucoFound, vectors = findArucoMarkers(frame, frame_number, config['aruco_size'])
    aruco_dists = arucoFound[2]

    # Estimate depth
    if config["depth_estimation"]["enable"]:
        disparity_map, scaled_disparity_map = \
            depth_estimator.predict(rgb_frame)

    # Pose estimation
    if config["pose_estimation"]["enable"]:
        pred_poses = pose_estimator.predict(rgb_frame)
        if config["pose_estimation"]["draw_2d"]:
            frame = pose_estimator.draw2D(pred_poses, frame)

        chest_points_2d, chest_points_3d = pose_estimator.get_chest(pred_poses)
        for i, human_point in enumerate(chest_points_2d):
            frame = cv2.circle(frame, human_point, radius=8,
                               color=(0, 0, 255), thickness=-1)
            cv2.putText(frame, str(i), (human_point[0] - 5, human_point[1] -5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    if len(arucoFound[0]) != 0:
        num = 1
        for i in range(len(arucoFound[0])):
            for j in range(i+1, len(arucoFound[0])):
                aruco_point1 = np.array(arucoFound[3][i])
                aruco_point2 = np.array(arucoFound[3][j])

                angle = angle_between(aruco_point1, aruco_point2)
                distance_beetween_markers = distance_beetween2objects(
                    aruco_dists[i], aruco_dists[j], angle)
                cv2.putText(frame, "Dist between aruco {} and {}: {:.2f}".format(arucoFound[1][i][0], arucoFound[1][j][0], distance_beetween_markers),
                            (10, int(frame.shape[0] - num*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                num +=1
                log[f"aruco_{arucoFound[1][i][0]}_{arucoFound[1][j][0]}"] = distance_beetween_markers

    if config["pose_estimation"]["enable"] and config["depth_estimation"]["enable"]:
        if len(arucoFound[0]) != 0:
            aruco_num = -1
            ID = 1
            for i, arucoId in enumerate(arucoFound[1]):
                if arucoId[0] == ID:
                    aruco_num = i
            if aruco_num == -1:
                print("No aruco with id", ID)
            aruco_point = np.mean(arucoFound[0][aruco_num][0], 0)
            aruco_point = (round(aruco_point[0]), round(aruco_point[1]))
            depth_aruco = scaled_disparity_map[aruco_point[1], aruco_point[0]]
            log[f"depth_aruco_{arucoFound[1][0][0]}"] = depth_aruco

            human_dists = []
            for i, human_point in enumerate(chest_points_2d):
                human_point_x = min(frame.shape[1] - 1, max(0, human_point[0]))
                human_point_y = min(frame.shape[0] - 1, max(0, human_point[1]))
                depth_human = scaled_disparity_map[human_point_y, human_point_x]
                human_dist = aruco_dists[0] *  depth_aruco / depth_human
                human_dists.append(human_dist)
                cv2.putText(frame, "Dist to human {}: {:.2f}".format(i, human_dist), (frame.shape[1] - 300, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 255, 0), 2)
                log[f"dist_to_human_{i}"] = human_dist
                log[f"depth_human_{i}"] = depth_human
                
            num = 2
            for i in range(len(chest_points_3d)):
                for j in range(i+1, len(chest_points_3d)):
                    human_point1 = np.array(chest_points_3d[i])
                    human_point2 = np.array(chest_points_3d[j])
                    angle = angle_between(human_point1, human_point2)
                    distance_beetween_humans = distance_beetween2objects(
                        human_dists[i], human_dists[j], angle)
                    cv2.putText(frame, "Dist between human {} and {}: {:.2f}".format(i, j, distance_beetween_humans),
                                (frame.shape[1] - 500, int(frame.shape[0] - num*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
                    num +=1
                    log[f"dist_between_human_{i}_{j}"] = distance_beetween_humans



    # Draw Aruco
    if len(arucoFound[0]) != 0:
        for i, (bbox, id, aruco_dist) in enumerate(zip(arucoFound[0], arucoFound[1], arucoFound[2])):
            
            drawArUco(frame, arucoFound, vectors)
            frame = arucoIndex(bbox, id, frame)
            cv2.putText(frame, "Dist to aruco {}: {:.2f}".format(id[0], aruco_dist), (10, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (255, 0, 0), 2)
            log[f"dist_to_aruco_{id[0]}"] = aruco_dist
    
    logger.info(log)

    cv2.imshow("Frame", frame)

    if config["depth_estimation"]["enable"] and config["depth_estimation"]["show_depth_window"]:
        cv2.imshow("Depth estimation", scaled_disparity_map)

    key = cv2.waitKey(1)
    if not ret or key == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
