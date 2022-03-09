import cv2
import yaml
import time
import tensorflow as tf
import numpy as np
import logging

from utils.vector_math import angle_between, distance_beetween2objects
from utils.custom_video_capture import CustomVideoCapture
from modules.depth_estimator import DepthEstimator
from modules.aruco.aruco_detector import ArucoDetector
from modules.pose_estimator import PoseEstimator


def get_dist(position): return (
    position[0]**2 + position[1]**2 + position[2]**2)**0.5


Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename="stats.log",
                    filemode="a",
                    format=Log_Format,
                    level=20)

logger = logging.getLogger()
logger.info("New experiment!!!!!!!!!")

with open("config.yaml") as file:
    config = yaml.full_load(file)

cap = CustomVideoCapture(0)

aruco_detector = ArucoDetector(config['aruco']['real_side_size'], config['aruco']['bits_marker_size'],
                               config['aruco']['total_markers'])

if config["depth_estimation"]["enable"]:
    depth_estimator = DepthEstimator()

if config["pose_estimation"]["enable"]:
    pose_estimator = PoseEstimator(
        use_poseviz=config["pose_estimation"]["poseviz"])

time.sleep(2)
tic = 0
while True:
    if config['print_fps']:
        fps = 1 / (time.time() - tic)
        tic = time.time()
        print("FPS: {}".format(fps))

    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find aruco markers
    aruco_markers = aruco_detector.find_aruco_markers(frame)

    # Estimate depth
    if config["depth_estimation"]["enable"]:
        disparity_map, scaled_disparity_map = \
            depth_estimator.predict(rgb_frame)

    if config["depth_estimation"]["enable"] and config["depth_estimation"]["show_depth_window"]:
        cv2.imshow("Depth estimation", scaled_disparity_map)

    # Pose estimation
    if config["pose_estimation"]["enable"]:
        pred_poses = pose_estimator.predict(rgb_frame)
        chest_points_2d, chest_points_3d = pose_estimator.get_chest(pred_poses)

        if config["pose_estimation"]["draw_2d"]:
            frame = pose_estimator.draw2D(pred_poses, frame)

            for i, human_point in enumerate(chest_points_2d):
                # Draw the human chest point
                frame = cv2.circle(frame, human_point, radius=8,
                                   color=(0, 0, 255), thickness=-1)
                # Put human "ID" near the chest point
                cv2.putText(frame, str(i), (human_point[0] - 5, human_point[1] - 5),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    log = {}
    bbox, ids, rvecs, tvecs = aruco_markers

    if len(bbox) != 0:
        xyz, roll_pitch_yaw = ArucoDetector.get_aruco_positions(rvecs, tvecs)

        num = 1
        # Calculate distance between ArUco markers
        for i in range(len(bbox)):
            for j in range(i+1, len(bbox)):
                aruco_point1 = np.array(xyz[i])
                aruco_point2 = np.array(xyz[j])
                aruco_dist1 = get_dist(aruco_point1)
                aruco_dist2 = get_dist(aruco_point2)

                angle = angle_between(aruco_point1, aruco_point2)
                distance_beetween_markers = distance_beetween2objects(
                    aruco_dist1, aruco_dist2, angle)
                cv2.putText(frame, "Dist between aruco {} and {}: {:.2f}".format(ids[i][0], ids[j][0], distance_beetween_markers),
                            (10, int(frame.shape[0] - num*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                num += 1
                log[f"aruco_{ids[i][0]}_{ids[j][0]}"] = distance_beetween_markers

        if config["pose_estimation"]["enable"] and config["depth_estimation"]["enable"]:
            aruco_point = np.mean(bbox[0][0], 0)
            aruco_point = (round(aruco_point[0]), round(aruco_point[1]))
            depth_aruco = scaled_disparity_map[aruco_point[1], aruco_point[0]]
            aruco_dist = get_dist(xyz[0])
            log[f"depth_aruco_{ids[0][0]}"] = depth_aruco

            human_dists = []
            # Calculate distance to human
            for i, human_point in enumerate(chest_points_2d):
                human_point_x = min(frame.shape[1] - 1, max(0, human_point[0]))
                human_point_y = min(frame.shape[0] - 1, max(0, human_point[1]))
                depth_human = scaled_disparity_map[human_point_y,
                                                   human_point_x]
                human_dist = aruco_dist * depth_aruco / depth_human
                human_dists.append(human_dist)
                cv2.putText(frame, "Dist to human {}: {:.2f}".format(i, human_dist), (frame.shape[1] - 300, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 255, 0), 2)
                log[f"dist_to_human_{i}"] = human_dist
                log[f"depth_human_{i}"] = depth_human

            num = 2
            # Calculate distance between humans
            for i in range(len(chest_points_3d)):
                for j in range(i+1, len(chest_points_3d)):
                    human_point1 = np.array(chest_points_3d[i])
                    human_point2 = np.array(chest_points_3d[j])
                    angle = angle_between(human_point1, human_point2)
                    distance_beetween_humans = distance_beetween2objects(
                        human_dists[i], human_dists[j], angle)
                    cv2.putText(frame, "Dist between human {} and {}: {:.2f}".format(i, j, distance_beetween_humans),
                                (frame.shape[1] - 500, int(frame.shape[0] - num*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
                    num += 1
                    log[f"dist_between_human_{i}_{j}"] = distance_beetween_humans

        # Draw Aruco
        for i in range(len(bbox)):
            aruco_detector.draw_arucos(frame, aruco_markers)
            aruco_dist = get_dist(xyz[i])
            cv2.putText(frame, "Dist to aruco {}: {:.2f}".format(ids[i][0], aruco_dist), (10, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (255, 0, 0), 2)
            log[f"dist_to_aruco_{ids[i][0]}"] = aruco_dist

        logger.info(log)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if not ret or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
