import cv2
import yaml
import time
import tensorflow as tf
import logging

from utils.vector_math import angle_between, distance_beetween2objects
from utils.custom_video_capture import CustomVideoCapture
from modules.depth_estimator import DepthEstimator
from modules.aruco.aruco_detector import ArucoDetector, ArucoMarker
from modules.pose_estimator import PoseEstimator


log_format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="stats.log", filemode="a",
                    format=log_format, level=20)
logger = logging.getLogger()
logger.info("Start a new session")

# Read the config file
with open("config.yaml") as file:
    config = yaml.full_load(file)

# Init video stream
cap = CustomVideoCapture(0)

# Init aruco detector
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

        if config["depth_estimation"]["show_depth_window"]:
            cv2.imshow("Depth estimation", scaled_disparity_map)

    # Pose estimation
    if config["pose_estimation"]["enable"]:
        pred_poses = pose_estimator.predict(rgb_frame)
        human_points = pose_estimator.get_human_points(pred_poses)

        if config["pose_estimation"]["draw_2d"]:
            # Draw skeleton
            frame = pose_estimator.draw_2d_skeleton(pred_poses, frame)

            # Draw "human points" - points on the chest
            for i, human_point in enumerate(human_points):
                human_point.draw(frame, i)

    # Draw all detected ArUco markers
    aruco_detector.draw_arucos(frame, aruco_markers)

    # ---------------------------------------------
    # Calculate distances
    # ---------------------------------------------
    log = {}

    # Calculate and print distance between ArUco markers
    num = 1
    for i in range(len(aruco_markers)):
        for j in range(i+1, len(aruco_markers)):
            # Get angle between two vectors
            angle = angle_between(aruco_markers[i].xyz, aruco_markers[j].xyz)
            # Calculate distance
            distance_beetween_markers = distance_beetween2objects(
                aruco_markers[i].dist, aruco_markers[j].dist, angle)
            cv2.putText(frame, "Dist between aruco {} and {}: {:.2f}".format(aruco_markers[i].id, aruco_markers[j].id, distance_beetween_markers),
                        (10, int(frame.shape[0] - num*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            log[f"aruco_{aruco_markers[i].id}_{aruco_markers[j].id}"] = distance_beetween_markers
            num += 1

    # Calculate distances to humans and between humans if pose estimation and depth estimation is available
    if len(aruco_markers) != 0 and config["pose_estimation"]["enable"] and config["depth_estimation"]["enable"]:
        # Get reference ArUco marker from which we will calculate the distance to humans
        try:
            # Try to get marker with specific id
            reference_marker = ArucoMarker.get_id(aruco_markers, 4)
        except ValueError as e:
            print(e)
            reference_marker = aruco_markers[0]

        # Calculate distance to each human
        for i, human_point in enumerate(human_points):

            estimated_dist = DepthEstimator.get_distance(disparity_map, reference_marker.dist,
                                                         reference_marker.center_point, human_point.point_2d)
            human_point.estimated_dist = estimated_dist
            cv2.putText(frame, "Dist to human {}: {:.2f}".format(i, estimated_dist), (frame.shape[1] - 300, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (255, 255, 0), 2)
            log[f"dist_to_human_{i}"] = estimated_dist

        # Calculate distance between humans
        num = 2
        for i in range(len(human_points)):
            for j in range(i+1, len(human_points)):
                angle = angle_between(
                    human_points[i].point_3d, human_points[j].point_3d)
                distance_beetween_humans = distance_beetween2objects(
                    human_points[i].estimated_dist, human_points[j].estimated_dist, angle)
                cv2.putText(frame, "Dist between human {} and {}: {:.2f}".format(i, j, distance_beetween_humans),
                            (frame.shape[1] - 500, int(frame.shape[0] - num*20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
                log[f"dist_between_human_{i}_{j}"] = distance_beetween_humans
                num += 1

    # Draw ArUco markers distance info
    for i, aruco_marker in enumerate(aruco_markers):
        cv2.putText(frame, "Dist to aruco {}: {:.2f}".format(aruco_marker.id, aruco_marker.dist), (10, int((i+1)*20)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 0), 2)
        log[f"dist_to_aruco_{aruco_marker.id}"] = aruco_marker.dist

    cv2.imshow("Frame", frame)
    logger.info(log)

    key = cv2.waitKey(1)
    if not ret or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
