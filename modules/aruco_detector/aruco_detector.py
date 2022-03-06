import cv2 
from cv2 import aruco
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import logging
import os 

"""
    This module detects ArUco markers and it's ID and
  calculates it's distance and orientation wrt to camera.
  All the information about markers is aslo being logged.
  On Ravil's machine works at 15 FPS.          
"""

#Creating and Configuring Logger

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = "logfile.log",
                    filemode = "w",
                    format = Log_Format, 
                    level = 20)

logger = logging.getLogger()
cur_dir = os.path.dirname(os.path.realpath(__file__))

def euler_from_quaternion(x, y, z, w):
    
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """

  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z # in radians

def findArucoMarkers(img, frame_number, aruco_marker_side_length, markerSize = 6, totalMarkers = 250, draw = True, drawID = True):


    """
      This function detects ArUco markers and it's ID and
      calculates it's distance and orientation wrt to camera
    """

    # Please, write the size of the marker here:
    # aruco_marker_side_length = 0.038
    camera_calibration_parameters_filename = os.path.join(cur_dir, 'calibration_chessboard.yaml')

    # Load the camera parameters from the saved file
    cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bbox, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam, cameraMatrix=mtx, distCoeff=dst)
    aruco_dists = []

    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, bbox)

        # Get the rotation and translation vectors
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
            bbox, aruco_marker_side_length, mtx, dst)

        for i, id in enumerate(ids):

            # Store the translation (i.e. position) information
            transform_translation_x = tvecs[i][0][0]
            transform_translation_y = tvecs[i][0][1]
            transform_translation_z = tvecs[i][0][2]
            aruco_dist =  (transform_translation_x**2 + transform_translation_y**2 + transform_translation_z**2)**(0.5)
            aruco_dists.append(aruco_dist)

            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()   

            # Quaternion format     
            transform_rotation_x = quat[0] 
            transform_rotation_y = quat[1] 
            transform_rotation_z = quat[2] 
            transform_rotation_w = quat[3] 

            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                           transform_rotation_y, 
                                                           transform_rotation_z, 
                                                           transform_rotation_w)

            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)

            # Logging
            log = { 'Frame №' : str(frame_number),
                        'Marker number' : str(id),
                        'Distance to the camera' : str(format(transform_translation_z)),
                        'roll_x': str(format(roll_x)),
                        'pitch_y': str(format(pitch_y)),
                        'yaw_z' : str(format(yaw_z))}

            logger.info(log)

            # Draw the axes on the marker
            cv2.aruco.drawAxis(img, mtx, dst, rvecs[i], tvecs[i], 0.05)

    return [bbox, ids, aruco_dists]

def arucoIndex(bbox, id, img, drawID = True):


    """
      This function prints the respective ID near the detected marker
    """

    tl = int(bbox[0][0][0]), int(bbox[0][0][1])

    if drawID:
        cv2.putText(img, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, 
                    (255, 0, 255), 2)
    return img

def main():
    cap = cv2.VideoCapture(0)
    frame = 0

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img, frame)

        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                img = arucoIndex(bbox, id, img)

        cv2.imshow("Image", img)
        frame = frame + 1

        # exit if "d" is pressed on the keyboard
        if cv2.waitKey(20) & 0xFF==ord('d'):
            break

if __name__ == "__main__":
    main()
