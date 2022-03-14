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
"""

INTRINSIC_PARAMS_FILENAME = 'intrinsic_params.yaml'


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

    return roll_x, pitch_y, yaw_z  # in radians


class ArucoMarker:
    def __init__(self, bbox, id, rvec, tvec):
        """ArUco marker instance

        Args:
            bbox (array): 4 points of ArUco corners
            id (int): ArUco marker id
            rvec (array): rotation vector
            tvec (array): translation vector
        """
        self.bbox = bbox
        self.id = id
        self.rvec = rvec
        self.tvec = tvec
        self.xyz, self.roll_pitch_yaw = \
            ArucoMarker.get_aruco_position(rvec, tvec)

        self.dist = self.__get_distance(self.xyz)

        # Center point of the marker in an image
        center_point = np.mean(bbox[0], 0)
        self.center_point = (round(center_point[0]), round(center_point[1]))

    def __get_distance(self, position):
        """Calculates ArUco marker distance wrt to camera

        Args:
            position (array): point in 3d space

        Returns:
            float: ArUco marker distance
        """
        return (position[0]**2 + position[1]**2 + position[2]**2)**0.5

    def get_aruco_position(rvec, tvec):
        """Calculates ArUco marker position and orientation wrt to camera

        Args:
            rvec (array): rotation vector
            tvec (array): translation vector

        Returns:
            array: position and orientation marker wrt to camera
        """

        # Store the translation (i.e. position) information
        transform_translation_x = tvec[0][0]
        transform_translation_y = tvec[0][1]
        transform_translation_z = tvec[0][2]
        xyz = [transform_translation_x,
               transform_translation_y, transform_translation_z]

        # Store the rotation information
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(
            np.array(rvec[0]))[0]
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
        roll_pitch_yaw = [roll_x, pitch_y, yaw_z]

        return [xyz, roll_pitch_yaw]

    def get_id(aruco_markers, id):
        """Get ArUco marker with specific ID

        Args:
            aruco_markers (array): array of ArucoMarker instances
            id (int): ArUco marker ID

        Raises:
            ValueError: if there is no ArUco marker with such ID

        Returns:
            ArucoMarker: ArucoMarker instance
        """
        for aruco_marker in aruco_markers:
            if aruco_marker.id == id:
                return aruco_marker
        raise ValueError(f'ArUco marker with id {id} is not found.')


class ArucoDetector:

    def __init__(self, aruco_size, marker_size=4, total_markers=50):
        """Initialization of ArUco marker detector

        Args:
            aruco_size (float, required): Size in meters of a square side of ArUco marker. 
            marker_size (int, optional): The 2D bit size of the ArUco marker. Defaults to 4.
            total_markers (int, optional): The total marker size of the dictionary. Defaults to 50.
        """

        # Load the camera parameters from the saved file
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        intrinsic_params_file = os.path.join(
            cur_dir, INTRINSIC_PARAMS_FILENAME)
        cv_file = cv2.FileStorage(intrinsic_params_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = cv_file.getNode('camera_matrix').mat()
        self.distortion_coefficients = cv_file.getNode(
            'distortion_coefficients').mat()
        cv_file.release()

        try:
            dict_name = f'DICT_{marker_size}X{marker_size}_{total_markers}'
            aruco_dict = getattr(aruco, dict_name)
        except AttributeError:
            print(f"There is no {dict_name} dictionary in ArUco OpenCV module")
            raise
        self.aruco_dict = aruco.Dictionary_get(aruco_dict)
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_size = aruco_size

    def find_aruco_markers(self, frame):
        """ This function detects ArUco markers and it's ID
        Args:
            frame (2D array): image

        Returns:
            array: [bounding boxes, ids, rotation vectors, translation vectors]
        """
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox, ids, rejected = aruco.detectMarkers(imgGray, self.aruco_dict, parameters=self.aruco_params,
                                                  cameraMatrix=self.camera_matrix, distCoeff=self.distortion_coefficients)

        rvecs = None
        tvecs = None
        if ids is not None:
            # Get the rotation and translation vectors
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(bbox, self.aruco_size,
                                                                           self.camera_matrix, self.distortion_coefficients)

        aruco_markers = []
        for i in range(len(bbox)):
            aruco_marker = ArucoMarker(bbox[i], ids[i][0], rvecs[i], tvecs[i])
            aruco_markers.append(aruco_marker)

        return aruco_markers

    def draw_arucos(self, frame, aruco_markers):
        """
          Draw bounding boxes and axes on detected ArUco markers
          and print the respective ID near the detected marker
        """

        for aruco_marker in aruco_markers:
            aruco.drawDetectedMarkers(frame, [aruco_marker.bbox])
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.distortion_coefficients,
                               aruco_marker.rvec, aruco_marker.tvec, 0.05)
            # Draw index
            # Get top left point from bbox
            point = (int(aruco_marker.bbox[0][0][0]),
                     int(aruco_marker.bbox[0][0][1]))
            cv2.putText(frame, str(aruco_marker.id), point, cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)


if __name__ == "__main__":

    # Creating and configuring logger
    Log_Format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename="logfile.log", filemode="a",
                        format=Log_Format, level=20)
    logger = logging.getLogger()
    logger.info("Let's start an experiment!!!")

    aruco_detector = ArucoDetector(aruco_size=0.094,
                                   marker_size=6, total_markers=50)
    cap = cv2.VideoCapture(0)
    frame_number = 0

    while True:
        success, frame = cap.read()

        aruco_markers = aruco_detector.find_aruco_markers(frame)
        aruco_detector.draw_arucos(frame, aruco_markers)

        frame_number = frame_number + 1
        cv2.imshow("Image", frame)

        # Exit if "d" is pressed on the keyboard
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

        # Logging
        for aruco_marker in aruco_markers:
            log = {'Frame №': str(frame_number),
                   'Marker id': str(aruco_marker.id),
                   'Distance to the camera': "{:.5f}".format(aruco_marker.dist),
                   'roll_x': "{:.5f}".format(aruco_marker.roll_pitch_yaw[0]),
                   'pitch_y': "{:.5f}".format(aruco_marker.roll_pitch_yaw[1]),
                   'yaw_z': "{:.5f}".format(aruco_marker.roll_pitch_yaw[2])}

            logger.info(log)
