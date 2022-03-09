import argparse
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

INTRINSIC_PARAMS_FILENAME = 'intrinsic_params.yaml'

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(
    description="""Callibration camera script. Starts a video stream from camera to collect
    data for callibration until n frames are collected or d button is pressed""")
parser.add_argument("--number_of_squares_x", type=int, default=10,
                    help="Chessboard dimension along x-axis")
parser.add_argument("--number_of_squares_y", type=int, default=8,
                    help="Chessboard dimension along y-axis")
parser.add_argument("--square_size", type=float, default=0.016,
                    help="Size in meters of a square side of the chessboard cell")
parser.add_argument("--number_of_frames", type=int, default=100,
                    help="Number of collected frames for callibration")
parser.add_argument("--pick_every", type=int, default=3,
                    help="Pick every n frame from video stream for calibration dataset.")
parser.add_argument("--display_images", type=int, default=False,
                    help="Display callibrated images. Used for testing.")

args = parser.parse_args()

print("Start to collect data")
print("Press d to stop")

dataset_folder = 'Calibration_dataset'
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# capture = cv2.VideoCapture(0)

# i = 0
# frame_num = 0
# while frame_num < args.number_of_frames:
#     ret, img = capture.read()
#     cv2.imshow('Calibration', img)

#     if i % args.pick_every == 0:
#         cv2.imwrite(os.path.join(dataset_folder, f'frame_{frame_num}.png'), img)
#         frame_num += 1

#         if frame_num % 20 == 0:
#             print(f"{frame_num} images collected")

#     if cv2.waitKey(20) & 0xFF == ord('d'):
#         break

#     i+=1

# capture.release()
# cv2.destroyAllWindows()

print("Finish to collect data")

nX = args.number_of_squares_x - 1  # Number of interior corners along x-axis
nY = args.number_of_squares_y - 1  # Number of interior corners along y-axis

# Set termination criteria. We stop either when an accuracy is reached or when
# we have finished a certain number of iterations.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define real world coordinates for points in the 3D coordinate frame
# Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)
object_points_3D[:, :2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2)
object_points_3D = object_points_3D * args.square_size

object_points = []
image_points = []

# Get the file path for images in the current directory
images = glob.glob(f'{dataset_folder}/*.png')

# Go through each chessboard image, one by one
print("Find chessboard corners")
for image_file in tqdm(images):

    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)

    if success == True:

        object_points.append(object_points_3D)
        corners_2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners_2)

        if args.display_images:
            # Draw the corners
            cv2.drawChessboardCorners(image, (nY, nX), corners_2, success)
            # Display the image. Used for testing.
            cv2.imshow("Image", image)
            cv2.waitKey(1000)

# Perform camera calibration to return the camera matrix, distortion coefficients, rotation and translation vectors etc
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                   image_points,
                                                   gray.shape[::-1],
                                                   None,
                                                   None)

# Save parameters to a YAML file
cv_file = cv2.FileStorage(INTRINSIC_PARAMS_FILENAME, cv2.FILE_STORAGE_WRITE)
cv_file.write('camera_matrix', mtx)
cv_file.write('distortion_coefficients', dist)
cv_file.release()

# Load the parameters from the saved file
cv_file = cv2.FileStorage(INTRINSIC_PARAMS_FILENAME, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode('camera_matrix').mat()
dst = cv_file.getNode('distortion_coefficients').mat()
cv_file.release()

# Display key parameter outputs of the camera calibration process
print("Camera matrix:")
print(mtx)

print("\n Distortion coefficients:")
print(dst)

print(f"Saved to {INTRINSIC_PARAMS_FILENAME}")

# Close all windows
cv2.destroyAllWindows()
