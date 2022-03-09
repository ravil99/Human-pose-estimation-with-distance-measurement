import argparse
import cv2
from cv2 import aruco
import os

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description="ArUCo markers generation script")
parser.add_argument("--id", type=int, required=True,
                    help="ID of ArUCo marker to generate")

parser.add_argument("--marker_size", type=int, default=4,
                    help="The 2D bit size of the ArUco marker")

parser.add_argument("--total_markers", type=int, default=50,
                    help="The total marker size of the dictionary")

parser.add_argument("--pixel_size", type=int, default=700,
                    help="Size of the generated ArUCo marker image in pixels")

args = parser.parse_args()

# Get ArUCo dictionary from OpenCV
try:
    dict_name = f'DICT_{args.marker_size}X{args.marker_size}_{args.total_markers}'
    aruco_dict = getattr(aruco, dict_name)
except AttributeError:
    print(f"There is no {dict_name} dictionary in ArUco OpenCV module")
    raise

# load the ArUCo dictionary
aruco_dict = aruco.Dictionary_get(aruco_dict)

# Generate ArUCo marker image
aruco_marker = aruco.drawMarker(aruco_dict, args.id, args.pixel_size)

# Save ArUCo image
folder = 'ArUCo_markers'
if not os.path.exists(folder):
    os.makedirs(folder)
filename = f'ArUCo_{dict_name}_id_{args.id}.png'
cv2.imwrite(os.path.join(folder, filename), aruco_marker)
