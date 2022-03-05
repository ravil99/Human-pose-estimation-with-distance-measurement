import cv2
import numpy as np
import yaml
import time
import torch
from custom_video_capture import CustomVideoCapture
from aruco_detector.aruco_detector import findArucoMarkers, arucoIndex

with open("config.yaml") as file:
    config = yaml.full_load(file)

cap = CustomVideoCapture(0)
time.sleep(2)
frame_number = 0
tic = 0
while True:
    if config['print_fps']:
        fps = 1 / (time.time() - tic)
        tic = time.time()
        print("FPS: {}".format(fps))

    ret, frame = cap.read()
    arucoFound = findArucoMarkers(frame, frame_number)



    #Draw Aruco
    if len(arucoFound[0]) != 0:
        for bbox, id in zip(arucoFound[0], arucoFound[1]):
            frame = arucoIndex(bbox, id, frame)

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if not ret or key == ord('q'):
        break
    
    frame_number += 1

cap.release()
cv2.destroyAllWindows()