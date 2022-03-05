# ArUco markers, made by Ravil Akhmethzanov

    There are 5 files in this folder

* __aruco_generator.py__ - This modules generates arbitrary number of ArUco markers  .
* __YAML_camera_calibration.py__ - this module performs camera calibration.
* __calibration_chessboard.yaml__ - this file contains camera distortion vector and calibration matrix.
* __aruco_detector.py__ - this module detects ArUco markers and it's ID and calculates it's distance and orientation with respect to camera.
  All the information about markers is aslo being logged. On Ravil's machine works at 15 FPS. 
* __logfile.log__ - structure of the log:

    1) № of frame
    2) Markers ID
    3) Distance to the camera
    4) roll_x
    5) pitch_y   
    6) yaw_z

