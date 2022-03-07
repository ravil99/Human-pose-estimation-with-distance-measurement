import cv2 
import numpy as np 
import glob 

# Chessboard dimensions
number_of_squares_X = 10 
number_of_squares_Y = 8 
nX = number_of_squares_X - 1 # Number of interior corners along x-axis
nY = number_of_squares_Y - 1 # Number of interior corners along y-axis
square_size = 0.016 # Size, in meters, of a square side 
  
# Set termination criteria. We stop either when an accuracy is reached or when
# we have finished a certain number of iterations.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
 
# Define real world coordinates for points in the 3D coordinate frame
# Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)  
object_points_3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) 
object_points_3D = object_points_3D * square_size
 
object_points = []
  
image_points = []
  
def main():

  capture = cv2.VideoCapture(0)
    
  for i in range(300):
    ret, img = capture.read()
    cv2.imshow('Calibration', img)
    cv2.imwrite(f'Calibration_dataset/frame_{i}.png', img)
    i+=1
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

  capture.release()
  cv2.destroyAllWindows

  # Get the file path for images in the current directory
  images = glob.glob('Calibration_dataset/*.png')
      
  # Go through each chessboard image, one by one
  for image_file in images:
   
    image = cv2.imread(image_file)  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)
      
    if success == True:
  
      object_points.append(object_points_3D)
      corners_2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)       
      image_points.append(corners_2)
  
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

  # Saving .npy to a zip
  np.savez('Parameters', matrix=mtx, distortion=dist,
         rotation=rvecs, translation=tvecs)
 
  # Save parameters to a YAML file
  cv_file = cv2.FileStorage('calibration_chessboard.yaml', cv2.FILE_STORAGE_WRITE)
  cv_file.write('K', mtx)
  cv_file.write('D', dist)
  cv_file.release()
  
  # Load the parameters from the saved file
  cv_file = cv2.FileStorage('calibration_chessboard.yaml', cv2.FILE_STORAGE_READ) 
  mtx = cv_file.getNode('K').mat()
  dst = cv_file.getNode('D').mat()
  cv_file.release()
   
  # Display key parameter outputs of the camera calibration process
  print("Camera matrix:") 
  print(mtx) 
  
  print("\n Distortion coefficient:") 
  print(dist) 
    
  # Close all windows
  cv2.destroyAllWindows() 
      
if __name__ == '__main__':
  print(__doc__)
  main()
