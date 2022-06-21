# HandGesture_Kinect

Tools

  1) Python
  2) OpenCV
  3) Norfair
  4) Pykinect2
  5) Pygame
  6) Pyautogui
 
Gestures Implemented
  1) Zoom in/out
  2) Click
  3) Panning
  4) Movement of the cursor

Implementation Details:
  A hand gesture implementation was made based on the contours obtained with infrared and depth frames taken from the camera looking at the table from above and at a distance of about 1.5 meters.
  
The summary of the steps that followed:
- In order not to be affected by the projection light, an infrared image is used and then it is binarized such that the bigger portion in the image is made white and the rest remains black. 
- In these pictures, firstly a convex hull is found and then convexity defects of this hull are obtained. Convexity defects are the points farthest from the convex points. 
- By using these points and measuring the angle two types of points are obtained: far points and peak points. While peak points are in places like fingertips, far points are located near to the palm. 
- To differentiate between the peak points that are found in the body and the hand, depth image is used.
- To get information of the direction that the people stands in the image the differences between x and y coordinates and the binary image are analyzed.
- To track the detected part (hand), norfair is used.
