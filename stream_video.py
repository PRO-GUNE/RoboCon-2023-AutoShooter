import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import FrameSourceTypes

# Initialize Kinect sensor
kinect = PyKinectV2()
kinect.start()

# Initialize OpenCV windows for color and depth streams
cv2.namedWindow('Color Stream', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth Stream', cv2.WINDOW_NORMAL)

while True:
    # Get color and depth frames from Kinect
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        depth_frame = kinect.get_last_depth_frame()
        depth_frame = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
        
        # Display color and depth frames
        cv2.imshow('Color Stream', color_frame)
        cv2.imshow('Depth Stream', depth_frame)

    # Exit loop on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()
kinect.close()
