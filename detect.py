import cv2
import freenect
import numpy as np

# Initialize OpenCV windows
cv2.namedWindow('Kinect Color Stream', cv2.WINDOW_NORMAL)
cv2.namedWindow('Kinect Depth Stream', cv2.WINDOW_NORMAL)

while True:
    # Capture frames
    rgb_frame, _ = freenect.sync_get_video()
    depth_frame, _ = freenect.sync_get_depth()

    # Convert frame to numpy arrays
    color_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    depth_image = depth_frame.astype(np.uint8)

    # Display color and depth frames
    cv2.imshow('Kinect Color Stream', color_image)
    cv2.imshow('Kinect Depth Stream', depth_image)

    # Exit loop on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()