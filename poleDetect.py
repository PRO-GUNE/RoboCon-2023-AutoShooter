import cv2
import freenect
import numpy as np

def trasformDistance(x):
    y = (0.000191945244034458*(x**3)) - (0.452976139846793*(x**2)) + (359.366764823288*x) - 94561.8031149121
    return y

# Initialize OpenCV windows
cv2.namedWindow('Kinect Color Stream', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Hand Detection', cv2.WINDOW_AUTOSIZE)

# Color thresholds
lower_color_bound = np.array([0,0,200], dtype=np.uint8)
upper_color_bound = np.array([180,30,255], dtype=np.uint8)

while True:
    # Capture frames
    rgb_frame, _ = freenect.sync_get_video()
    depth_frame, _ = freenect.sync_get_depth()

    # Convert frame to numpy arrays
    color_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    depth_image = depth_frame.astype(np.uint64)

    # Apply color segmentation to detect ball
    hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color_bound, upper_color_bound)

    # Find circular objects using Hough Circle Transform
    circular_objects = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=50, maxRadius=150)

    # Iterate through the contours and calculate the center and depth
    if circular_objects is not None:
        # Filter small contours
        circular_objects = np.uint16(np.around(circular_objects))
        for circle in circular_objects[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            if center[0] < 640 and center[1] < 480:
                # Get the corresponding depth from the depth frame
                depth = depth_image[center[1], center[0]]

                # Display depth above the bounding box (in mm)
                distance = trasformDistance(depth)
                cv2.putText(color_image, f"Distance: {distance: .2f} mm", (center[0], center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Draw the bounding circle
                cv2.circle(color_image, center, radius, (0,255,0), 2)

    # Display frames
    cv2.imshow('Kinect Color Stream', color_image)
    cv2.imshow('Hand Detection', mask)

    # Exit loop on key press
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()