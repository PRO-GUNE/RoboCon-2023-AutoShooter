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
lower_color_bound = np.array([0,50,50], dtype=np.uint8)
upper_color_bound = np.array([20,255,255], dtype=np.uint8)

while True:
    # Capture frames
    rgb_frame, _ = freenect.sync_get_video()
    depth_frame, _ = freenect.sync_get_depth()

    # Convert frame to numpy arrays
    color_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    depth_image = depth_frame.astype(np.uint64)

    # Apply color segmentation to detect hand
    hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color_bound, upper_color_bound)

    # Find contours of the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and calculate the center and depth
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) > 100:
            # Get bounding rectangle and its center
            x,y,w,h = cv2.boundingRect(contour)
            center_x = x+w // 2
            center_y = y+h // 2

            # Get the corresponding depth from the depth frame
            depth = depth_image[center_y, center_x]

            # Display depth above the bounding box (in mm)
            distance = trasformDistance(depth)
            cv2.putText(color_image, f"Distance: {distance: .2f} mm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Draw bounding box around the hand
            cv2.rectangle(color_image, (x,y), (x+w, y+h), (0,255,0), 2)

    # Display frames
    cv2.imshow('Kinect Color Stream', color_image)
    cv2.imshow('Hand Detection', mask)

    # Exit loop on key press
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()