import cv2
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os
import time
from filterpy.kalman import KalmanFilter  # Correct import with the right case

# Define the coordinates for the sleeper polygon
sleeper_coordinates = [
    (15, 479),   # Bottom-left corner
    (626, 479),  # Bottom-right corner
    (518, 118),  # Top-right corner
    (126, 123)   # Top-left corner
]

def define_polygonal_roi(frame, shape="trapezium"):
    height, width = frame.shape[:2]

    if shape == "trapezium":
        # Define vertices for a horizontal trapezoidal ROI
        vertices = np.array([[
            (50, height),               # Bottom-left corner
            (width - 50, height),       # Bottom-right corner
            (width // 2 + 100, height // 3 + 100),  # Top-right corner (wider)
            (width // 2 - 100, height // 3 + 100)   # Top-left corner (wider)
        ]], dtype=np.int32)

    return vertices

def is_object_in_polygonal_roi(box, roi_vertices):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    
    # Check if any of the bounding box points are inside the ROI
    for point in points:
        if cv2.pointPolygonTest(roi_vertices, point, False) >= 0:
            return True
    return False

def initialize_realsense():
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align

def save_frame_image(frame, output_folder, count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(output_folder, f"detected_object_{count}.jpg")
    cv2.imwrite(filename, frame)

def blackout_outside_polygon(frame, roi_vertices, sleeper_vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    sleeper_mask = np.zeros_like(frame)
    cv2.fillPoly(sleeper_mask, sleeper_vertices, (255, 255, 255))
    final_mask = cv2.bitwise_or(mask, sleeper_mask)
    masked_frame = cv2.bitwise_and(frame, final_mask)

    return masked_frame

# Initialize YOLO model
model = YOLO("/home/raillabs/labelImg/ultralytics/runs/detect/train7/weights/best.pt")
model.to('cuda')

# Class names for YOLO
classNames = ["Human Obstacle", "Animal Obstacle", "Obstacle"]

# Define trapezoidal ROI vertices
roi_vertices = define_polygonal_roi(np.zeros((480, 640, 3), dtype=np.uint8), shape="trapezium")
sleeper_vertices = np.array([sleeper_coordinates], dtype=np.int32)

# Initialize RealSense
pipeline, align = initialize_realsense()

# Define output folder path
output_folder = '/media/raillabs/16 GB/newblackout_images'
image_counter = 0
last_save_time = 0
save_interval = 1

# Initialize Kalman Filter
kalman = KalmanFilter(dim_x=4, dim_z=2)  # State vector: [x, y, dx/dt, dy/dt]

# Define the initial state
kalman.x = np.array([[0], [0], [0], [0]])  # [x, y, dx/dt, dy/dt]

# Define the state transition matrix (A)
kalman.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

# Define the measurement function (H)
kalman.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

# Define the measurement noise covariance (R)
kalman.R = np.eye(2) * 10

# Define the process noise covariance (Q)
kalman.Q = np.eye(4) * 0.1

# Define the initial uncertainty (P)
kalman.P = np.eye(4) * 500


# Initialize flags
flag1 = 1  # No objects detected
flag2 = 2  # Object detected outside trapezoidal ROI
flag3 = 3  # Object detected inside trapezoidal ROI

# Background subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize RealSense
pipeline, align = initialize_realsense()

while True:
    # Capture frames from RealSense
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    # Apply background subtraction
    fgmask = bg_subtractor.apply(color_image)

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(color_image, (5, 5), 0)

    # Perform object detection
    results = model.predict(img_blur, stream=True, conf=0.75)

    # Reset flags for each frame
    flag1 = 1
    flag2 = 2
    flag3 = 3

    for i, r in enumerate(results):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Kalman filter prediction and correction
            kalman.predict()
            kalman.update(np.array([cx, cy], dtype=np.float32))
            pred_cx, pred_cy = kalman.x[:2]

            # Determine if the object is inside the ROI
            if is_object_in_trapezoidal_roi(box, roi_vertices):
                color = (0, 0, 255)  # Red color for inside ROI
                flag2 = 0  # Reset flag2 since an object is inside ROI
                flag3 = 3  # Set flag3 since object is inside ROI
            else:
                color = (255, 0, 0)  # Blue color for outside ROI
                flag3 = 0  # Reset flag3 since an object is outside ROI
                flag2 = 2  # Set flag2 since object is outside ROI

            flag1 = 0  # Set flag1 since an object is detected

            # Draw bounding box on color image
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

            # Calculate distance to the object
            depth_value = depth_image[cy, cx]  # Depth value at the center of the bounding box
            distance = depth_value * 0.001  # Convert to meters (depth values are in millimeters)

            # Display class name, confidence, and distance on color image
            confidence = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]}: {confidence} - Dist: {distance:.2f}m"
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display distance on depth colormap
            cv2.putText(depth_colormap, f"Dist: {distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the ROI boundary for visualization
    cv2.polylines(color_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the frames
    cv2.imshow('Color Image with ROI', color_image)
    cv2.imshow('Depth Image', depth_colormap)

    # Print corresponding outputs based on flags
    if flag1 == 1:
        print("1")
    elif flag2 == 2:
        print("2")
    elif flag3 == 3:
        print("3")

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
pipeline.stop()
cv2.destroyAllWindows()
