import cv2
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import os
import time

# Define the coordinates for the sleeper polygon
sleeper_coordinates = [  
    (15, 479),   # Bottom-left corner
    (626, 479),  # Bottom-right corner
    (518, 118),  # Top-right corner
    (126, 123)   # Top-left corner
]

def define_polygonal_roi(frame, shape="trapezium"):
    height, width = frame.shape[:2]
    bottom_margin = 0  # Align bottom of the trapezium with the bottom of the frame
    
    # Adjust this value based on how wide the bottom should be
    margin = 50  # You can modify this value for your needs
    
    if shape == "trapezium":
        # Define vertices for a trapezoidal ROI (track)
        vertices = np.array([[
            (margin, height - bottom_margin),  # Bottom-left corner
            (width - margin, height - bottom_margin),  # Bottom-right corner
            (width // 2 + 150, height // 3),  # Top-right corner
            (width // 2 - 150, height // 3)  # Top-left corner
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
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Save the image
    filename = os.path.join(output_folder, f"detected_object_{count}.jpg")
    cv2.imwrite(filename, frame)

def blackout_outside_polygon(frame, roi_vertices, sleeper_vertices):
    # Create a mask the same size as the frame
    mask = np.zeros_like(frame)

    # Fill the trapezoidal ROI with white color (255)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))

    # Create a mask for the sleeper polygon
    sleeper_mask = np.zeros_like(frame)
    cv2.fillPoly(sleeper_mask, sleeper_vertices, (255, 255, 255))

    # Combine masks to create the final mask
    final_mask = cv2.bitwise_or(mask, sleeper_mask)
    
    # Apply the combined mask to the frame
    masked_frame = cv2.bitwise_and(frame, final_mask)

    return masked_frame

# Initialize YOLO model
model = YOLO("/home/raillabs/labelImg/ultralytics/runs/detect/train7/weights/best.pt")
model.to('cuda')  # Use 'cuda' for GPU, 'cpu' for CPU

# Class names for YOLO
classNames = ["Human Obstacle", "Animal Obstacle", "Obstacle"]

# Define trapezoidal ROI vertices once outside the loop
roi_vertices = define_polygonal_roi(np.zeros((480, 640, 3), dtype=np.uint8), shape="trapezium")

# Define sleeper polygon vertices using provided coordinates
sleeper_vertices = np.array([sleeper_coordinates], dtype=np.int32)

# Initialize Kalman Filter
kalman = KalmanFilter(dim_x=4, dim_z=2)
kalman.transition_matrices = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.measurement_matrices = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.process_noise_covariance = 0.03 * np.eye(4, dtype=np.float32)
kalman.measurement_noise_covariance = 1e-1 * np.eye(2, dtype=np.float32)

# Initialize RealSense
pipeline, align = initialize_realsense()

# Define output folder path
output_folder = '/media/raillabs/16 GB/newblackout_images'
image_counter = 0  # Counter for image filenames

# Initialize variables for saving images
last_save_time = 0  # To track the last save time
save_interval = 1  # Interval in seconds between saves

while True:
    # Capture frames from RealSense
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    
    # Blackout the areas outside the trapezoidal ROI and sleeper polygon
    blackout_image = blackout_outside_polygon(color_image, roi_vertices, sleeper_vertices)

    # Perform object detection on the blacked-out image
    results = model.predict(blackout_image, stream=True, conf=0.75)

    for i, r in enumerate(results):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Determine if the object is inside the ROI
            if is_object_in_polygonal_roi(box, roi_vertices):
                color = (0, 0, 255)  # Red for inside ROI
            else:
                color = (255, 0, 0)  # Blue for outside ROI

            # Draw bounding box on blackout image
            cv2.rectangle(blackout_image, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(blackout_image, (cx, cy), 5, (0, 255, 0), -1)

            # Display class name and confidence on blackout image
            confidence = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]}: {confidence}"
            cv2.putText(blackout_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Logic to stop or slow down trolley
            if color == (0, 0, 255):  # Red box indicates inside the ROI
                print("Object inside the red bounding box. Stop the trolley.")
                # Add your stopping logic here
            elif color == (255, 0, 0):  # Blue box indicates outside the ROI
                print("Object outside the ROI. Slow down the trolley.")
                # Add your slowing down logic here

    # Draw the ROI boundary for visualization
    cv2.polylines(blackout_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)  # Trapezoidal ROI
    cv2.polylines(blackout_image, [sleeper_vertices], isClosed=True, color=(255, 255, 255), thickness=2)  # Sleeper Polygon

    # Display the result with the blackout applied
    cv2.imshow('Blackout Image with ROI', blackout_image)

    # Save image if needed
    current_time = time.time()
    if current_time - last_save_time >= save_interval:
        save_frame_image(blackout_image, output_folder, image_counter)
        last_save_time = current_time
        image_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()

