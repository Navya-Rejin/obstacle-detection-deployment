import cv2
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os
import time

# Define the coordinates for the sleeper polygon
sleeper_coordinates = [  
    (-15, 479),   # Bottom-left corner
    (650, 479),  # Bottom-right corner
    (518, 118),  # Top-right corner
    (116, 123)   # Top-left corner
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

def is_point_inside_polygon(x, y, roi_mask):
    # Check if the given point (x, y) is inside the polygon using the mask
    return roi_mask[y, x] == 255

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

# Initialize YOLO model
model = YOLO("/home/raillabs/labelImg/ultralytics/runs/detect/train7/weights/best.pt")
model.to('cuda')  # Use 'cuda' for GPU, 'cpu' for CPU

# Class names for YOLO
classNames = ["Human Obstacle", "Animal Obstacle", "Obstacle"]

# Define trapezoidal ROI vertices once outside the loop
roi_vertices = define_polygonal_roi(np.zeros((480, 640, 3), dtype=np.uint8), shape="trapezium")

# Define sleeper polygon vertices using provided coordinates
sleeper_vertices = np.array([sleeper_coordinates], dtype=np.int32)

# Create an empty mask for the ROI
roi_mask = np.zeros((480, 640), dtype=np.uint8)
cv2.fillPoly(roi_mask, [roi_vertices], 255)  # Fill ROI region with white

# Initialize RealSense
pipeline, align = initialize_realsense()

# Define output folder path
output_folder = '/media/raillabs/16 GB/new_images'
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
    
    # Perform object detection on the color image
    results = model.predict(color_image, stream=True, conf=0.75)

    for i, r in enumerate(results):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box

            # Check if the center of the object is inside the ROI
            if is_point_inside_polygon(cx, cy, roi_mask):
                color = (0, 0, 255)  # Red for inside ROI
                print("Object inside the red bounding box. Stop the trolley.")
                # Add your stopping logic here
            else:
                color = (255, 0, 0)  # Blue for outside ROI
                print("Object outside the ROI. Slow down the trolley.")
                # Add your slowing down logic here

            # Draw bounding box on the color image
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)

            # Draw center point
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

            # Display class name and confidence on color image
            confidence = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]}: {confidence}"
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw the ROI boundary for visualization
    cv2.polylines(color_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)  # Trapezoidal ROI
    cv2.polylines(color_image, [sleeper_vertices], isClosed=True, color=(255, 255, 255), thickness=2)  # Sleeper Polygon

    # Display the result
    cv2.imshow('Image with ROI', color_image)

    # Save image if needed
    current_time = time.time()
    if current_time - last_save_time >= save_interval:
        save_frame_image(color_image, output_folder, image_counter)
        last_save_time = current_time
        image_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()

