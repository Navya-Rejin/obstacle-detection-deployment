import cv2
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import os
import time

def define_polygonal_roi(frame, shape="trapezium"):
    height, width = frame.shape[:2]
    bottom_margin = 0  # Align bottom of the trapezium with the bottom of the frame
    
    # Increase the margin to widen the trapezoid, especially the bottom part, to cover sleepers
    margin = 50  # Adjust this value based on how wide the bottom should be
    
    if shape == "trapezium":
        # Define vertices for a trapezoidal ROI with a steeper top
        vertices = np.array([[
            (margin, height - bottom_margin),  # Bottom-left corner (close to the left edge)
            (width - margin, height - bottom_margin),  # Bottom-right corner (close to the right edge)
            (width // 2 + 100, height // 4),  # Top-right corner (adjust this value to control width and steepness)
            (width // 2 - 100, height // 4)  # Top-left corner
        ]], dtype=np.int32)
    else:
        raise ValueError("Shape must be 'trapezium'")

    return vertices


def blackout_outside_roi(frame, vertices, blackout_margin=50):
    mask = np.zeros_like(frame)
    
    # Blackout the region just outside the trapezoid with a smaller margin
    margin_vertices = np.array([[
        (vertices[0][0][0] + blackout_margin, vertices[0][0][1] - blackout_margin),
        (vertices[0][1][0] - blackout_margin, vertices[0][1][1] - blackout_margin),
        (vertices[0][2][0] - blackout_margin, vertices[0][2][1] + blackout_margin),
        (vertices[0][3][0] + blackout_margin, vertices[0][3][1] + blackout_margin)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, margin_vertices, (255, 255, 255))
    blackout = cv2.bitwise_and(frame, mask)
    return blackout

def is_object_in_polygonal_roi(box, roi_vertices):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, cy)]

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

# Initialize YOLO model
model = YOLO("/home/raillabs/labelImg/ultralytics/runs/detect/train7/weights/best.pt")
model.to('cuda')

# Class names for YOLO
classNames = ["Human Obstacle", "Animal Obstacle", "Obstacle"]

roi_vertices = define_polygonal_roi(np.zeros((480, 640, 3), dtype=np.uint8), shape="trapezium")

# Initialize Kalman Filter
kalman = KalmanFilter(dim_x=4, dim_z=2)
kalman.transition_matrices = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.measurement_matrices = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.process_noise_covariance = 0.03 * np.eye(4, dtype=np.float32)
kalman.measurement_noise_covariance = 1e-1 * np.eye(2, dtype=np.float32)

flag1, flag2, flag3 = 1, 2, 3
object_detected_inside_roi = False

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

pipeline, align = initialize_realsense()
output_folder = '/media/raillabs/16 GB/widen_images'
image_counter = 0
last_save_time = 0
save_interval = 20

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    fgmask = bg_subtractor.apply(color_image)
    img_blur = cv2.GaussianBlur(color_image, (5, 5), 0)
    results = model.predict(img_blur, stream=True, conf=0.75)

    flag1, flag2, flag3 = 1, 2, 3
    object_detected_inside_roi = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            kalman.predict()
            kalman.update(np.array([cx, cy], dtype=np.float32))
            pred_cx, pred_cy = kalman.x[:2]

            if is_object_in_polygonal_roi(box, roi_vertices):
                color = (0, 0, 255)
                flag2 = 0
                flag3 = 3
                object_detected_inside_roi = True
            else:
                color = (255, 0, 0)
                flag3 = 0
                flag2 = 2

            flag1 = 0

            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

            depth_value = depth_image[cy, cx]
            distance = depth_value * 0.001

            confidence = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]}: {confidence} - Dist: {distance:.2f}m"
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(depth_colormap, f"Dist: {distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.polylines(color_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)

    # Apply blackout on a smaller region outside the ROI
    color_image = blackout_outside_roi(color_image, roi_vertices, blackout_margin=50)

    cv2.imshow('Color Image with ROI', color_image)
    cv2.imshow('Depth Image', depth_colormap)

    if flag1 == 1:
        print("1")
    elif flag2 == 2:
        print("2")
    elif flag3 == 3:
        print("3")

    if object_detected_inside_roi:
        current_time = time.time()
        if current_time - last_save_time > save_interval:
            save_frame_image(color_image, output_folder, image_counter)
            image_counter += 1
            last_save_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

