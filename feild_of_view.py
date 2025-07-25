import pyrealsense2 as rs
import math

# Initialize the pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Get stream profile and intrinsic parameters
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)  # Fetch depth stream
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# Field of view (FOV) in degrees
fov_x = math.degrees(2 * math.atan2(intrinsics.width / 2, intrinsics.fx))
fov_y = math.degrees(2 * math.atan2(intrinsics.height / 2, intrinsics.fy))

print(f"Horizontal FOV: {fov_x} degrees")
print(f"Vertical FOV: {fov_y} degrees")

pipeline.stop()

