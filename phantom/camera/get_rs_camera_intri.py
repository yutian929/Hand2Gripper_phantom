import pyrealsense2 as rs
import json
import os
import argparse
import numpy as np

def get_realsense_intrinsics(width=640, height=480, fps=30):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start streaming
    print(f"Starting RealSense pipeline with resolution {width}x{height}...")
    profile = pipeline.start(config)

    try:
        # Get stream profile and intrinsics
        stream_profile = profile.get_stream(rs.stream.color)
        intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()

        # Calculate FOV
        h_fov = np.degrees(2 * np.arctan(intrinsics.width / (2 * intrinsics.fx)))
        v_fov = np.degrees(2 * np.arctan(intrinsics.height / (2 * intrinsics.fy)))
        
        # Diagonal FOV
        diagonal = np.sqrt(intrinsics.width**2 + intrinsics.height**2)
        f_avg = (intrinsics.fx + intrinsics.fy) / 2
        d_fov = np.degrees(2 * np.arctan(diagonal / (2 * f_avg)))

        # Distortion coefficients
        # RealSense usually provides 5 coefficients. 
        # We pad to 12 to match your requested format (if that's what downstream tools expect).
        disto = [float(c) for c in intrinsics.coeffs]
        if len(disto) < 12:
            disto.extend([0.0] * (12 - len(disto)))

        # Construct dictionary in the requested format
        intrinsics_dict = {
            "left": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.ppx,
                "cy": intrinsics.ppy,
                "disto": disto,
                "v_fov": float(v_fov),
                "h_fov": float(h_fov),
                "d_fov": float(d_fov)
            },
            "right": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.ppx,
                "cy": intrinsics.ppy,
                "disto": disto,
                "v_fov": float(v_fov),
                "h_fov": float(h_fov),
                "d_fov": float(d_fov)
            }
        }

        return intrinsics_dict

    finally:
        # Stop streaming
        pipeline.stop()

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Intrinsics saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get RealSense Camera Intrinsics")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--output", type=str, default="camera_intrinsics_d435i.json", help="Output JSON filename")
    
    args = parser.parse_args()
    
    try:
        intrinsics = get_realsense_intrinsics(width=args.width, height=args.height)
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, args.output)
        
        save_to_json(intrinsics, output_file)
        
        print("Intrinsics Content:")
        print(json.dumps(intrinsics, indent=4))
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure RealSense camera is connected.")
