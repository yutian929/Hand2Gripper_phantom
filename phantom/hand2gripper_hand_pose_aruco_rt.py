#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import cv2
import sys
from pathlib import Path
import trimesh
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import pyrealsense2 as rs
except Exception as e:
    raise RuntimeError("pyrealsense2 未安装或无法导入，请先安装 RealSense SDK + pyrealsense2。") from e

# Imports from phantom package
try:
    from phantom.detectors.detector_hamer import DetectorHamer
    from phantom.utils.pcd_utils import (
        get_visible_points, 
        get_pcd_from_points, 
        icp_registration, 
        get_point_cloud_of_segmask, 
        get_3D_points_from_pixels
    )
    from phantom.utils.transform_utils import transform_pts
except ImportError as e:
    print(f"Error importing phantom modules: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Warning: MediaPipe not found.")

# -----------------------------
# Hand skeleton connections (MediaPipe style 21 pts)
# -----------------------------
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]


# -----------------------------
# Utilities
# -----------------------------
def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


def robust_depth_at(depth_m, x, y, k=5, max_valid_m=3.0):
    """
    depth_m: float32 depth in meters, aligned to color
    Return robust depth value near (x,y) using median in a kxk window
    """
    h, w = depth_m.shape[:2]
    x = clamp_int(x, 0, w - 1)
    y = clamp_int(y, 0, h - 1)

    r = k // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)

    patch = depth_m[y0:y1, x0:x1].reshape(-1)
    patch = patch[np.isfinite(patch)]
    patch = patch[(patch > 0.0) & (patch < max_valid_m)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def deproject_pixel_to_3d(rs_intrinsics, x, y, depth_m):
    """
    Use RealSense intrinsics to deproject (x,y,depth) -> (X,Y,Z) in meters
    """
    pt = rs.rs2_deproject_pixel_to_point(rs_intrinsics, [float(x), float(y)], float(depth_m))
    return np.array(pt, dtype=np.float32)  # [X,Y,Z] meters

def get_intrinsics_as_dict(intr):
    """
    Convert RealSense intrinsics to a dictionary format for HaMeR.
    """
    return dict(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
        disto=[float(d) for d in intr.coeffs],
        v_fov=np.degrees(2 * np.arctan(intr.height / (2 * intr.fy))),
        h_fov=np.degrees(2 * np.arctan(intr.width / (2 * intr.fx))),
        d_fov=np.degrees(2 * np.arctan(intr.height / (2 * np.sqrt(intr.fx ** 2 + intr.fy ** 2))))
    )

def get_simple_mask(depth_m, bbox):
    """Generate a simple mask based on depth thresholding within bbox."""
    mask = np.zeros_like(depth_m, dtype=bool)
    x1, y1, x2, y2 = bbox
    
    # Ensure bbox is within bounds
    h, w = depth_m.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return mask

    # Crop depth
    depth_crop = depth_m[y1:y2, x1:x2]
    
    # Filter invalid depth
    valid_depth = depth_crop[depth_crop > 0]
    if len(valid_depth) == 0:
        return mask
        
    # Find median depth of the object (assuming hand is the primary object in bbox)
    median_d = np.median(valid_depth)
    
    # Threshold: +/- 15cm from median
    mask_crop = (depth_crop > (median_d - 0.15)) & (depth_crop < (median_d + 0.15))
    
    mask[y1:y2, x1:x2] = mask_crop
    return mask

def depth_alignment(hamer_out, depth_m, img_rgb, intrinsics_dict, detector_hamer):
    """
    Stronger depth alignment using ICP.
    """
    if hamer_out is None:
        return None, None

    # Create mesh
    mesh = trimesh.Trimesh(hamer_out["verts"].copy(), detector_hamer.faces_left.copy(), process=False)
    
    # Get mask from bbox and depth
    bbox = hamer_out.get('bbox')
    if bbox is None:
        return hamer_out["kpts_3d"], None
        
    mask = get_simple_mask(depth_m, bbox)
    
    # Get point cloud from depth
    pcd = get_point_cloud_of_segmask(mask, depth_m, img_rgb, intrinsics_dict, visualize=False)
    
    if len(pcd.points) < 10:
        return hamer_out["kpts_3d"], None

    # Ray casting for visibility
    visible_hamer_vertices, _ = get_visible_points(mesh, origin=np.array([0,0,0]))
    
    # Project to 2D to filter by image bounds
    visible_points_2d = detector_hamer.project_3d_kpt_to_2d(
        (visible_hamer_vertices - hamer_out["T_cam_pred"].cpu().numpy()).astype(np.float32),
        hamer_out["img_w"], hamer_out["img_h"], hamer_out["scaled_focal_length"],
        hamer_out["camera_center"], hamer_out["T_cam_pred"]
    )
    
    # Filter valid
    h, w = depth_m.shape
    valid_mask = (visible_points_2d[:, 0] >= 0) & (visible_points_2d[:, 0] < w) & \
                 (visible_points_2d[:, 1] >= 0) & (visible_points_2d[:, 1] < h)
    
    visible_points_2d = visible_points_2d[valid_mask]
    visible_hamer_vertices = visible_hamer_vertices[valid_mask]
    
    if len(visible_points_2d) == 0:
         return hamer_out["kpts_3d"], None

    # Get 3D points from depth
    visible_points_3d = get_3D_points_from_pixels(visible_points_2d, depth_m, intrinsics_dict)
    
    # Initial transform (Translation)
    translation = np.nanmedian(visible_points_3d - visible_hamer_vertices, axis=0)
    T_0 = np.eye(4)
    if not np.isnan(translation).any():
        T_0[:3, 3] = translation
        
    # ICP
    visible_hamer_pcd = get_pcd_from_points(visible_hamer_vertices, colors=np.ones_like(visible_hamer_vertices) * [0, 1, 0])
    try:
        aligned_hamer_pcd, T = icp_registration(visible_hamer_pcd, pcd, voxel_size=0.005, init_transform=T_0)
    except Exception as e:
        print(f"ICP failed: {e}")
        T = T_0
        
    # Transform keypoints
    kpts_3d = transform_pts(hamer_out["kpts_3d"], T)
    return kpts_3d, None


def draw_hand_overlay(img_bgr, kps_xy, raw_pts3d=None, aligned_pts3d=None, show_3d_indices=(0, 4, 8)):
    """
    在 BGR 图上只画指定关键点；返回 4, 8 点的 Raw 和 Aligned 坐标文本。
    """
    if kps_xy is None:
        return img_bgr, []

    h, w = img_bgr.shape[:2]
    info_lines = []

    # draw points (only selected indices)
    for i in show_3d_indices:
        if i >= len(kps_xy):
            continue
        
        x, y = kps_xy[i]
        if x is None or y is None:
            continue
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < w and 0 <= y < h):
            continue

        # Draw simple circle
        cv2.circle(img_bgr, (x, y), 5, (0, 0, 255), -1)
        
    # Prepare text info for 4 and 8
    target_indices = [4, 8]
    for i in target_indices:
        if i >= len(kps_xy): continue
        
        line_parts = []
        line_parts.append(f"H[{i}]")
        
        if raw_pts3d is not None:
            X, Y, Z = raw_pts3d[i]
            line_parts.append(f"Raw({X:.3f},{Y:.3f},{Z:.3f})")
            
        if aligned_pts3d is not None:
            X, Y, Z = aligned_pts3d[i]
            line_parts.append(f"Align({X:.3f},{Y:.3f},{Z:.3f})")
            
        if len(line_parts) > 1:
            info_lines.append(" ".join(line_parts))

    return img_bgr, info_lines


def setup_aruco(dict_name="DICT_4X4_50"):
    """
    OpenCV aruco detector setup
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("你的 OpenCV 没有 aruco 模块，请安装 opencv-contrib-python。")

    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_map.get(dict_name, cv2.aruco.DICT_4X4_50))
    params = cv2.aruco.DetectorParameters()

    # OpenCV 4.7+ has ArucoDetector, fallback otherwise
    detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    return aruco_dict, params, detector


def detect_aruco(img_bgr, aruco_dict, params, detector=None):
    """
    return: corners(list), ids(np.ndarray or None)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def draw_aruco_overlay(img_bgr, corners, ids, depth_m, rs_intrinsics, camera_matrix, dist_coeffs, marker_length=0.05):
    """
    Draw marker axes and center pixel; return 3D info for top-left display.
    """
    info_lines = []
    if ids is None or len(ids) == 0:
        return img_bgr, info_lines

    # Draw axes
    try:
        # estimatePoseSingleMarkers returns rvecs, tvecs, objPoints
        # Note: This function might be unavailable in some opencv-python-headless versions or require contrib
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.drawFrameAxes(img_bgr, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length)
    except Exception as e:
        # Fallback if pose estimation fails or function missing
        pass

    h, w = img_bgr.shape[:2]
    for i in range(len(ids)):
        c = corners[i].reshape(4, 2)
        cx = int(round(np.mean(c[:, 0])))
        cy = int(round(np.mean(c[:, 1])))
        cx = clamp_int(cx, 0, w - 1)
        cy = clamp_int(cy, 0, h - 1)

        # Draw center point
        cv2.circle(img_bgr, (cx, cy), 6, (255, 255, 0), -1)

        d = robust_depth_at(depth_m, cx, cy, k=7)
        if d is not None:
            X, Y, Z = deproject_pixel_to_3d(rs_intrinsics, cx, cy, d)
            info_lines.append(f"ArUco[{int(ids[i])}]: ({X:.3f}, {Y:.3f}, {Z:.3f}) m")
        else:
            info_lines.append(f"ArUco[{int(ids[i])}]: Depth Invalid")

    return img_bgr, info_lines


# -----------------------------
# Detector adapter (plug your self.detector_hamer here)
# -----------------------------
class HamerDetectorWrapper:
    """
    Wrapper for DetectorHamer to include MediaPipe bbox detection and return HaMeR output.
    """
    def __init__(self, intrinsics_dict):
        self.detector = DetectorHamer()
        self.intrinsics_dict = intrinsics_dict
        
        if HAS_MEDIAPIPE:
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            print("MediaPipe not available, fallback to center crop bbox.")

    def get_bbox(self, img_rgb):
        if not HAS_MEDIAPIPE:
            h, w = img_rgb.shape[:2]
            margin = 100
            return np.array([margin, margin, w-margin, h-margin])

        results = self.hands_detector.process(img_rgb)
        if results.multi_hand_landmarks:
            h, w = img_rgb.shape[:2]
            landmarks = results.multi_hand_landmarks[0].landmark
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            pad = 30
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)
            return np.array([x_min, y_min, x_max, y_max])
        return None

    def detect_hand_keypoints(self, img_bgr):
        """
        Returns hamer_out dict if successful, else None.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bbox = self.get_bbox(img_rgb)
        
        if bbox is None:
            return None
            
        bboxes = bbox[None, ...]
        hand_side = "right"
        is_right = np.array([True])
        
        try:
            hamer_out = self.detector.detect_hand_keypoints(
                img_rgb,
                hand_side=hand_side,
                bboxes=bboxes,
                is_right=is_right,
                kpts_2d_only=False,
                camera_params=self.intrinsics_dict,
                visualize=False
            )
            
            if hamer_out and hamer_out.get("success"):
                hamer_out['bbox'] = bbox
                return hamer_out
        except Exception as e:
            print(f"HaMeR detection error: {e}")
            
        return None


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--aruco_dict", type=str, default="DICT_4X4_50")
    parser.add_argument("--marker_length", type=float, default=0.05, help="ArUco marker side length in meters")
    parser.add_argument("--device_sn", type=str, default="", help="可选：指定 RealSense 序列号")
    args = parser.parse_args()

    # ---- RealSense pipeline ----
    pipeline = rs.pipeline()
    config = rs.config()
    if args.device_sn.strip():
        config.enable_device(args.device_sn.strip())
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per unit
    print(f"[INFO] depth_scale = {depth_scale} m/unit")

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Get intrinsics for deprojection (color stream)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()  # rs.intrinsics
    intr_dict = get_intrinsics_as_dict(intr)

    # Prepare OpenCV camera matrix and dist coeffs
    camera_matrix = np.array([
        [intr_dict['fx'], 0, intr_dict['cx']],
        [0, intr_dict['fy'], intr_dict['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.array(intr_dict['disto'], dtype=np.float32)

    # ---- detector (replace this with your self.detector_hamer) ----
    # Initialize Wrapper
    print("Initializing HaMeR Detector...")
    detector_wrapper = HamerDetectorWrapper(intr_dict)

    # ---- ArUco ----
    aruco_dict, aruco_params, aruco_detector = setup_aruco(args.aruco_dict)

    # FPS counter
    last_t = time.time()
    fps = 0.0

    print("[INFO] Press 'q' or ESC to quit.")
    print("[INFO] Press 's' to save current frame (raw image).")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())  # BGR
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # uint16 -> float
            depth_m = depth_raw * float(depth_scale)
            # clip depth for robustness
            depth_m[(depth_m <= 0.0) | (depth_m > args.max_depth_m)] = np.nan

            # ---- (2) hand keypoints detection ----
            # kps_xy = detector.detect_hand_keypoints(color_img)
            hamer_out = detector_wrapper.detect_hand_keypoints(color_img)
            
            kps_xy = None
            raw_pts3d = None
            aligned_pts3d = None
            
            if hamer_out is not None:
                kps_xy = hamer_out['kpts_2d']
                raw_pts3d = hamer_out['kpts_3d']
                
                # ---- (3) depth alignment -> 3D points ----
                # pts3d, _depths = depth_alignment(kps_xy, depth_m, intr)
                # Use ICP alignment
                img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                aligned_pts3d, _ = depth_alignment(hamer_out, depth_m, img_rgb, intr_dict, detector_wrapper.detector)

            # ---- (4) visualize 21 keypoints + 0,4,8 3D coords ----
            vis = color_img.copy()
            vis, hand_info = draw_hand_overlay(vis, kps_xy, raw_pts3d=raw_pts3d, aligned_pts3d=aligned_pts3d, show_3d_indices=(0, 4, 8))

            # ---- (5) aruco detect + label center + center 3D ----
            corners, ids = detect_aruco(vis, aruco_dict, aruco_params, detector=aruco_detector)
            vis, aruco_info = draw_aruco_overlay(vis, corners, ids, depth_m, intr, camera_matrix, dist_coeffs, args.marker_length)

            # ---- Draw Info Text at Top-Left ----
            y_offset = 30
            
            # FPS
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 1e-6:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25

            # Hand Info
            for line in hand_info:
                cv2.putText(vis, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 25
            
            # ArUco Info
            for line in aruco_info:
                cv2.putText(vis, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                y_offset += 25

            cv2.imshow("D435 Hand Keypoints + ArUco", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.png"
                cv2.imwrite(filename, vis)
                print(f"[INFO] Saved raw image to {filename}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
