#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record RGB (MP4) + depth (Single Numpy .npy file) + camera intrinsics (JSON) from Intel RealSense D435 at 30 FPS.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import json

import pyrealsense2 as rs


def get_intrinsics_as_dict(intr):
    """
    Convert RealSense intrinsics to a dictionary format.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="output", help="Output directory")
    ap.add_argument("--fps", type=int, default=30, choices=[15, 30, 60], help="Stream FPS")
    ap.add_argument("--width", type=int, default=640, help="Color width")
    ap.add_argument("--height", type=int, default=480, help="Color height")
    ap.add_argument("--duration", type=float, default=5.0, help="Record duration in seconds")
    ap.add_argument("--warmup", type=int, default=15, help="Warmup frames to stabilize exposure")
    ap.add_argument("--show", action="store_true", help="Show the realtime video")
    args = ap.parse_args()

    out_video = Path(args.output, "video.mp4")
    out_depth = Path(args.output, "depth.npy")  # Save all depth frames into this single file
    out_intrinsics = Path(args.output, "camera_intrinsics.json")  # Camera intrinsics output
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- RealSense setup ----------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    # 对齐：将深度对齐到彩色（保证像素对应）
    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intr = get_intrinsics_as_dict(color_stream.get_intrinsics())
    depth_intr = get_intrinsics_as_dict(depth_stream.get_intrinsics())

    # Video writer（OpenCV：BGR -> mp4v）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_video), fourcc, args.fps, (args.width, args.height))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different codec or install ffmpeg/GStreamer enabled OpenCV.")

    # ---------------- Warmup ----------------
    for _ in range(max(0, args.warmup)):
        pipeline.wait_for_frames()

    # ---------------- Record loop ----------------
    depth_frames = []  # list of (H, W) float32 meters
    t0 = time.time()
    target_frames = int(args.duration * args.fps)

    print(f"Recording for ~{args.duration:.1f}s at {args.fps} FPS ...")
    n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # 对齐深度到彩色
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 取数据
            color_bgr = np.asanyarray(color_frame.get_data())            # (H, W, 3) uint8
            depth_raw = np.asanyarray(depth_frame.get_data())            # (H, W) uint16（单位：device units）
            depth_m = depth_raw.astype(np.float32) * depth_scale         # 转换为米

            # 写视频（BGR）
            vw.write(color_bgr)

            # 记录深度数据（以米为单位）
            depth_frames.append(depth_m)

            if args.show:
                color_show = cv2.putText(color_bgr, f"Frame: {n}/{target_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("RealSense Stream", color_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            n += 1
            # 基于帧数或时间终止
            if n >= target_frames or (time.time() - t0) >= args.duration:
                break

    finally:
        vw.release()
        pipeline.stop()

    depth_stack = np.stack(depth_frames, axis=0)  # (N, H, W)

    # ---------------- Save depth ----------------
    np.save(out_depth, depth_stack)  # Save depth stack as single .npy file

    # ---------------- Save camera intrinsics ----------------
    camera_intrinsics = {
        "color": color_intr,
        "depth": depth_intr
    }
    with open(out_intrinsics, "w") as json_file:
        json.dump(camera_intrinsics, json_file, indent=4)

    print(f"Done.\nRGB video: {out_video}\nDepth saved to: {out_depth}")
    print(f"Camera intrinsics saved to: {out_intrinsics}")
    print(f"Depth shape: {depth_stack.shape}, dtype={depth_stack.dtype}, scale(m/unit)={depth_scale}")


if __name__ == "__main__":
    main()
