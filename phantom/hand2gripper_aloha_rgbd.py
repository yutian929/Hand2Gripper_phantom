#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA风格主从示教控制器 + RealSense RGBD 录制版
为 3D-Diffusion-Policy 采集数据

功能：
1. 启动进入预览模式，显示 RGB 和 Depth
2. 按 'Space' 键：开始录制 (按 --record_freq 设定的频率抽帧保存)
3. 按 'Space' 键：停止并保存
4. 按 'Q' 键：退出

保存内容：
- video.mp4: RGB视频
- depth.npy: 深度图序列 (N, H, W) 单位 mm 或 m (取决于RealSense设置，通常为 1e-3, 即 mm)
- intrinsics.npy: 相机内参矩阵 (3, 3)
- 左右臂的Pose (T matrices) 和 Gripper Width

使用方法：
    python hand2gripper_aloha_rgbd.py --save_dir ./data/my_dataset_raw --record_freq 10
"""

import os
import time
import signal
import sys
import threading
import numpy as np
import cv2
import argparse
import json
import pyrealsense2 as rs
from typing import Optional, List

# 保持原有 import
from fucking_arx_mujoco.real.real_single_arm import RealSingleArm

def get_7d_from_T(T: np.ndarray, gripper: float) -> np.ndarray:
    if T is None: return np.zeros(7)
    pos = T[:3, 3]
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    rvec = rvec.flatten()
    return np.concatenate([pos, rvec, [gripper]])

def get_next_episode_dir(root_dir):
    if not os.path.exists(root_dir): os.makedirs(root_dir)
    idx = 1
    while True:
        dir_name = f"episode_{idx:03d}"
        full_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(full_path): return full_path, idx
        idx += 1

def draw_ui_overlay(img, arm_info_text, is_recording, frame_count, current_episode_idx, record_freq):
    h, w = img.shape[:2]
    # 状态指示
    if is_recording:
        cv2.circle(img, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(img, f"REC @ {record_freq}Hz", (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f"Ep: {current_episode_idx:03d} | Saved: {frame_count}", (w - 300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    else:
        cv2.putText(img, "PREVIEW", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Next Ep: {current_episode_idx:03d}", (w - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    instr_color = (0, 255, 255)
    cv2.putText(img, "Controls:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "[Space] Rec/Save  |  [Q] Quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instr_color, 2)

    y_start = 85
    for line in arm_info_text:
        cv2.putText(img, line, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 20
    return img

class AlohaRGBDTeachingSystem:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.is_recording = False
        self.current_episode_dir = None
        self.current_episode_idx = 1
        
        self.record_interval = 1.0 / self.args.record_freq
        self.last_record_time = 0.0
        
        _, self.current_episode_idx = get_next_episode_dir(self.args.save_dir)

        self.current_state = {
            "T_L": np.eye(4), "width_L": 0.0,
            "T_R": np.eye(4), "width_R": 0.0,
        }
        self.state_lock = threading.Lock()

        # Data Buffers
        self.data_T_L = []
        self.data_T_R = []
        self.data_width_L = []
        self.data_width_R = []
        self.data_depth = [] # Store depth frames in memory
        self.intrinsics_matrix = None
        self.depth_scale = 0.001 # Default to 1mm

        self.video_writer = None

        self._init_arms()
        self._init_camera()
        signal.signal(signal.SIGINT, self._signal_handler)

    def _init_arms(self):
        print("=" * 60)
        print("正在初始化机械臂...")
        try:
            self.master_l = RealSingleArm(can_port=self.args.master_l, arm_type=0, max_velocity=300, max_acceleration=800)
            self.master_r = RealSingleArm(can_port=self.args.master_r, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower_l = RealSingleArm(can_port=self.args.follower_l, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower_r = RealSingleArm(can_port=self.args.follower_r, arm_type=0, max_velocity=300, max_acceleration=800)
            print("机械臂初始化完成。")
        except Exception as e:
            print(f"机械臂初始化失败: {e}")
            sys.exit(1)

    def _init_camera(self):
        print("正在初始化 RealSense RGBD...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置 RGB 和 Depth 流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
            
            # 获取内参
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.intrinsics_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            print(f"相机内参:\n{self.intrinsics_matrix}")

            # 获取 Depth Scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth Scale: {self.depth_scale} meters")

            # 对齐 Depth 到 RGB
            self.align = rs.align(rs.stream.color)
            
            print("RealSense 初始化完成 (RGBD Aligned)。")
        except Exception as e:
            print(f"相机启动失败: {e}")
            self.pipeline = None

    def _signal_handler(self, signum, frame):
        print("\n收到退出信号...")
        self.running = False

    def enable_gravity_compensation(self):
        self.master_l.arm.gravity_compensation()
        self.master_r.arm.gravity_compensation()

    def go_home_all(self):
        self.master_l.go_home()
        self.master_r.go_home()
        self.follower_l.go_home()
        self.follower_r.go_home()
        time.sleep(2)

    def control_loop(self):
        dt = 1.0 / self.args.freq
        while self.running:
            t_start = time.time()
            try:
                # 获取 Teacher 的状态
                m1_j = self.master_l.get_joint_positions()
                m2_j = self.master_r.get_joint_positions()
                m1_g = self.master_l.get_gripper_width(teacher=True)
                m2_g = self.master_r.get_gripper_width(teacher=True)

                # 控制 Follower
                if m1_j is not None:
                    self.follower_l.set_joint_positions(m1_j)
                    self.follower_l.set_gripper_width(m1_g)
                if m2_j is not None:
                    self.follower_r.set_joint_positions(m2_j)
                    self.follower_r.set_gripper_width(m2_g)

                # 记录 Follower 状态 (用于训练)
                f1_pose = self.follower_l.get_gripper_pose() # 4x4 matrix
                f2_pose = self.follower_r.get_gripper_pose()
                f1_width = self.follower_l.get_gripper_width()
                f2_width = self.follower_r.get_gripper_width()

                with self.state_lock:
                    if f1_pose is not None: self.current_state["T_L"] = f1_pose
                    if f2_pose is not None: self.current_state["T_R"] = f2_pose
                    self.current_state["width_L"] = f1_width
                    self.current_state["width_R"] = f2_width

            except Exception: pass 
            
            elapsed = time.time() - t_start
            if elapsed < dt: time.sleep(dt - elapsed)

    def start_recording(self):
        self.current_episode_dir, self.current_episode_idx = get_next_episode_dir(self.args.save_dir)
        os.makedirs(self.current_episode_dir, exist_ok=True)
        print(f"\n>>> [START] 开始录制 Episode: {self.current_episode_idx}")
        print(f"    - 保存目录: {self.current_episode_dir}")

        video_path = os.path.join(self.current_episode_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.args.record_freq)
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

        # Reset Buffers
        self.data_T_L = []
        self.data_T_R = []
        self.data_width_L = []
        self.data_width_R = []
        self.data_depth = [] # (H, W) uint16

        self.last_record_time = 0.0 
        self.is_recording = True

    def stop_and_save(self):
        if not self.is_recording: return

        print(f"\n>>> [STOP] 正在保存 Episode {self.current_episode_idx}...")
        self.is_recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        save_path = self.current_episode_dir
        if len(self.data_T_L) > 0:
            # Save Trajectories
            np.save(os.path.join(save_path, "left_arm_T.npy"), np.array(self.data_T_L))
            np.save(os.path.join(save_path, "right_arm_T.npy"), np.array(self.data_T_R))
            np.save(os.path.join(save_path, "left_target_width.npy"), np.array(self.data_width_L))
            np.save(os.path.join(save_path, "right_target_width.npy"), np.array(self.data_width_R))
            
            # Save Depth
            print(f"    - 保存深度数据 (Frames: {len(self.data_depth)})...")
            np.save(os.path.join(save_path, "depth.npy"), np.array(self.data_depth))
            
            # Save Intrinsics and Depth Scale
            meta = {
                "intrinsics": self.intrinsics_matrix.tolist(),
                "depth_scale": self.depth_scale
            }
            with open(os.path.join(save_path, "camera_meta.json"), 'w') as f:
                json.dump(meta, f, indent=4)

            print(f"    - Episode {self.current_episode_idx} 保存完成。")
        else:
            print("    [警告] 数据为空，丢弃。")

        self.current_episode_idx += 1

    def run(self):
        self.go_home_all()
        print(">>> 机械臂已就绪。")
        self.enable_gravity_compensation()
        
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()

        print("\n" + "="*50)
        print(f"  录制频率: {self.args.record_freq} Hz")
        print("  [Space] 开始/停止  |  [Q] 退出")
        print("="*50 + "\n")

        frame_count = 0
        
        try:
            while self.running:
                # 1. 获取对齐的 RGBD 帧
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame: continue
                
                color_img = np.asanyarray(color_frame.get_data()) # BGR
                depth_img = np.asanyarray(depth_frame.get_data()) # UINT16 (mm)

                # 2. 获取当前机器人状态
                with self.state_lock:
                    curr_T_L = self.current_state["T_L"].copy()
                    curr_T_R = self.current_state["T_R"].copy()
                    curr_w_L = self.current_state["width_L"]
                    curr_w_R = self.current_state["width_R"]

                # 3. 录制逻辑
                if self.is_recording:
                    curr_time = time.time()
                    if curr_time - self.last_record_time >= self.record_interval:
                        
                        # Save RGB to Video
                        if self.video_writer is not None:
                            self.video_writer.write(color_img)
                        
                        # Save Depth to Memory
                        self.data_depth.append(depth_img.copy())
                        
                        # Save Robot State
                        self.data_T_L.append(curr_T_L)
                        self.data_T_R.append(curr_T_R)
                        self.data_width_L.append(curr_w_L)
                        self.data_width_R.append(curr_w_R)
                        
                        self.last_record_time = curr_time
                        frame_count += 1
                else:
                    frame_count = 0 

                # 4. 可视化
                # Depth colorization for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                
                pose_7d_L = get_7d_from_T(curr_T_L, curr_w_L)
                pose_7d_R = get_7d_from_T(curr_T_R, curr_w_R)
                arm_text = [
                    f"L: {np.round(pose_7d_L[:3], 3)} G:{pose_7d_L[6]:.2f}",
                    f"R: {np.round(pose_7d_R[:3], 3)} G:{pose_7d_R[6]:.2f}"
                ]

                # 拼接 RGB 和 Depth 显示
                display_img = np.hstack((color_img, depth_colormap))
                # Resize for display if too large
                display_img = cv2.resize(display_img, (0,0), fx=0.8, fy=0.8)
                
                draw_ui_overlay(display_img, arm_text, self.is_recording, frame_count, self.current_episode_idx, self.args.record_freq)
                cv2.imshow("Capture (RGB | Depth)", display_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_and_save()
                elif key == ord('h') or key == ord('H'):
                    if not self.is_recording:
                        print("\n>>> [HOME] 正在回零...")
                        self.go_home_all()
                        self.enable_gravity_compensation()
                        print(">>> 回零完成，已进入泄力模式。")
                elif key == ord('q') or key == ord('Q'):
                    if self.is_recording: self.stop_and_save()
                    self.running = False

        finally:
            self.running = False
            self.pipeline.stop()
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            print("程序退出。")

def main():
    parser = argparse.ArgumentParser(description='DP3 Data Collection: Aloha + RealSense RGBD')
    parser.add_argument('--master_l', type=str, default='can0')
    parser.add_argument('--master_r', type=str, default='can2')
    parser.add_argument('--follower_l', type=str, default='can3')
    parser.add_argument('--follower_r', type=str, default='can1')
    parser.add_argument('--save_dir', type=str, default='./data/hand2gripper_demo_dp3_raw', help='数据保存根目录')
    parser.add_argument('--freq', type=float, default=50.0, help='机械臂底层控制 Hz')
    parser.add_argument('--record_freq', type=float, default=10.0, help='数据采样 Hz')
    
    args = parser.parse_args()
    
    sys = AlohaRGBDTeachingSystem(args)
    sys.run()

if __name__ == "__main__":
    main()
