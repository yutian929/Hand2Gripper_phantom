#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA风格主从示教控制器 + RealSense 交互式录制
功能：
1. 启动即显示 RealSense 画面和机械臂状态 (预览模式)
2. 按 'R' 键开始录制 (MP4 + NPY)
3. 按 'Q' 键停止并保存退出
4. 实时可视化 7D Pose 和 UI 交互

使用方法：
    python hand2gripper_aloha_rs_v2.py --save_dir ./my_data_02
"""

import os
import time
import signal
import sys
import threading
import numpy as np
import cv2
import argparse
from typing import Optional, List
import pyrealsense2 as rs

# 保持原有 import (请确保路径正确)
from fucking_arx_mujoco.real.real_single_arm import RealSingleArm

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def get_7d_from_T(T: np.ndarray, gripper: float) -> np.ndarray:
    """
    将 4x4 变换矩阵转换为 7D 向量 [x, y, z, rx, ry, rz, gripper]
    """
    if T is None:
        return np.zeros(7)
    pos = T[:3, 3]
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    rvec = rvec.flatten()
    return np.concatenate([pos, rvec, [gripper]])

def draw_ui_overlay(img, arm_info_text, is_recording, frame_count):
    """绘制UI界面（无黑色背景遮挡，直接文字叠加）"""
    h, w = img.shape[:2]

    # 录制状态 (右上角)
    if is_recording:
        cv2.circle(img, (w - 30, 30), 10, (0, 0, 255), -1) # 红点
        cv2.putText(img, "REC", (w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f"Frames: {frame_count}", (w - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(img, "PREVIEW", (w - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 按键说明 (左上角)
    instr_color = (0, 255, 255) # 黄色
    cv2.putText(img, "Controls:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "[R] Start Record  |  [Q] Quit & Save", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, instr_color, 2)

    # 机械臂数据 (下方)
    y_start = 80
    for line in arm_info_text:
        cv2.putText(img, line, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 20

    return img

# -----------------------------------------------------------------------------
# 核心控制器类
# -----------------------------------------------------------------------------

class AlohaTeachingSystem:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.is_recording = False # 初始为预览模式
        
        # 机械臂状态缓存
        self.current_state = {
            "T_L": np.eye(4), "width_L": 0.0,
            "T_R": np.eye(4), "width_R": 0.0,
            "joints_M1": None, "joints_M2": None
        }
        self.state_lock = threading.Lock()

        # 数据容器
        self.data_T_L = []
        self.data_T_R = []
        self.data_width_L = []
        self.data_width_R = []
        self.video_writer = None

        # 初始化设备
        self._init_arms()
        self._init_camera()
        
        signal.signal(signal.SIGINT, self._signal_handler)

    def _init_arms(self):
        print("=" * 60)
        print("正在初始化机械臂...")
        try:
            self.master1 = RealSingleArm(can_port=self.args.master1, arm_type=0, max_velocity=300, max_acceleration=800)
            self.master2 = RealSingleArm(can_port=self.args.master2, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower1 = RealSingleArm(can_port=self.args.follower1, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower2 = RealSingleArm(can_port=self.args.follower2, arm_type=0, max_velocity=300, max_acceleration=800)
            print("机械臂初始化完成。")
        except Exception as e:
            print(f"机械臂初始化失败: {e}")
            sys.exit(1)

    def _init_camera(self):
        print("正在初始化 RealSense...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            self.pipeline.start(config)
            print("RealSense 初始化完成。")
        except Exception as e:
            print(f"相机启动失败: {e}")
            self.pipeline = None

    def _signal_handler(self, signum, frame):
        print("\n收到退出信号...")
        self.running = False

    def enable_gravity_compensation(self):
        print("启用主臂重力补偿...")
        self.master1.arm.gravity_compensation()
        self.master2.arm.gravity_compensation()

    def go_home_all(self):
        print("机械臂回零位...")
        self.master1.go_home()
        self.master2.go_home()
        self.follower1.go_home()
        self.follower2.go_home()
        time.sleep(2)

    # -------------------------------------------------------------------------
    # 控制线程
    # -------------------------------------------------------------------------
    def control_loop(self):
        dt = 1.0 / self.args.freq
        while self.running:
            t_start = time.time()
            try:
                # 1. 读主臂
                m1_j = self.master1.get_joint_positions()
                m2_j = self.master2.get_joint_positions()
                m1_g = self.master1.get_gripper_width(teacher=True)
                m2_g = self.master2.get_gripper_width(teacher=True)

                # 2. 写从臂
                if m1_j is not None:
                    self.follower1.set_joint_positions(m1_j)
                    self.follower1.set_gripper_width(m1_g)
                if m2_j is not None:
                    self.follower2.set_joint_positions(m2_j)
                    self.follower2.set_gripper_width(m2_g)

                # 3. 读从臂状态 (缓存供录制)
                f1_pose = self.follower1.get_gripper_pose()
                f2_pose = self.follower2.get_gripper_pose()
                f1_width = self.follower1.get_gripper_width()
                f2_width = self.follower2.get_gripper_width()

                with self.state_lock:
                    if f1_pose is not None: self.current_state["T_L"] = f1_pose
                    if f2_pose is not None: self.current_state["T_R"] = f2_pose
                    self.current_state["width_L"] = f1_width
                    self.current_state["width_R"] = f2_width

            except Exception as e:
                pass 
            
            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # -------------------------------------------------------------------------
    # 主循环 (UI + 录制)
    # -------------------------------------------------------------------------
    def run(self):
        # 1. 机械臂归位并启用重力补偿
        self.go_home_all()
        print(">>> 机械臂已就绪，主臂进入泄力模式。")
        self.enable_gravity_compensation()
        
        # 2. 启动控制后台线程
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()

        # 3. 进入主循环
        print("\n" + "="*50)
        print("  程序已启动！")
        print("  [画面窗口中] 按 'R' 开始录制")
        print("  [画面窗口中] 按 'Q' 结束并保存")
        print("="*50 + "\n")

        frame_count = 0
        
        try:
            while self.running:
                # --- 获取相机帧 ---
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                frame_img = np.asanyarray(color_frame.get_data())

                # --- 获取机械臂状态快照 ---
                with self.state_lock:
                    curr_T_L = self.current_state["T_L"].copy()
                    curr_T_R = self.current_state["T_R"].copy()
                    curr_w_L = self.current_state["width_L"]
                    curr_w_R = self.current_state["width_R"]

                # --- 录制逻辑 ---
                if self.is_recording:
                    if self.video_writer is not None:
                        self.video_writer.write(frame_img)
                    
                    self.data_T_L.append(curr_T_L)
                    self.data_T_R.append(curr_T_R)
                    self.data_width_L.append(curr_w_L)
                    self.data_width_R.append(curr_w_R)
                    frame_count += 1

                # --- 可视化 ---
                # 计算 Pose 数值文本
                pose_7d_L = get_7d_from_T(curr_T_L, curr_w_L)
                pose_7d_R = get_7d_from_T(curr_T_R, curr_w_R)
                
                arm_text = [
                    f"L Pos: {np.round(pose_7d_L[:3], 3)}",
                    f"L Rot: {np.round(pose_7d_L[3:6], 2)} G: {pose_7d_L[6]:.3f}",
                    f"R Pos: {np.round(pose_7d_R[:3], 3)}",
                    f"R Rot: {np.round(pose_7d_R[3:6], 2)} G: {pose_7d_R[6]:.3f}"
                ]

                # 绘制 UI
                display_img = frame_img.copy()
                draw_ui_overlay(display_img, arm_text, self.is_recording, frame_count)
                
                cv2.imshow("ALOHA Data Collection", display_img)
                
                # --- 按键处理 ---
                key = cv2.waitKey(1) & 0xFF
                
                # 按 'r' 开始录制 (仅当未录制时)
                if key == ord('r') or key == ord('R'):
                    if not self.is_recording:
                        print("\n>>> 开始录制! <<<")
                        os.makedirs(self.args.save_dir, exist_ok=True)
                        video_path = os.path.join(self.args.save_dir, "video.mp4")
                        # 初始化 VideoWriter
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
                        self.is_recording = True
                        frame_count = 0 # 重置计数器
                
                # 按 'q' 退出
                elif key == ord('q') or key == ord('Q'):
                    self.running = False

        finally:
            self.running = False
            # 等待线程
            if control_thread.is_alive():
                control_thread.join(timeout=1.0)
            
            # 清理资源
            self.pipeline.stop()
            if self.video_writer is not None:
                self.video_writer.release()

            # 保存数据
            if len(self.data_T_L) > 0:
                print(f"\n正在保存 {len(self.data_T_L)} 帧数据到: {self.args.save_dir}")
                np.save(os.path.join(self.args.save_dir, "hand2gripper_train_base_L_T_ee_L.npy"), np.array(self.data_T_L))
                np.save(os.path.join(self.args.save_dir, "hand2gripper_train_base_R_T_ee_R.npy"), np.array(self.data_T_R))
                np.save(os.path.join(self.args.save_dir, "hand2gripper_train_gripper_width_left.npy"), np.array(self.data_width_L))
                np.save(os.path.join(self.args.save_dir, "hand2gripper_train_gripper_width_right.npy"), np.array(self.data_width_R))
                print("保存完成。")
            else:
                print("\n未录制任何数据。")
            
            cv2.destroyAllWindows()
            print("程序退出。")

def main():
    parser = argparse.ArgumentParser(description='ALOHA + RealSense 交互式录制')
    parser.add_argument('--master1', type=str, default='can0')
    parser.add_argument('--master2', type=str, default='can2')
    parser.add_argument('--follower1', type=str, default='can3')
    parser.add_argument('--follower2', type=str, default='can1')
    parser.add_argument('--save_dir', type=str, default='./aloha/demo_01', help='保存路径')
    parser.add_argument('--freq', type=float, default=50.0, help='控制频率 Hz')
    
    args = parser.parse_args()
    
    sys = AlohaTeachingSystem(args)
    sys.run()

if __name__ == "__main__":
    main()