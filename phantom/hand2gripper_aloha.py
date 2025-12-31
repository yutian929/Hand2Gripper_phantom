#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA风格主从示教控制器 + RealSense 连续录制版
功能：
1. 启动进入预览模式
2. 按 'R' 键：开始录制当前 Episode (自动新建序号文件夹，如 01, 02...)
3. 按 'S' 键：停止当前录制并保存数据，重置状态准备下一条
4. 按 'Q' 键：完全退出程序
5. 实时可视化 7D Pose 和 UI 交互

使用方法：
    python hand2gripper_aloha_rs_v2.py --save_dir ../data/hand2gripper_demo_aloha/demo
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

# 保持原有 import
from fucking_arx_mujoco.real.real_single_arm import RealSingleArm

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def get_7d_from_T(T: np.ndarray, gripper: float) -> np.ndarray:
    """将 4x4 变换矩阵转换为 7D 向量 [x, y, z, rx, ry, rz, gripper]"""
    if T is None:
        return np.zeros(7)
    pos = T[:3, 3]
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    rvec = rvec.flatten()
    return np.concatenate([pos, rvec, [gripper]])

def get_next_episode_dir(root_dir):
    """自动获取下一个可用的 episode 目录 (例如: root/01, root/02 ...)"""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    idx = 1
    while True:
        # 格式化为 01, 02, ... 99
        dir_name = f"{idx:02d}"
        full_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(full_path):
            return full_path, idx
        idx += 1

def draw_ui_overlay(img, arm_info_text, is_recording, frame_count, current_episode_idx):
    """绘制UI界面"""
    h, w = img.shape[:2]

    # 状态指示
    if is_recording:
        cv2.circle(img, (w - 30, 30), 10, (0, 0, 255), -1) # 红点
        cv2.putText(img, "REC", (w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f"Ep: {current_episode_idx:02d} | Frames: {frame_count}", (w - 250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    else:
        cv2.putText(img, "PREVIEW", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Next Ep: {current_episode_idx:02d}", (w - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 按键说明
    instr_color = (0, 255, 255) # 黄色
    cv2.putText(img, "Controls:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # 更新按键说明
    cv2.putText(img, "[R] Start  |  [S] Save & Next  |  [Q] Quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instr_color, 2)

    # 机械臂数据
    y_start = 85
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
        self.is_recording = False
        self.current_episode_dir = None
        self.current_episode_idx = 1
        
        # 预先计算下一个序号，仅用于显示
        _, self.current_episode_idx = get_next_episode_dir(self.args.save_dir)

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
            self.master_l = RealSingleArm(can_port=self.args.master_l, arm_type=0, max_velocity=300, max_acceleration=800)
            self.master_r = RealSingleArm(can_port=self.args.master_r, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower_l = RealSingleArm(can_port=self.args.follower_l, arm_type=0, max_velocity=300, max_acceleration=800)
            self.follower_r = RealSingleArm(can_port=self.args.follower_r, arm_type=0, max_velocity=300, max_acceleration=800)
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
        self.master_l.arm.gravity_compensation()
        self.master_r.arm.gravity_compensation()

    def go_home_all(self):
        print("机械臂回零位...")
        self.master_l.go_home()
        self.master_r.go_home()
        self.follower_l.go_home()
        self.follower_r.go_home()
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
                m1_j = self.master_l.get_joint_positions()
                m2_j = self.master_r.get_joint_positions()
                m1_g = self.master_l.get_gripper_width(teacher=True)
                m2_g = self.master_r.get_gripper_width(teacher=True)

                # 2. 写从臂
                if m1_j is not None:
                    self.follower_l.set_joint_positions(m1_j)
                    self.follower_l.set_gripper_width(m1_g)
                if m2_j is not None:
                    self.follower_r.set_joint_positions(m2_j)
                    self.follower_r.set_gripper_width(m2_g)

                # 3. 读从臂状态 (缓存供录制)
                f1_pose = self.follower_l.get_gripper_pose()
                f2_pose = self.follower_r.get_gripper_pose()
                f1_width = self.follower_l.get_gripper_width()
                f2_width = self.follower_r.get_gripper_width()

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
    # 录制管理
    # -------------------------------------------------------------------------
    def start_recording(self):
        """开始新的录制片段"""
        # 1. 确定保存路径
        self.current_episode_dir, self.current_episode_idx = get_next_episode_dir(self.args.save_dir)
        os.makedirs(self.current_episode_dir, exist_ok=True)
        print(f"\n>>> [START] 开始录制 Episode: {self.current_episode_idx} (路径: {self.current_episode_dir})")

        # 2. 初始化视频录制
        video_path = os.path.join(self.current_episode_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))

        # 3. 清空数据缓存
        self.data_T_L = []
        self.data_T_R = []
        self.data_width_L = []
        self.data_width_R = []
        
        self.is_recording = True

    def stop_and_save(self):
        """停止录制并保存当前片段"""
        if not self.is_recording:
            return

        print(f"\n>>> [STOP] 正在保存 Episode {self.current_episode_idx}...")
        self.is_recording = False
        
        # 1. 停止视频
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # 2. 保存 Numpy 数据
        save_path = self.current_episode_dir
        if len(self.data_T_L) > 0:
            np.save(os.path.join(save_path, "hand2gripper_train_base_L_T_ee_L.npy"), np.array(self.data_T_L))
            np.save(os.path.join(save_path, "hand2gripper_train_base_R_T_ee_R.npy"), np.array(self.data_T_R))
            np.save(os.path.join(save_path, "hand2gripper_train_gripper_width_left.npy"), np.array(self.data_width_L))
            np.save(os.path.join(save_path, "hand2gripper_train_gripper_width_right.npy"), np.array(self.data_width_R))
            print(f"    - 已保存 {len(self.data_T_L)} 帧数据。")
            print(f"    - 保存完成！准备下一条。")
        else:
            print("    [警告] 数据为空，未保存 NPY。")

        # 3. 更新显示的序号 (指向下一个)
        self.current_episode_idx += 1

    # -------------------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------------------
    def run(self):
        self.go_home_all()
        print(">>> 机械臂已就绪，主臂进入泄力模式。")
        self.enable_gravity_compensation()
        
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()

        print("\n" + "="*50)
        print("  程序已启动！连续采集模式")
        print("  按 [R] 开始录制")
        print("  按 [S] 停止并保存 (自动进入下一个序号)")
        print("  按 [Q] 退出程序")
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
                else:
                    frame_count = 0 # 预览模式重置计数

                # --- 可视化 ---
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
                draw_ui_overlay(display_img, arm_text, self.is_recording, frame_count, self.current_episode_idx)
                
                cv2.imshow("ALOHA Data Collection", display_img)
                
                # --- 按键处理 ---
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r') or key == ord('R'):
                    if not self.is_recording:
                        self.start_recording()
                
                elif key == ord('s') or key == ord('S'):
                    if self.is_recording:
                        self.stop_and_save()
                
                elif key == ord('q') or key == ord('Q'):
                    # 如果正在录制，先保存
                    if self.is_recording:
                        self.stop_and_save()
                    self.running = False

        finally:
            self.running = False
            if control_thread.is_alive():
                control_thread.join(timeout=1.0)
            
            self.pipeline.stop()
            if self.video_writer is not None:
                self.video_writer.release()
            
            cv2.destroyAllWindows()
            print("程序退出。")

def main():
    parser = argparse.ArgumentParser(description='ALOHA + RealSense 连续录制版')
    parser.add_argument('--master_l', type=str, default='can0')
    parser.add_argument('--master_r', type=str, default='can2')
    parser.add_argument('--follower_l', type=str, default='can3')
    parser.add_argument('--follower_r', type=str, default='can1')
    # 这里 save_dir 指向根目录，程序会自动在下面创建 01, 02, 03...
    parser.add_argument('--save_dir', type=str, default='../data/hand2gripper_demo_aloha/demo', help='数据保存根目录')
    parser.add_argument('--freq', type=float, default=50.0, help='控制频率 Hz')
    
    args = parser.parse_args()
    
    sys = AlohaTeachingSystem(args)
    sys.run()
    sys.go_home_all() # 在 run 里面已经调用了，这里可以注释掉，或者保留作为双重确认

if __name__ == "__main__":
    main()