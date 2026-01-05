#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA 关键帧示教控制器 (Keyframe Mode) - 多片段版
功能：
1. 启动即进入准备状态 (自动新建序号文件夹)
2. 按 'Space' (空格键)：捕捉当前这一帧 (保存图像 + 机械臂状态)
3. 按 'S' 键：保存当前片段(Episode)，并自动新建下一个文件夹开始新的录制
4. 按 'Q' 键：保存当前片段并退出程序

使用方法：
    python hand2gripper_aloha_keyframe.py --save_dir ../data/hand2gripper_demo_aloha/demo
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
        dir_name = f"{idx:02d}"
        full_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(full_path):
            return full_path, idx
        idx += 1

def draw_ui_overlay(img, arm_info_text, keyframe_count, current_episode_idx, last_action_text=""):
    """绘制UI界面"""
    h, w = img.shape[:2]

    # 状态指示
    cv2.circle(img, (w - 30, 30), 10, (0, 255, 0), -1) # 绿点 (Ready)
    cv2.putText(img, "KEYFRAME MODE", (w - 240, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 显示当前 Episode 和已采集帧数
    cv2.putText(img, f"Ep: {current_episode_idx:02d} | Frames: {keyframe_count}", (w - 320, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # 交互反馈 (例如显示 "Captured!" 或 "Saved!")
    if last_action_text:
        # 居中显示大字
        text_size = cv2.getTextSize(last_action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(img, last_action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 按键说明
    instr_color = (0, 255, 255) # 黄色
    cv2.putText(img, "Controls:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 更新按键说明
    cv2.putText(img, "[SPACE] Capture Frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instr_color, 2)
    cv2.putText(img, "[S]     Save & Next Ep", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instr_color, 2)
    cv2.putText(img, "[Q]     Quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instr_color, 2)

    # 机械臂数据
    y_start = 130
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
        self.current_episode_dir = None
        self.current_episode_idx = 0

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
        self.save_episode_data() # 退出前尝试保存
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
    # 控制线程 (保持原有逻辑，用于同步主从臂)
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
    def start_new_episode(self):
        """开启一个新的录制 Episode"""
        self.current_episode_dir, self.current_episode_idx = get_next_episode_dir(self.args.save_dir)
        os.makedirs(self.current_episode_dir, exist_ok=True)
        
        print(f"\n>>> [NEW EPISODE] 准备录制 Episode: {self.current_episode_idx}")
        print(f"    路径: {self.current_episode_dir}")

        # 初始化视频录制 (5fps 播放关键帧，方便查看)
        video_path = os.path.join(self.current_episode_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 5, (640, 480))
        
        # 清空数据缓存
        self.data_T_L = []
        self.data_T_R = []
        self.data_width_L = []
        self.data_width_R = []

    def capture_keyframe(self, frame_img, T_L, T_R, w_L, w_R):
        """按下空格时调用：保存单帧数据"""
        if self.video_writer is not None:
            self.video_writer.write(frame_img)
        
        self.data_T_L.append(T_L)
        self.data_T_R.append(T_R)
        self.data_width_L.append(w_L)
        self.data_width_R.append(w_R)
        
        count = len(self.data_T_L)
        print(f"    [Frame {count}] Captured.")
        return count

    def save_episode_data(self):
        """按下S或Q时调用：保存当前Episode的数据到硬盘"""
        if not self.current_episode_dir:
            return

        print(f"\n>>> [SAVE] 正在保存 Episode {self.current_episode_idx}...")
        
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
            print(f"    - 成功保存 {len(self.data_T_L)} 帧数据。")
        else:
            print("    [提示] 本片段没有录制任何帧，跳过保存 .npy。")
            # 可选：如果完全是空的，删除这个文件夹
            try:
                # os.rmdir(save_path) # 如果需要自动删除空文件夹，取消注释
                pass
            except:
                pass

    # -------------------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------------------
    def run(self):
        self.go_home_all()
        print(">>> 机械臂已就绪，主臂进入泄力模式。")
        self.enable_gravity_compensation()
        
        # 启动时，先开启第一个 Episode
        self.start_new_episode()
        
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()

        print("\n" + "="*50)
        print("  【关键帧采集模式】")
        print("  按 [SPACE] (空格) : 拍一帧 (记录当前状态)")
        print("  按 [S]            : 保存当前片段 -> 下一片段 (重置)")
        print("  按 [Q]            : 保存并退出")
        print("="*50 + "\n")

        feedback_timer = 0
        last_action_text = ""
        
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

                # --- 可视化 ---
                pose_7d_L = get_7d_from_T(curr_T_L, curr_w_L)
                pose_7d_R = get_7d_from_T(curr_T_R, curr_w_R)
                
                arm_text = [
                    f"L Pos: {np.round(pose_7d_L[:3], 3)}",
                    f"L Rot: {np.round(pose_7d_L[3:6], 2)} G: {pose_7d_L[6]:.3f}",
                    f"R Pos: {np.round(pose_7d_R[:3], 3)}",
                    f"R Rot: {np.round(pose_7d_R[3:6], 2)} G: {pose_7d_R[6]:.3f}"
                ]

                # 反馈文字计时器
                if feedback_timer > 0:
                    feedback_timer -= 1
                else:
                    last_action_text = ""

                # 绘制 UI
                display_img = frame_img.copy()
                draw_ui_overlay(
                    display_img, 
                    arm_text, 
                    len(self.data_T_L), 
                    self.current_episode_idx,
                    last_action_text
                )
                
                cv2.imshow("ALOHA Keyframe Collection", display_img)
                
                # --- 按键处理 ---
                key = cv2.waitKey(1) & 0xFF
                
                # 1. 按空格: 拍一帧
                if key == 32: 
                    self.capture_keyframe(frame_img, curr_T_L, curr_T_R, curr_w_L, curr_w_R)
                    last_action_text = "CAPTURED!"
                    feedback_timer = 10 
                
                # 2. 按S: 保存这一条，开始下一条
                elif key == ord('s') or key == ord('S'):
                    self.save_episode_data()   # 保存当前
                    self.start_new_episode()   # 开启下一条(清零)
                    last_action_text = "SAVED & NEXT!"
                    feedback_timer = 20
                    self.go_home_all()
                    self.enable_gravity_compensation()

                
                # 3. 按Q: 退出
                elif key == ord('q') or key == ord('Q'):
                    self.save_episode_data()
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
    parser = argparse.ArgumentParser(description='ALOHA + RealSense 关键帧采集版')
    parser.add_argument('--master_l', type=str, default='can0')
    parser.add_argument('--master_r', type=str, default='can2')
    parser.add_argument('--follower_l', type=str, default='can3')
    parser.add_argument('--follower_r', type=str, default='can1')
    parser.add_argument('--save_dir', type=str, default='../data/hand2gripper_demo_aloha/demo', help='数据保存根目录')
    parser.add_argument('--freq', type=float, default=50.0, help='控制频率 Hz')
    
    args = parser.parse_args()
    
    sys = AlohaTeachingSystem(args)
    sys.run()

if __name__ == "__main__":
    main()