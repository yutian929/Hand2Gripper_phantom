import os
import time
import numpy as np
import cv2
import mediapy as media
from typing import List

from fucking_arx_mujoco.real.camera.camera_utils import load_camera_intrinsics, load_eye_to_hand_matrix
from fucking_arx_mujoco.real.real_single_arm import RealSingleArm

def load_traj_and_widths(traj_path: str, width_path: str) -> tuple[List[np.ndarray], np.ndarray]:
    T_list = np.load(traj_path)
    if T_list.ndim == 3:
        T_list = [T_list[i] for i in range(T_list.shape[0])]
    widths = np.load(width_path)
    return T_list, widths

def apply_7d_bias(T_ee: np.ndarray, gripper: float, bias: np.ndarray) -> tuple[np.ndarray, float]:
    # bias: [x, y, z, rx, ry, rz, gripper]
    T_biased = T_ee.copy()
    pos = T_ee[:3, 3] + bias[:3]
    rmat = T_ee[:3, :3]
    rpy = cv2.Rodrigues(rmat)[0].flatten() + bias[3:6]
    rmat_biased, _ = cv2.Rodrigues(rpy)
    T_biased[:3, 3] = pos
    T_biased[:3, :3] = rmat_biased
    gripper_biased = gripper + bias[6]
    return T_biased, gripper_biased

def main():
    # 配置
    DATA_DIR = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/1"
    LEFT_TRAJ = os.path.join(DATA_DIR, "inpaint_processor", "hand2gripper_train_base_L_T_ee_L.npy")
    RIGHT_TRAJ = os.path.join(DATA_DIR, "inpaint_processor", "hand2gripper_train_base_R_T_ee_R.npy")
    LEFT_WIDTH = os.path.join(DATA_DIR, "inpaint_processor", "hand2gripper_train_gripper_width_left.npy")
    RIGHT_WIDTH = os.path.join(DATA_DIR, "inpaint_processor", "hand2gripper_train_gripper_width_right.npy")
    OVERLAY_VIDEO = os.path.join(DATA_DIR, "video_overlay_Arx5_shoulders.mkv")

    CAN_PORT_LEFT = "can1"
    CAN_PORT_RIGHT = "can3"
    MAX_VEL, MAX_ACC = 100, 300
    EE_POSE_L_BIAS = np.array([0, 0, 0, 0, 0, 0, 0])
    EE_POSE_R_BIAS = np.array([0, 0, 0.07, 0, 0, 0, 0])

    # 加载轨迹和夹爪宽度
    T_ee_L_list, gripper_L = load_traj_and_widths(LEFT_TRAJ, LEFT_WIDTH)
    T_ee_R_list, gripper_R = load_traj_and_widths(RIGHT_TRAJ, RIGHT_WIDTH)
    N = min(len(T_ee_L_list), len(T_ee_R_list))

    # 加载overlay视频帧（直接作为仿真可视化）
    overlay_frames = list(media.read_video(OVERLAY_VIDEO))
    assert len(overlay_frames) == N, f"Overlay video frames {len(overlay_frames)} != traj {N}"

    # 初始化真实机械臂
    real_arm_L = real_arm_R = None
    print(f"[INFO] 连接左臂: {CAN_PORT_LEFT}")
    real_arm_L = RealSingleArm(can_port=CAN_PORT_LEFT, max_velocity=MAX_VEL, max_acceleration=MAX_ACC)
    print(f"[INFO] 连接右臂: {CAN_PORT_RIGHT}")
    real_arm_R = RealSingleArm(can_port=CAN_PORT_RIGHT, max_velocity=MAX_VEL, max_acceleration=MAX_ACC)
    time.sleep(1)

    # 可视化与联动
    print("[INFO] 操作: q=退出, 空格=暂停, a/d=前后帧, r=执行当前帧, h=回零")
    paused, idx = True, 0

    while idx < N:
        overlay = overlay_frames[idx].copy()
        T_ee_L = T_ee_L_list[idx]
        T_ee_R = T_ee_R_list[idx]
        gw_L = gripper_L[idx]
        gw_R = gripper_R[idx]

        # 7D显示
        pos_L = T_ee_L[:3, 3]
        pos_R = T_ee_R[:3, 3]
        euler_L = cv2.Rodrigues(T_ee_L[:3, :3])[0].flatten() if not np.isnan(T_ee_L).any() else np.zeros(3)
        euler_R = cv2.Rodrigues(T_ee_R[:3, :3])[0].flatten() if not np.isnan(T_ee_R).any() else np.zeros(3)
        pose7d_L = np.concatenate([pos_L, euler_L, [gw_L]])
        pose7d_R = np.concatenate([pos_R, euler_R, [gw_R]])
        cv2.putText(overlay, f"{idx}/{N-1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"L7D: [{', '.join(f'{v:.3f}' for v in pose7d_L)}]", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(overlay, f"R7D: [{', '.join(f'{v:.3f}' for v in pose7d_R)}]", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(overlay, f"{'PAUSED' if paused else 'RUNNING'}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Sim2Real Overlay", overlay)

        key = cv2.waitKey(0 if paused else 100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a') and paused:
            idx = max(0, idx - 1)
            continue
        elif key == ord('d') and paused:
            idx = min(N - 1, idx + 1)
            continue
        elif key == ord('r'):
            # 执行当前帧
            if real_arm_L and not np.isnan(T_ee_L).any():
                T_ee_L_biased, gw_L_biased = apply_7d_bias(T_ee_L, gw_L, EE_POSE_L_BIAS)
                print(f"[INFO] 左臂执行帧 {idx} (with bias)")
                real_arm_L.move_to(T_ee_L_biased, gw_L_biased, is_gripper_pose=True)
            if real_arm_R and not np.isnan(T_ee_R).any():
                T_ee_R_biased, gw_R_biased = apply_7d_bias(T_ee_R, gw_R, EE_POSE_R_BIAS)
                print(f"[INFO] 右臂执行帧 {idx} (with bias)")
                real_arm_R.move_to(T_ee_R_biased, gw_R_biased, is_gripper_pose=True)
            continue
        elif key == ord('h'):
            if real_arm_L:
                print("[INFO] 左臂回零")
                real_arm_L.go_home()
            if real_arm_R:
                print("[INFO] 右臂回零")
                real_arm_R.go_home()
            continue

        if not paused:
            if real_arm_L and not np.isnan(T_ee_L).any():
                T_ee_L_biased, gw_L_biased = apply_7d_bias(T_ee_L, gw_L, EE_POSE_L_BIAS)
                real_arm_L.move_to(T_ee_L_biased, gw_L_biased, is_gripper_pose=True)
            if real_arm_R and not np.isnan(T_ee_R).any():
                T_ee_R_biased, gw_R_biased = apply_7d_bias(T_ee_R, gw_R, EE_POSE_R_BIAS)
                real_arm_R.move_to(T_ee_R_biased, gw_R_biased, is_gripper_pose=True)
            idx += 1

    cv2.destroyAllWindows()
    if real_arm_L:
        print("[INFO] 左臂回零")
        real_arm_L.go_home()
    if real_arm_R:
        print("[INFO] 右臂回零")
        real_arm_R.go_home()
    time.sleep(2)

if __name__ == "__main__":
    main()
