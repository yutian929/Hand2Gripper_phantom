import cv2
import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation as R
from dual_arm_controller_with_ee_width import DualArmController
from test_single_arm import load_and_transform_data, pose_to_matrix, matrix_to_pose

# =========================================================================
# 核心功能：把仿真机械臂 贴图到 背景视频上
# =========================================================================
def overlay_robot_on_video(robot_frames, robot_masks, bg_video_path, output_path="final_output.mp4"):
    print(f"\n[Pipeline] 开始合成视频...")
    
    # 1. 打开背景视频
    if not os.path.exists(bg_video_path):
        print(f"[Error] 找不到背景视频: {bg_video_path}")
        return
    
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频文件: {bg_video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    bg_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bg_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. 准备输出视频写入器
    # 默认输出 mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (bg_w, bg_h))
    
    num_sim_frames = len(robot_frames)
    print(f"[Pipeline] 背景尺寸: {bg_w}x{bg_h}, 仿真帧数: {num_sim_frames}")
    
    # 确保仿真帧数不超过背景视频长度
    frame_limit = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(num_sim_frames, frame_limit)
    
    for i in range(limit):
        ret, bg_frame = cap.read()
        if not ret:
            break 
            
        # --- A. 获取仿真的 RGB 和 Mask ---
        sim_rgb = robot_frames[i]     # RGB
        sim_mask = robot_masks[i]     # Mask (ID图)
        
        # 将 RGB 转为 BGR (OpenCV格式)
        robot_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
        
        # Resize 到背景视频大小
        # 注意: cv2.resize 的 dsize 参数是 (width, height)
        if (robot_bgr.shape[1] != bg_w) or (robot_bgr.shape[0] != bg_h):
            robot_bgr_resized = cv2.resize(robot_bgr, (bg_w, bg_h))
            sim_mask_resized = cv2.resize(sim_mask, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)
        else:
            robot_bgr_resized = robot_bgr
            sim_mask_resized = sim_mask
        
        # --- B. 处理 Mask ---
        # MuJoCo Mask 中: 0是背景，>0 是物体(机械臂)
        if len(sim_mask_resized.shape) == 3:
            geom_ids = sim_mask_resized[:, :, 0]
        else:
            geom_ids = sim_mask_resized
            
        # 生成二值掩码: 1表示机械臂，0表示背景
        mask_binary = (geom_ids > 0).astype(float) 
        mask_3ch = cv2.merge([mask_binary, mask_binary, mask_binary]) # 扩展成3通道
        
        # --- C. 像素合成 (Alpha Blending思路) ---
        # 公式: 结果 = (机械臂 * Mask) + (背景 * (1 - Mask))
        foreground = robot_bgr_resized.astype(float) * mask_3ch
        background = bg_frame.astype(float) * (1.0 - mask_3ch)
        
        final_frame = (foreground + background).astype(np.uint8)
        
        # --- D. 写入结果 ---
        out.write(final_frame)
        
        # 实时显示 (按ESC退出)
        cv2.imshow("Merging...", final_frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[Success] 合成完成! 视频已保存为: {output_path}")

# =========================================================================
# 主程序
# =========================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [配置 1] XML 与 数据路径
    xml_path = os.path.join(current_dir, "R5", "R5a", "meshes", "dual_arm_scene.xml")
    data_path_L = os.path.join(current_dir, "actions_left_shoulders.npz")
    data_path_R = os.path.join(current_dir, "actions_right_shoulders.npz")
    
    # [配置 2] 这里的名字改为了你的 .mkv 文件
    bg_video_path = os.path.join(current_dir, "video_human_inpaint.mkv")

    # 1. 准备数据
    print(">>> 正在加载数据...")
    Mat_base_L_T_camera = np.array([[1., 0., 0., 0.], [0., 1., 0., -0.25], [0., 0., 1., 0.2], [0., 0., 0., 1.]])
    Mat_base_R_T_camera = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.25], [0., 0., 1., 0.2], [0., 0., 0., 1.]])
    
    seqs_L_raw = load_and_transform_data(data_path_L) # (N, 6)
    seqs_R_raw = load_and_transform_data(data_path_R) # (N, 6)
    
    dual_robot = DualArmController(xml_path, max_steps=100)
    
    Mat_world_T_base_L = pose_to_matrix(dual_robot._get_base_pose_world("L"))
    Mat_world_T_base_R = pose_to_matrix(dual_robot._get_base_pose_world("R"))
    
    def transform_seq(seq, Mat_base_T_cam, Mat_world_T_base):
        Mat_cam_T_seq = np.array([pose_to_matrix(p) for p in seq])
        Mat_world_T_seq = Mat_world_T_base @ Mat_base_T_cam @ Mat_cam_T_seq
        return np.array([matrix_to_pose(m) for m in Mat_world_T_seq])

    seqs_L_world = transform_seq(seqs_L_raw, Mat_base_L_T_camera, Mat_world_T_base_L)
    seqs_R_world = transform_seq(seqs_R_raw, Mat_base_R_T_camera, Mat_world_T_base_R)

    # 2. 加入 ee_width
    N = len(seqs_L_world)
    gripper_widths = np.zeros((N, 1))
    gripper_widths[:N//2] = 0.044 
    gripper_widths[N//2:] = 0.0   
    
    seqs_L_final = np.hstack([seqs_L_world, gripper_widths])
    seqs_R_final = np.hstack([seqs_R_world, gripper_widths])
    
    Mat_world_T_camera = Mat_world_T_base_L @ Mat_base_L_T_camera
    cam_pose = matrix_to_pose(Mat_world_T_camera)
    camera_poses_world = np.tile(cam_pose, (N, 1))

    # 3. [Task 2] 获取 RGB 和 Mask
    print(">>> [Task 2] 正在运行仿真生成帧...")
    
    # 尝试读取视频尺寸，确保仿真渲染尺寸和视频一致
    try:
        tmp_cap = cv2.VideoCapture(bg_video_path)
        if tmp_cap.isOpened():
            v_w = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tmp_cap.release()
            print(f"检测到视频分辨率: {v_w}x{v_h}，将使用此分辨率渲染仿真。")
        else:
            v_w, v_h = 1280, 720
            print("无法读取视频分辨率，默认使用 1280x720。")
    except:
        v_w, v_h = 1280, 720
    
    sim_frames, sim_masks = dual_robot.move_trajectory_with_camera(
        seqs_L_final, 
        seqs_R_final, 
        camera_poses_world, 
        max_steps_per_point=50, 
        kinematic_only=True,
        width=v_w,  # 动态适配视频宽度
        height=v_h  # 动态适配视频高度
    )
    
    # 4. [Task 3] 贴图合成
    print(">>> [Task 3] 正在合成视频...")
    if len(sim_frames) > 0:
        overlay_robot_on_video(sim_frames, sim_masks, bg_video_path)
    else:
        print("[Error] 仿真没有生成任何帧。")
