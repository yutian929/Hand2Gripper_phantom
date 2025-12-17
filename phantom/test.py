import numpy as np
import cv2
import mediapy as media
import os

# ================= 配置区域 (已替换为你提供的路径) =================
NPZ_PATH = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/2/inpaint_processor/training_data_shoulders.npz"
VIDEO_PATH = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/2/video_overlay_Arx5_shoulders.mkv"
OUTPUT_PATH = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/2/test.mp4"
# =================================================================

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(NPZ_PATH):
        print(f"Error: NPZ file not found at {NPZ_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # 2. 加载数据
    print(f"Loading data from {NPZ_PATH}...")
    try:
        data = np.load(NPZ_PATH, allow_pickle=True)
        # 处理可能的保存格式差异：如果是直接 savez 保存的，直接取键
        # 如果保存的是一个对象，可能需要 data['arr_0'].item()['action_pos_left']
        # 这里假设是标准的 savez 格式
        if 'action_pos_left' in data:
            pos_left = data['action_pos_left']
            pos_right = data['action_pos_right']
        else:
            # 备用读取逻辑 (针对某些 save 实现)
            keys = list(data.keys())
            print(f"Keys in npz: {keys}")
            # 尝试解包第一个对象
            inner = data[keys[0]].item()
            pos_left = inner['action_pos_left']
            pos_right = inner['action_pos_right']
            
        print(f"Data shape: Left {pos_left.shape}, Right {pos_right.shape}")
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return

    # 3. 加载视频
    print(f"Loading video from {VIDEO_PATH}...")
    video = media.read_video(VIDEO_PATH)
    frames = list(video)
    print(f"Video frames: {len(frames)}")

    # 4. 定义变换矩阵 (必须与生成视频时的参数一致)
    # Camera 相对于 Base 的位置 (你的代码里的硬编码)
    Mat_base_L_T_camera = np.array([
        [1., 0., 0., 0.], 
        [0., 1., 0., -0.25], 
        [0., 0., 1., 0.2], 
        [0., 0., 0., 1.]
    ])
    
    # 我们需要将 World(Base) 坐标转回 Camera 坐标来投影
    # T_cam_base = inv(T_base_cam)
    Mat_camera_T_base_L = np.linalg.inv(Mat_base_L_T_camera)

    # 5. 相机内参 (假设是 640x480 的标准 D435 参数，与你渲染时一致)
    # 如果你的图像分辨率不是 640x480，这里可能需要按比例缩放
    H, W = frames[0].shape[:2]
    fx = 640
    fy = 640
    cx = 320
    cy = 240
    
    # 如果视频被裁切过(Square Crop)，需要调整内参
    # 假设你的 output_resolution 是 256 或其他正方形，且生成视频时做了 Center Crop
    # 这里我们先按原始 640x480 逻辑写，如果点偏了，说明这里需要针对 Crop 做偏移
    # 你的代码里有: blended = cv2.resize(blended, (self.output_resolution, self.output_resolution))
    # 这是一个潜在的坑。为了验证原始数据，我们假设内参适配了 Resize。
    # 简易修正：根据图像大小自动缩放内参
    scale_x = W / 640.0
    scale_y = H / 480.0
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    viz_frames = []
    print("Processing verification frames...")
    
    # 循环处理
    num_frames = min(len(frames), len(pos_left))
    for i in range(num_frames):
        img = frames[i].copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- 绘制左臂 (红色) ---
        pt_world_L = pos_left[i] # [x, y, z]
        pt_world_L_h = np.append(pt_world_L, 1.0)
        
        # 变换到相机系
        pt_cam_L = Mat_camera_T_base_L @ pt_world_L_h
        
        # 投影
        if pt_cam_L[2] > 0.01:
            u = int(pt_cam_L[0] * fx / pt_cam_L[2] + cx)
            v = int(pt_cam_L[1] * fy / pt_cam_L[2] + cy)
            
            # 绘制十字准星
            cv2.drawMarker(img, (u, v), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(img, "L", (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- 绘制右臂 (蓝色) ---
        pt_world_R = pos_right[i]
        pt_world_R_h = np.append(pt_world_R, 1.0)
        
        # 你的代码里 World 系原点就是 Left Base，所以直接用同一个矩阵
        pt_cam_R = Mat_camera_T_base_L @ pt_world_R_h
        
        if pt_cam_R[2] > 0.01:
            u = int(pt_cam_R[0] * fx / pt_cam_R[2] + cx)
            v = int(pt_cam_R[1] * fy / pt_cam_R[2] + cy)
            
            # 绘制十字准星
            cv2.drawMarker(img, (u, v), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(img, "R", (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        viz_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 6. 保存结果
    print(f"Saving verification video to {OUTPUT_PATH}...")
    media.write_video(OUTPUT_PATH, viz_frames, fps=15)
    print("Done! Please check the output video.")

if __name__ == "__main__":
    main()