import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_trajectory(traj_dir):
    """
    从指定目录加载轨迹文件 (.npy)。
    假设文件名为 0.npy, 1.npy, ...
    """
    if not os.path.exists(traj_dir):
        print(f"目录不存在: {traj_dir}")
        return None

    # 按数字顺序排序
    traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.npy")), 
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if not traj_files:
        print(f"在 {traj_dir} 中未找到 .npy 文件")
        return None

    trajectory = [np.load(f) for f in traj_files]
    return np.array(trajectory)

def plot_trajectory(trajectory, title="Camera Trajectory"):
    """
    绘制相机轨迹及其方向。
    trajectory: (N, 4, 4) 的numpy数组
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 提取位置
    positions = trajectory[:, :3, 3]
    
    # 绘制轨迹线
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=title, linewidth=2)
    
    # 标注起点和终点
    if len(positions) > 0:
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', marker='o', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', marker='s', s=100, label='End')

    # 绘制相机位姿方向 (每隔一定帧数)
    if len(positions) > 1:
        max_range = np.max(np.max(positions, axis=0) - np.min(positions, axis=0))
        axis_length = max_range * 0.1 if max_range > 0 else 0.1
    else:
        axis_length = 0.1
    
    step = max(1, len(trajectory) // 20) # 最多绘制20个坐标系
    for i in range(0, len(trajectory), step):
        T_w_c = trajectory[i]
        origin = T_w_c[:3, 3]
        R_w_c = T_w_c[:3, :3]
        # 绘制Z轴 (蓝色) 表示相机朝向
        ax.quiver(origin[0], origin[1], origin[2], R_w_c[0, 2], R_w_c[1, 2], R_w_c[2, 2], color='b', length=axis_length, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    # 保持各轴比例一致
    try:
        ax.set_aspect('equal', adjustable='box')
    except:
        pass # 某些matplotlib版本可能不支持3d下的equal aspect
        
    plt.show()

if __name__ == "__main__":
    # 定义轨迹文件夹路径
    TRAJ_DIR = "/home/yutian/ros2_ws/camera_data/traj"
    
    print(f"正在加载轨迹: {TRAJ_DIR}")
    trajectory = load_trajectory(TRAJ_DIR)
    
    if trajectory is not None and len(trajectory) > 0:
        print(f"加载成功，共 {len(trajectory)} 帧。")
        plot_trajectory(trajectory, title="Recorded Trajectory")
    else:
        print("无法加载轨迹或轨迹为空。")
