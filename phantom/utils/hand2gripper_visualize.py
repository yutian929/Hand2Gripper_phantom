import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection # <--- 修正: 导入 Line3DCollection
from matplotlib.colors import Normalize, Colormap

def vis_hand_2D(image: np.ndarray, joints_2d: np.ndarray, bbox: List[float], 
                is_right: bool, keypoint_color: Tuple[int, int, int] = (0, 255, 0),
                bbox_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    在图像上可视化单个手的2D关键点
    
    Args:
        image: 输入图像 (BGR格式)
        joints_2d: 单个手的2D关键点 (N, 2) 的numpy数组
        bbox: 边界框 [x1, y1, x2, y2]
        is_right: 手部类型 (True=右手, False=左手)
        keypoint_color: 关键点颜色 (BGR)
        bbox_color: 边界框颜色 (BGR)
    
    Returns:
        可视化后的图像
    """
    vis_image = image.copy()
    
    # 绘制关键点
    if joints_2d is not None and len(joints_2d) > 0:
        for j, (x, y) in enumerate(joints_2d):
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(vis_image, (int(x), int(y)), 3, keypoint_color, -1)
                # 可选：添加关键点编号
                cv2.putText(vis_image, str(j), (int(x)+5, int(y)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, keypoint_color, 1)
    
    # 绘制边界框
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 2)
        
        # 添加手部类型标签
        hand_type = "Right" if is_right else "Left"
        cv2.putText(vis_image, hand_type, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
    
    return vis_image

def vis_hand_mesh(image: np.ndarray, rendered_mesh: np.ndarray, bbox: List[float], 
                 is_right: bool, bbox_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    在图像上可视化手部mesh渲染结果
    
    Args:
        image: 输入图像 (BGR格式)
        rendered_mesh: 渲染的mesh图像 (H, W, 4) 包含RGBA通道
        bbox: 边界框 [x1, y1, x2, y2]
        is_right: 手部类型 (True=右手, False=左手)
        bbox_color: 边界框颜色 (BGR)
    
    Returns:
        可视化后的图像
    """
    # 将输入图像转换为RGB格式并归一化
    input_img = image.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    
    # 将mesh渲染结果叠加到原图上
    if rendered_mesh.shape[2] == 4:  # RGBA
        # Alpha blending
        input_img_overlay = input_img[:, :, :3] * (1 - rendered_mesh[:, :, 3:]) + rendered_mesh[:, :, :3] * rendered_mesh[:, :, 3:]
    else:  # RGB
        input_img_overlay = rendered_mesh[:, :, :3]
    
    # 转换回BGR格式
    vis_image = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
    
    # 绘制边界框
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 2)
        
        # 添加手部类型标签
        hand_type = "Right" if is_right else "Left"
        cv2.putText(vis_image, f"{hand_type} Hand Mesh", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
    
    return vis_image

def vis_hand_2D_skeleton(image: np.ndarray, joints_2d: np.ndarray, bbox: List[float], 
                             is_right: bool, keypoint_color: Tuple[int, int, int] = (0, 255, 0),
                             bbox_color: Tuple[int, int, int] = (255, 0, 0),
                             skeleton_color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    在图像上可视化手部2D关键点并绘制骨架连接
    
    Args:
        image: 输入图像 (BGR格式)
        joints_2d: 单个手的2D关键点 (21, 2) 的numpy数组
        bbox: 边界框 [x1, y1, x2, y2]
        is_right: 手部类型 (True=右手, False=左手)
        keypoint_color: 关键点颜色 (BGR)
        bbox_color: 边界框颜色 (BGR)
        skeleton_color: 骨架连接线颜色 (BGR)
    
    Returns:
        可视化后的图像
    """
    vis_image = vis_hand_2D(image, joints_2d, bbox, is_right, keypoint_color, bbox_color)
    
    # MANO手部骨架连接定义
    skeleton_connections = [
        # 拇指
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 食指
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 中指
        (0, 9), (9, 10), (10, 11), (11, 12),
        # 无名指
        (0, 13), (13, 14), (14, 15), (15, 16),
        # 小指
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # 绘制骨架连接线
    if joints_2d is not None and len(joints_2d) >= 21:
        for start_idx, end_idx in skeleton_connections:
            if (start_idx < len(joints_2d) and end_idx < len(joints_2d) and
                not (np.isnan(joints_2d[start_idx]).any() or np.isnan(joints_2d[end_idx]).any())):
                start_point = (int(joints_2d[start_idx][0]), int(joints_2d[start_idx][1]))
                end_point = (int(joints_2d[end_idx][0]), int(joints_2d[end_idx][1]))
                cv2.line(vis_image, start_point, end_point, skeleton_color, 2)
    
    return vis_image

def vis_hand_2D_skeleton_without_bbox(image: np.ndarray, joints_2d: np.ndarray, is_right: bool,
                                 keypoint_color: Tuple[int, int, int] = (0, 255, 0),
                                 skeleton_color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    在图像上可视化手部2D关键点并绘制骨架连接（不绘制边界框）
    """
    vis_image = image.copy()
    # 绘制关键点
    if joints_2d is not None and len(joints_2d) > 0:
        for j, (x, y) in enumerate(joints_2d):
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(vis_image, (int(x), int(y)), 3, keypoint_color, -1)
                # 可选：添加关键点编号
                cv2.putText(vis_image, str(j), (int(x)+5, int(y)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, keypoint_color, 1)
    # 添加手部类型标签
    hand_type = "Right" if is_right else "Left"
    cv2.putText(vis_image, f"{hand_type}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 定义每个手指的连接和颜色
    # 颜色顺序: 拇指(绿), 食指(红), 中指(蓝), 无名指(粉), 小指(黄)
    finger_connections = [
        # 拇指
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        # 食指
        [(0, 5), (5, 6), (6, 7), (7, 8)],
        # 中指
        [(0, 9), (9, 10), (10, 11), (11, 12)],
        # 无名指
        [(0, 13), (13, 14), (14, 15), (15, 16)],
        # 小指
        [(0, 17), (17, 18), (18, 19), (19, 20)],
    ]
    finger_colors = [
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 0, 0),   # Blue
        (255, 0, 255), # Pink (Magenta)
        (0, 255, 255)  # Yellow
    ]

    # 绘制骨架连接线
    if joints_2d is not None and len(joints_2d) >= 21:
        for i, connections in enumerate(finger_connections):
            color = finger_colors[i]
            for start_idx, end_idx in connections:
                if (start_idx < len(joints_2d) and end_idx < len(joints_2d) and
                    not (np.isnan(joints_2d[start_idx]).any() or np.isnan(joints_2d[end_idx]).any())):
                    start_point = (int(joints_2d[start_idx][0]), int(joints_2d[start_idx][1]))
                    end_point = (int(joints_2d[end_idx][0]), int(joints_2d[end_idx][1]))
                    cv2.line(vis_image, start_point, end_point, color, 2)

    return vis_image


def vis_hand_2D_skeleton_contact(image: np.ndarray, joints_2d: np.ndarray, bbox: List[float], 
                                 is_right: bool, contact_joint_out: np.ndarray,
                                 eval_threshold: float = 0.5,
                                 keypoint_color: Tuple[int, int, int] = (0, 255, 0),
                                 bbox_color: Tuple[int, int, int] = (255, 0, 0),
                                 skeleton_color: Tuple[int, int, int] = (0, 255, 255),
                                 contact_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    在图像上可视化手部2D关键点并绘制骨架连接，并根据接触关节点输出绘制接触关节点
    """
    vis_image = vis_hand_2D_skeleton(image, joints_2d, bbox, is_right, keypoint_color, bbox_color, skeleton_color)
    # 对contact_joint_out，用numpy，做softmax
    contact_joint_out = np.exp(contact_joint_out) / np.sum(np.exp(contact_joint_out))
    # 根据contact_joint_out，绘制接触关节点
    for i in range(len(contact_joint_out)):
        if contact_joint_out[i] > eval_threshold:
            cv2.circle(vis_image, (int(joints_2d[i][0]), int(joints_2d[i][1])), 3, contact_color, -1)
    return vis_image

def vis_selected_gripper(image: np.ndarray, joints_2d: np.ndarray, base_left_right_joint_ids: np.array, gripper_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    在图像上可视化选择gripper
    """
    vis_image = image.copy()
    # 绘制base关节点
    cv2.circle(vis_image, (int(joints_2d[base_left_right_joint_ids[0]][0]), int(joints_2d[base_left_right_joint_ids[0]][1])), 3, gripper_color, -1)
    cv2.putText(vis_image, "B", (int(joints_2d[base_left_right_joint_ids[0]][0]), int(joints_2d[base_left_right_joint_ids[0]][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gripper_color, 2)
    # 绘制left关节点
    cv2.circle(vis_image, (int(joints_2d[base_left_right_joint_ids[1]][0]), int(joints_2d[base_left_right_joint_ids[1]][1])), 3, gripper_color, -1)
    cv2.putText(vis_image, "L", (int(joints_2d[base_left_right_joint_ids[1]][0]), int(joints_2d[base_left_right_joint_ids[1]][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gripper_color, 2)
    # 绘制right关节点
    cv2.circle(vis_image, (int(joints_2d[base_left_right_joint_ids[2]][0]), int(joints_2d[base_left_right_joint_ids[2]][1])), 3, gripper_color, -1)
    cv2.putText(vis_image, "R", (int(joints_2d[base_left_right_joint_ids[2]][0]), int(joints_2d[base_left_right_joint_ids[2]][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gripper_color, 2)
    return vis_image

def vis_segmentation(image: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.5, mask_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    在图像上可视化分割掩膜
    
    Args:
        image: 输入图像 (BGR格式)
        seg_mask: 分割掩膜 (H, W) 的numpy数组，布尔类型或二值图
        alpha: 掩膜透明度
        mask_color: 掩膜颜色 (BGR)
    
    Returns:
        可视化后的图像
    """
    vis_image = image.copy()
    
    # 创建彩色掩膜
    colored_mask = np.zeros_like(vis_image, dtype=np.uint8)
    colored_mask[seg_mask > 0] = mask_color
    
    # 融合图像和掩膜
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, colored_mask, alpha, 0)
    
    return vis_image

def _plot_single_trajectory(ax, npz_path: str, trajectory_color: str, label: str):
    """
    在给定的 3D 坐标轴上绘制单条末端执行器轨迹。
    """
    try:
        data = np.load(npz_path)
        ee_pts = data['ee_pts']
        ee_oris = data['ee_oris']
        ee_widths = data['ee_widths']
        print(f"成功加载 '{label}' 数据，共 {len(ee_pts)} 帧。")
    except Exception as e:
        print(f"加载文件 '{npz_path}' 失败: {e}")
        return None, None

    # --- 绘制轨迹线 ---
    # 使用单一颜色绘制轨迹，而不是根据宽度变化
    ax.plot(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2], color=trajectory_color, label=label, linewidth=2)

    # --- 在轨迹上绘制姿态坐标系 ---
    step = max(1, len(ee_pts) // 15)
    axis_length = 0.005

    for i in range(0, len(ee_pts), step):
        origin = ee_pts[i]
        rot_matrix = ee_oris[i]
        
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]
        
        # 仅绘制X轴（接近方向）以保持清晰
        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=axis_length, color=trajectory_color, normalize=True)

    # 标记起点和终点
    ax.scatter(ee_pts[0, 0], ee_pts[0, 1], ee_pts[0, 2], color=trajectory_color, marker='o', s=100, depthshade=False)
    ax.scatter(ee_pts[-1, 0], ee_pts[-1, 1], ee_pts[-1, 2], color=trajectory_color, marker='x', s=100, depthshade=False)
    
    return ee_pts

def visualize_multiple_trajectories(npz_paths: List[str], labels: List[str]):
    """
    加载并可视化多个包含末端执行器轨迹的 .npz 文件以进行对比。
    """
    if len(npz_paths) != len(labels):
        raise ValueError("npz_paths 和 labels 的数量必须一致。")

    # 设置 Matplotlib 3D 绘图环境
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    all_points = []

    # 循环绘制每条轨迹
    for i, (path, label) in enumerate(zip(npz_paths, labels)):
        color = colors[i % len(colors)]
        points = _plot_single_trajectory(ax, path, color, label)
        if points is not None:
            all_points.append(points)

    if not all_points:
        print("没有成功加载任何轨迹数据，无法生成图像。")
        return

    # --- 设置图表样式 ---
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector Trajectory Comparison')

    # 自动调整坐标轴范围以保证所有内容可见
    all_points_np = np.concatenate(all_points, axis=0)
    max_range = np.array([all_points_np[:,0].max()-all_points_np[:,0].min(), 
                          all_points_np[:,1].max()-all_points_np[:,1].min(), 
                          all_points_np[:,2].max()-all_points_np[:,2].min()]).max() / 2.0
    mid_x = (all_points_np[:,0].max()+all_points_np[:,0].min()) * 0.5
    mid_y = (all_points_np[:,1].max()+all_points_np[:,1].min()) * 0.5
    mid_z = (all_points_np[:,2].max()+all_points_np[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.grid(True)
    plt.show()

def analyze_and_correct_trajectory(paths_to_visualize: List[str]):
        """
        分析两个轨迹文件之间的坐标系差异，并计算修正矩阵。
        """
        # 2. 分析坐标系差异
        print("\n" + "="*30)
        print("Analyzing Coordinate System Difference")
        print("="*30)

        origin_data = np.load(paths_to_visualize[0])
        hand2gripper_data = np.load(paths_to_visualize[1])

        # --- 使用所有帧的平均值来计算 R_fix，以获得更鲁棒的结果 ---
        R_origin_all = origin_data['ee_oris']
        R_hand2gripper_all = hand2gripper_data['ee_oris']

        # 逐帧计算 R_fix 并求和
        sum_R_fix = np.zeros((3, 3))
        num_frames = len(R_origin_all)
        for i in range(num_frames):
            R_o = R_origin_all[i]
            R_h = R_hand2gripper_all[i]
            R_fix_i = R_h.T @ R_o
            sum_R_fix += R_fix_i

        # 计算平均矩阵
        avg_R_fix = sum_R_fix / num_frames

        # 平均后的矩阵不一定是正交的，需要通过SVD进行重正交化
        U, S, Vt = np.linalg.svd(avg_R_fix)
        R_fix = U @ Vt  # 这就是最接近平均旋转的有效旋转矩阵

        print("Calculated Average Fix Rotation Matrix (R_fix) over all frames:\n", np.round(R_fix, 2))

        # 寻找最接近的90度旋转矩阵
        # R_fix 的每一列告诉我们，Original 的 (X, Y, Z) 轴在 Hand2Gripper 坐标系中是如何表示的。
        # 例如, R_fix 的第一列 [c11, c21, c31] 是 Original 的 X 轴在 H2G 坐标系下的坐标。
        # 如果 c11≈1, c21≈0, c31≈0，说明 Original.X 和 H2G.X 是同一个方向。
        # 如果 c11≈0, c21≈-1, c31≈0，说明 Original.X 和 H2G.Y 的反方向是同一个方向。
        
        axis_map = ['X', 'Y', 'Z']
        print("\n--- Interpretation ---")
        print("This matrix describes how to get from 'Hand2Gripper' coords to 'Original' coords.")
        
        # 分析 R_fix 的每一列
        for i in range(3):
            orig_axis = axis_map[i]
            col = R_fix[:, i]
            # 找到绝对值最大的元素，确定是哪个轴
            dominant_axis_idx = np.argmax(np.abs(col))
            dominant_axis_val = col[dominant_axis_idx]
            
            h2g_axis = axis_map[dominant_axis_idx]
            direction = "positive" if dominant_axis_val > 0 else "negative"
            
            print(f"Original '{orig_axis}' axis corresponds to the {direction} '{h2g_axis}' axis of Hand2Gripper.")

        print("\n--- Suggested Solution ---")
        print("Based on the analysis, you likely need to apply a fixed rotation.")
        print("For example, if 'Original X' is 'negative Y' and 'Original Y' is 'positive X',")
        print("this corresponds to a -90 degree rotation around the Z axis.")
        print("Please check the interpretation above to determine the exact rotation needed.")

        # 3. 自动计算并应用修正矩阵
        print("\n" + "="*30)
        print("Automatically Calculating and Verifying Correction")
        print("="*30)

        # --- 自动计算修正矩阵 ---
        R_correction_auto = np.zeros((3, 3))
        # 遍历 R_fix 的每一列 (代表 Original 的 X, Y, Z 轴)
        for i in range(3):
            col = R_fix[:, i]
            # 找到绝对值最大的元素的索引
            dominant_axis_idx = np.argmax(np.abs(col))
            # 获取该元素的符号
            sign = np.sign(col[dominant_axis_idx])
            # 在新矩阵的对应位置设置 +1 或 -1
            R_correction_auto[dominant_axis_idx, i] = sign

        print("Automatically Calculated Correction Matrix (R_correction_auto):\n", R_correction_auto)

        # 应用修正
        R_hand2gripper_all = hand2gripper_data['ee_oris']
        R_corrected_all = R_hand2gripper_all @ R_correction_auto

        # 重新计算并打印前几帧的角度差
        R_origin_all = origin_data['ee_oris']
        
        print("\n--- New Orientation Difference after Auto-Correction ---")
        total_error = 0
        for i in range(len(R_origin_all)):
            R_o = R_origin_all[i]
            R_c = R_corrected_all[i]
            
            # 计算两个旋转矩阵之间的角度差
            delta_R = R_o @ R_c.T
            # clip a_cos to [-1, 1] to avoid numerical errors
            trace_val = (np.trace(delta_R) - 1) / 2.0
            angle_rad = np.arccos(np.clip(trace_val, -1.0, 1.0))
            angle_deg = np.rad2deg(angle_rad)
            total_error += angle_deg
            if i < 5: # 只打印前5帧
                print(f"Frame {i}: New Difference = {angle_deg:.2f} degrees")
        
        avg_error = total_error / len(R_origin_all)
        print(f"\nAverage difference over all frames: {avg_error:.2f} degrees")


        # 可选：将修正后的轨迹保存，用于可视化对比
        corrected_path = "/tmp/corrected_trajectory.npz"
        np.savez(
            corrected_path,
            ee_pts=hand2gripper_data['ee_pts'],
            ee_oris=R_corrected_all,
            ee_widths=hand2gripper_data['ee_widths']
        )
        print(f"\nSaved corrected trajectory to {corrected_path}")
        print("You can now visualize the 'Original' vs 'Corrected' trajectories.")

        # 可视化对比 Original 和 Corrected
        visualize_multiple_trajectories(
            [paths_to_visualize[0], corrected_path],
            ["Original Trajectory", "Corrected Hand2Gripper"]
        )




if __name__ == '__main__':
    # 定义要对比的轨迹文件路径和标签
    paths_to_visualize = [
        [
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed_origin/epic/0/smoothing_processor/smoothed_actions_left_shoulders.npz",
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed/epic/0/smoothing_processor/smoothed_actions_left_shoulders.npz"
        ],
        [
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed_origin/epic/0/smoothing_processor/smoothed_actions_right_shoulders.npz",
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed/epic/0/smoothing_processor/smoothed_actions_right_shoulders.npz"
        ],
        [
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed_origin/epic/1/smoothing_processor/smoothed_actions_left_shoulders.npz",
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed/epic/1/smoothing_processor/smoothed_actions_left_shoulders.npz"
        ],
        [
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed_origin/epic/1/smoothing_processor/smoothed_actions_right_shoulders.npz",
        "/home/yutian/projs/Hand2Gripper_phantom/data/processed/epic/1/smoothing_processor/smoothed_actions_right_shoulders.npz"
        ],
    ]
    
    labels_for_paths = [
        [
        "Original Trajectory 0 left_shoulders",
        "Hand2Gripper Trajectory 0 left_shoulders"
        ],
        [
        "Original Trajectory 0 right_shoulders",
        "Hand2Gripper Trajectory 0 right_shoulders"
        ],
                [
        "Original Trajectory 1 left_shoulders",
        "Hand2Gripper Trajectory 1 left_shoulders"
        ],
        [
        "Original Trajectory 1 right_shoulders",
        "Hand2Gripper Trajectory 1 right_shoulders"
        ],

    ]

    # 可视化对比两条轨迹
    assert len(paths_to_visualize) == len(labels_for_paths)
    for paths_to_visualize, labels_for_paths in zip(paths_to_visualize, labels_for_paths):
        visualize_multiple_trajectories(paths_to_visualize, labels_for_paths)
        analyze_and_correct_trajectory(paths_to_visualize)