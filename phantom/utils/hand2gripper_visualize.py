import cv2
import numpy as np
from typing import List, Tuple

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