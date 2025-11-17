import os
import mediapy as media
import numpy as np
from python_orb_slam3 import ORBExtractor
import cv2
import json

class ORB_SLAM3_RGBD_VO:
    """
    一个使用ORB特征和深度信息的RGB-D视觉里程计类。
    """
    def __init__(self, rgb_video_path, depth_npy_path, camera_intri_json_path):
        """
        初始化RGB-D视觉里程计。

        :param rgb_video_path: RGB视频文件路径。
        :param depth_npy_path: 深度图.npy文件路径 (N, H, W)。
        :param camera_intri_json_path: 相机内参JSON文件路径。
        """

        print("正在加载数据...")
        self.rgb_frames = media.read_video(rgb_video_path)
        self.depth_frames = np.load(depth_npy_path)
        
        assert len(self.rgb_frames) == len(self.depth_frames), \
            f"RGB帧数 ({len(self.rgb_frames)}) 与 深度帧数 ({len(self.depth_frames)}) 不一致！"
        
        print(f"数据加载成功，共 {len(self.rgb_frames)} 帧。")

        # 加载相机内参
        with open(camera_intri_json_path, "r") as f:
            intrinsics = json.load(f)
        color_intr = intrinsics["color"]
        self.K = np.array([[color_intr["fx"], 0, color_intr["cx"]],
                                   [0, color_intr["fy"], color_intr["cy"]],
                                   [0, 0, 1]])
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.orb_extractor = ORBExtractor()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 初始化状态
        self.prev_data = None
        self.T_w_c = np.eye(4)  # 世界坐标系到相机坐标系的变换 (初始相机在原点)
        self.trajectory = [self.T_w_c] # 存储完整的4x4位姿矩阵

    def _get_3d_points(self, keypoints, depth_frame):
        """根据关键点和深度图计算3D坐标。"""
        points_3d = []
        valid_indices = []
        for i, kp in enumerate(keypoints):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            
            # 检查坐标是否在图像范围内
            if 0 <= v < depth_frame.shape[0] and 0 <= u < depth_frame.shape[1]:
                d = depth_frame[v, u]
                # 假设深度单位是米，且0是无效深度
                if d > 0:
                    z = d
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    points_3d.append([x, y, z])
                    valid_indices.append(i)
        
        return np.array(points_3d, dtype=np.float32), valid_indices

    def run_with_mask(self, arm_masks):
        """
        运行带有动态掩码的视觉里程计流程，以忽略手臂等运动物体。
        :param arm_masks: (N, H, W) 的布尔数组，True表示需要屏蔽的区域。
        """
        assert len(self.rgb_frames) == len(arm_masks), \
            f"帧数 ({len(self.rgb_frames)}) 与 掩码数 ({len(arm_masks)}) 不一致！"

        for i in range(len(self.rgb_frames)):
            rgb_frame = self.rgb_frames[i]
            depth_frame = self.depth_frames[i]
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            mask = arm_masks[i]

            # 在特征提取前，将掩码区域涂黑
            gray_frame[mask] = 0

            # 1. 提取特征
            kps, des = self.orb_extractor.detectAndCompute(gray_frame)

            if self.prev_data is None:
                # 如果是第一帧，只保存数据
                # 注意：这里我们保存未被掩码的kps和des，以便下一帧匹配
                # 但用于获取3D点的深度图和kps需要来自原始数据
                kps_full, des_full = self.orb_extractor.detectAndCompute(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY))
                self.prev_data = {'kps': kps_full, 'des': des_full, 'depth': depth_frame, 'mask': mask}
                continue

            # 2. 特征匹配
            # 注意：前一帧的描述子也应该从没有手臂的区域提取
            prev_gray_masked = cv2.cvtColor(self.rgb_frames[i-1], cv2.COLOR_BGR2GRAY)
            prev_gray_masked[self.prev_data['mask']] = 0
            prev_kps_masked, prev_des_masked = self.orb_extractor.detectAndCompute(prev_gray_masked)

            if prev_des_masked is None or des is None:
                print("警告: 当前帧或前一帧在掩码后没有提取到特征点，跳过。")
                self.prev_data = {'kps': kps, 'des': des, 'depth': depth_frame, 'mask': mask}
                continue

            matches = self.bf_matcher.match(prev_des_masked, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # 3. 获取3D-2D点对
            # 从前一帧获取3D点 (使用未被掩码的完整关键点列表)
            prev_kps_matched = [prev_kps_masked[m.queryIdx] for m in matches]
            points_3d_prev, valid_indices = self._get_3d_points(prev_kps_matched, self.prev_data['depth'])
            
            if len(valid_indices) < 10:
                print("警告: 有效的3D匹配点过少，跳过此帧。")
                self.prev_data = {'kps': kps, 'des': des, 'depth': depth_frame, 'mask': mask}
                continue

            # 获取当前帧对应的2D点
            curr_kps_matched = [kps[m.trainIdx] for m in matches]
            points_2d_curr = np.array([curr_kps_matched[i].pt for i in valid_indices], dtype=np.float32)

            # 4. 使用PnP求解相对位姿
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d_prev, points_2d_curr, self.K, None)
                if not success:
                    raise Exception("solvePnPRansac 失败")
                
                R, _ = cv2.Rodrigues(rvec)
                
                T_relative = np.eye(4)
                T_relative[:3, :3] = R
                T_relative[:3, 3] = tvec.flatten()

                # 5. 更新全局位姿
                self.T_w_c = self.T_w_c @ np.linalg.inv(T_relative)
                self.trajectory.append(self.T_w_c)

            except Exception as e:
                print(f"位姿估计失败: {e}")

            # 更新前一帧数据
            self.prev_data = {'kps': kps, 'des': des, 'depth': depth_frame, 'mask': mask}
        
        return np.array(self.trajectory)

    def run(self):
        """
        运行整个视觉里程计流程。
        """
        for i in range(len(self.rgb_frames)):
            # print(f"处理第 {i+1}/{len(self.rgb_frames)} 帧...")
            rgb_frame = self.rgb_frames[i]
            depth_frame = self.depth_frames[i]
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

            # 1. 提取特征
            kps, des = self.orb_extractor.detectAndCompute(gray_frame)

            if self.prev_data is None:
                # 如果是第一帧，只保存数据
                self.prev_data = {'kps': kps, 'des': des, 'depth': depth_frame}
                continue

            # 2. 特征匹配
            matches = self.bf_matcher.match(self.prev_data['des'], des)
            matches = sorted(matches, key=lambda x: x.distance) # 排序

            # 3. 获取3D-2D点对
            # 从前一帧获取3D点
            prev_kps_matched = [self.prev_data['kps'][m.queryIdx] for m in matches]
            points_3d_prev, valid_indices = self._get_3d_points(prev_kps_matched, self.prev_data['depth'])
            
            if len(valid_indices) < 10:
                print("警告: 有效的3D匹配点过少，跳过此帧。")
                self.prev_data = {'kps': kps, 'des': des, 'depth': depth_frame}
                continue

            # 获取当前帧对应的2D点
            curr_kps_matched = [kps[m.trainIdx] for m in matches]
            points_2d_curr = np.array([curr_kps_matched[i].pt for i in valid_indices], dtype=np.float32)

            # 4. 使用PnP求解相对位姿
            try:
                # 求解的是当前相机坐标系相对于前一帧相机坐标系的变换
                success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d_prev, points_2d_curr, self.K, None)
                if not success:
                    raise Exception("solvePnPRansac 失败")
                
                R, _ = cv2.Rodrigues(rvec)
                
                # 构建相对变换矩阵 T_curr_prev
                T_relative = np.eye(4)
                T_relative[:3, :3] = R
                T_relative[:3, 3] = tvec.flatten()

                # 5. 更新全局位姿
                # T_w_c_curr = T_w_c_prev @ T_prev_curr
                # T_prev_curr 是 T_curr_prev 的逆
                self.T_w_c = self.T_w_c @ np.linalg.inv(T_relative)
                self.trajectory.append(self.T_w_c)

            except Exception as e:
                print(f"位姿估计失败: {e}")

            # 更新前一帧数据
            self.prev_data = {'kps': kps, 'des': des, 'depth': depth_frame}
        
        return np.array(self.trajectory)

    @staticmethod
    def plot_trajectory(trajectory):
        """
        绘制相机轨迹及其方向。
        trajectory: (N, 4, 4) 的numpy数组
        """
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 提取位置
        positions = trajectory[:, :3, 3]
        
        # 绘制轨迹线
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Camera Trajectory")
        
        # 标注起点和终点
        if len(positions) > 0:
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', marker='o', s=100, label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', marker='s', s=100, label='End')

        # 绘制相机位姿方向 (每隔一定帧数)
        # 根据轨迹的运动范围动态调整坐标轴长度
        if len(positions) > 1:
            max_range = np.max(np.max(positions, axis=0) - np.min(positions, axis=0))
            axis_length = max_range * 0.1 if max_range > 0 else 0.1
        else:
            axis_length = 0.1 # 如果只有一个点，使用默认长度
        
        step = max(1, len(trajectory) // 20) # 最多绘制20个坐标系
        for i in range(0, len(trajectory), step):
            T_w_c = trajectory[i]
            origin = T_w_c[:3, 3]
            R_w_c = T_w_c[:3, :3]
            # 绘制X, Y, Z轴
            # ax.quiver(origin[0], origin[1], origin[2], R_w_c[0, 0], R_w_c[1, 0], R_w_c[2, 0], color='r', length=axis_length)
            # ax.quiver(origin[0], origin[1], origin[2], R_w_c[0, 1], R_w_c[1, 1], R_w_c[2, 1], color='g', length=axis_length)
            ax.quiver(origin[0], origin[1], origin[2], R_w_c[0, 2], R_w_c[1, 2], R_w_c[2, 2], color='b', length=axis_length)


        ax.set_title("RGB-D VO Camera Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        # 保持各轴比例一致
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    @staticmethod
    def plot_trajectories_comparison(traj1, label1, traj2, label2):
        """
        在一个图上绘制并比较两个相机轨迹。
        traj1: 第一个轨迹 (N, 4, 4) 的numpy数组
        label1: 第一个轨迹的标签
        traj2: 第二个轨迹 (M, 4, 4) 的numpy数组
        label2: 第二个轨迹的标签
        """
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制第一个轨迹
        positions1 = traj1[:, :3, 3]
        ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], label=label1, color='blue', linestyle='--')
        if len(positions1) > 0:
            ax.scatter(positions1[0, 0], positions1[0, 1], positions1[0, 2], color='green', marker='o', s=150, label='Start')
            ax.scatter(positions1[-1, 0], positions1[-1, 1], positions1[-1, 2], color='blue', marker='s', s=100, label=f'{label1} End')

        # 绘制第二个轨迹
        positions2 = traj2[:, :3, 3]
        ax.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], label=label2, color='red')
        if len(positions2) > 0:
            ax.scatter(positions2[-1, 0], positions2[-1, 1], positions2[-1, 2], color='red', marker='X', s=100, label=f'{label2} End')

        ax.set_title("Trajectory Comparison")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.show()


if __name__ == "__main__":
    # 1. 定义文件路径
    RGB_VIDEO_PATH = "/home/yutian/Hand2Gripper_phantom/data/raw/epic/0/video_L.mp4"
    DEPTH_NPY_PATH = "/home/yutian/Hand2Gripper_phantom/data/raw/epic/0/depth.npy"
    CAMERA_INTRINSICS_PATH = "/home/yutian/Hand2Gripper_phantom/data/raw/epic/0/camera_intrinsics.json"
    MASKS_ARM_NPY_PATH = "/home/yutian/Hand2Gripper_phantom/data/processed/epic/0/segmentation_processor/masks_arm.npy"

    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [RGB_VIDEO_PATH, DEPTH_NPY_PATH, CAMERA_INTRINSICS_PATH]):
        print("错误: 请确保所有文件路径都正确，并且文件存在。")
        print(f"RGB视频: {RGB_VIDEO_PATH}")
        print(f"深度NPY: {DEPTH_NPY_PATH}")
        print(f"相机内参: {CAMERA_INTRINSICS_PATH}")
        exit()

    trajectory_no_mask = None
    trajectory_with_mask = None

    # --- 场景1: 不使用掩码运行 ---
    print("\n" + "="*50)
    print("场景 1: 不使用掩码运行视觉里程计")
    print("="*50)
    try:
        # 初始化并运行VO
        vo_no_mask = ORB_SLAM3_RGBD_VO(RGB_VIDEO_PATH, DEPTH_NPY_PATH, CAMERA_INTRINSICS_PATH)
        trajectory_no_mask = vo_no_mask.run()
        print("--- 无掩码VO运行结束 ---\n")

        # 打印并可视化结果
        if len(trajectory_no_mask) > 1:
            print(f"相机运动轨迹 (共 {len(trajectory_no_mask)} 个位姿):")
            print("最后一个位姿矩阵 T_w_c:")
            print(trajectory_no_mask[-1])
            # ORB_SLAM3_RGBD_VO.plot_trajectory(trajectory_no_mask) # 单独绘图(可选)
        else:
            print("未能成功生成轨迹。")
            trajectory_no_mask = None

    except Exception as e:
        print(f"无掩码测试过程中发生错误: {e}")


    # --- 场景2: 使用掩码运行 ---
    if os.path.exists(MASKS_ARM_NPY_PATH):
        print("\n" + "="*50)
        print("场景 2: 使用动态掩码运行视觉里程计")
        print("="*50)
        try:
            arm_masks = np.load(MASKS_ARM_NPY_PATH)
            print(f"已加载掩码，形状为: {arm_masks.shape}, 类型为: {arm_masks.dtype}")

            # 初始化并运行VO
            vo_with_mask = ORB_SLAM3_RGBD_VO(RGB_VIDEO_PATH, DEPTH_NPY_PATH, CAMERA_INTRINSICS_PATH)
            trajectory_with_mask = vo_with_mask.run_with_mask(arm_masks)
            print("--- 带掩码VO运行结束 ---\n")

            # 打印并可视化结果
            if len(trajectory_with_mask) > 1:
                print(f"相机运动轨迹 (共 {len(trajectory_with_mask)} 个位姿):")
                print("最后一个位姿矩阵 T_w_c:")
                print(trajectory_with_mask[-1])
                # ORB_SLAM3_RGBD_VO.plot_trajectory(trajectory_with_mask) # 单独绘图(可选)
            else:
                print("未能成功生成轨迹。")
                trajectory_with_mask = None

        except Exception as e:
            print(f"带掩码测试过程中发生错误: {e}")

    # --- 场景3: 比较两个轨迹 ---
    if trajectory_no_mask is not None and trajectory_with_mask is not None:
        print("\n" + "="*50)
        print("场景 3: 比较两种情况下的轨迹")
        print("="*50)
        ORB_SLAM3_RGBD_VO.plot_trajectories_comparison(
            trajectory_no_mask, "Without Mask",
            trajectory_with_mask, "With Mask"
        )
    else:
        print("\n无法生成对比图，因为至少有一个轨迹未能成功计算。")