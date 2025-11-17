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


class VOTester:
    """
    用于测试ORB_SLAM3_RGBD_VO的类。
    """
    def __init__(self, rgb_video_path, depth_npy_path, camera_intri_json_path):
        """
        初始化测试器。

        :param rgb_video_path: RGB视频文件路径。
        :param depth_npy_path: 深度图.npy文件路径。
        :param camera_matrix: 3x3相机内参矩阵。
        """
        self.rgb_video_path = rgb_video_path
        self.depth_npy_path = depth_npy_path
        self.camera_intri_json_path = camera_intri_json_path

    def _prepare_data(self):
        """检查数据文件，如果深度文件不存在，则创建一个假的深度文件。"""
        if not os.path.exists(self.rgb_video_path):
            raise FileNotFoundError(f"错误: 找不到RGB视频文件 '{self.rgb_video_path}'。")

        if not os.path.exists(self.depth_npy_path):
            print(f"警告: 找不到深度文件 '{self.depth_npy_path}'。")
            print("正在创建一个假的深度文件用于演示...")
            
            try:
                video_frames = media.read_video(self.rgb_video_path)
                # 创建一个所有点深度都为2米的假深度数据
                fake_depth = np.ones((len(video_frames), video_frames.shape[1], video_frames.shape[2]), dtype=np.float32) * 2.0
                np.save(self.depth_npy_path, fake_depth)
                print(f"已创建假的深度文件: '{self.depth_npy_path}'")
            except Exception as e:
                print(f"创建假的深度文件失败: {e}")
                raise

    def run_test(self):
        """
        运行完整的VO测试流程。
        """
        try:
            self._prepare_data()

            # 初始化并运行VO
            print("\n--- 开始运行RGB-D VO ---")
            vo = ORB_SLAM3_RGBD_VO(self.rgb_video_path, self.depth_npy_path, self.camera_intri_json_path)
            trajectory = vo.run()
            print("--- RGB-D VO 运行结束 ---\n")

            # 打印并可视化结果
            print(f"相机运动轨迹 (共 {len(trajectory)} 个位姿):")
            print("最后一个位姿矩阵 T_w_c:")
            print(trajectory[-1])
            ORB_SLAM3_RGBD_VO.plot_trajectory(trajectory)

        except Exception as e:
            print(f"测试过程中发生错误: {e}")


if __name__ == "__main__":
    # 1. 定义路径和相机参数
    # !!! 请务必修改为您的真实文件路径 !!!
    RGB_VIDEO_PATH = "/home/yutian/Hand2Gripper_phantom/output/video_L.mp4"
    DEPTH_NPY_PATH = "/home/yutian/Hand2Gripper_phantom/output/depth.npy"
    CAMERA_INTRINSICS_PATH = "/home/yutian/Hand2Gripper_phantom/output/camera_intrinsics.json" 

    # 使用您自己的相机内参
    # K_MATRIX = np.array([[1057.7322998046875, 0, 972.5150756835938],
    #                      [0, 1057.7322998046875, 552.568359375],
    #                      [0, 0, 1]])

    # 2. 初始化并运行测试
    tester = VOTester(RGB_VIDEO_PATH, DEPTH_NPY_PATH, CAMERA_INTRINSICS_PATH)
    tester.run_test()