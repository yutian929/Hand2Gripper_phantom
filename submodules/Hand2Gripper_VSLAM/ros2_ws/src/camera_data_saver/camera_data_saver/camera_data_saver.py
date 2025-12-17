import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import os
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R  # 使用scipy来处理四元数
import message_filters

def camera_info_to_dict(camera_info):
    """将CameraInfo消息转换为字典"""
    return {
        'height': camera_info.height,
        'width': camera_info.width,
        'distortion_model': camera_info.distortion_model,
        'd': list(camera_info.d),  # 畸变系数
        'k': list(camera_info.k),  # 内参矩阵
        'r': list(camera_info.r),  # 去畸变矩阵
        'p': list(camera_info.p),  # 投影矩阵
        'binning_x': camera_info.binning_x,
        'binning_y': camera_info.binning_y,
        'roi': {
            'x_offset': camera_info.roi.x_offset,
            'y_offset': camera_info.roi.y_offset,
            'height': camera_info.roi.height,
            'width': camera_info.roi.width,
            'do_rectify': camera_info.roi.do_rectify
        }
    }

class CameraDataSaver(Node):
    def __init__(self):
        super().__init__('camera_data_saver')

        # 设置订阅话题
        self.bridge = CvBridge()
        self.image_count = 0
        self.max_images = 1000
        self.is_alive = True

        # 创建保存文件夹
        self.root_dir = 'camera_data'
        os.makedirs(self.root_dir, exist_ok=True)
        self.save_dir_color = os.path.join(self.root_dir, 'color')
        self.save_dir_depth = os.path.join(self.root_dir, 'depth')
        self.save_dir_traj = os.path.join(self.root_dir, 'traj')
        os.makedirs(self.save_dir_color, exist_ok=True)
        os.makedirs(self.save_dir_depth, exist_ok=True)
        os.makedirs(self.save_dir_traj, exist_ok=True)

        # QoS策略
        qos_profile = QoSProfile(depth=10)

        # 订阅相机信息，只保存一次
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, qos_profile)
        self.camera_info_saved = False

        # 用于暂存同步消息
        self.latest_rgb_msg = None
        self.latest_depth_msg = None

        # 使用message_filters同步RGB和深度图像
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        # 创建Transform监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 定时器，每0.1秒调用一次保存函数
        self.timer = self.create_timer(0.1, self.timer_save_callback)

    def camera_info_callback(self, msg: CameraInfo):
        """接收相机信息并保存一次"""
        if not self.camera_info_saved:
            camera_info_dict = camera_info_to_dict(msg)
            camera_info_filename = os.path.join(self.root_dir, 'camera_info.json')
            with open(camera_info_filename, 'w') as f:
                json.dump(camera_info_dict, f, indent=4)
            self.camera_info_saved = True
            self.get_logger().info("Camera info saved.")
            # 销毁订阅，因为它只需要一次
            self.destroy_subscription(self.camera_info_sub)

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        """同步回调，暂存对齐的消息"""
        self.latest_rgb_msg = rgb_msg
        self.latest_depth_msg = depth_msg

    def timer_save_callback(self):
        """定时器回调，用于保存数据"""
        if self.image_count >= self.max_images:
            if self.is_alive:  # 避免在节点关闭时重复打印
                self.get_logger().info("Reached max image count. Shutting down.")
                self.is_alive = False
                self.destroy_node()
            return

        if not self.camera_info_saved:
            self.get_logger().warn("Waiting for camera info...")
            return

        # 检查是否有新的同步消息
        if self.latest_rgb_msg is None or self.latest_depth_msg is None:
            self.get_logger().warn("No sync data updated...")
            return

        # 读取并清除暂存的消息，以避免重复保存
        rgb_msg = self.latest_rgb_msg
        depth_msg = self.latest_depth_msg
        self.latest_rgb_msg = None
        self.latest_depth_msg = None

        try:
            # 获取与图像时间戳对齐的位姿
            transform = self.tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
            pose = transform.transform.translation
            rotation = transform.transform.rotation

            # 获取4x4矩阵位姿
            pose_matrix = self.get_pose_matrix(pose, rotation)

            # 转换图像
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            depth_image = depth_image / 1000.0  # 转换为米

            # 保存RGB图像
            rgb_filename = os.path.join(self.save_dir_color, f'{self.image_count}.png')
            cv2.imwrite(rgb_filename, rgb_image)

            # 保存深度图像
            depth_filename = os.path.join(self.save_dir_depth, f'{self.image_count}.npy')
            np.save(depth_filename, depth_image)

            # 保存相机位姿
            pose_filename = os.path.join(self.save_dir_traj, f'{self.image_count}.npy')
            np.save(pose_filename, pose_matrix)

            self.get_logger().info(f"Saved aligned data frame {self.image_count}")

            # 更新计数器
            self.image_count += 1

        except Exception as e:
            self.get_logger().error(f"Error saving data: {e}")

    def get_pose_matrix(self, translation, rotation):
        """将位姿转换为4x4矩阵"""
        pose_matrix = np.eye(4)
        pose_matrix[0, 3] = translation.x
        pose_matrix[1, 3] = translation.y
        pose_matrix[2, 3] = translation.z
        pose_matrix[:3, :3] = self.quaternion_to_rotation_matrix(rotation)
        return pose_matrix

    def quaternion_to_rotation_matrix(self, q):
        """将四元数转换为旋转矩阵"""
        r = R.from_quat([q.x, q.y, q.z, q.w])  # 使用scipy处理四元数
        return r.as_matrix()  # 返回3x3的旋转矩阵

def main(args=None):
    rclpy.init(args=args)
    node = CameraDataSaver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
