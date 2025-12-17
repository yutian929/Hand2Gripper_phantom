import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. 声明数据路径参数
    # 默认路径设为你刚才提到的目录，运行时也可通过 data_dir:=... 覆盖
    data_dir_arg = DeclareLaunchArgument(
        'data_dir',
        default_value='/home/yutian/ros2_ws/disturb', 
        description='Path to directory containing video_L.mp4, depth.npy, camera_intrinsics.json and segmentation_processor/masks_arm.npy'
    )
    
    data_dir = LaunchConfiguration('data_dir')

    # 2. RTAB-MAP 通用参数配置
    # frame_id: 机器人的基坐标系 (通常是 camera_link)
    # subscribe_depth: 使用 RGB-D 模式
    # approx_sync: False (我们使用 Python 脚本实现了精确的时间戳同步)
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': True,
        'wait_imu_to_init': False,
        'queue_size': 30,
        'publish_tf': True  # 让 rtabmap 发布 map -> odom 的 TF
    }]

    # 3. Topic 重映射配置
    # 左边是 RTAB-MAP 节点的标准输入名，右边是我们播放节点发出的实际 Topic 名
    rtabmap_remappings = [
        ('rgb/image',       '/camera/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/camera/color/camera_info'),
        ('depth/image',     '/camera/camera/aligned_depth_to_color/image_raw')
    ]

    return LaunchDescription([
        data_dir_arg,

        # ---------------------------------------------------------
        # Node A: 静态 TF 发布 (Critical: 坐标系变换)
        # ---------------------------------------------------------
        # 将 camera_link (X前, Y左, Z上) 转换为 camera_color_optical_frame (Z前, X右, Y下)
        # 四元数 [-0.5, 0.5, -0.5, 0.5] 是标准的 ROS 到 Optical 的旋转变换
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_link_to_optical',
            arguments=['0', '0', '0', '-0.5', '0.5', '-0.5', '0.5', 'camera_link', 'camera_color_optical_frame']
        ),

        # ---------------------------------------------------------
        # Node B: 自定义数据播放器 + 记录器 (替换了原来的 player)
        # ---------------------------------------------------------
        Node(
            package='rgbd_playback',
            # [修改] 可执行文件改为 setup.py 里注册的新名字
            executable='player_mapper_masker', 
            name='rgbd_player_mapper_masker_node',
            output='screen',
            parameters=[{
                'data_dir': data_dir,
                'frequency': 30.0,
                # [注意] 这里移除了 loop 参数，因为目的是离线评估，跑完一次就保存退出
                'output_json': 'traj.json' # 结果会保存在 data_dir 下
            }],
            # Remappings 保持不变
            remappings=[
                ('camera/color/image_raw', '/camera/camera/color/image_raw'),
                ('camera/aligned_depth_to_color/image_raw', '/camera/camera/aligned_depth_to_color/image_raw'),
                ('camera/color/camera_info', '/camera/camera/color/camera_info')
            ]
        ),

        # ---------------------------------------------------------
        # Node C: RGBD Odometry (前端：视觉里程计)
        # ---------------------------------------------------------
        # 输入：RGB图 + 深度图
        # 输出：/odom (TF: odom -> camera_link)
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings
        ),

        # ---------------------------------------------------------
        # Node D: RTAB-MAP SLAM (后端：建图与闭环)
        # ---------------------------------------------------------
        # 输入：RGB图 + 深度图 + 里程计
        # 输出：/mapData, /grid_map (TF: map -> odom)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings,
            arguments=['-d'] # -d 参数表示启动时清空数据库，开始新的建图
        ),

        # ---------------------------------------------------------
        # Node E: Visualization (可视化界面)
        # ---------------------------------------------------------
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings
        ),
    ])