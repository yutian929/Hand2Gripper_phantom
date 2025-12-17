import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    data_dir_arg = DeclareLaunchArgument(
        'data_dir',
        default_value='/home/yutian/Hand2Gripper_phantom/data/raw/epic/0', 
        description='Path to the directory containing video_L.mp4, depth.npy and json'
    )

    # 1. RGBD 播放器节点
    player_node = Node(
        package='rgbd_playback',
        executable='player',
        name='rgbd_player_node',
        output='screen',
        parameters=[{
            'data_dir': LaunchConfiguration('data_dir'),
            'frequency': 30.0,
            'loop': True
        }],
        remappings=[
            ('camera/color/image_raw', '/camera/camera/color/image_raw'),
            ('camera/aligned_depth_to_color/image_raw', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('camera/color/camera_info', '/camera/camera/color/camera_info')
        ]
    )

    # 2. 静态 TF 发布 (关键修改)
    # 目标：将 camera_link (X前,Y左,Z上) 转换为 camera_color_optical_frame (Z前,X右,Y下)
    # 这里的参数顺序是: x y z qx qy qz qw
    # 四元数 [-0.5, 0.5, -0.5, 0.5] 正是对应： Roll=-90, Pitch=0, Yaw=-90 的旋转
    # 变换结果：Link的X轴 变成了 Optical的Z轴
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_broadcaster',
        arguments = ['0', '0', '0', '-0.5', '0.5', '-0.5', '0.5', 'camera_link', 'camera_color_optical_frame']
    )

    return LaunchDescription([
        data_dir_arg,
        player_node,
        static_tf
    ])