# 文件路径: Hand2Gripper_Project/Hand2Gripper_phantom/submodules/my_robo/setup.py

import setuptools
import os

# --- 1. 外部依赖：根据您提供的所有导入语句确定 ---
# 请根据您的实际运行代码，确认并添加版本限制（如 >=1.19.0）
REQUIRED_PACKAGES = [
    "numpy",         
    "scipy",         
    "mujoco",        
    "opencv-python", # for cv2
    "matplotlib",    # for plotting (plt, Axes3D)
    # 假设您的项目还依赖于以下库，如果不需要可以删除
    "mediapy",
    "omegaconf",
]

# --- 2. 核心 setuptools 配置 ---
setuptools.setup(
    # 核心信息
    name="Hand2Gripper_RobotInpaint_ARX", 
    version="0.1.0",
    author="Hand2Gripper Project Contributors",
    description="MuJoCo Dual Arm Controller for the Hand2Gripper Project's ARX configuration.",
    
    # 查找 my_robo 目录下的所有 Python 包（例如 raw_mujoco）
    packages=setuptools.find_packages(),
    
    # 依赖关系
    install_requires=REQUIRED_PACKAGES,
    
    # --- 数据文件打包配置 ---
    # 包含非 Python 文件
    include_package_data=True, 
    
    # 确保 MuJoCo 模型文件 (XML, meshes) 被包含
    # 注意：这里的路径是相对于 setup.py 文件所在的 my_robo/ 目录
    package_data={
        # 针对所有的包 (例如 raw_mujoco)，包含通用数据文件
        "": [
            "*.xml", 
            "*.json", 
            "*.txt",
            "raw_mujoco/**/*.xml" # 确保 raw_mujoco 及其子目录下的 xml 文件被包含
        ],
        # 针对 my_robo 包，确保 meshes 目录被包含
        "my_robo": [
            "meshes/*", 
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License", 
    ],
    python_requires='>=3.8',
)