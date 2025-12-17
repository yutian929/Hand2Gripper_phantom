from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rgbd_playback'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ------------------- 新增下面这一行 -------------------
        # 意思：把 'launch' 文件夹下的所有 .py 文件，安装到 'share/包名/launch' 目录下
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # -----------------------------------------------------
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yutian',
    maintainer_email='yutian929@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'player = rgbd_playback.player_node:main',
            'player_mapper = rgbd_playback.player_mapper_node:main',
            'player_mapper_masker = rgbd_playback.player_mapper_masker_node:main',  # 新增的可执行文件
        ],
    },
)
