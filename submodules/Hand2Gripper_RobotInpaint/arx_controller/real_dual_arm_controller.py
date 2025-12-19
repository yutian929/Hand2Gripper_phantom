import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from bimanual import SingleArm

class RealDualArmController:
    def __init__(self, left_can='can0', right_can='can1', arm_type=0):
        """
        初始化双臂控制器，配置 CAN 通信
        :param left_can: 左臂 CAN 端口名称 (如 'can0')
        :param right_can: 右臂 CAN 端口名称 (如 'can1')
        :param arm_type: 机械臂类型 ID (默认为 0)
        """
        # 配置左臂
        self.left_config = {
            "can_port": left_can,
            "type": arm_type
        }
        print(f"[RealDualArm] Connecting Left Arm on {left_can}...")
        self.left_arm = SingleArm(self.left_config)
        
        # 配置右臂
        self.right_config = {
            "can_port": right_can,
            "type": arm_type
        }
        print(f"[RealDualArm] Connecting Right Arm on {right_can}...")
        self.right_arm = SingleArm(self.right_config)
        
        # 可以在这里添加初始化指令，如进入位置控制模式
        # self.left_arm.set_mode_position() 
        pass

    def _base_T_ee_to_flange_init_T_flange(self, M_base_T_ee):
        """将基座坐标系转换为法兰坐标系 (假设有固定偏移)"""
        # 这里假设一个简单的偏移矩阵作为示例
        offset_flange_init_T_base = np.eye(4)
        offset_flange_init_T_base[0, 3] = -0.1  # X 轴上偏移 0.1 米
        offset_flange_init_T_base[2, 3] = -0.1  # Z 轴上偏移 0.1 米
        offset_ee_T_flange = np.eye(4)
        offset_ee_T_flange[0, 3] = -0.15  # 末端执行器到法兰的偏移 0.15 米
        M_flange_init_T_flange = offset_flange_init_T_base @ M_base_T_ee @ offset_ee_T_flange
        return M_flange_init_T_flange
        

    def _matrix_to_xyzrpy(self, matrix):
        """将 4x4 矩阵转换为 [x, y, z, r, p, y]"""
        pos = matrix[:3, 3]
        rot = R.from_matrix(matrix[:3, :3])
        rpy = rot.as_euler('xyz', degrees=False)
        return np.concatenate([pos, rpy])

    def execute_trajectory(self, left_ee_poses_base, right_ee_poses_base, left_widths, right_widths, dt=0.1):
        """
        执行双臂轨迹
        :param left_poses: 左臂 4x4 变换矩阵列表【末端执行器】
        :param right_poses: 右臂 4x4 变换矩阵列表【末端执行器】
        :param left_widths: 左臂夹爪宽度列表
        :param right_widths: 右臂夹爪宽度列表
        :param dt: 控制周期 (秒)
        """
        steps = min(len(left_ee_poses_base), len(right_ee_poses_base))
        print(f"[RealDualArm] Starting execution of {steps} steps...")
        left_poses = [self._base_T_ee_to_flange_init_T_flange(mat) for mat in left_ee_poses_base]
        right_poses = [self._base_T_ee_to_flange_init_T_flange(mat) for mat in right_ee_poses_base]
        print(f"[base]: {left_ee_poses_base[0]}\n[flange]: {left_poses[0]}")
        print(f"[base]: {right_ee_poses_base[0]}\n[flange]: {right_poses[0]}")
        
        for i in range(steps):
            loop_start = time.time()
            
            # --- 左臂控制 ---
            l_mat = left_poses[i]
            l_width = left_widths[i]
            l_xyzrpy = self._matrix_to_xyzrpy(l_mat)
            
            # 调用底层接口
            self.left_arm.set_ee_pose_xyzrpy(l_xyzrpy)
            self.left_arm.set_catch_pos(l_width)

            # --- 右臂控制 ---
            r_mat = right_poses[i]
            r_width = right_widths[i]
            r_xyzrpy = self._matrix_to_xyzrpy(r_mat)
            
            self.right_arm.set_ee_pose_xyzrpy(r_xyzrpy)
            self.right_arm.set_catch_pos(r_width)

            # --- 频率控制 ---
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print("[RealDualArm] Trajectory execution finished.")

    def get_current_poses(self):
        """获取当前双臂位姿 (4x4 Matrix)"""
        # 左臂
        l_xyzrpy = self.left_arm.get_ee_pose_xyzrpy() # [x,y,z,r,p,y]
        l_mat = np.eye(4)
        l_mat[:3, 3] = l_xyzrpy[:3]
        l_mat[:3, :3] = R.from_euler('xyz', l_xyzrpy[3:], degrees=False).as_matrix()
        
        # 右臂
        r_xyzrpy = self.right_arm.get_ee_pose_xyzrpy()
        r_mat = np.eye(4)
        r_mat[:3, 3] = r_xyzrpy[:3]
        r_mat[:3, :3] = R.from_euler('xyz', r_xyzrpy[3:], degrees=False).as_matrix()
        
        return l_mat, r_mat

if __name__ == "__main__":
    # 示例用法
    controller = RealDualArmController(left_can='can1', right_can='can3')
    
    # 获取当前位姿
    l_curr, r_curr = controller.get_current_poses()
    print("Left Current:\n", l_curr)
    print("Right Current:\n", r_curr)
    # 定义简单的轨迹【基座坐标系下】
    left_pose = np.eye(4)
    left_pose[:3, 3] = [0.3, 0.0, 0.1]
    right_pose = np.eye(4)
    right_pose[:3, 3] = [0.3, 0.0, 0.2]
    left_poses = [left_pose]
    right_poses = [right_pose]
    left_widths = [0.05]
    right_widths = [0.05]
    while True:
        controller.execute_trajectory(left_poses, right_poses, left_widths, right_widths, dt=0.1)
