import time
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from bimanual import SingleArm

# Default calibration data if file is missing
DEFAULT_GRIPPER_CALIBRATION = {
    "-1.0": {"raw_data": [0.0, 0.0], "mean": 0.0},
    "-0.5": {"raw_data": [0.0, 0.0], "mean": 0.0},
    "0.0": {"raw_data": [0.0, 0.0], "mean": 0.0},
    "0.5": {"raw_data": [4.0, 4.0], "mean": 4.0},
    "1.0": {"raw_data": [13.0, 12.0], "mean": 12.5},
    "1.5": {"raw_data": [21.0, 20.0], "mean": 20.5},
    "2.0": {"raw_data": [30.0, 30.0], "mean": 30.0},
    "2.5": {"raw_data": [38.0, 39.0], "mean": 38.5},
    "3.0": {"raw_data": [48.0, 47.0], "mean": 47.5},
    "3.5": {"raw_data": [57.0, 56.0], "mean": 56.5},
    "4.0": {"raw_data": [65.0, 65.0], "mean": 65.0},
    "4.5": {"raw_data": [74.0, 73.0], "mean": 73.5},
    "5.0": {"raw_data": [82.0, 82.0], "mean": 82.0}
}

class RealDualArmController:
    def __init__(self, left_can='can0', right_can='can1', arm_type=0, calib_path="gripper_calibration.json"):
        """
        初始化双臂控制器，配置 CAN 通信
        :param left_can: 左臂 CAN 端口名称 (如 'can0')
        :param right_can: 右臂 CAN 端口名称 (如 'can1')
        :param arm_type: 机械臂类型 ID (默认为 0)
        :param calib_path: 夹爪校准文件路径
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
        
        # 加载夹爪校准数据
        self.calib_points = []
        calib_data = None
        
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
                print(f"[RealDualArm] Loaded gripper calibration from {calib_path}")
            except Exception as e:
                print(f"[RealDualArm] Error loading calibration file: {e}")
        
        if calib_data is None:
            print(f"[RealDualArm] Using default built-in calibration data.")
            calib_data = DEFAULT_GRIPPER_CALIBRATION

        try:
            points = []
            for k, v in calib_data.items():
                set_val = float(k)
                real_val = v["mean"]
                points.append((real_val, set_val))
            
            # 排序: 先按真实值升序, 再按设定值升序
            points.sort(key=lambda x: (x[0], x[1]))
            self.calib_points = points
        except Exception as e:
            print(f"[RealDualArm] Error processing calibration data: {e}")
            self.calib_points = []

    def go_home(self):
        """重置双臂控制器状态"""
        self.left_arm.go_home()
        self.right_arm.go_home()

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

    def _get_set_width_from_real(self, target_real_mm):
        """通过分段线性插值计算设定宽度"""
        if not self.calib_points:
            return target_real_mm
            
        points = self.calib_points
        
        if target_real_mm <= points[0][0]:
            return points[0][1]
        if target_real_mm >= points[-1][0]:
            return points[-1][1]
            
        for i in range(len(points) - 1):
            r1, s1 = points[i]
            r2, s2 = points[i+1]
            
            if r1 <= target_real_mm <= r2:
                if abs(r2 - r1) < 1e-6:
                    continue # 跳过重复点
                ratio = (target_real_mm - r1) / (r2 - r1)
                return s1 + ratio * (s2 - s1)
        return points[-1][1]

    def execute_trajectory(self, left_ee_poses_base, right_ee_poses_base, left_widths, right_widths, dt=0.1):
        """
        执行双臂轨迹
        :param left_poses: 左臂 4x4 变换矩阵列表【末端执行器】
        :param right_poses: 右臂 4x4 变换矩阵列表【末端执行器】
        :param left_widths: 左臂夹爪宽度列表 (真实宽度 mm)
        :param right_widths: 右臂夹爪宽度列表 (真实宽度 mm)
        :param dt: 控制周期 (秒)
        """
        steps = min(len(left_ee_poses_base), len(right_ee_poses_base))
        print(f"[RealDualArm] Starting execution of {steps} steps...")
        left_poses = [self._base_T_ee_to_flange_init_T_flange(mat) for mat in left_ee_poses_base]
        right_poses = [self._base_T_ee_to_flange_init_T_flange(mat) for mat in right_ee_poses_base]
        # print(f"[base]: {left_ee_poses_base[0]}\n[flange]: {left_poses[0]}")
        # print(f"[base]: {right_ee_poses_base[0]}\n[flange]: {right_poses[0]}")
        
        for i in range(steps):
            loop_start = time.time()
            
            # --- 左臂控制 ---
            l_mat = left_poses[i]
            if not np.any(np.isnan(l_mat)):
                try:
                    l_width_real = left_widths[i]
                    l_width_set = self._get_set_width_from_real(l_width_real)
                    l_xyzrpy = self._matrix_to_xyzrpy(l_mat)
                    
                    # 调用底层接口
                    self.left_arm.set_ee_pose_xyzrpy(l_xyzrpy)
                    self.left_arm.set_catch_pos(l_width_set)
                except Exception as e:
                    print(f"[Left Arm Error] Step {i}: {e}")

            # --- 右臂控制 ---
            r_mat = right_poses[i]
            if not np.any(np.isnan(r_mat)):
                try:
                    r_width_real = right_widths[i]
                    r_width_set = self._get_set_width_from_real(r_width_real)
                    r_xyzrpy = self._matrix_to_xyzrpy(r_mat)
                    
                    self.right_arm.set_ee_pose_xyzrpy(r_xyzrpy)
                    self.right_arm.set_catch_pos(r_width_set)
                except Exception as e:
                    print(f"[Right Arm Error] Step {i}: {e}")

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

def gripper_calibration():
    # gripper_calibration示例用法
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
    
    # 准备校准数据
    calibration_data = {}
    
    # 生成测试序列
    # 前向: -1 到 5, 步长 0.5
    forward_widths = np.arange(-1, 5.1, 0.5)
    # 后向: 5 到 -1, 步长 0.5
    backward_widths = np.arange(5, -1.1, -0.5)
    
    sequences = [
        ("Forward", forward_widths),
        ("Backward", backward_widths)
    ]
    
    print("Starting Gripper Calibration...")
    
    for direction, widths in sequences:
        print(f"--- Starting {direction} Sequence ---")
        for w in widths:
            w = float(np.round(w, 1)) # 避免浮点数精度问题
            print(f"Setting gripper width to: {w}")
            
            # 执行指令 (保持位姿不变，只改变夹爪宽度)
            # 注意：这里直接调用底层接口或者临时绕过校准逻辑，
            # 但因为我们修改了execute_trajectory，它会尝试转换。
            # 为了校准，我们需要直接发送set值。
            # 这里为了简单，我们直接调用底层arm对象
            
            l_xyzrpy = controller._matrix_to_xyzrpy(controller._base_T_ee_to_flange_init_T_flange(left_pose))
            r_xyzrpy = controller._matrix_to_xyzrpy(controller._base_T_ee_to_flange_init_T_flange(right_pose))
            
            controller.left_arm.set_ee_pose_xyzrpy(l_xyzrpy)
            controller.left_arm.set_catch_pos(w)
            controller.right_arm.set_ee_pose_xyzrpy(r_xyzrpy)
            controller.right_arm.set_catch_pos(w)
            
            # 等待稳定
            time.sleep(0.5)
            
            # 获取用户输入
            while True:
                try:
                    user_input = input(f"[{direction}] Set: {w}. Enter measured width: ")
                    real_val = float(user_input)
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # 记录数据
            key = f"{w:.1f}"
            if key not in calibration_data:
                calibration_data[key] = {"raw_data": [], "mean": 0.0}
            calibration_data[key]["raw_data"].append(real_val)
            
    # 计算平均值
    for key in calibration_data:
        raw = calibration_data[key]["raw_data"]
        if raw:
            calibration_data[key]["mean"] = sum(raw) / len(raw)
            
    # 保存结果
    output_file = "gripper_calibration.json"
    with open(output_file, "w") as f:
        json.dump(calibration_data, f, indent=4)
        
    print(f"Calibration finished. Data saved to {output_file}")
    
    controller.go_home()
    time.sleep(1)

if __name__ == "__main__":
    # 示例用法
    controller = RealDualArmController(left_can='can1', right_can='can3')
    
    # 定义位姿
    left_pose = np.eye(4)
    left_pose[:3, 3] = [0.3, 0.0, 0.1]
    right_pose = np.eye(4)
    right_pose[:3, 3] = [0.3, 0.0, 0.2]

    # 测试目标宽度 (真实宽度 mm)
    target_widths = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    
    print("Starting Gripper Control with Real Widths...")
    for real_w in target_widths:
        print(f"Target Real Width: {real_w} mm")
        
        # 现在 execute_trajectory 内部会自动转换 real_w -> set_w
        controller.execute_trajectory([left_pose], [right_pose], [real_w], [real_w], dt=0.1)
        time.sleep(1.0)
        
    controller.go_home()
    time.sleep(1)

