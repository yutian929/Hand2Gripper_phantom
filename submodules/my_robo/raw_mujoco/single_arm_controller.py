import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import bimanual
from scipy.spatial.transform import Rotation as R

class SingleArmController:
    def __init__(self, xml_path, end_effector_site="end_effector", base_link_name="base_link"):
        """
        初始化机器人控制器
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 保存关键名称
        self.ee_site_name = end_effector_site
        self.base_link_name = base_link_name
        
        # 1. 初始化常量与配置
        self.GRIPPER_OPEN_VAL = 0.044
        self.POSITION_THRESHOLD = 0.01  # 到达判定阈值 (m)
        
        # 2. 自动读取运动学参数
        self._init_kinematics_params()
        
        # 3. 复位机器人
        self.reset()

    def _init_kinematics_params(self):
        """[Internal] 从 XML 读取基座位置和末端偏移"""
        # A. 读取基座在世界坐标系下的位置
        try:
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
            if base_id == -1: raise ValueError(f"Body {self.base_link_name} not found")
            self.base_pos_world = self.model.body_pos[base_id].copy()
            print(f"[Init] Base Position (World): {self.base_pos_world}")
        except Exception as e:
            print(f"[Error] Init Base: {e}")
            self.base_pos_world = np.zeros(3)

        # B. 读取末端 Site 相对于法兰盘(Link6)的局部偏移
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
            if site_id == -1: raise ValueError(f"Site {self.ee_site_name} not found")
            self.ee_offset_local = self.model.site_pos[site_id].copy()
            print(f"[Init] EE Offset (Local): {self.ee_offset_local}")
        except Exception as e:
            print(f"[Error] Init EE Offset: {e}")
            self.ee_offset_local = np.zeros(3)

    def reset(self):
        """重置机器人状态"""
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)

    # ========================================================
    # 核心算法原子函数 (Internal Helpers)
    # ========================================================

    def _fk_base(self, joint_angles):
        """
        [FK] 正运动学: 关节空间 -> 基座坐标系
        Args:
            joint_angles (np.array): 6个关节角
        Returns:
            pose_base (np.array): [x, y, z, rx, ry, rz] 相对于基座的位姿
        """
        try:
            return bimanual.forward_kinematics(joint_angles)
        except Exception as e:
            print(f"[FK Error] {e}")
            return np.zeros(6)

    def _ik_base(self, target_pose_base):
        """
        [IK] 逆运动学: 基座坐标系 -> 关节空间
        Args:
            target_pose_base (np.array): [x, y, z, rx, ry, rz] 相对于基座的目标法兰盘位姿
        Returns:
            joint_angles (np.array or None): 目标关节角
        """
        try:
            return bimanual.inverse_kinematics(target_pose_base)
        except Exception as e:
            print(f"[IK Error] {e}")
            return None

    def _get_ee_pose_world(self):
        """
        [State] 获取当前末端(Site)在【世界坐标系】下的真实位姿
        Returns:
            pose_world (np.array): [x, y, z, rx, ry, rz]
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        
        # 1. 位置
        pos = self.data.site_xpos[site_id]
        
        # 2. 姿态 (Matrix -> Euler)
        mat = self.data.site_xmat[site_id].reshape(3, 3)
        r = R.from_matrix(mat)
        euler = r.as_euler('xyz', degrees=False)
        
        return np.concatenate([pos, euler])

    def _pose_tf_world2base(self, target_pose_world):
        """
        [Transform] 坐标变换: 指尖目标(世界) -> 法兰目标(基座)
        逻辑:
        1. Tip(World) -> Flange(World): 减去末端偏移(需考虑旋转)
        2. Flange(World) -> Flange(Base): 减去基座位置
        3. Manual Calibration: 应用特定的机械臂偏移补偿
        """
        # 解包
        target_pos_world = target_pose_world[:3]
        target_euler_world = target_pose_world[3:]
        
        # 1. 计算末端偏移在世界系下的向量 (Rotation * Offset_Local)
        r = R.from_euler('xyz', target_euler_world, degrees=False)
        rot_matrix = r.as_matrix()
        offset_world = rot_matrix @ self.ee_offset_local
        
        # 2. 计算法兰盘在世界系位置
        flange_pos_world = target_pos_world - offset_world
        
        # 3. 计算法兰盘在基座系位置 (减去基座坐标)
        flange_pos_base = flange_pos_world - self.base_pos_world
        
        # 4. 【特定补偿】特定的偏移修正
        flange_pos_base[0] -= 0.1
        flange_pos_base[2] -= 0.16
        
        # 5. 组合 (假设基座与世界无相对旋转)
        target_pose_base = np.concatenate([flange_pos_base, target_euler_world])
        
        return target_pose_base

    def _update_mocap_marker(self, pos):
        """[Visual] 更新可视化小球位置"""
        try:
            target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
            if target_body_id != -1:
                mocap_id = self.model.body_mocapid[target_body_id]
                self.data.mocap_pos[mocap_id] = pos
        except: pass

    # ========================================================
    # 主调用接口 (Public API)
    # ========================================================

    def move_ee_pose_world(self, target_pose_world, max_steps=3000):
        """
        移动末端到指定的世界坐标位姿
        
        Args:
            target_pose_world (np.array): [x, y, z, rx, ry, rz]
            max_steps (int): 最大仿真步数
            
        Returns:
            success (bool): 是否到达
            final_error (float): 最终的位置误差(m)
        """
        print(f"\n[Command] Move to World Pose: {np.round(target_pose_world[:3], 3)}")
        
        # 1. 可视化目标
        self._update_mocap_marker(target_pose_world[:3])
        
        # 2. 坐标转换 (World -> Base)
        target_pose_base = self._pose_tf_world2base(target_pose_world)
        
        # 3. 逆运动学求解
        target_joints = self._ik_base(target_pose_base)
        
        if target_joints is None:
            print("[Error] IK Solution Not Found")
            return False, 999.0
            
        print(f"[IK] Solved Joints: {np.round(target_joints, 3)}")
        
        # 4. 执行控制循环
        success = False
        final_error = 999.0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count = 0
            while viewer.is_running() and step_count < max_steps:
                loop_start = time.time()
                
                # --- A. 下发控制 ---
                self.data.ctrl[:6] = target_joints
                if self.model.nu >= 8:
                    self.data.ctrl[6:] = self.GRIPPER_OPEN_VAL # 保持张开

                # --- B. 物理步进 ---
                mujoco.mj_step(self.model, self.data)
                step_count += 1
                
                # --- C. 误差检测 ---
                # 获取当前真实的指尖世界坐标
                current_pose = self._get_ee_pose_world()
                current_pos_world = current_pose[:3]
                
                # 计算位置误差
                error = np.linalg.norm(current_pos_world - target_pose_world[:3])
                final_error = error
                
                # 判定到达
                if error < self.POSITION_THRESHOLD:
                    success = True
                    break
                
                # --- D. 渲染与同步 ---
                viewer.sync()
                
                # 简单的帧率控制
                time_until_next = self.model.opt.timestep - (time.time() - loop_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                    
        # 5. 打印结果
        status_str = "SUCCESS ✅" if success else "FAILED ❌"
        print(f"[Result] {status_str} | Steps: {step_count} | Error: {final_error:.5f}m")
        
        return success, final_error

# ========================================================
# 使用示例
# ========================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "R5/R5a/meshes/mjmodel.xml")
    
    # 实例化控制器
    try:
        robot = SingleArmController(xml_path, end_effector_site="end_effector")
        
        # 定义目标 (世界坐标系)
        # 假设基座在 z=1.0，我们去前方 0.3m, 高度 1.3m 的地方
        target_world = np.array([0.3, 0.3, 1.3, np.pi/2, 0.0, np.pi/2])
        
        # 调用接口
        is_reached, err = robot.move_ee_pose_world(target_world, max_steps=10_000)
        
    except Exception as e:
        print(f"Main Error: {e}")