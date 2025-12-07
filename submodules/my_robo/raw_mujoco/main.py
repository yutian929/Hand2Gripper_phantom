import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import bimanual  # 假设你已经安装了这个包

# ================= 配置区域 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(current_dir, "R5/R5a/meshes/mjmodel.xml")

# 目标位置: [x, y, z, rx, ry, rz]
TARGET_POSE = np.array([0.2, 0.2, 0.2, 1.0, 0.0, 0.0])

# 夹爪配置
GRIPPER_OPEN_VAL = 0.044  # 全开 (m)
GRIPPER_CLOSE_VAL = 0.0   # 全闭 (m)

# ================= 你的算法接口区域 =================

def forward_kinematics(joint_angles):
    try:
        return bimanual.forward_kinematics(joint_angles)
    except Exception as e:
        print(f"FK Error: {e}")
        return np.zeros(6)

def inverse_kinematics(target_pose, current_joint_guess=None):
    try:
        # 如果 bimanual 库需要初始猜测，请修改此处调用方式
        return bimanual.inverse_kinematics(target_pose)
    except Exception as e:
        print(f"IK Error: {e}")
        return None

# ================= 主控制循环 =================

def main():
    # 1. 加载 MuJoCo 模型
    if not os.path.exists(XML_PATH):
        print(f"错误: 找不到文件 {XML_PATH}")
        return
        
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 2. 初始姿态设置 (防止炸机的关键步骤)
    # 定义一个安全的初始姿态
    start_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # [物理状态] 设置关节当前角度
    data.qpos[:6] = start_qpos
    
    # [控制器] !!!关键!!! 设置电机目标位置等于当前位置
    # 如果不写这一行，第一帧时误差 = init_qpos - 0，巨大的误差会导致瞬间爆炸
    data.ctrl[:6] = start_qpos 
    
    # 运行一步让物理引擎同步状态
    mujoco.mj_step(model, data)
    print("机械臂已初始化到安全位置。")

    # 3. 计算 IK
    print("正在计算 IK...")
    target_joints = inverse_kinematics(TARGET_POSE, current_joint_guess=start_qpos)

    if target_joints is None:
        print("错误: IK 未找到解，程序退出。")
        return
    print(f"目标关节角: {target_joints}")

    # ==========================================
    # 4. 轨迹规划配置
    # ==========================================
    trajectory_duration = 1.0  # 3秒完成动作
    is_target_reached = False  # 状态标记
    
    # 必须使用 .copy()，否则 Python 可能会传递引用，导致逻辑错误
    q_start = start_qpos.copy()
    q_end = target_joints.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 记录仿真开始时的物理时间
        sim_start_time = data.time 
        
        print("开始平滑运动控制...")
        
        while viewer.is_running():
            step_start = time.time()
            
            # 计算已经流逝的物理时间
            current_sim_time = data.time - sim_start_time
            
            # --- 插值控制逻辑 ---
            if current_sim_time < trajectory_duration:
                # 计算进度 (0.0 到 1.0)
                progress = current_sim_time / trajectory_duration
                
                # 线性插值公式: P_current = P_start + t * (P_end - P_start)
                current_target = q_start + progress * (q_end - q_start)
                
                # 赋值给控制器
                data.ctrl[:6] = current_target
            else:
                # 时间到，保持在终点
                data.ctrl[:6] = q_end
                if not is_target_reached:
                    print(f"运动完成 (用时 {trajectory_duration}s)，位置保持中...")
                    is_target_reached = True

            # --- 夹爪控制 ---
            if model.nu >= 8:
                # 保持夹爪张开
                data.ctrl[6] = GRIPPER_OPEN_VAL
                data.ctrl[7] = GRIPPER_OPEN_VAL

            # --- 物理步进 ---
            try:
                mujoco.mj_step(model, data)
                # 修正: 使用 np.round 来控制小数位数
                if int(time.time()) % 10 == 0:
                    print(f"data.qpos = {np.round(data.qpos, 2)}")
            except Exception as e:
                print(f"仿真出错 (可能是数值不稳定): {e}")
                break

            # --- 渲染同步 ---
            viewer.sync()

            # --- 帧率控制 (保持真实时间流逝) ---
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()