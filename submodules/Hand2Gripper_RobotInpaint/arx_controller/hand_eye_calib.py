import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from bimanual import SingleArm

# ================= 配置区域 =================
ARM_CONFIG = {
    "can_port": "can1", 
    "type": 0
}

# 标定板参数
MARKER_SIZE = 0.05      # 50mm
MARKER_DICT = aruco.DICT_4X4_50
MARKER_ID = 0           

# ================= 工具函数 =================

def get_realsense_intrinsics(profile):
    """ 获取内参并返回 Camera Matrix 和 Dist Coeffs """
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)
    print(f"内参获取成功: {intr.width}x{intr.height}, fx={intr.fx:.1f}, fy={intr.fy:.1f}")
    return camera_matrix, dist_coeffs

def euler_to_matrix(rpy):
    """ 欧拉角(xyz)转旋转矩阵 """
    r = R.from_euler('xyz', rpy, degrees=False)
    return r.as_matrix()

def get_marker_pose(frame, dictionary, parameters, camera_matrix, dist_coeffs):
    """ 检测标定板位姿 """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    if ids is not None and MARKER_ID in ids:
        index = np.where(ids == MARKER_ID)[0][0]
        corner = corners[index]
        marker_points = np.array([
            [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
            [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
            [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
            [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
        ], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(marker_points, corner, camera_matrix, dist_coeffs)
        if success:
            return True, rvec, tvec, corner
    return False, None, None, None

def main():
    # 1. 连接机械臂
    print("正在连接机械臂...")
    arm = SingleArm(ARM_CONFIG)
    arm.gravity_compensation() 
    print("机械臂已就绪 (重力补偿模式)")

    # 2. 开启 RealSense
    print("正在开启 RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    camera_matrix, dist_coeffs = get_realsense_intrinsics(profile)

    # ArUco 配置
    aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)
    aruco_params = aruco.DetectorParameters()

    # 数据缓存
    R_gripper2base = [] 
    t_gripper2base = [] 
    R_target2cam = []   
    t_target2cam = []   

    print("\n========= 操作指南 =========")
    print(" [S] 采集数据 (建议 >15 组，多旋转)")
    print(" [C] 计算并保存结果")
    print(" [Q] 退出")
    print("===========================\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            frame = np.asanyarray(color_frame.get_data())
            found, rvec_cam, tvec_cam, corners = get_marker_pose(frame, aruco_dict, aruco_params, camera_matrix, dist_coeffs)

            # 界面显示
            display_frame = frame.copy()
            if found:
                cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, rvec_cam, tvec_cam, 0.03)
                cv2.aruco.drawDetectedMarkers(display_frame, [corners], np.array([[MARKER_ID]]))
                cv2.putText(display_frame, "DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "NO TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"Samples: {len(R_gripper2base)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Calibration', display_frame)

            key = cv2.waitKey(1) & 0xFF

            # === 采集 (Sampler) ===
            if key == ord('s'):
                if found:
                    ee_pose = arm.get_ee_pose_xyzrpy()
                    
                    # 机械臂位姿 (Base -> End)
                    t_g2b = np.array(ee_pose[:3]).reshape(3, 1)
                    R_g2b = euler_to_matrix(ee_pose[3:])

                    # 视觉位姿 (Camera -> Target)
                    R_t2c, _ = cv2.Rodrigues(rvec_cam)
                    t_t2c = tvec_cam

                    R_gripper2base.append(R_g2b)
                    t_gripper2base.append(t_g2b)
                    R_target2cam.append(R_t2c)
                    t_target2cam.append(t_t2c)
                    print(f"[OK] 已采集第 {len(R_gripper2base)} 组")
                else:
                    print("[WARN] 未检测到标定板")

            # === 计算 (Compute) ===
            elif key == ord('c'):
                if len(R_gripper2base) < 5:
                    print("数据不足，请继续采集。")
                    continue
                
                print("\n正在计算 Eye-to-World 矩阵...")
                try:
                    # 1. 求解 Base -> Camera (Optical)
                    R_b2c_opt, t_b2c_opt = cv2.calibrateHandEye(
                        R_gripper2base, t_gripper2base, 
                        R_target2cam, t_target2cam, 
                        method=cv2.CALIB_HAND_EYE_TSAI
                    )
                    
                    # 组装 4x4 矩阵 (Optical Frame)
                    Mat_base_T_camera_optical = np.eye(4)
                    Mat_base_T_camera_optical[:3, :3] = R_b2c_opt
                    Mat_base_T_camera_optical[:3, 3] = t_b2c_opt.flatten()

                    # 2. 转换为 Camera Link (Physical)
                    # 修正矩阵: Optical (RD Fwd) -> Link (FwdL Up)
                    # X_opt = -Y_link, Y_opt = -Z_link, Z_opt = X_link
                    fix_matrix = np.array([
                        [0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [1,  0,  0,  0],
                        [0,  0,  0,  1]
                    ])
                    
                    # 计算 Link 矩阵: T_base_link = T_base_opt * T_opt_link (correction)
                    Mat_base_T_camera_link = Mat_base_T_camera_optical @ fix_matrix

                    # 3. 结果保存
                    result = {
                        "canera_intri": {
                            "matrix": camera_matrix.tolist(),
                            "dist_coeffs": dist_coeffs.tolist()
                        },
                        "Mat_base_T_camera_optical": Mat_base_T_camera_optical.tolist(),
                        "Mat_base_T_camera_link": Mat_base_T_camera_link.tolist()
                    }

                    filename = f"calib_result_{ARM_CONFIG['can_port']}.json"
                    with open(filename, "w") as f:
                        json.dump(result, f, indent=4)
                    
                    print("\n========= 计算成功 =========")
                    print(f"结果已保存至: {filename}")
                    print("包含键值: canera_intri, Mat_base_T_camera_optical, Mat_base_T_camera_link")
                    print("===========================")

                except Exception as e:
                    print(f"计算失败: {e}")

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()