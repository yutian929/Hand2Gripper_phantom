#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

# =========================
# 1) ArUco 参数（按你的）
# =========================
MARKER_SIZE = 0.05                 # 50mm -> meters
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0

# =========================
# 2) 机械臂接口（按你工程里的用法）
# =========================
from bimanual import SingleArm

ARM_CONFIG = {
    "can_port": "can3",
    "type": 0,
}

# =========================
# 2.5) Free-drag 配置（新增）
# =========================
# 有些控制器需要更高频的 gravity_compensation 才“够泄力”
# 你的摄像头循环大约 30Hz；如果感觉泄力不够，可以把这里调到 2~5
GC_REPEAT_PER_LOOP = 1

# =========================
# 3) SE(3) 4x4 工具
# =========================
def T_from_Rt(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def T_inv(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti

def make_valid_rotation(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    T2 = T.copy()
    T2[:3, :3] = Rm
    return T2

def rvec_tvec_to_T_cam_marker(rvec, tvec):
    # solvePnP 输出 rvec/tvec: X_cam = R * X_marker + t  (marker -> cam)
    Rm = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    return T_from_Rt(Rm, np.asarray(tvec, dtype=np.float64).reshape(3))

def pose_to_T_base_flange(xyzrpy, degrees: bool, mode: str):
    """
    xyzrpy: [x, y, z, roll, pitch, yaw]
    mode 用来尝试不同欧拉组合：
      - "xyz_rpy": R = from_euler('xyz', [roll,pitch,yaw])
      - "zyx_ypr": R = from_euler('zyx', [yaw,pitch,roll])
    """
    x, y, z, roll, pitch, yaw = map(float, xyzrpy)

    if mode == "xyz_rpy":
        Rm = R.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
    elif mode == "zyx_ypr":
        Rm = R.from_euler("zyx", [yaw, pitch, roll], degrees=degrees).as_matrix()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return T_from_Rt(Rm, [x, y, z])

# =========================
# 4) solvePnP 估 ArUco 位姿（OpenCV 4.12 兼容）
# =========================
def estimate_pose_from_aruco_corners(corner_4x2, marker_size, K, dist):
    """
    corner_4x2: (4,2) 图像角点 (来自 detectMarkers)
    返回 rvec,tvec，使得 marker -> cam
    """
    L = float(marker_size)

    # 假设 detectMarkers 给角点顺序：左上, 右上, 右下, 左下
    objp = np.array([
        [-L/2,  L/2, 0.0],
        [ L/2,  L/2, 0.0],
        [ L/2, -L/2, 0.0],
        [-L/2, -L/2, 0.0],
    ], dtype=np.float64)

    imgp = np.asarray(corner_4x2, dtype=np.float64).reshape(4, 2)

    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
    if not ok:
        return None, None
    return rvec.reshape(3), tvec.reshape(3)

# =========================
# 5) 同时解 X,Y： A_i Y = X B_i
# =========================
def se3vec_to_T(v6):
    v6 = np.asarray(v6, dtype=np.float64).reshape(6)
    rvec = v6[:3]
    t = v6[3:]
    Rm = R.from_rotvec(rvec).as_matrix()
    return T_from_Rt(Rm, t)

def solve_XY(A_list, B_list, w_trans=5.0, loss="huber"):
    assert len(A_list) == len(B_list) and len(A_list) >= 10

    A_list = [make_valid_rotation(np.asarray(A, np.float64)) for A in A_list]
    B_list = [make_valid_rotation(np.asarray(B, np.float64)) for B in B_list]

    x0 = np.zeros(12, dtype=np.float64)  # X(6)+Y(6) 初值单位阵

    def residuals(x):
        X = se3vec_to_T(x[0:6])   # T_base_cam
        Y = se3vec_to_T(x[6:12])  # T_flange_marker
        res = []
        for A, B in zip(A_list, B_list):
            E = T_inv(X @ B) @ (A @ Y)  # 理想=I
            r_err = R.from_matrix(E[:3, :3]).as_rotvec()
            t_err = E[:3, 3]
            res.append(np.hstack([r_err, w_trans * t_err]))
        return np.concatenate(res)

    sol = least_squares(residuals, x0, method="trf", loss=loss, f_scale=1.0)
    X = make_valid_rotation(se3vec_to_T(sol.x[0:6]))
    Y = make_valid_rotation(se3vec_to_T(sol.x[6:12]))
    return X, Y, sol

def evaluate(A_list, B_list, X, Y):
    ang_deg, trans = [], []
    for A, B in zip(A_list, B_list):
        E = T_inv(X @ B) @ (A @ Y)
        a = np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec()) * 180.0 / np.pi
        t = np.linalg.norm(E[:3, 3])
        ang_deg.append(a)
        trans.append(t)
    ang_deg = np.array(ang_deg); trans = np.array(trans)
    return dict(
        ang_mean=float(ang_deg.mean()), ang_median=float(np.median(ang_deg)), ang_max=float(ang_deg.max()),
        trans_mean=float(trans.mean()), trans_median=float(np.median(trans)), trans_max=float(trans.max()),
    )

def try_solve_with_best_pose_convention(raw_ee_list, B_list):
    """
    自动尝试：度/弧度 + 两种常见欧拉组合，选残差更小的。
    """
    candidates = []
    raw_max = float(np.max(np.abs(np.array(raw_ee_list)[:, 3:])))
    deg_guess = raw_max > 6.5

    for degrees in ([deg_guess, not deg_guess] if deg_guess else [False, True]):
        for mode in ["xyz_rpy", "zyx_ypr"]:
            A_list = [pose_to_T_base_flange(p, degrees=degrees, mode=mode) for p in raw_ee_list]
            X, Y, sol = solve_XY(A_list, B_list, w_trans=5.0, loss="huber")
            stats = evaluate(A_list, B_list, X, Y)
            candidates.append((stats["ang_mean"], stats["trans_mean"], degrees, mode, X, Y, sol, stats))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0]

# =========================
# 6) 只在 OpenCV 画面上叠字显示 ee_pose（按你要求）
# =========================
def draw_text(vis, lines, org=(10, 90), scale=0.6, thickness=2, color=(255, 255, 255), line_gap=26):
    x, y = org
    for i, s in enumerate(lines):
        cv2.putText(vis, s, (x, y + i * line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def get_T_link_optical():
    """
    返回从 Optical Frame 到 Link Frame 的变换矩阵 T_link_optical
    P_link = T_link_optical @ P_optical
    
    Standard ROS convention:
      Optical: X-Right, Y-Down, Z-Forward
      Link:    X-Forward, Y-Left, Z-Up
    """
    T = np.eye(4, dtype=np.float64)
    # Rotation:
    # Link X (1,0,0) <-> Optical Z (0,0,1)
    # Link Y (0,1,0) <-> Optical -X (-1,0,0) -> Wait, Y-Left means Y is + in Left? 
    # Usually ROS body frame: X-Forward, Y-Left, Z-Up.
    # Optical X (Right) -> Link -Y
    # Optical Y (Down)  -> Link -Z
    # Optical Z (Forward)-> Link X
    T[:3, :3] = np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0]
    ])
    return T

# =========================
# 7) 主程序：D435采图 + ArUco检测 + Free-drag + 采集/解算
# =========================
def main(file_name="eye_to_hand_result_right.npz"):
    print("Connecting arm...")
    arm = SingleArm(ARM_CONFIG)

    print("\n========================================")
    print("  机械臂进入【重力补偿模式】(free-drag)")
    print("  请用手拖动机械臂到不同姿态采样")
    print("  按键：S=保存样本  C=解算  Q=退出")
    print("========================================\n")

    print("Starting RealSense D435...")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    # RealSense 内参
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs[:5], dtype=np.float64)

    print("K=\n", K)
    print("dist=", dist)

    # ArUco detector
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(MARKER_DICT)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    raw_ee_list = []
    B_list = []

    last_ee = None
    last_ee_ts = 0.0

    try:
        while True:
            # --- Free-drag: 必须循环调用以保持泄力状态 :contentReference[oaicite:1]{index=1} ---
            for _ in range(GC_REPEAT_PER_LOOP):
                arm.gravity_compensation()

            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())
            vis = img.copy()

            # 读当前 ee_pose（用于叠字显示，也用于按 S 保存）
            try:
                last_ee = np.array(arm.get_ee_pose_xyzrpy(), dtype=np.float64)
                last_ee_ts = time.time()
            except Exception:
                pass

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            found = False
            rvec_sel, tvec_sel = None, None

            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten().tolist()
                aruco.drawDetectedMarkers(vis, corners, ids)

                if MARKER_ID in ids_flat:
                    idx = ids_flat.index(MARKER_ID)
                    c = corners[idx].reshape(4, 2)

                    rvec_sel, tvec_sel = estimate_pose_from_aruco_corners(c, MARKER_SIZE, K, dist)
                    if rvec_sel is not None:
                        found = True
                        cv2.drawFrameAxes(vis, K, dist, rvec_sel, tvec_sel, 0.04)

            # --- 叠字（只做 text overlay） ---
            lines = [
                f"samples={len(B_list)}   [S]save  [C]calib  [Q]quit",
                f"aruco(ID={MARKER_ID})={'OK' if found else 'NO'}",
            ]
            if last_ee is not None:
                xyz = last_ee[:3]
                rpy = last_ee[3:]
                # 同时显示“当作度”的版本，方便肉眼判断
                rpy_deg = rpy * 180.0 / np.pi if np.max(np.abs(rpy)) <= 6.5 else rpy
                lines += [
                    f"EE xyz: [{xyz[0]: .4f}, {xyz[1]: .4f}, {xyz[2]: .4f}] (m?)",
                    f"EE rpy raw: [{rpy[0]: .4f}, {rpy[1]: .4f}, {rpy[2]: .4f}]",
                    f"EE rpy ~deg: [{rpy_deg[0]: .1f}, {rpy_deg[1]: .1f}, {rpy_deg[2]: .1f}] deg",
                ]
            else:
                lines += ["EE pose: N/A"]

            if found:
                lines += [f"tvec(m): [{tvec_sel[0]: .3f}, {tvec_sel[1]: .3f}, {tvec_sel[2]: .3f}]"]

            draw_text(vis, lines, org=(10, 30), scale=0.6, thickness=2)

            cv2.imshow("D435 ArUco + EE Pose (free-drag)", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q')]:
                break

            if key in [ord('s'), ord('S')]:
                if not found:
                    print("[WARN] 没看到目标 ArUco，无法保存。")
                    continue
                if last_ee is None:
                    print("[WARN] 未读到 ee_pose，无法保存。")
                    continue

                B = rvec_tvec_to_T_cam_marker(rvec_sel, tvec_sel)
                raw_ee_list.append(last_ee.copy())
                B_list.append(B)
                print(f"[OK] saved #{len(B_list)}  ee={np.round(last_ee,4)}  tvec={np.round(tvec_sel,4)}")

            if key in [ord('c'), ord('C')]:
                if len(B_list) < 10:
                    print("[WARN] 样本太少：至少10帧，建议20~40帧。")
                    continue

                print("\n[INFO] Solving (auto-try pose convention)...")
                ang_mean, trans_mean, degrees, mode, X, Y, sol, stats = try_solve_with_best_pose_convention(raw_ee_list, B_list)

                print("\n=== BEST choice ===")
                print("degrees =", degrees, "mode =", mode)
                print("rotation mean(deg) =", stats["ang_mean"], "translation mean(m) =", stats["trans_mean"])
                print("\n=== X = T_base_cam (cam_optical -> base) ===\n", X)
                print("\n=== Y = T_flange_marker (marker on flange) ===\n", Y)
                
                # --- 计算 T_base_link ---
                # X 是 T_base_optical
                # 关系: T_base_optical = T_base_link @ T_link_optical
                # 所以: T_base_link = T_base_optical @ inv(T_link_optical)
                T_link_optical = get_T_link_optical()
                T_base_link = X @ np.linalg.inv(T_link_optical)
                print("\n=== T_base_link (camera_link -> base) ===\n", T_base_link)

                print("\nOptimizer:", sol.success, sol.message)

                np.savez(
                    file_name,
                    T_base_cam=X,            # Optical Frame
                    T_base_link=T_base_link, # Link Frame (Converted)
                    T_flange_marker=Y,
                    K=K,
                    dist=dist,
                    raw_ee=np.array(raw_ee_list),
                    B_list=np.array(B_list),
                    degrees=degrees,
                    mode=mode
                )
                print(f"\n[OK] Saved: {file_name}\n")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(file_name="eye_to_hand_result_right_20251218.npz")
