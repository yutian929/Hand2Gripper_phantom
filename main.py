def test_real_dual_arm_controller():
    from hand2gripper_robot_inpaint.arx_controller.real_dual_arm_controller import RealDualArmController
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    Mat_base_L_T_ee_L = np.load("/home/yutian/Hand2Gripper_phantom/data/processed/epic/4/inpaint_processor/hand2gripper_train_base_L_T_ee_L.npy")
    Mat_base_R_T_ee_R = np.load("/home/yutian/Hand2Gripper_phantom/data/processed/epic/4/inpaint_processor/hand2gripper_train_base_R_T_ee_R.npy")
    gripper_widths_L = np.load("/home/yutian/Hand2Gripper_phantom/data/processed/epic/4/inpaint_processor/hand2gripper_train_gripper_width_left.npy")
    gripper_widths_R = np.load("/home/yutian/Hand2Gripper_phantom/data/processed/epic/4/inpaint_processor/hand2gripper_train_gripper_width_right.npy")
    # Initialize controller
    # Note: Adjust 'can0'/'can1' to match your actual hardware ports
    controller = RealDualArmController(left_can='can1', right_can='can3')

    # Execute trajectory
    # dt=0.04 corresponds to roughly 25Hz
    controller.execute_trajectory(Mat_base_L_T_ee_L, Mat_base_R_T_ee_R, gripper_widths_L, gripper_widths_R, dt=0.1)
    




def test_hand2gripper_robot_inpaint_arx_controller():
    import numpy as np
    import cv2
    import os
    from scipy.spatial.transform import Rotation as R
    from hand2gripper_robot_inpaint.arx_controller.single_arm_controller import SingleArmController
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def load_and_transform_data(filepath):
        """
        Load .npz data and transform from Optical Frame to Camera Link Frame.
        Returns:
            target_seq_world (np.array): Shape (N, 6) -> [x, y, z, rx, ry, rz]
        """
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found.")
            return None

        try:
            data = np.load(filepath)
            ee_pts = data["ee_pts"]   # (N, 3)
            ee_oris = data["ee_oris"] # (N, 3, 3)
        except Exception as e:
            print(f"Error loading npz: {e}")
            return None

        # Transformation Matrix: Optical -> Camera Link
        # x' = z, y' = -x, z' = -y
        R_transform = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

        # 1. Transform Positions
        # P_new = R_transform * P_old
        ee_pts_transformed = (R_transform @ ee_pts.T).T

        # 2. Transform Orientations
        # R_new = R_transform * R_old
        ee_oris_transformed = np.matmul(R_transform, ee_oris)

        # Additional 180 degree rotation around Z axis
        R_z180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        ee_oris_transformed = np.matmul(ee_oris_transformed, R_z180)

        # 3. Convert Rotation Matrices to Euler Angles (xyz)
        # We need (N, 6) for the controller: [x, y, z, rx, ry, rz]
        N = len(ee_pts)
        target_seq = np.zeros((N, 6))
        
        target_seq[:, :3] = ee_pts_transformed
        
        # Batch convert rotations
        r = R.from_matrix(ee_oris_transformed)
        euler_angles = r.as_euler('xyz', degrees=False)
        target_seq[:, 3:] = euler_angles

        return target_seq

    # ---------------------------------------------------------
    # Helper Functions: 矩阵与Pose互转
    # ---------------------------------------------------------
    def pose_to_matrix(pose):
        """
        [x, y, z, rx, ry, rz] -> 4x4 Homogeneous Matrix
        """
        t = pose[:3]
        euler = pose[3:]
        
        r = R.from_euler('xyz', euler, degrees=False)
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = t
        return mat

    def matrix_to_pose(mat):
        """
        4x4 Homogeneous Matrix -> [x, y, z, rx, ry, rz]
        """
        t = mat[:3, 3]
        rot_mat = mat[:3, :3]
        
        r = R.from_matrix(rot_mat)
        euler = r.as_euler('xyz', degrees=False)
        
        return np.concatenate([t, euler])

    def visualize_trajectory(target_seq, title="Trajectory Visualization"):
        """
        Visualize the trajectory using Matplotlib.
        Args:
            target_seq (np.array): Shape (N, 6) -> [x, y, z, rx, ry, rz]
        """
        points = target_seq[:, :3]
        euler_angles = target_seq[:, 3:]
        
        # Convert Euler to Rotation Matrices for visualization
        r = R.from_euler('xyz', euler_angles, degrees=False)
        matrices = r.as_matrix()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot World Frame Axes at origin
        ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', arrow_length_ratio=0.3)
        ax.text(0.1, 0, 0, 'X', color='r')
        ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', arrow_length_ratio=0.3)
        ax.text(0, 0.1, 0, 'Y', color='g')
        ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', arrow_length_ratio=0.3)
        ax.text(0, 0, 0.1, 'Z', color='b')

        # Plot path line
        ax.plot(points[:, 0], points[:, 1], points[:, 2], label='Trajectory Path', linewidth=1, color='gray')
        
        # Plot points and orientation
        # Downsample for clarity (every 5th point)
        indices = np.arange(0, len(points), 5) 
        
        for i in indices:
            pt = points[i]
            rot = matrices[i]
            
            # Plot point
            ax.scatter(pt[0], pt[1], pt[2], c='r', marker='o', s=20)
            
            # Plot orientation axes
            length = 0.02
            # X axis (Red)
            ax.quiver(pt[0], pt[1], pt[2], rot[0, 0], rot[1, 0], rot[2, 0], length=length, color='r')
            # Y axis (Green)
            ax.quiver(pt[0], pt[1], pt[2], rot[0, 1], rot[1, 1], rot[2, 1], length=length, color='g')
            # Z axis (Blue)
            ax.quiver(pt[0], pt[1], pt[2], rot[0, 2], rot[1, 2], rot[2, 2], length=length, color='b')
            
            # Label start and end
            if i == 0:
                ax.text(pt[0], pt[1], pt[2], 'Start', color='black', fontweight='bold')
            if i == indices[-1]:
                ax.text(pt[0], pt[1], pt[2], 'End', color='black', fontweight='bold')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Equal aspect ratio hack
        all_points = points
        if len(all_points) > 0:
            max_range = np.array([
                all_points[:,0].max()-all_points[:,0].min(), 
                all_points[:,1].max()-all_points[:,1].min(), 
                all_points[:,2].max()-all_points[:,2].min()
            ]).max() / 2.0
            
            mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
            mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
            mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.legend()
        plt.show()
    ### main ###
    camera_pose_in_base = np.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    Mat_base_T_camera = pose_to_matrix(camera_pose_in_base)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "submodules/Hand2Gripper_RobotInpaint/arx_controller", "R5/R5a/meshes/single_arm_scene.xml")
    data_path = os.path.join(current_dir, "submodules/Hand2Gripper_RobotInpaint/arx_controller", "free_hand_N_to_F_smoothed_actions_right_in_camera_optical_frame.npz")

    robot = SingleArmController(xml_path)
    robot_base_pose_world = robot._get_base_pose_world()
    Mat_world_T_base = pose_to_matrix(robot_base_pose_world)
    print(f"Mat_world_T_base:\n{Mat_world_T_base}")
    
    seqs_in_camera_link = load_and_transform_data(data_path)
    Mat_camera_T_seqs = np.array([pose_to_matrix(pose) for pose in seqs_in_camera_link])
    print(f"Mat_camera_T_seqs shape: {Mat_camera_T_seqs.shape}")
    # visualize_trajectory(seqs_in_camera_link, title="Trajectory in Camera Link Frame")

    Mat_world_T_seqs = Mat_world_T_base @ Mat_base_T_camera @ Mat_camera_T_seqs
    seqs_in_world = np.array([matrix_to_pose(mat) for mat in Mat_world_T_seqs])
    print(f"seqs_in_world shape: {seqs_in_world.shape}")
    # visualize_trajectory(seqs_in_world, title="Trajectory in World Frame")

    # assume camera base fixed
    Mat_world_T_camera = Mat_world_T_base @ Mat_base_T_camera
    camera_poses_in_world = np.array([matrix_to_pose(Mat_world_T_camera) for _ in range(len(seqs_in_world))])
    print(f"camera_poses_in_world shape: {camera_poses_in_world.shape}")
    # visualize_trajectory(camera_poses_in_world, title="Camera Poses in World Frame")

    if seqs_in_world is not None:
        try:
            print("########## Executing move_trajectory ##########")
            robot.move_trajectory(seqs_in_world, kinematic_only=True)
            print("########## Executing move_trajectory_with_camera ##########")
            frames, masks = robot.move_trajectory_with_camera(seqs_in_world, camera_poses_in_world, kinematic_only=True)
            
            for i, (frame, mask) in enumerate(zip(frames, masks)):
                cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                geom_ids = mask[:, :, 0]
                # ID 0 -> Black [0,0,0], Others -> White [255,255,255]
                mask_vis = np.zeros((geom_ids.shape[0], geom_ids.shape[1], 3), dtype=np.uint8)
                mask_vis[geom_ids > 0] = [255, 255, 255]
                
                cv2.imshow("Mask", mask_vis)
                cv2.waitKey(100)
            
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Execution Error: {e}")
    else:
        print("Failed to load trajectory data.")

def test_mujoco_viewer():
    import mujoco
    
    pass


def test_robo_suite():
    from robosuite.robots import register_robot_class
    from robosuite.models.robots import Panda
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    import mujoco


    # @register_robot_class("WheeledRobot")
    # class MobilePanda(Panda):
    #     @property
    #     def default_base(self):
    #         return "OmronMobileBase"

    #     @property
    #     def default_arms(self):
    #         return {"right": "Panda"}

    # Create environment
    env = suite.make(
        env_name="Lift",
        robots=["Arx5","Arx5"],
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        use_camera_obs=False,
        control_freq=20,
    )

    # Run the simulation, and visualize it
    env.reset()
    mujoco.viewer.launch(env.sim.model._model, env.sim.data._data)


def test_python_orb_slam3():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from python_orb_slam3 import ORBExtractor
    import mediapy as media

    # Initialize ORB extractor
    orb_extractor = ORBExtractor()

    # Video file path
    video_path = "/home/yutian/projs/Hand2Gripper_phantom/data/raw/0/video_L.mp4"

    # Read video using mediapy
    video = media.read_video(video_path)

    # Initialize previous frame variables
    prev_frame = None
    prev_kps = None
    prev_des = None
    T_w_c = np.eye(4)  # Initial pose (identity matrix, world is the origin)

    # Create a list to store camera trajectory (in 3D)
    camera_trajectory = []

    # Process each frame
    frame_idx = 0
    for frame in video:
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract features (keypoints and descriptors) from the current frame
        kps, des = orb_extractor.detectAndCompute(gray_frame)

        if prev_frame is not None:
            # Match features between the current and previous frame
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_des, des)

            # Extract matched points
            pts_prev = np.float32([prev_kps[m.queryIdx].pt for m in matches])
            pts_curr = np.float32([kps[m.trainIdx].pt for m in matches])

            # Camera intrinsic matrix (replace with your actual camera parameters)
            K = np.array([[1057.7322998046875, 0, 972.5150756835938],
                          [0, 1057.7322998046875, 552.568359375],
                          [0, 0, 1]])

            # Compute the essential matrix and recover pose (rotation, translation)
            E, mask = cv2.findEssentialMat(pts_prev, pts_curr, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, K)

            # Update global camera pose by multiplying the transformation matrix
            T_c_prev_c = np.eye(4)
            T_c_prev_c[:3, :3] = R
            T_c_prev_c[:3, 3] = t.flatten()

            # Accumulate the camera pose to get the global trajectory
            T_w_c = T_w_c @ T_c_prev_c
            camera_trajectory.append(T_w_c[:3, 3])  # Store the position (X, Y, Z)

            # Visualize feature matches (not for the first frame)
            matches_img = cv2.drawMatches(prev_frame, prev_kps, frame, kps, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # Uncomment to see the matches for each frame
            # plt.imshow(matches_img)
            # plt.title(f"Frame {frame_idx}")
            # plt.show()

        else:
            # For the first frame, set initial pose to origin and skip matching
            camera_trajectory.append(T_w_c[:3, 3])

        # Update previous frame variables
        prev_frame = frame
        prev_kps = kps
        prev_des = des

        # Increment frame index
        frame_idx += 1

    # Convert camera trajectory list to numpy array
    camera_trajectory = np.array(camera_trajectory)

    # Plot 3D camera trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(camera_trajectory[:, 0], camera_trajectory[:, 1], camera_trajectory[:, 2], label="Camera Trajectory")
    ax.set_title("Camera Trajectory")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Z position")
    plt.show()

def test_hand2gripper():
    """
    测试 Hand2Gripper 模型的完整推理流程。
    """
    # 1. 导入推理模块
    # 将其放在函数内部，以避免在运行其他测试时产生不必要的导入
    import os
    import numpy as np
    import cv2
    import torch
    from hand2gripper.inference import Hand2GripperInference

    # 2. 定义路径和参数 (替换 argparse)
    checkpoint_path = "submodules/Hand2Gripper_hand2gripper/hand2gripper/hand2gripper.pt"
    input_path = "/home/yutian/projs/Hand2Gripper_phantom/data/processed/epic/0/hand2gripper_annotator_processor/left/50.npz"
    output_path = "test_hand2gripper_output.png"

    # 3. 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型权重文件: {checkpoint_path}")
        return
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入数据文件: {input_path}")
        return

    # 4. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 5. 初始化推理引擎
    print("正在初始化 Hand2Gripper 推理引擎...")
    try:
        inference_engine = Hand2GripperInference(checkpoint_path, device=device)
    except Exception as e:
        print(f"初始化模型时出错: {e}")
        return

    # 6. 加载数据
    print(f"正在从 {input_path} 加载数据...")
    try:
        data = np.load(input_path, allow_pickle=True)
        color_np = data["img_rgb"]
        bbox_np = data["bbox"]
        kpts_3d_np = data["kpts_3d"]
        contact_np = data["contact_logits"]
        is_right_np = data["is_right"]
        kpts_2d_np = data["kpts_2d"]  # 用于可视化
    except Exception as e:
        print(f"加载数据时出错。请确保 {input_path} 是一个有效的 .npz 文件并包含所需键。错误: {e}")
        return

    # 7. 执行推理
    print("正在执行推理...")
    pred_triple = inference_engine.predict(
        color=color_np,
        bbox=bbox_np,
        keypoints_3d=kpts_3d_np,
        contact=contact_np,
        is_right=is_right_np
    )

    # 8. 打印结果
    print("\n" + "="*30)
    print("      推理结果")
    print("="*30)
    print(f"预测的抓手三元组 (Base, Left, Right): {pred_triple}")
    print(f"  - Base 关节点 ID:  {pred_triple[0]}")
    print(f"  - Left 关节点 ID:  {pred_triple[1]}")
    print(f"  - Right 关节点 ID: {pred_triple[2]}")
    print("="*30)

    # 9. 可视化并保存结果
    print(f"\n正在生成可视化结果并保存到 {output_path}...")
    
    # 准备用于 OpenCV 的图像 (HWC, uint8, BGR)
    if color_np.dtype != np.uint8:
        vis_img = (color_np).astype(np.uint8)
    else:
        vis_img = color_np.copy()
    
    if vis_img.shape[0] == 3:  # CHW -> HWC
        vis_img = np.transpose(vis_img, (1, 2, 0))
    
    # RGB -> BGR for OpenCV
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # 绘制结果
    vis_img_result = inference_engine.vis_output(vis_img, kpts_2d_np, pred_triple)
    
    # 保存图像
    try:
        cv2.imwrite(output_path, vis_img_result)
        print(f"可视化结果已成功保存。")
    except Exception as e:
        print(f"保存可视化结果时出错: {e}")


def test_haco():
    from hand2gripper_haco import HACOContactEstimator, WILORHandDetector, HACOContactEstimatorWithoutRenderer
    import os
    import cv2
    # Hand Detection Configuration
    detector_type = 'wilor'  # Options: 'wilor' or 'mediapipe'
    # - 'wilor': Uses YOLO-based WILOR detector for hand detection
    # - 'mediapipe': Uses Google MediaPipe for hand landmark detection
    
    # detector_path = os.path.join(os.environ['HACO_BASE_DATA_PATH'], 'demo_data', 'wilor_detector.pt')  # Path to WILOR detector model
    detector_path = os.path.join('/home/yutian/projs/Hand2Gripper_phantom/submodules/Hand2Gripper_HACO/base_data', 'demo_data', 'wilor_detector.pt')
    # - Required when detector_type='wilor'
    # - Should point to the trained YOLO model file (.pt)
    # - Default: 'data/base_data/demo_data/wilor_detector.pt'
    
    # Contact Estimation Configuration
    backbone = 'hamer'  # Options: 'hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'handoccnet', 'hrnet-w48', 'hrnet-w32', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'
    # - 'hamer': Vision Transformer backbone (recommended, best performance)
    # - 'vit-*': Different Vision Transformer variants
    # - 'resnet-*': ResNet backbone variants
    # - 'hrnet-*': High-Resolution Network variants
    # - 'handoccnet': Hand occlusion network
    
    # checkpoint_path = os.path.join(os.environ['HACO_BASE_DATA_PATH'], 'release_checkpoint', 'haco_final_hamer_checkpoint.ckpt')  # Path to HACO model checkpoint
    checkpoint_path = os.path.join('/home/yutian/projs/Hand2Gripper_phantom/submodules/Hand2Gripper_HACO/base_data', 'release_checkpoint', 'haco_final_hamer_checkpoint.ckpt')
    # - Path to the trained HACO model file (.pth or .ckpt)
    # - Leave empty '' if no checkpoint is available (will use random weights)
    # - Example: 'checkpoints/haco_hamer_best.pth'
    
    experiment_dir = 'experiments_demo_image'  # Experiment directory for configuration
    # - Directory where experiment configurations and logs are stored
    # - Used for model configuration and logging
    # - Default: 'experiments_demo_image'
    # breakpoint()
    print("Initializing hand detector...")
    # Initialize WILOR hand detector
    # Parameters:
    # - detector_type: Type of detector ('wilor' or 'mediapipe')
    # - detector_path: Path to detector model file
    hand_detector = WILORHandDetector(
        detector_type=detector_type,
        detector_path=detector_path
    )
    
    print("Initializing contact estimator...")
    # Initialize HACO contact estimator
    # Parameters:
    # - backbone: Model backbone type (see options above)
    # - checkpoint_path: Path to trained model checkpoint
    # - experiment_dir: Directory for experiment configuration
    contact_estimator = HACOContactEstimator(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        experiment_dir=experiment_dir
    )

    img_rgb = cv2.imread('epic_kitch_demo.jpg')[..., ::-1]  # Load demo image and convert BGR to RGB
    H, W, _ = img_rgb.shape
    res = contact_estimator.predict_contact(img_rgb, bbox=[W//2, 2*H//3, 3*W//4, H])  # Predict contact using bounding box covering center of image
    cv2.imshow("Contact Visualization", res['contact_rendered'][..., ::-1])  # Display contact visualization (convert RGB to BGR for OpenCV)
    cv2.waitKey(1000)  # Wait for a key press to close the window
    print("All components initialized successfully!")

    contact_estimator_without_renderer = HACOContactEstimatorWithoutRenderer(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        experiment_dir=experiment_dir
    )
    res_no_renderer = contact_estimator_without_renderer.predict_contact(img_rgb, bbox=[W//2, 2*H//3, 3*W//4, H])
    print(res_no_renderer.keys())




def test_wilor_hand_pose3d_estimation_pipeline():
    import torch
    import cv2
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    img_path = "epic_kitch_demo.jpg"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = pipe.predict(image)
    # breakpoint()
    print(outputs)

    # (Pdb) type(outputs)
    # <class 'list'>
    # (Pdb) len(outputs)
    # 2
    # (Pdb) type(outputs[0])
    # <class 'dict'>
    # (Pdb) outputs[0].keys()
    # dict_keys(['hand_bbox', 'is_right', 'wilor_preds'])
    # (Pdb) outputs[0]['hand_bbox']
    # [638.0, 732.0, 882.0, 1035.0]
    # (Pdb) outputs[0]['is_right']
    # 0.0
    # (Pdb) type(outputs[0]['wilor_preds'])
    # <class 'dict'>
    # (Pdb) outputs[0]['wilor_preds'].keys()
    # dict_keys(['global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_keypoints_3d', 'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length', 'pred_keypoints_2d'])


def test_wilor_hand_pose3d_estimation_pipeline_hand_detection():
    import torch
    import cv2
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    img_path = "epic_kitch_demo.jpg"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = pipe.hand_detector(image, conf=0.3, verbose=True)[0]
    breakpoint()
    print(detections)

    # (Pdb) type(detections)
    # <class 'ultralytics.engine.results.Results'>
    # (Pdb) dir(detections)
    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_keys', 'boxes', 'cpu', 'cuda', 'keypoints', 'masks', 'names', 'new', 'numpy', 'obb', 'orig_img', 'orig_shape', 'path', 'plot', 'probs', 'save', 'save_crop', 'save_dir', 'save_txt', 'show', 'speed', 'summary', 'to', 'tojson', 'update', 'verbose']
    # (Pdb) len(detections)
    # 2
    # (Pdb) type(detections[0])
    # <class 'ultralytics.engine.results.Results'>
    # (Pdb) dir(detections[0])
    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_keys', 'boxes', 'cpu', 'cuda', 'keypoints', 'masks', 'names', 'new', 'numpy', 'obb', 'orig_img', 'orig_shape', 'path', 'plot', 'probs', 'save', 'save_crop', 'save_dir', 'save_txt', 'show', 'speed', 'summary', 'to', 'tojson', 'update', 'verbose']
    # (Pdb) detections[0]._keys
    # ('boxes', 'masks', 'probs', 'keypoints', 'obb')
    # (Pdb) detections[0].boxes
    # ultralytics.engine.results.Boxes object with attributes:
    # cls: tensor([0.], device='cuda:0')
    # conf: tensor([0.8516], device='cuda:0')
    # data: tensor([[6.3800e+02, 7.3200e+02, 8.8200e+02, 1.0350e+03, 8.5157e-01, 0.0000e+00]], device='cuda:0')
    # id: None
    # is_track: False
    # orig_shape: (1080, 1920)
    # shape: torch.Size([1, 6])
    # xywh: tensor([[760.0000, 883.5000, 244.0000, 303.0000]], device='cuda:0')
    # xywhn: tensor([[0.3958, 0.8181, 0.1271, 0.2806]], device='cuda:0')
    # xyxy: tensor([[ 638.,  732.,  882., 1035.]], device='cuda:0')
    # xyxyn: tensor([[0.3323, 0.6778, 0.4594, 0.9583]], device='cuda:0')


if __name__ == "__main__":
    # test_robo_suite()
    # test_hand2gripper_robot_inpaint_arx_controller()
    test_real_dual_arm_controller()