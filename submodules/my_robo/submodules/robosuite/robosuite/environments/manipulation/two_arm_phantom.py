from collections import OrderedDict
import os
import json
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.objects import BallObject

def _load_hand2gripper_params(key: str):
    """
    Load global parameters for the environment.

    Args:
        key (str): The parameter key to load.
    Returns:
        The value of the global parameter.
    """
    hand2gripper_config_path = os.environ.get("HAND2GRIPPER_CONFIG_PATH", "hand2gripper_config.json")
    if os.path.exists(hand2gripper_config_path): 
        with open(hand2gripper_config_path, 'r') as f:
            config = json.load(f)
            val = config.get(key, None)
            if val is not None:
                return val
    # Default values if config file or key does not exist
    default_params = {
        "robots-left-base_x": 0,
        "robots-left-base_y": -0.2,
        "robots-left-base_z": 0.8,
        "robots-right-base_x": 0,
        "robots-right-base_y": 0.2,
        "robots-right-base_z": 0.8,

        "camera-zed-pos_x": -1.0,
        "camera-zed-pos_y": 0,
        "camera-zed-pos_z": 1.3,
        "camera-zed-quat_w": 0.5,
        "camera-zed-quat_x": -0.5,
        "camera-zed-quat_y": -0.5,
        "camera-zed-quat_z": 0.5,
        "camera-zed-offset_horizon": 45.0,
        "camera-zed-fx": 615.973876953125,
        "camera-zed-fy": 614.9360961914062,
        "camera-zed-cx": 321.73553466796875,
        "camera-zed-cy": 238.9122314453125,
        "camera-zed-size_w": 0.0010408,
        "camera-zed-size_h": 0.00077925,
        "camera-zed-fov": 55.0,
        "visualize-target-spheres": True,
    }
    return default_params.get(key, None)

class TwoArmPhantom(TwoArmEnv):
    """
    This class corresponds to the lifting task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot(s)
            Note: Must be either 2 robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment if two robots inputted. Can be either:

            :`'parallel'`: Sets up the two robots next to each other on the -x side of the table
            :`'opposed'`: Sets up the two robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" "opposed" if two robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        return 0


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            elif self.env_configuration == "phantom_parallel":  # "phantom_parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                left_robot, right_robot = self.robots
                base_pose_left = np.array((
                    _load_hand2gripper_params("robots-left-base_x"),
                    _load_hand2gripper_params("robots-left-base_y"),
                    _load_hand2gripper_params("robots-left-base_z"),
                ))
                base_pose_right = np.array((
                    _load_hand2gripper_params("robots-right-base_x"),
                    _load_hand2gripper_params("robots-right-base_y"),
                    _load_hand2gripper_params("robots-right-base_z"),
                ))
                left_robot.robot_model.set_base_xpos(base_pose_left)
                right_robot.robot_model.set_base_xpos(base_pose_right)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
        
        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        zed_camera_pos = np.array((
            _load_hand2gripper_params("camera-zed-pos_x"),
            _load_hand2gripper_params("camera-zed-pos_y"),
            _load_hand2gripper_params("camera-zed-pos_z"),
        ))
        zed_camera_quat = np.array((
            _load_hand2gripper_params("camera-zed-quat_x"),
            _load_hand2gripper_params("camera-zed-quat_y"),
            _load_hand2gripper_params("camera-zed-quat_z"),
            _load_hand2gripper_params("camera-zed-quat_w"),
        ))
        zed_camera_offset_horizon = _load_hand2gripper_params("camera-zed-offset_horizon")
        
        if zed_camera_offset_horizon is not None:
            # Rotate around local X axis to look down
            # Positive offset means looking down (rotating negative angle around X)
            rot_angle = np.deg2rad(zed_camera_offset_horizon)
            rot_quat = T.axisangle2quat(np.array([0, rot_angle, 0]))
            zed_camera_quat = T.quat_multiply(zed_camera_quat, rot_quat)

        # Prepare camera attributes
        camera_attribs = {
            "resolution": f"{self.camera_widths[0]} {self.camera_heights[0]}",
            # "principalpixel": f"{_load_hand2gripper_params('camera-zed-cx')} {_load_hand2gripper_params('camera-zed-cy')}",  # (cx, cy)
            # "focalpixel": f"{_load_hand2gripper_params('camera-zed-fx')} {_load_hand2gripper_params('camera-zed-fy')}",  # (fx, fy)
            # 'sensorsize': f"{_load_hand2gripper_params('camera-zed-size_w')} {_load_hand2gripper_params('camera-zed-size_h')}",  # (size_w, size_h)
            "fovy": f"{_load_hand2gripper_params('camera-zed-fov')}",  # field of view
        }

        mujoco_arena.set_camera(
            camera_name="zed",
            pos=zed_camera_pos,
            quat=zed_camera_quat,
            camera_attribs=camera_attribs,
        )
        print(f"Set zed camera pos to {zed_camera_pos}")

        # Check if we should visualize target spheres
        visualize_spheres = _load_hand2gripper_params("visualize-target-spheres")
        mujoco_objects = []

        if visualize_spheres:
            # Add target spheres
            # We use obj_type="all" to ensure the body has mass (density is applied to collision geoms).
            # Then we manually disable collisions by setting contype/conaffinity to 0.
            self.target_sphere_0 = BallObject(
                name="target_sphere_0",
                size=[0.02],
                rgba=[1, 0, 0, 0.5],
                obj_type="all",
                joints=[dict(type="free", damping="0.0005")],
                density=1000,
            )
            # Disable collision for sphere 0
            for geom in self.target_sphere_0.get_obj().findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            # Disable gravity for sphere 0 (gravcomp=1 cancels gravity)
            self.target_sphere_0.get_obj().set("gravcomp", "1")

            self.target_sphere_1 = BallObject(
                name="target_sphere_1",
                size=[0.02],
                rgba=[0, 1, 0, 0.5],
                obj_type="all",
                joints=[dict(type="free", damping="0.0005")],
                density=1000,
            )
            # Disable collision for sphere 1
            for geom in self.target_sphere_1.get_obj().findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            # Disable gravity for sphere 1
            self.target_sphere_1.get_obj().set("gravcomp", "1")

            mujoco_objects = [self.target_sphere_0, self.target_sphere_1]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        return False
