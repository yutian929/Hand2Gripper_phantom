from setuptools import setup, find_packages

setup(
    name="hand2gripper_robot_inpaint",
    version="0.1.0",
    packages=["hand2gripper_robot_inpaint", "hand2gripper_robot_inpaint.arx_controller"],
    package_dir={
        "hand2gripper_robot_inpaint": ".",
        "hand2gripper_robot_inpaint.arx_controller": "./arx_controller"
    },
    install_requires=[
        # Add dependencies here, e.g., 'numpy', 'mujoco'
    ],
)
