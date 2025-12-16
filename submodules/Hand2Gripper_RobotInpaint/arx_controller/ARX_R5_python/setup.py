from setuptools import setup, find_packages

setup(
    name="arx_r5_python",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    include_package_data=True,
    description="Python interface for ARX R5 robot kinematics",
)
