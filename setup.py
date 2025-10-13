from pathlib import Path

from setuptools import find_packages
from setuptools import setup

dir_path = Path(__file__).resolve().parent


def read_requirements_file(filename):
    req_file = dir_path.joinpath(filename)
    with req_file.open('r') as f:
        return [line.strip() for line in f]


packages = find_packages(exclude=[])
pkgs = []
for p in packages:
    if p == 'symdex' or p.startswith('symdex.'):
        pkgs.append(p)

setup(
    name='symdex',
    author='Zechu Li',
    license='MIT',
    packages=pkgs,
    install_requires=[
        "coacd",
        "pytransform3d==3.8.0",
        "pybullet==3.2.5",
        "escnn==1.0.11",
        "robot-descriptions==1.13.0",
        "pin==2.7.0",
        "hydra-core",
        "loguru",
        "ray",
        #"wandb",
        "cloudpickle",
        "scipy",
        "shortuuid",
        "ninja",
        "open3d",
        "orjson",
        "zarr",
    ],
    include_package_data=True,
)
