from setuptools import setup, find_packages

setup(
    name="genesis_ILDP",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "zarr",
        "numcodecs",
    ],
    python_requires=">=3.8",
    author="Research Team",
    description="Genesis ILDP package for imitation learning and diffusion policy",
)