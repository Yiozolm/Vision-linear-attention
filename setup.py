from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')

setup(
    name="vision-linear-attention",
    version="0.1.0",
    packages=find_packages(),
    author="Fangzhou Yi",
    author_email="m202310581@xs.ustb.edu.cn",
    description="A collection of vision linear attention models",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Yiozolm/Vision-linear-attention",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "torch>=2.5.1",
        "ninja",
        "timm",
        "einops",
        # Note: mamba-ssm requires manual installation from .whl files
    ],
    python_requires=">=3.11",
)