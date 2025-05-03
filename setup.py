from setuptools import setup, find_packages

setup(
    name="trt_inference",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "tensorrt>=8.0",
        "pycuda>=2021.1",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for TensorRT engine creation and inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trt_inference",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)