from setuptools import setup, find_packages

setup(
    name="gesture_recognition",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "opencv-python>=4.9.0.80",
        "torch>=2.7.0",
        "torchvision>=0.18.0",
        "tensorflow>=2.16.1",
        "keras>=2.16.1",
        "scikit-learn>=1.4.1.post1",
        "pillow>=10.2.0",
        "python-dotenv>=1.0.1",
        "tqdm>=4.66.2",
        "matplotlib>=3.8.3",
        "ultralytics>=8.1.28",
        "mediapipe>=0.10.11"
    ],
    python_requires=">=3.13.3",
    author="Your Name",
    description="Hand Gesture Recognition System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 