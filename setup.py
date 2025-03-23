from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="yolov5_dior",
    version="0.1.0",
    author="Abdulbaki Kocabasa, Yue Yao",
    author_email="bakikocabasa@gmail.com",
    description="YOLOv5 Object Detection on DIOR Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Yolov5-Object-Detection-Application-on-DIOR-Dataset",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
) 