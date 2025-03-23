# YOLOv5 Object Detection on DIOR Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-red.svg)](https://pytorch.org/)

This repository contains the implementation of YOLOv5 object detection models (YOLOv5s and YOLOv5m) on the DIOR (Dataset for Object Detection in Optical Remote Sensing Images) dataset. The project aims to detect various objects in remote sensing imagery.

## Overview

The DIOR dataset contains 23,463 images and 190,288 instances covering 20 object categories. This implementation uses the YOLOv5 architecture to perform object detection on this dataset, focusing on recognizing objects such as airplanes, airports, bridges, ships, and more.

## Repository Structure

```
.
├── data/                  # Data directory
│   ├── raw/               # Raw data, including original images and annotations
│   └── processed/         # Processed data ready for training
├── docs/                  # Documentation
├── models/                # Model directory
│   └── weights/           # Pre-trained model weights
├── src/                   # Source code
│   ├── preprocessing/     # Scripts for data preparation and preprocessing
│   ├── training/          # Training scripts and notebooks
│   ├── evaluation/        # Evaluation scripts
│   └── utils/             # Utility functions
├── .gitattributes         # Git attributes file
├── .gitignore             # Git ignore file
├── LICENSE                # License file
├── README.md              # Project README
└── requirements.txt       # Python dependencies
```

## Features

- Data preparation and preprocessing for the DIOR dataset
- Implementation of YOLOv5s (small) and YOLOv5m (medium) models
- Training and evaluation pipelines
- Conversion tools for annotation formats (XML to TXT)
- Dataset splitting utilities

## Object Categories

The DIOR dataset contains the following 20 object categories:
- Airplane
- Airport
- Baseball field
- Basketball court
- Bridge
- Chimney
- Dam
- Expressway service area
- Expressway toll station
- Golf field
- Ground track field
- Harbor
- Overpass
- Ship
- Stadium
- Storage tank
- Tennis court
- Train station
- Vehicle
- Windmill

## Installation and Usage

### Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/Yolov5-Object-Detection-Application-on-DIOR-Dataset.git
cd Yolov5-Object-Detection-Application-on-DIOR-Dataset
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### DIOR Dataset Preparation

1. **Automatic Download** (Recommended):
   ```bash
   # Download and prepare the DIOR dataset automatically
   python -m src.utils.download_dataset --output_dir data/raw
   ```

2. **Manual Download**:
   - Download the DIOR dataset from the [official source](http://www.escience.cn/people/gongcheng/DIOR.html).
     - You need to request access by contacting the dataset authors.
     - Alternatively, you can download it from [this direct link](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC).

3. **Dataset Structure**:
   - The dataset contains three main parts:
     - `Images/`: Contains all the image files (JPG format)
     - `Annotations/`: Contains XML annotation files in two folders:
       - `Horizontal Bounding Boxes/`: Annotations with horizontal bounding boxes
       - `Oriented Bounding Boxes/`: Annotations with oriented bounding boxes

4. **Dataset Organization**:
   - Place the dataset files in the proper structure:
   ```
   data/
   └── raw/
       ├── Images/
       │   └── *.JPG files
       └── Annotations/
           ├── Horizontal Bounding Boxes/
           │   └── *.xml files
           └── Oriented Bounding Boxes/
               └── *.xml files
   ```

5. **Process the Dataset**:
   ```bash
   # Convert XML annotations to TXT format (YOLOv5 compatible)
   python -m src.preprocessing.xml_to_txt --xml_dir data/raw/Annotations/Horizontal\ Bounding\ Boxes --output_dir data/processed/labels

   # Split the dataset into train/val/test sets
   python -m src.preprocessing.split_dataset --images_dir data/raw/Images --labels_dir data/processed/labels --output_dir data/processed/dataset
   ```

6. **One-step Processing** (After downloading the dataset):
   ```bash
   # Process the dataset in one step
   python -m src.preprocessing.prepare_data
   ```

### Training

To train the YOLOv5 model on the DIOR dataset:

1. Using the Python script:
```bash
# Train YOLOv5s
python -m src.training.train --model yolov5s --img_size 640 --batch_size 16 --epochs 100

# Train YOLOv5m
python -m src.training.train --model yolov5m --img_size 640 --batch_size 8 --epochs 100
```

2. Using the training notebook:
```bash
# Run the training notebook
jupyter notebook src/training/Final_Project_Run.ipynb
```

The notebook provides a step-by-step guide to:
- Set up the environment
- Download YOLOv5 repository
- Prepare the dataset
- Configure training parameters
- Run the training process
- Monitor training metrics
- Validate the model

### Inference

To run inference with a trained model:

1. Using the Python script (Recommended):
```bash
# Run inference on images or videos
python -m src.evaluation.inference --weights models/weights/yolov5s_dior.pt --source path/to/images --img_size 640 --conf_thres 0.25

# Run inference on a video
python -m src.evaluation.inference --weights models/weights/yolov5s_dior.pt --source path/to/video.mp4 --img_size 640

# Save detection results as text files
python -m src.evaluation.inference --weights models/weights/yolov5s_dior.pt --source path/to/images --save_txt --save_conf
```

2. Using YOLOv5 directly:
```bash
# Clone YOLOv5 repository if not already done
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Run inference
python detect.py --weights ../models/weights/yolov5s_dior.pt --source path/to/test/images --img 640
```

### Evaluation

Evaluate the trained model:

1. Using the Python script:
```bash
# Evaluate on validation set
python -m src.training.validate --weights models/weights/yolov5s_dior.pt --data data/processed/dataset/dior.yaml --img_size 640

# Evaluate on test set
python -m src.training.validate --weights models/weights/yolov5s_dior.pt --data data/processed/dataset/dior.yaml --img_size 640 --task test
```

2. Using YOLOv5 directly:
```bash
# Clone YOLOv5 repository if not already done
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Run validation
python val.py --weights ../models/weights/yolov5s_dior.pt --data ../data/processed/dataset/dior.yaml --img 640
```

## Results

The YOLOv5 models trained on the DIOR dataset achieved the following performance:

| Model    | mAP@0.5 | mAP@0.5:0.95 |
|----------|---------|--------------|
| YOLOv5s  | 76.8%   | 46.3%        |
| YOLOv5m  | 81.2%   | 50.5%        |

## Examples

![Example Detection](docs/example_detection.png)

## Contributors

- [Abdulbaki Kocabasa](mailto:bakikocabasa@gmail.com)
- [Yue Yao](mailto:yue.yao@tu-braunschweig.de)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [DIOR Dataset](http://www.escience.cn/people/gongcheng/DIOR.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- Technical University of Braunschweig - Institute for Geophysics and Extraterrestrial Physics
