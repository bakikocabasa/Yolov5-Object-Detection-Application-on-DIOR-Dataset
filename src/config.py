"""Configuration settings for the YOLOv5 DIOR project."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WEIGHTS_DIR = os.path.join(MODELS_DIR, "weights")

# Dataset settings
NUM_CLASSES = 20
CLASS_NAMES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

# Model settings
MODEL_CONFIGS = {
    "yolov5s": {
        "weights": os.path.join(WEIGHTS_DIR, "yolov5s_dior.pt"),
        "batch_size": 16,
        "img_size": 640,
    },
    "yolov5m": {
        "weights": os.path.join(WEIGHTS_DIR, "yolov5m_dior.pt"),
        "batch_size": 8,
        "img_size": 640,
    }
}

# Training settings
EPOCHS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005 