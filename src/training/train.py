"""
Training script for YOLOv5 on the DIOR dataset.
This script clones the YOLOv5 repository and runs the training process.
"""

import os
import argparse
import subprocess
import yaml
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, CLASS_NAMES

def setup_yolov5(yolov5_dir):
    """
    Clone YOLOv5 repository if it doesn't exist and install dependencies.
    
    Args:
        yolov5_dir (str): Directory to clone YOLOv5 repository
    """
    if not os.path.exists(yolov5_dir):
        print(f"Cloning YOLOv5 repository to {yolov5_dir}...")
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5", yolov5_dir],
            check=True
        )
        
        # Install YOLOv5 dependencies
        subprocess.run(
            ["pip", "install", "-r", os.path.join(yolov5_dir, "requirements.txt")],
            check=True
        )
    else:
        print(f"YOLOv5 repository already exists at {yolov5_dir}")

def create_dataset_yaml(dataset_dir, yaml_path):
    """
    Create a YAML file for the dataset configuration.
    
    Args:
        dataset_dir (str): Directory containing the split dataset
        yaml_path (str): Path to save the YAML file
    """
    dataset_config = {
        'path': dataset_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset YAML configuration at {yaml_path}")

def train_model(args):
    """
    Train the YOLOv5 model.
    
    Args:
        args: Command-line arguments
    """
    # Setup YOLOv5
    setup_yolov5(args.yolov5_dir)
    
    # Create dataset YAML
    yaml_path = os.path.join(args.dataset_dir, "dior.yaml")
    create_dataset_yaml(args.dataset_dir, yaml_path)
    
    # Build training command
    cmd = [
        "python", os.path.join(args.yolov5_dir, "train.py"),
        "--img", str(args.img_size),
        "--batch", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--data", yaml_path,
        "--cfg", os.path.join(args.yolov5_dir, "models", f"{args.model}.yaml"),
        "--weights", args.weights,
        "--name", args.name,
        "--project", args.project_dir
    ]
    
    # Add optional arguments
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Print the command
    print("Running training command:")
    print(" ".join(cmd))
    
    # Run the training
    subprocess.run(cmd, check=True)
    
    print(f"Training completed. Results saved to {os.path.join(args.project_dir, args.name)}")

def main():
    """Main function to train YOLOv5 on the DIOR dataset."""
    parser = argparse.ArgumentParser(description="Train YOLOv5 on the DIOR dataset")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(PROCESSED_DATA_DIR, "dataset"),
                        help="Directory containing the split dataset")
    parser.add_argument("--model", type=str, default="yolov5s", choices=["yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                        help="YOLOv5 model size")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--weights", type=str, default="",
                        help="Initial weights path or empty for scratch")
    parser.add_argument("--name", type=str, default="dior_yolov5",
                        help="Training run name")
    parser.add_argument("--project_dir", type=str, default=os.path.join(MODELS_DIR, "runs"),
                        help="Directory to save training results")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use for training (e.g., '0' for GPU)")
    parser.add_argument("--yolov5_dir", type=str, default=os.path.join(MODELS_DIR, "yolov5"),
                        help="Directory to clone/use YOLOv5 repository")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Train the model
    train_model(args)

if __name__ == "__main__":
    main() 