"""
Validation script for YOLOv5 models on the DIOR dataset.
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
    Create a YAML file for the dataset configuration if it doesn't exist.
    
    Args:
        dataset_dir (str): Directory containing the split dataset
        yaml_path (str): Path to save the YAML file
    """
    if os.path.exists(yaml_path):
        print(f"Dataset YAML already exists at {yaml_path}")
        return
        
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

def validate_model(args):
    """
    Validate the YOLOv5 model.
    
    Args:
        args: Command-line arguments
    """
    # Setup YOLOv5
    setup_yolov5(args.yolov5_dir)
    
    # Create dataset YAML if it doesn't exist
    yaml_path = args.data
    if not os.path.exists(yaml_path):
        dataset_dir = os.path.dirname(yaml_path)
        create_dataset_yaml(dataset_dir, yaml_path)
    
    # Build validation command
    cmd = [
        "python", os.path.join(args.yolov5_dir, "val.py"),
        "--img", str(args.img_size),
        "--batch", str(args.batch_size),
        "--data", yaml_path,
        "--weights", args.weights,
        "--task", args.task,
        "--name", args.name,
        "--project", args.project_dir
    ]
    
    # Add optional arguments
    if args.device:
        cmd.extend(["--device", args.device])
    if args.conf_thres:
        cmd.extend(["--conf-thres", str(args.conf_thres)])
    if args.iou_thres:
        cmd.extend(["--iou-thres", str(args.iou_thres)])
    if args.verbose:
        cmd.append("--verbose")
    
    # Print the command
    print("Running validation command:")
    print(" ".join(cmd))
    
    # Run the validation
    subprocess.run(cmd, check=True)
    
    print(f"Validation completed. Results saved to {os.path.join(args.project_dir, args.name)}")

def main():
    """Main function to validate YOLOv5 on the DIOR dataset."""
    parser = argparse.ArgumentParser(description="Validate YOLOv5 on the DIOR dataset")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights")
    parser.add_argument("--data", type=str, default=os.path.join(PROCESSED_DATA_DIR, "dataset", "dior.yaml"),
                        help="Path to dataset YAML file")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for validation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for validation")
    parser.add_argument("--task", type=str, default="val", choices=["val", "test"],
                        help="Validation task: 'val' for validation set, 'test' for test set")
    parser.add_argument("--name", type=str, default="dior_validation",
                        help="Validation run name")
    parser.add_argument("--project_dir", type=str, default=os.path.join(MODELS_DIR, "runs", "val"),
                        help="Directory to save validation results")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use for validation (e.g., '0' for GPU)")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output")
    parser.add_argument("--yolov5_dir", type=str, default=os.path.join(MODELS_DIR, "yolov5"),
                        help="Directory to clone/use YOLOv5 repository")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Validate the model
    validate_model(args)

if __name__ == "__main__":
    main() 