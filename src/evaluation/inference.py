"""
Inference script for running YOLOv5 models on new images.
"""

import os
import argparse
import subprocess
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODELS_DIR

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

def run_inference(args):
    """
    Run inference with YOLOv5 model on new images.
    
    Args:
        args: Command-line arguments
    """
    # Setup YOLOv5
    setup_yolov5(args.yolov5_dir)
    
    # Build inference command
    cmd = [
        "python", os.path.join(args.yolov5_dir, "detect.py"),
        "--img", str(args.img_size),
        "--conf", str(args.conf_thres),
        "--iou", str(args.iou_thres),
        "--weights", args.weights,
        "--source", args.source,
        "--name", args.name,
        "--project", args.project_dir
    ]
    
    # Add optional arguments
    if args.device:
        cmd.extend(["--device", args.device])
    if args.save_txt:
        cmd.append("--save-txt")
    if args.save_conf:
        cmd.append("--save-conf")
    if args.nosave:
        cmd.append("--nosave")
    if args.classes:
        cmd.extend(["--classes"] + [str(c) for c in args.classes])
    if args.agnostic_nms:
        cmd.append("--agnostic-nms")
    if args.augment:
        cmd.append("--augment")
    
    # Print the command
    print("Running inference command:")
    print(" ".join(cmd))
    
    # Run the inference
    subprocess.run(cmd, check=True)
    
    print(f"Inference completed. Results saved to {os.path.join(args.project_dir, args.name)}")

def main():
    """Main function to run inference with YOLOv5 on new images."""
    parser = argparse.ArgumentParser(description="Run inference with YOLOv5 model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Source image or directory")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for inference")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use for inference (e.g., '0' for GPU)")
    parser.add_argument("--name", type=str, default="dior_inference",
                        help="Inference run name")
    parser.add_argument("--project_dir", type=str, default=os.path.join(MODELS_DIR, "runs", "detect"),
                        help="Directory to save inference results")
    parser.add_argument("--save_txt", action="store_true",
                        help="Save results to *.txt")
    parser.add_argument("--save_conf", action="store_true",
                        help="Save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true",
                        help="Do not save images/videos")
    parser.add_argument("--classes", type=int, nargs="+",
                        help="Filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic_nms", action="store_true",
                        help="Class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="Augmented inference")
    parser.add_argument("--yolov5_dir", type=str, default=os.path.join(MODELS_DIR, "yolov5"),
                        help="Directory to clone/use YOLOv5 repository")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Run inference
    run_inference(args)

if __name__ == "__main__":
    main() 