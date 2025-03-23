"""
Utility script to extract a frame from a video file.
"""

import os
import argparse
import cv2
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))


def extract_frame(video_path, output_dir, frame_number=0):
    """
    Extract a single frame from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frame
        frame_number (int): Frame number to extract (0 for the first frame)
    
    Returns:
        str: Path to the saved frame image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video has {total_frames} frames")
    
    # Validate frame number
    if frame_number >= total_frames:
        print(f"Error: Requested frame {frame_number} exceeds total frames {total_frames}")
        cap.release()
        return None
    
    # Set position to the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Check if frame was read successfully
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return None
    
    # Generate output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_number}.jpg")
    
    # Save the frame
    cv2.imwrite(output_path, frame)
    print(f"Frame saved to {output_path}")
    
    # Release the video capture object
    cap.release()
    
    return output_path


def main():
    """Main function to extract a frame from a video file."""
    parser = argparse.ArgumentParser(description="Extract a frame from a video file")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the video file")
    parser.add_argument("--output_dir", type=str, default="docs/images",
                        help="Directory to save the extracted frame")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame number to extract (0 for the first frame)")
    
    args = parser.parse_args()
    
    extract_frame(args.video, args.output_dir, args.frame)


if __name__ == "__main__":
    main() 