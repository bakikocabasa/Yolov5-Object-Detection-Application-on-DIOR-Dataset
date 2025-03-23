"""
Utility script to create a mock detection example image.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
import random

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import CLASS_NAMES


def draw_detection_box(image, x, y, w, h, label, color, conf=None):
    """
    Draw a detection box on the image.
    
    Args:
        image (numpy.ndarray): Image to draw on
        x, y, w, h (float): Box coordinates (center_x, center_y, width, height) normalized [0-1]
        label (str): Class label
        color (tuple): Box color (B, G, R)
        conf (float, optional): Confidence score
    """
    # Convert normalized coordinates to pixel coordinates
    img_h, img_w = image.shape[:2]
    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)
    
    # Ensure coordinates are within image boundaries
    x1 = max(0, min(x1, img_w-1))
    y1 = max(0, min(y1, img_h-1))
    x2 = max(0, min(x2, img_w-1))
    y2 = max(0, min(y2, img_h-1))
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    if conf is not None:
        label_text = f"{label} {conf:.2f}"
    else:
        label_text = label
    
    # Draw label background
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (x1, y1-text_size[1]-4), (x1+text_size[0]+2, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label_text, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def create_example_detection(input_image, output_path, num_detections=5):
    """
    Create a mock detection example image.
    
    Args:
        input_image (str): Path to the input image
        output_path (str): Path to save the example detection image
        num_detections (int): Number of mock detections to add
    """
    # Read the input image
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Could not read image {input_image}")
        return None
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define detection colors
    colors = {
        'airplane': (255, 0, 0),      # Blue
        'ship': (0, 255, 0),          # Green
        'storage tank': (0, 0, 255),  # Red
        'harbor': (255, 255, 0),      # Cyan
        'vehicle': (255, 0, 255),     # Magenta
    }
    
    # Define mock detections (class, x, y, w, h, conf)
    mock_detections = []
    
    # Either use predefined mock detections or generate random ones
    if input_image.endswith('sample_video_frame_100.jpg'):
        # For our specific sample frame, add some realistic detections
        mock_detections = [
            ('harbor', 0.5, 0.5, 0.8, 0.6, 0.93),
            ('ship', 0.3, 0.6, 0.04, 0.03, 0.87),
            ('ship', 0.62, 0.58, 0.03, 0.02, 0.82),
            ('ship', 0.7, 0.48, 0.04, 0.03, 0.89),
            ('vehicle', 0.85, 0.73, 0.02, 0.01, 0.76)
        ]
    else:
        # Generate random detections
        available_classes = list(colors.keys())
        for _ in range(num_detections):
            class_name = random.choice(available_classes)
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            w = random.uniform(0.05, 0.2)
            h = random.uniform(0.05, 0.2)
            conf = random.uniform(0.7, 0.95)
            mock_detections.append((class_name, x, y, w, h, conf))
    
    # Draw the detections on the image
    for class_name, x, y, w, h, conf in mock_detections:
        color = colors.get(class_name, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw_detection_box(image, x, y, w, h, class_name, color, conf)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Example detection image saved to {output_path}")
    
    return output_path


def main():
    """Main function to create an example detection image."""
    parser = argparse.ArgumentParser(description="Create a mock detection example image")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default="docs/images/example_detection.jpg",
                        help="Path to save the example detection image")
    parser.add_argument("--num_detections", type=int, default=5,
                        help="Number of mock detections to add")
    
    args = parser.parse_args()
    
    create_example_detection(args.input, args.output, args.num_detections)


if __name__ == "__main__":
    main() 