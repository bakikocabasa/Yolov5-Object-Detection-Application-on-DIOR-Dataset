"""
Utility script to create an overview image of the DIOR dataset classes.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import CLASS_NAMES


def create_class_grid(output_path, grid_size=(5, 4), cell_size=(150, 150)):
    """
    Create a grid showing the 20 DIOR classes with their names.
    
    Args:
        output_path (str): Path to save the grid image
        grid_size (tuple): Grid dimensions (rows, columns)
        cell_size (tuple): Size of each cell (width, height)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define grid parameters
    grid_h, grid_w = grid_size
    cell_w, cell_h = cell_size
    img_w, img_h = grid_w * cell_w, grid_h * cell_h
    
    # Create a white image
    grid_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # Define colors for each class (BGR format)
    colors = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 0),     # Dark Blue
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Dark Red
        (128, 128, 0),   # Dark Cyan
        (128, 0, 128),   # Dark Magenta
        (0, 128, 128),   # Dark Yellow
        (64, 0, 0),      # Very Dark Blue
        (0, 64, 0),      # Very Dark Green
        (0, 0, 64),      # Very Dark Red
        (64, 64, 0),     # Very Dark Cyan
        (64, 0, 64),     # Very Dark Magenta
        (0, 64, 64),     # Very Dark Yellow
        (192, 192, 192), # Silver
        (128, 128, 128)  # Gray
    ]
    
    # Draw grid lines
    for i in range(1, grid_h):
        y = i * cell_h
        cv2.line(grid_img, (0, y), (img_w, y), (200, 200, 200), 1)
    
    for j in range(1, grid_w):
        x = j * cell_w
        cv2.line(grid_img, (x, 0), (x, img_h), (200, 200, 200), 1)
    
    # Draw class names and icons
    for idx, class_name in enumerate(CLASS_NAMES):
        if idx >= grid_h * grid_w:
            break
            
        # Calculate position in grid
        row = idx // grid_w
        col = idx % grid_w
        
        # Calculate top-left corner of cell
        x1 = col * cell_w
        y1 = row * cell_h
        
        # Draw a colored rectangle as class icon
        icon_size = min(cell_w, cell_h) // 2
        icon_x = x1 + (cell_w - icon_size) // 2
        icon_y = y1 + cell_h // 4 - icon_size // 2
        
        cv2.rectangle(grid_img, 
                     (icon_x, icon_y), 
                     (icon_x + icon_size, icon_y + icon_size), 
                     colors[idx], 
                     -1)  # Filled rectangle
        
        # Draw class name
        font_scale = 0.5
        text = class_name.capitalize()
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_x = x1 + (cell_w - text_size[0]) // 2
        text_y = y1 + cell_h - cell_h // 4
        
        cv2.putText(grid_img, 
                   text, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, 
                   (0, 0, 0), 
                   1)
    
    # Save the image
    cv2.imwrite(output_path, grid_img)
    print(f"DIOR dataset overview image saved to {output_path}")
    
    return output_path


def main():
    """Main function to create a DIOR dataset overview image."""
    parser = argparse.ArgumentParser(description="Create an overview image of the DIOR dataset classes")
    parser.add_argument("--output", type=str, default="docs/images/dior_classes.jpg",
                        help="Path to save the overview image")
    parser.add_argument("--grid_rows", type=int, default=5,
                        help="Number of rows in the grid")
    parser.add_argument("--grid_cols", type=int, default=4,
                        help="Number of columns in the grid")
    parser.add_argument("--cell_width", type=int, default=150,
                        help="Width of each cell in pixels")
    parser.add_argument("--cell_height", type=int, default=150,
                        help="Height of each cell in pixels")
    
    args = parser.parse_args()
    
    create_class_grid(
        args.output, 
        grid_size=(args.grid_rows, args.grid_cols), 
        cell_size=(args.cell_width, args.cell_height)
    )


if __name__ == "__main__":
    main() 