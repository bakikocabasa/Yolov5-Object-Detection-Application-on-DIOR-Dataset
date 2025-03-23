"""
Script for splitting the DIOR dataset into training, validation, and testing sets.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import argparse
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR

def split_dataset(images_dir, labels_dir, output_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and testing sets.
    
    Args:
        images_dir (str): Directory containing image files
        labels_dir (str): Directory containing label files
        output_dir (str): Directory to save the split dataset
        test_size (float): Proportion of the dataset to include in the test split
        val_size (float): Proportion of the dataset to include in the validation split
        random_state (int): Random seed for reproducibility
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_label_dir = os.path.join(output_dir, "train", "labels")
    val_img_dir = os.path.join(output_dir, "valid", "images")
    val_label_dir = os.path.join(output_dir, "valid", "labels")
    test_img_dir = os.path.join(output_dir, "test", "images")
    test_label_dir = os.path.join(output_dir, "test", "labels")
    
    # Create all required directories
    for directory in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split data into train+val and test sets
    train_val_files, test_files = train_test_split(
        image_files, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Split train+val into train and val sets
    train_files, val_files = train_test_split(
        train_val_files, 
        test_size=val_size/(1-test_size),  # Adjust val_size to account for the test split
        random_state=random_state
    )
    
    print(f"Dataset split: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} testing")
    
    # Copy files to respective directories
    for files, img_dir, label_dir in [
        (train_files, train_img_dir, train_label_dir),
        (val_files, val_img_dir, val_label_dir),
        (test_files, test_img_dir, test_label_dir)
    ]:
        for file in files:
            # Copy image
            shutil.copy(
                os.path.join(images_dir, file),
                os.path.join(img_dir, file)
            )
            
            # Copy label (assuming same filename with .txt extension)
            label_file = os.path.splitext(file)[0] + '.txt'
            if os.path.exists(os.path.join(labels_dir, label_file)):
                shutil.copy(
                    os.path.join(labels_dir, label_file),
                    os.path.join(label_dir, label_file)
                )
            else:
                print(f"Warning: Label file {label_file} not found")
    
    # Create dataset files listing
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for file in train_files:
            f.write(os.path.join('train', 'images', file) + '\n')
    
    with open(os.path.join(output_dir, 'valid.txt'), 'w') as f:
        for file in val_files:
            f.write(os.path.join('valid', 'images', file) + '\n')
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for file in test_files:
            f.write(os.path.join('test', 'images', file) + '\n')
    
    print(f"Dataset split complete. Files saved to {output_dir}")

def main():
    """Main function to split the dataset."""
    parser = argparse.ArgumentParser(description="Split the DIOR dataset into train/val/test sets")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing image files")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory containing label files")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROCESSED_DATA_DIR, "dataset"),
                        help="Directory to save the split dataset")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the validation split")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    split_dataset(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        args.test_size,
        args.val_size,
        args.random_state
    )

if __name__ == "__main__":
    main() 