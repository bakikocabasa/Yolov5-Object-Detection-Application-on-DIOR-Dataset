"""
Utility script to download the DIOR dataset from Google Drive.
"""

import os
import sys
import argparse
from pathlib import Path
import gdown
import zipfile
import shutil

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR

# Google Drive links for the DIOR dataset
GDRIVE_LINKS = {
    "images": "https://drive.google.com/uc?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC",
    "annotations": "https://drive.google.com/uc?id=1BNdl9n_dHcXnmzn-qLyuYTfa3olWD7F8"
}

def download_file(url, output_path):
    """
    Download a file from Google Drive.
    
    Args:
        url (str): Google Drive URL
        output_path (str): Path to save the downloaded file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    print(f"Downloaded to {output_path}")

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Directory to extract to
    """
    os.makedirs(extract_to, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def download_and_prepare_dataset(output_dir, keep_zip=False):
    """
    Download and prepare the DIOR dataset.
    
    Args:
        output_dir (str): Directory to save the dataset
        keep_zip (bool): Whether to keep the zip files after extraction
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download images
    images_zip = os.path.join(output_dir, "Images.zip")
    download_file(GDRIVE_LINKS["images"], images_zip)
    
    # Download annotations
    annotations_zip = os.path.join(output_dir, "Annotations.zip")
    download_file(GDRIVE_LINKS["annotations"], annotations_zip)
    
    # Extract images
    images_dir = os.path.join(output_dir, "Images")
    extract_zip(images_zip, images_dir)
    
    # Extract annotations
    annotations_dir = os.path.join(output_dir, "Annotations")
    extract_zip(annotations_zip, annotations_dir)
    
    # Clean up zip files if not keeping them
    if not keep_zip:
        os.remove(images_zip)
        os.remove(annotations_zip)
    
    print("Dataset download and preparation complete!")
    print(f"Images are in: {images_dir}")
    print(f"Annotations are in: {annotations_dir}")

def main():
    """Main function to download and prepare the DIOR dataset."""
    parser = argparse.ArgumentParser(description="Download and prepare the DIOR dataset")
    parser.add_argument("--output_dir", type=str, default=RAW_DATA_DIR,
                        help="Directory to save the dataset")
    parser.add_argument("--keep_zip", action="store_true", 
                        help="Keep the zip files after extraction")
    
    args = parser.parse_args()
    
    download_and_prepare_dataset(args.output_dir, args.keep_zip)

if __name__ == "__main__":
    main() 