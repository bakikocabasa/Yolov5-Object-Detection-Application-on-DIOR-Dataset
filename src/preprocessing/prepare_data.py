"""
Data preparation script for the DIOR dataset.
This script converts XML annotation files to formats suitable for YOLOv5.
"""

import os
import zipfile
from xml.etree import ElementTree
import csv
import argparse
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_csv_from_xml(xml_path, output_dir):
    """Convert XML annotation files to CSV format."""
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_path):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ElementTree.parse(os.path.join(xml_path, xml_file))
        root = tree.getroot()
        
        # Extract image information
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        with open(os.path.join(output_dir, f"{os.path.splitext(xml_file)[0]}.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class', 'x', 'y', 'width', 'height'])
            
            for obj in root.findall('object'):
                obj_class = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to center coordinates and normalized width/height
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                writer.writerow([obj_class, x_center, y_center, w, h])
                
    print(f"Converted XML files to CSV in {output_dir}")

def main(args):
    """Main function to prepare the DIOR dataset."""
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Extract annotations if zip file exists
    annotations_zip = os.path.join(RAW_DATA_DIR, "Annotations.zip")
    if os.path.exists(annotations_zip):
        print(f"Extracting {annotations_zip}...")
        extract_zip(annotations_zip, RAW_DATA_DIR)
    
    # Convert horizontal bounding box annotations to CSV
    hbb_path = os.path.join(RAW_DATA_DIR, "Annotations", "Horizontal Bounding Boxes")
    if os.path.exists(hbb_path):
        print("Converting Horizontal Bounding Boxes to CSV...")
        create_csv_from_xml(
            hbb_path, 
            os.path.join(PROCESSED_DATA_DIR, "Horizontal Bounding Boxes CSV")
        )
    
    # Convert oriented bounding box annotations to CSV
    obb_path = os.path.join(RAW_DATA_DIR, "Annotations", "Oriented Bounding Boxes")
    if os.path.exists(obb_path):
        print("Converting Oriented Bounding Boxes to CSV...")
        create_csv_from_xml(
            obb_path, 
            os.path.join(PROCESSED_DATA_DIR, "Oriented Bounding Boxes CSV")
        )
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DIOR dataset for YOLOv5")
    parser.add_argument("--raw_dir", type=str, default=RAW_DATA_DIR, 
                        help="Directory containing the raw data")
    parser.add_argument("--processed_dir", type=str, default=PROCESSED_DATA_DIR, 
                        help="Directory to save processed data")
    args = parser.parse_args()
    
    main(args) 