"""
XML to TXT Converter for DIOR dataset annotations.
Converts XML annotation files to the format required by YOLOv5.
"""

import os
import xml.etree.ElementTree as ET
from decimal import Decimal
import argparse
from pathlib import Path
import sys

# Add the src directory to the path to import the config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES

def convert_xml_to_txt(xml_dir, output_dir, labels=None):
    """
    Convert XML annotation files to TXT format for YOLOv5.
    
    Args:
        xml_dir (str): Directory containing XML annotation files
        output_dir (str): Directory to save the converted TXT files
        labels (list): List of class labels
    """
    if labels is None:
        labels = CLASS_NAMES
    
    os.makedirs(output_dir, exist_ok=True)
    
    for fp in os.listdir(xml_dir):
        if not fp.endswith('.xml'):
            continue
            
        try:
            root = ET.parse(os.path.join(xml_dir, fp)).getroot()
            
            # Get image dimensions
            sz = root.find('size')
            width = float(sz.find('width').text)
            height = float(sz.find('height').text)
            filename = root.find('filename').text
            
            # Create output text file
            with open(os.path.join(output_dir, fp.split('.')[0] + '.txt'), 'w') as f:
                for child in root.findall('object'):  # Find all objects in the image
                    sub = child.find('bndbox')  # Find bounding box information
                    sub_label = child.find('name')
                    
                    # Extract coordinates
                    xmin = float(sub.find('xmin').text)
                    ymin = float(sub.find('ymin').text)
                    xmax = float(sub.find('xmax').text)
                    ymax = float(sub.find('ymax').text)
                    
                    try:
                        # Convert to YOLOv5 format (normalized center x, center y, width, height)
                        x_center = Decimal(str(round(float((xmin + xmax) / (2 * width)), 6))).quantize(Decimal('0.000000'))
                        y_center = Decimal(str(round(float((ymin + ymax) / (2 * height)), 6))).quantize(Decimal('0.000000'))
                        w = Decimal(str(round(float((xmax - xmin) / width), 6))).quantize(Decimal('0.000000'))
                        h = Decimal(str(round(float((ymax - ymin) / height), 6))).quantize(Decimal('0.000000'))
                        
                        # Find class index
                        for idx, label in enumerate(labels):
                            if sub_label.text == label:
                                f.write(' '.join([str(idx), str(x_center), str(y_center), str(w), str(h) + '\n']))
                                break
                    except ZeroDivisionError:
                        print(f"{filename} has a problem with dimensions")
            
            print(f"Converted {fp} to TXT format")
        except Exception as e:
            print(f"Error processing {fp}: {e}")
    
    print(f"Conversion complete. Saved TXT files to {output_dir}")

def main():
    """Main function to convert XML annotations to TXT format."""
    parser = argparse.ArgumentParser(description="Convert XML annotations to TXT format for YOLOv5")
    parser.add_argument("--xml_dir", type=str, 
                        default=os.path.join(RAW_DATA_DIR, "Annotations", "Horizontal Bounding Boxes"),
                        help="Directory containing XML annotation files")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(PROCESSED_DATA_DIR, "labels"),
                        help="Directory to save the converted TXT files")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert XML files to TXT
    convert_xml_to_txt(args.xml_dir, args.output_dir)

if __name__ == "__main__":
    main() 