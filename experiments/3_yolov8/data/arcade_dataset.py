"""
Convert ARCADE dataset from COCO format to YOLO format for YOLOv8 segmentation.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import shutil


class ArcadeToYOLO:
    """Convert ARCADE COCO annotations to YOLO segmentation format."""

    def __init__(self, arcade_root, output_root, task='stenosis'):
        """
        Initialize converter.

        Args:
            arcade_root: Path to ARCADE dataset root
            output_root: Path to output YOLO dataset
            task: 'stenosis' or 'syntax'
        """
        self.arcade_root = Path(arcade_root)
        self.output_root = Path(output_root)
        self.task = task

        self.splits = ['train', 'val']
        self.output_root.mkdir(parents=True, exist_ok=True)

    def convert(self):
        """Convert ARCADE to YOLO format."""
        print(f"Converting ARCADE {self.task} task to YOLO format...")

        # Create YOLO directory structure
        self._create_directories()

        # Convert each split
        for split in self.splits:
            print(f"\nProcessing {split} split...")
            self._convert_split(split)

        # Create data.yaml
        self._create_yaml()

        print(f"\n✓ Conversion complete! Dataset saved to {self.output_root}")

    def _create_directories(self):
        """Create YOLO directory structure."""
        for split in self.splits:
            (self.output_root / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_root / split / 'labels').mkdir(parents=True, exist_ok=True)

    def _convert_split(self, split):
        """Convert a single split."""
        images_dir = self.arcade_root / self.task / split / 'images'
        ann_file = self.arcade_root / self.task / split / 'annotations' / f'{split}.json'

        if not ann_file.exists():
            print(f"  Annotation file not found: {ann_file}")
            return

        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # Create category mapping (YOLO expects indices 0 to num_classes-1)
        # Sort by category ID to ensure consistent mapping
        sorted_categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        cat_id_to_yolo_id = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}

        # Process images
        for img_info in coco_data['images']:
            img_id = img_info['id']
            img_filename = img_info['file_name']

            # Copy image
            src_img = images_dir / img_filename
            dst_img = self.output_root / split / 'images' / img_filename

            if src_img.exists():
                shutil.copy(src_img, dst_img)

            # Get annotations for this image
            anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

            if len(anns) == 0:
                # Create empty label file
                label_file = self.output_root / split / 'labels' / (Path(img_filename).stem + '.txt')
                label_file.touch()
                continue

            # Convert annotations to YOLO format
            label_lines = []
            for ann in anns:
                yolo_id = cat_id_to_yolo_id[ann['category_id']]

                # Get segmentation mask
                if 'segmentation' in ann and len(ann['segmentation']) > 0:
                    seg = ann['segmentation'][0]  # Get first polygon

                    # Normalize coordinates
                    img_h = img_info['height']
                    img_w = img_info['width']

                    normalized_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / img_w
                        y = seg[i+1] / img_h
                        normalized_seg.extend([x, y])

                    # Create label line: class_id x1 y1 x2 y2 ... (normalized coordinates)
                    label_line = str(yolo_id) + ' ' + ' '.join(f'{coord:.6f}' for coord in normalized_seg)
                    label_lines.append(label_line)

            # Write label file
            if label_lines:
                label_file = self.output_root / split / 'labels' / (Path(img_filename).stem + '.txt')
                with open(label_file, 'w') as f:
                    f.write('\n'.join(label_lines))

        print(f"  ✓ {split} split: {len(coco_data['images'])} images processed")

    def _create_yaml(self):
        """Create data.yaml for YOLO."""
        # Load category names from train annotations
        ann_file = self.arcade_root / self.task / 'train' / 'annotations' / 'train.json'
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        num_classes = len(coco_data['categories'])

        # Sort categories by ID to match the mapping in _convert_split
        # YOLO expects class indices 0 to num_classes-1
        sorted_categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        class_names = {idx: cat['name'] for idx, cat in enumerate(sorted_categories)}

        yaml_content = f"""
# ARCADE {self.task} dataset for YOLOv8 segmentation
path: {self.output_root}
train: train/images
val: val/images

nc: {num_classes}
names:
"""

        for class_id in sorted(class_names.keys()):
            yaml_content += f"  {class_id}: {class_names[class_id]}\n"

        yaml_file = self.output_root / 'data.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)

        print(f"\n✓ Created {yaml_file}")


if __name__ == '__main__':
    # Example usage
    converter = ArcadeToYOLO(
        arcade_root='/path/to/ARCADE',
        output_root='/path/to/yolo_dataset',
        task='stenosis'
    )
    converter.convert()
