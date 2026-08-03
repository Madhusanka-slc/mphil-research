"""
ARCADE dataset loader for UCNet.

The ARCADE challenge (MICCAI 2023) ships COCO-style annotations: each image
has polygon segmentations, and each polygon carries a category id naming the
coronary segment. This loader:

  1. reads the COCO json,
  2. rasterises the polygons into a (H, W) class-index mask
     (0 = background, 1..num_classes = segments),
  3. builds the model INPUT as the original XCA image overlaid on the binary
     vessel mask (the paper uses a binary vessel segmentation as the cGAN
     conditioning variable, with the original image superimposed).

Directory layout expected (ARCADE "syntax"/segmentation task):

    root/
      train/
        images/*.png
        annotations/train.json
      val/
        images/*.png
        annotations/val.json

Pass the split's image dir and json path directly if your layout differs.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None


def _get_annotation_area(ann):
    """Get annotation area for sorting."""
    return ann.get("area", 0)


class ArcadeSegmentDataset(Dataset):
    def __init__(self, image_dir, ann_file, img_size=512,
                 category_map=None, augment=False):
        assert Image is not None, "Pillow is required: pip install pillow"
        self.image_dir = image_dir
        self.img_size = img_size
        self.augment = augment

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {im["id"]: im for im in coco["images"]}
        self.anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_img[ann["image_id"]].append(ann)

        # category id -> contiguous class index (1..K); 0 stays background
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        if category_map is None:
            self.cat2class = {c["id"]: i + 1 for i, c in enumerate(cats)}
        else:
            self.cat2class = category_map
        self.class_names = {0: "background"}
        for c in cats:
            if c["id"] in self.cat2class:
                self.class_names[self.cat2class[c["id"]]] = c["name"]
        self.num_classes = max(self.cat2class.values()) + 1  # incl background

        self.ids = [i for i in self.images if i in self.anns_by_img]

    def __len__(self):
        return len(self.ids)

    def _rasterise(self, anns, w, h):
        """Polygons -> class mask and binary vessel mask. Larger segments drawn first."""
        mask = Image.new("I", (w, h), 0)
        binary = Image.new("L", (w, h), 0)
        dm, db = ImageDraw.Draw(mask), ImageDraw.Draw(binary)
        for ann in sorted(anns, key=_get_annotation_area, reverse=True):
            cls = self.cat2class.get(ann["category_id"])
            if cls is None:
                continue
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                dm.polygon(pts, fill=int(cls))
                db.polygon(pts, fill=255)
        return np.array(mask, dtype=np.int64), np.array(binary, dtype=np.uint8)

    def __getitem__(self, idx):
        info = self.images[self.ids[idx]]
        img_path = os.path.join(self.image_dir, info["file_name"])
        img = Image.open(img_path).convert("L")
        w0, h0 = img.size

        anns = self.anns_by_img[self.ids[idx]]
        cls_mask, binary = self._rasterise(anns, w0, h0)

        # resize to square target
        S = self.img_size
        img = img.resize((S, S), Image.BILINEAR)
        cls_img = Image.fromarray(cls_mask.astype(np.int32), mode="I").resize((S, S), Image.NEAREST)
        bin_img = Image.fromarray(binary).resize((S, S), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        binary = np.asarray(bin_img, dtype=np.float32) / 255.0
        target = np.asarray(cls_img, dtype=np.int64)

        if self.augment and np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k).copy()
            binary = np.rot90(binary, k).copy()
            target = np.rot90(target, k).copy()

        # INPUT = original image overlaid on binary vessel mask (condition).
        # 3 channels: [original, binary condition, original*binary overlay]
        overlay = img * binary
        x = np.stack([img, binary, overlay], axis=0)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(target).long(),
        )
