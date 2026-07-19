"""
Inference + visualisation for a trained UCNet.

Loads a checkpoint, runs the generator on an ARCADE split, and writes a
colour-coded overlay per image (one colour per coronary segment, matching
the spirit of Table 1's visualisation colours).

Usage:
    python inference.py --checkpoint checkpoints/ucnet_best.pth \
        --images /path/ARCADE/val/images \
        --ann    /path/ARCADE/val/annotations/val.json \
        --out    predictions
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from data.arcade_dataset import ArcadeSegmentDataset
from models.generator import ImprovedUNetGenerator
from utils.metrics import SegmentationMeter


def color_palette(n):
    rng = np.random.RandomState(42)
    pal = rng.randint(40, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)  # background
    return pal


def colorize(mask, palette):
    return palette[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--img-size", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    ds = ArcadeSegmentDataset(args.images, args.ann, img_size=args.img_size,
                              category_map=ckpt.get("cat2class"))
    num_classes = ds.num_classes

    G = ImprovedUNetGenerator(in_channels=3, out_channels=num_classes).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    palette = color_palette(num_classes)
    meter = SegmentationMeter(num_classes, class_names=ds.class_names)

    with torch.no_grad():
        for i in range(len(ds)):
            x, target = ds[i]
            logits = G(x.unsqueeze(0).to(device))
            meter.update(logits, target.unsqueeze(0))
            pred = logits.argmax(1)[0].cpu().numpy()

            base = (x[0].numpy() * 255).astype(np.uint8)
            base_rgb = np.stack([base] * 3, axis=-1)
            pred_rgb = colorize(pred, palette)
            blend = (0.5 * base_rgb + 0.5 * pred_rgb).astype(np.uint8)
            Image.fromarray(blend).save(os.path.join(args.out, f"pred_{i:04d}.png"))

    summary, per_class = meter.compute()
    print("Overall:", {k: round(v, 2) for k, v in summary.items()})
    print("\nPer-segment F1:")
    for name, m in sorted(per_class.items(), key=lambda kv: -kv[1]["f1"]):
        print(f"  {name:32s} F1={m['f1']:6.2f}  IoU={m['iou']:6.2f}")


if __name__ == "__main__":
    main()
