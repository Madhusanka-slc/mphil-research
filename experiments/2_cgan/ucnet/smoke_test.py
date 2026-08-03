"""Quick end-to-end smoke test on tiny synthetic data (no real dataset needed)."""
import json
import os
import tempfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.arcade_dataset import ArcadeSegmentDataset
from models.generator import ImprovedUNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import UCNetLoss
from utils.metrics import SegmentationMeter


def make_fake_arcade(root, n_images=4, n_classes=5, size=512):
    img_dir = os.path.join(root, "images"); os.makedirs(img_dir, exist_ok=True)
    images, annotations = [], []
    cats = [{"id": c, "name": f"segment_{c}"} for c in range(1, n_classes + 1)]
    ann_id = 1
    for i in range(n_images):
        fn = f"img_{i}.png"
        Image.fromarray(np.random.randint(0, 255, (size, size), np.uint8)).save(os.path.join(img_dir, fn))
        images.append({"id": i, "file_name": fn, "width": size, "height": size})
        for c in range(1, n_classes + 1):
            cx, cy = (int(v) for v in np.random.randint(20, size - 20, 2))
            poly = [cx - 10, cy - 10, cx + 10, cy - 10, cx + 10, cy + 10, cx - 10, cy + 10]
            annotations.append({"id": ann_id, "image_id": i, "category_id": c,
                                "segmentation": [poly], "area": 400.0, "bbox": [cx-10, cy-10, 20, 20]})
            ann_id += 1
    ann = {"images": images, "annotations": annotations, "categories": cats}
    ann_path = os.path.join(root, "ann.json")
    with open(ann_path, "w") as f:
        json.dump(ann, f)
    return img_dir, ann_path


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, ann = make_fake_arcade(tmp, n_images=4, n_classes=5, size=512)
        ds = ArcadeSegmentDataset(img_dir, ann, img_size=512, augment=True)
        print(f"dataset len={len(ds)} num_classes={ds.num_classes} names={ds.class_names}")
        loader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)

        nc = ds.num_classes
        G = ImprovedUNetGenerator(3, nc, base_filters=16).to(device)
        D = PatchGANDiscriminator(3, nc, base_filters=16).to(device)
        crit = UCNetLoss(class_weights=[1.0] * nc).to(device)
        opt_g = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))

        gp = sum(p.numel() for p in G.parameters())
        dp = sum(p.numel() for p in D.parameters())
        print(f"generator params={gp:,}  discriminator params={dp:,}")

        for step, (x, target) in enumerate(loader):
            x, target = x.to(device), target.to(device)
            print(f"  batch x={tuple(x.shape)} target={tuple(target.shape)} "
                  f"classes_present={target.unique().tolist()}")

            real_1h = torch.nn.functional.one_hot(target, nc).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                fake = torch.softmax(G(x), 1)
            d_real, d_fake = D(x, real_1h), D(x, fake.detach())
            print(f"  disc out shape={tuple(d_real.shape)}")
            loss_d = crit.discriminator_loss(d_real, d_fake)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            logits = G(x)
            d_fake_g = D(x, torch.softmax(logits, 1))
            loss_g, parts = crit.generator_loss(logits, target, d_fake_g)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            parts_str = {k: round(v, 4) for k, v in parts.items()}
            print(f"  loss_d={loss_d.item():.4f}  loss_g parts={parts_str}")

            # exact segment-loss metric (Eq. 1)
            hard = crit.seg.hard_ratio(logits, target)
            print(f"  hard segment M/N={hard:.4f}")
            break

        # metrics sanity
        meter = SegmentationMeter(nc, class_names=ds.class_names)
        G.eval()
        with torch.no_grad():
            for x, target in loader:
                meter.update(G(x.to(device)), target.to(device))
        summary, per_class = meter.compute()
        print("metrics summary:", {k: round(v, 2) for k, v in summary.items()})
        print("PASS: full pipeline (data -> G/D -> losses -> backward -> metrics) works.")


if __name__ == "__main__":
    run()
