"""
Training entry point for UCNet on the ARCADE dataset.

Reproduces the paper's setup (Sec. 4.3): Adam, lr 2e-4, batch size 4,
400 epochs, random-rotation augmentation, single-GPU. The cGAN loop
alternates a discriminator step and a generator step; the generator is
optimised with the combined UCNet loss (Eq. 9) plus the adversarial term.

Usage:
    python train.py \
        --train-images /path/ARCADE/train/images \
        --train-ann    /path/ARCADE/train/annotations/train.json \
        --val-images   /path/ARCADE/val/images \
        --val-ann      /path/ARCADE/val/annotations/val.json \
        --epochs 400 --batch-size 4 --lr 2e-4
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.arcade_dataset import ArcadeSegmentDataset
from models.generator import ImprovedUNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import UCNetLoss
from utils.metrics import SegmentationMeter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--eval-every", type=int, default=10)
    # loss weights (Eq. 9)
    p.add_argument("--alpha", type=float, default=1.0)   # segment
    p.add_argument("--beta", type=float, default=1.0)    # focal
    p.add_argument("--lam", type=float, default=10.0)    # L1
    p.add_argument("--gamma", type=float, default=1.0)   # dice
    return p.parse_args()


def build_class_weights(dataset):
    """Inverse-frequency class weights to fight imbalance (Table 1)."""
    counts = torch.zeros(dataset.num_classes)
    for _, target in dataset:
        vals, c = torch.unique(target, return_counts=True)
        counts[vals] += c
    counts = counts.clamp(min=1)
    w = 1.0 / counts
    w[0] = w[0] * 0.1  # de-emphasise background
    return (w / w.sum()).tolist()


def evaluate(generator, loader, num_classes, class_names, device):
    generator.eval()
    meter = SegmentationMeter(num_classes, class_names=class_names)
    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            logits = generator(x)
            meter.update(logits, target)
    return meter.compute()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ArcadeSegmentDataset(args.train_images, args.train_ann,
                                    img_size=args.img_size, augment=True)
    val_ds = ArcadeSegmentDataset(args.val_images, args.val_ann,
                                  img_size=args.img_size, augment=False,
                                  category_map=train_ds.cat2class)
    num_classes = train_ds.num_classes
    print(f"Classes (incl background): {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers)

    print("Computing class weights ...")
    class_weights = build_class_weights(train_ds)

    G = ImprovedUNetGenerator(in_channels=3, out_channels=num_classes).to(device)
    D = PatchGANDiscriminator(in_channels=3, seg_channels=num_classes).to(device)

    criterion = UCNetLoss(alpha=args.alpha, beta=args.beta, lam=args.lam,
                          gamma=args.gamma, class_weights=class_weights).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def to_onehot(target):
        return torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        running = {}
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)

            # ---- Discriminator step ----
            with torch.no_grad():
                fake_logits = G(x)
                fake_seg = torch.softmax(fake_logits, dim=1)
            real_seg = to_onehot(target)
            d_real = D(x, real_seg)
            d_fake = D(x, fake_seg.detach())
            loss_d = criterion.discriminator_loss(d_real, d_fake)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- Generator step ----
            logits = G(x)
            d_fake_for_g = D(x, torch.softmax(logits, dim=1))
            loss_g, parts = criterion.generator_loss(logits, target, d_fake_for_g)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            running["d"] = running.get("d", 0.0) + loss_d.item()

        n = len(train_loader)
        msg = " ".join(f"{k}={v / n:.4f}" for k, v in running.items())
        print(f"[epoch {epoch:3d}/{args.epochs}] {msg}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            summary, _ = evaluate(G, val_loader, num_classes,
                                  val_ds.class_names, device)
            print(f"  >> val F1={summary['f1']:.2f} IoU={summary['iou']:.2f} "
                  f"Acc={summary['accuracy']:.2f} Sen={summary['sensitivity']:.2f} "
                  f"Spe={summary['specificity']:.2f} Prec={summary['precision']:.2f}")
            if summary["f1"] > best_f1:
                best_f1 = summary["f1"]
                torch.save({"G": G.state_dict(), "D": D.state_dict(),
                            "cat2class": train_ds.cat2class,
                            "epoch": epoch, "f1": best_f1},
                           os.path.join(args.out, "ucnet_best.pth"))
                print(f"  >> saved new best (F1={best_f1:.2f})")

    print(f"Done. Best val F1 = {best_f1:.2f}")


if __name__ == "__main__":
    main()
