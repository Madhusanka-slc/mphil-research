"""
Evaluation metrics for UCNet (Sec. 4.2).

Computes, per class and averaged: accuracy, sensitivity (recall),
specificity, precision (PPV), IoU, F1 -- exactly the six metrics reported in
the paper (Eqs. 10-15). Confusion counts are accumulated over the whole test
set before the ratios are computed, so rare classes are handled correctly.
"""

import numpy as np
import torch


class SegmentationMeter:
    def __init__(self, num_classes, ignore_background=True, class_names=None):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.class_names = class_names or {}
        self.reset()

    def reset(self):
        n = self.num_classes
        self.tp = np.zeros(n, dtype=np.int64)
        self.fp = np.zeros(n, dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)
        self.tn = np.zeros(n, dtype=np.int64)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(dim=1).cpu().numpy().ravel()
        gt = target.cpu().numpy().ravel()
        for c in range(self.num_classes):
            p, g = pred == c, gt == c
            self.tp[c] += np.sum(p & g)
            self.fp[c] += np.sum(p & ~g)
            self.fn[c] += np.sum(~p & g)
            self.tn[c] += np.sum(~p & ~g)

    def _safe(self, num, den):
        return np.where(den > 0, num / np.maximum(den, 1), 0.0)

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        acc = self._safe(tp + tn, tp + tn + fp + fn)
        prec = self._safe(tp, tp + fp)
        sens = self._safe(tp, tp + fn)
        spec = self._safe(tn, tn + fp)
        iou = self._safe(tp, tp + fp + fn)
        f1 = self._safe(2 * tp, 2 * tp + fp + fn)

        classes = range(1, self.num_classes) if self.ignore_background else range(self.num_classes)
        present = [c for c in classes if (tp[c] + fn[c]) > 0]

        per_class = {}
        for c in present:
            per_class[self.class_names.get(c, f"class_{c}")] = {
                "accuracy": acc[c] * 100, "sensitivity": sens[c] * 100,
                "specificity": spec[c] * 100, "precision": prec[c] * 100,
                "f1": f1[c] * 100, "iou": iou[c] * 100,
            }

        def avg(arr):
            return float(np.mean([arr[c] for c in present]) * 100) if present else 0.0

        summary = {
            "accuracy": avg(acc), "sensitivity": avg(sens),
            "specificity": avg(spec), "precision": avg(prec),
            "f1": avg(f1), "iou": avg(iou),
        }
        return summary, per_class
