#!/usr/bin/env python3
"""Scores one or more models on <dir>/<bin>/*.jpg, on the board (TIDL) or on
the PC (ONNX on CPU), with the same code path the app uses:

    python3 training/evaluate.py --data training/kendi_test models/waste_mnv2_v5 models/waste_mnv2_v7
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))


def load(path):
    try:
        import cv2
        return cv2.imread(path)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cpu", action="store_true", help="TIDL yerine CPU")
    args = ap.parse_args()

    from classifier import WasteClassifier

    files = sorted(glob.glob(os.path.join(args.data, "*", "*.jpg")))
    truth = [f.split(os.sep)[-2] for f in files]
    images = [load(f) for f in files]
    for m in args.models:
        clf = WasteClassifier(m, not args.cpu, 224)
        times, pred = [], []
        for im in images:
            t = time.perf_counter()
            pred.append(clf.classify(im)[0])
            times.append(time.perf_counter() - t)
        ok = sum(p == t for p, t in zip(pred, truth))
        print(f"\n{m} [{clf.backend}]: {ok}/{len(files)} = %{100 * ok / len(files):.1f}"
              f"  ort {1000 * np.mean(times[1:]):.1f} ms")
        labels = clf.labels
        print("gerçek \\ tahmin " + " ".join(f"{l[:7]:>7s}" for l in labels) + "   doğru")
        recalls = []
        for t in labels:
            row = [sum(1 for p, g in zip(pred, truth) if g == t and p == l) for l in labels]
            n = sum(row)
            if n:
                recalls.append(row[labels.index(t)] / n)
                print(f"{t:16s} " + " ".join(f"{c:7d}" for c in row) + f"   %{100 * recalls[-1]:.0f}")
        # The camera set is mostly glass; the per-class mean is not swayed by that.
        print(f"sınıf ortalaması %{100 * np.mean(recalls):.1f}")


if __name__ == "__main__":
    main()
