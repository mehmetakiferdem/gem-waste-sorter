#!/usr/bin/env python3
"""Merges ready-made waste datasets into our six bins, split train/val/test.

Any dataset laid out as <root>/<class name>/<image> works; the class name is
matched by keyword (RealWaste, TrashNet, Kaggle "Garbage Classification"):

    python3 prepare_dataset.py --src ~/indirilen/realwaste-main/RealWaste \
                               --src ~/indirilen/trashnet --out dataset

Your own photos taken with the IMX219 on the real tray help the most: put them
in kendi/<bin key>/ (plastik, organik, ...) and pass --src kendi.
"""

import argparse
import hashlib
import os
import random
import shutil

BINS = ["plastik", "organik", "kagit", "cam", "metal", "diger"]

# First keyword found in the lower-cased folder name wins; order matters
# ("paper" must not catch "cardboard" checks, "glass" before "trash").
KEYWORDS = [
    ("plastik", ["plastic", "plastik"]),
    ("organik", ["food", "organic", "organik", "vegetation", "biological"]),
    ("kagit", ["cardboard", "paper", "kagit", "karton"]),
    ("cam", ["glass", "cam"]),
    ("metal", ["metal"]),
    ("diger", ["trash", "misc", "textile", "clothes", "shoes", "diger"]),
]
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def bin_of(name):
    n = name.lower()
    if n in BINS:
        return n
    for key, words in KEYWORDS:
        if any(w in n for w in words):
            return key
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", action="append", default=[], metavar="DIR=N",
                    help="a --src whose train images are linked N times, e.g. kendi=4: "
                         "a few hundred own photos must not drown in RealWaste")
    args = ap.parse_args()

    repeat = {os.path.abspath(k): int(v) for k, v in (r.split("=") for r in args.repeat)}
    files = {b: [] for b in BINS}
    weight = {}
    for src in args.src:
        for d in sorted(os.listdir(src)):
            path = os.path.join(src, d)
            if not os.path.isdir(path):
                continue
            b = bin_of(d)
            if b is None:
                print(f"  atlandı: {path} (sınıf eşlenemedi)")
                continue
            imgs = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
                    if f.lower().endswith(EXT)]
            print(f"  {d:28s} -> {b:8s} {len(imgs):5d}")
            files[b] += imgs
            for f in imgs:
                weight[f] = repeat.get(os.path.abspath(src), 1)

    rnd = random.Random(args.seed)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for b, imgs in files.items():
        rnd.shuffle(imgs)
        n_val, n_test = int(len(imgs) * args.val), int(len(imgs) * args.test)
        splits = {"val": imgs[:n_val], "test": imgs[n_val:n_val + n_test],
                  "train": imgs[n_val + n_test:]}
        for split, lst in splits.items():
            d = os.path.join(args.out, split, b)
            os.makedirs(d, exist_ok=True)
            for f in lst:
                # Hash the source path so equal names from two datasets don't clash.
                h = hashlib.sha1(f.encode()).hexdigest()[:10]
                ext = os.path.splitext(f)[1].lower()
                for k in range(weight[f] if split == "train" else 1):
                    os.symlink(os.path.abspath(f), os.path.join(d, f"{h}_{k}{ext}"))
    print("\nkutu      train   val  test")
    for b in BINS:
        c = [len(os.listdir(os.path.join(args.out, s, b))) for s in ("train", "val", "test")]
        print(f"{b:8s} {c[0]:6d} {c[1]:5d} {c[2]:5d}")


if __name__ == "__main__":
    main()
