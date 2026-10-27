#!/usr/bin/env python3
"""Collects training photos of one bin from this camera, on this tray.

    sudo python3 app/collect.py plastik          # Ctrl+C to stop
    sudo python3 app/collect.py organik --every 0.5
    sudo python3 app/collect.py kagit --merkez   # item the detector can't find

Saves a crop every `--every` seconds to data/kendi/<bin>/ and shows the
picture on the dashboard port. In mode "elde" the crop is the detector's box
around the held item (exactly what the classifier sees live); in mode "tepsi"
it is the ROI. Turn and move the item between shots, and use
several different items per bin. Then on the PC:

    scp -r gemstone@192.168.7.2:~/gem-waste-sorter/data/kendi training/
    python3 prepare_dataset.py --src <RealWaste> --src kendi --out dataset

Stop the sorter first (sudo pkill -INT -f app/main.py): one camera, one user.
"""

import argparse
import os
import signal
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import web                                          # noqa: E402
from camera import Camera                           # noqa: E402
from detector import DEFAULT_MODEL, ObjectFinder, crop  # noqa: E402
from main import ROOT, roi_box                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/atik.yaml"))
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--merkez", action="store_true",
                    help="save the centre of the frame instead of the detector's box "
                         "(crumpled paper, bags, cans: things COCO doesn't know)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.bin not in cfg["bins"]:
        sys.exit(f"bilinmeyen kutu {args.bin!r}; seçenekler: {', '.join(cfg['bins'])}")

    out = os.path.join(ROOT, "data", "kendi", args.bin)
    os.makedirs(out, exist_ok=True)
    count = len(os.listdir(out))
    shots = {"n": count}
    finder = None
    if cfg.get("mode") == "elde" and not args.merkez:
        el = cfg.get("elde", {})
        finder = ObjectFinder(el.get("detector", DEFAULT_MODEL), el.get("threshold", 0.35),
                              el.get("max_area", 0.7), cfg["model"]["use_tidl"])
    cam = Camera(cfg["camera"])
    if cfg.get("web_port"):
        web.start(cfg["web_port"], lambda: {"toplanan": shots["n"], "kutu": args.bin})
    stop = []
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    print(f"[TOPLA] {args.bin} -> {out} ({count} fotoğraf var). Ctrl+C ile bitir.")

    seq, last = 0, 0.0
    try:
        while not stop and cam.running:
            seq, frame = cam.read(seq)
            if frame is None:
                time.sleep(0.005)
                continue
            if finder:
                h, w = frame.shape[:2]
                held = finder.pick(finder.detect(frame), w, h)
                box = held[2] if held else None
            elif args.merkez:
                h, w = frame.shape[:2]
                box = (w // 2 - h // 3, h // 6, w // 2 + h // 3, h * 5 // 6)
            else:
                box = roi_box(frame, cfg["roi"])
            now = time.time()
            if box and now - last >= args.every:
                last = now
                roi = crop(frame, box) if finder else frame[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(os.path.join(out, f"{args.bin}_{int(now * 1000)}.jpg"), roi,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                shots["n"] += 1
            view = frame.copy()
            if box:
                cv2.rectangle(view, box[:2], box[2:], (0, 255, 0), 2)
            cv2.putText(view, f"TOPLA {args.bin}: {shots['n']}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            web.publish(view)
    finally:
        cam.close()
        print(f"[BITTI] {args.bin}: {shots['n']} fotoğraf")


if __name__ == "__main__":
    main()
