#!/usr/bin/env python3
"""Waste sorter: camera -> presence -> classifier -> vote -> bin + CO2 ledger."""

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
from classifier import WasteClassifier              # noqa: E402
from climate import ClimateLedger                   # noqa: E402
from decision import UNSURE, DecisionMaker          # noqa: E402
from presence import PresenceDetector               # noqa: E402
from sorter import Sorter                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def roi_box(frame, roi):
    h, w = frame.shape[:2]
    return int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)


def draw(frame, box, state, label, conf, last, bins, fps, backend, probs=None, labels=()):
    x0, y0, x1, y1 = box
    col = tuple(bins[label]["color"]) if label in bins else (200, 200, 200)
    cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
    # cv2 fonts have no Turkish letters, so the overlay uses the ASCII keys.
    if label:
        cv2.putText(frame, f"{label} {conf * 100:.0f}%", (x0 + 8, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    if probs is not None:
        for row, i in enumerate(probs.argsort()[::-1][:3]):
            cv2.putText(frame, f"{labels[i]:8s}{probs[i] * 100:3.0f}%", (x1 + 10, y0 + 28 + 28 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{state}  {fps:4.1f} fps  {backend}", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if last:
        cv2.rectangle(frame, (10, 10), (330, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"KARAR: {last}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    tuple(bins[last]["color"]) if last in bins else (0, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=os.path.join(ROOT, "configs/atik.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.chdir(ROOT)

    bins = cfg["bins"]
    clf = WasteClassifier(cfg["model"]["dir"], cfg["model"]["use_tidl"], cfg["model"]["input_size"])
    unknown = set(clf.labels) - set(bins)
    if unknown:
        sys.exit(f"labels.txt içindeki {sorted(unknown)} configs 'bins' altında yok")
    presence = PresenceDetector(**cfg["presence"])
    decider = DecisionMaker(**cfg["decision"])
    ledger = ClimateLedger(bins, cfg.get("log_csv"))
    sorter = Sorter(cfg.get("sorter", {}), bins)
    cam = Camera(cfg["camera"])
    if cfg.get("web_port"):
        web.start(cfg["web_port"], ledger.snapshot)

    stop = []
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))

    print(f"[BASLA] model={clf.backend} sınıflar={clf.labels}. Tepsi BOŞ olmalı, "
          f"{presence.warmup_frames} kare arka plan öğreniliyor.")
    seq, last, last_t, fps = 0, None, time.time(), 0.0
    try:
        while not stop and cam.running:
            seq, frame = cam.read(seq)
            if frame is None:
                time.sleep(0.003)
                continue
            frame = frame.copy()
            box = roi_box(frame, cfg["roi"])
            roi = frame[box[1]:box[3], box[0]:box[2]]

            present = presence.update(roi)
            label, conf, probs = None, 0.0, None
            if present:
                label, conf, probs = clf.classify(roi)
            d = decider.update(present, label, conf)
            if decider.needs_rebase:
                decider.needs_rebase = False
                presence.reset()
                print("[UYARI] sahne boşalmadı, boş sahne yeniden öğreniliyor")
            if d:
                ledger.record(d.label, d.confidence, d.agreement)
                if d.label != UNSURE:
                    sorter.route(d.label)
                last = d.label
                print(f"[KARAR] {d.label} güven={d.confidence:.2f} uyum={d.agreement:.2f}")
            elif decider.state == "BOS":
                last = None

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now
            state = decider.state if presence.ready else "ARKA PLAN"
            draw(frame, box, state, label, conf, last, bins, fps, clf.backend, probs, clf.labels)
            ledger.live = {"durum": state, "nesne": present,
                           "olasilik": {} if probs is None else
                           {l: round(float(p), 3) for l, p in zip(clf.labels, probs)}}
            web.publish(frame)
    finally:
        cam.close()
        sorter.close()
        print("[BITTI]")


if __name__ == "__main__":
    main()
