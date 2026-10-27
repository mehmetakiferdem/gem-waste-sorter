"""Is there something on the tray? Difference against an empty-tray model.

The classifier has no "empty" class, so without this it would happily call a
bare tray "kağıt". A running-average background is cheap on the A53 at this
size and good enough under steady indoor light.
"""

import cv2
import numpy as np

SIZE = (160, 120)


class PresenceDetector:
    def __init__(self, diff_threshold=22, area_fraction=0.03, learn_rate=0.02,
                 warmup_frames=30):
        self.diff_threshold = diff_threshold
        self.area_fraction = area_fraction
        self.learn_rate = learn_rate
        self.warmup_frames = warmup_frames
        self.bg = None
        self.seen = 0
        self.changed = 0.0      # last changed fraction, for the overlay

    @staticmethod
    def _prep(roi_bgr):
        g = cv2.cvtColor(cv2.resize(roi_bgr, SIZE), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)

    @property
    def ready(self):
        return self.seen >= self.warmup_frames

    def update(self, roi_bgr):
        g = self._prep(roi_bgr)
        self.seen += 1
        if self.bg is None:
            self.bg = g
            return False
        if not self.ready:
            cv2.accumulateWeighted(g, self.bg, 0.2)
            return False
        diff = cv2.absdiff(g, self.bg)
        self.changed = float(np.count_nonzero(diff > self.diff_threshold)) / diff.size
        present = self.changed >= self.area_fraction
        if not present:
            # Follow slow light changes only while the tray is empty, or a
            # resting item would fade into the background.
            cv2.accumulateWeighted(g, self.bg, self.learn_rate)
        return present

    def reset(self):
        self.bg = None
        self.seen = 0
