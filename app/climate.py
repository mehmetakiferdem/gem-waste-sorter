"""Counts sorted items and estimates the emissions they avoided.

Each decision is appended to a CSV so the totals survive a restart and can be
analysed later (spreadsheet, pandas).
"""

import csv
import os
import threading
import time

FIELDS = ["zaman", "sinif", "guven", "uyum", "kutle_kg", "co2_kg"]


class ClimateLedger:
    def __init__(self, bins, csv_path=None):
        self.bins = bins
        self.csv_path = csv_path
        self.counts = {k: 0 for k in bins}
        self.counts["emin_degil"] = 0
        self.mass_kg = 0.0
        self.co2_kg = 0.0
        self.recent = []            # newest first, for the dashboard
        self._lock = threading.Lock()
        if csv_path:
            self._replay()

    def estimate(self, label):
        b = self.bins.get(label)
        if not b:
            return 0.0, 0.0
        mass = float(b.get("mass_kg", 0.0))
        return mass, mass * float(b.get("co2_kg_per_kg", 0.0))

    def record(self, label, confidence=0.0, agreement=0.0, when=None):
        when = time.time() if when is None else when
        mass, co2 = self.estimate(label)
        row = {"zaman": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when)),
               "sinif": label, "guven": round(confidence, 3),
               "uyum": round(agreement, 3), "kutle_kg": round(mass, 4),
               "co2_kg": round(co2, 4)}
        with self._lock:
            self._add(row)
            if self.csv_path:
                self._append(row)
        return row

    def snapshot(self):
        with self._lock:
            total = sum(self.counts.values())
            sorted_ok = total - self.counts["emin_degil"] - self.counts.get("diger", 0)
            return {
                "sayilar": dict(self.counts),
                "toplam": total,
                "geri_kazanilan": sorted_ok,
                "kutle_kg": round(self.mass_kg, 3),
                "co2_kg": round(self.co2_kg, 3),
                "son": list(self.recent),
            }

    def _add(self, row):
        label = row["sinif"]
        self.counts[label] = self.counts.get(label, 0) + 1
        self.mass_kg += float(row["kutle_kg"])
        self.co2_kg += float(row["co2_kg"])
        self.recent.insert(0, row)
        del self.recent[20:]

    def _append(self, row):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def _replay(self):
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            self._add(row)
