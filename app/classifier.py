"""Waste classifier: ONNX model on the C7x through TIDL, CPU as fallback.

The model (training/train.py) takes float32 RGB 0..255, NCHW, and does its own
normalisation, so preprocessing here is only resize + transpose.
"""

import os

import cv2
import numpy as np


class WasteClassifier:
    def __init__(self, model_dir, use_tidl=True, input_size=224):
        import onnxruntime as ort

        self.size = input_size
        with open(os.path.join(model_dir, "labels.txt")) as f:
            self.labels = [l.strip() for l in f if l.strip()]

        model = os.path.join(model_dir, "model.onnx")
        artifacts = os.path.join(model_dir, "artifacts")
        so = ort.SessionOptions()
        if use_tidl and "TIDLExecutionProvider" in ort.get_available_providers() \
                and os.path.isdir(artifacts):
            # Same options as TI's edgeai_dl_inferer; TIDL wants ORT's own
            # graph rewrites off so the graph matches the compiled artifacts.
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opts = {"platform": "J7", "version": "8.2", "artifacts_folder": artifacts}
            self.session = ort.InferenceSession(
                model, sess_options=so,
                providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
                provider_options=[opts, {}])
            self.backend = "TIDL/C7x"
        else:
            if use_tidl:
                print("[UYARI] TIDL yok ya da artifacts/ eksik, CPU'da çalışılıyor (yavaş)")
            self.session = ort.InferenceSession(model, sess_options=so,
                                                providers=["CPUExecutionProvider"])
            self.backend = "CPU"
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, roi_bgr):
        """Returns (label, confidence, probabilities)."""
        # INTER_AREA: the ROI is ~3x larger than the input, and the default
        # bilinear aliases; on the test split it cost 4 points of accuracy.
        small = cv2.resize(roi_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input_name: x})[0][0].astype(np.float64)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        return self.labels[i], float(p[i]), p
