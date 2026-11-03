"""Finds the item held up to the camera: TI's COCO SSDLite-MobDet on the C7x.

The waste classifier then looks only at that box, so the background no longer
matters. The detector also knows common food (orange, banana, apple ...),
which is a reliable "organik" on its own.
"""

import glob
import os

import cv2
import numpy as np

# TF object detection COCO label map, ids 1..90 -> index id-1 ("" = unused id).
COCO90 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack",
    "umbrella", "", "", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "", "dining table", "", "", "toilet", "",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# Never the item: the person holding it, and the room around it (a ceiling
# air conditioner grille came back as "microwave" and "keyboard", the ceiling
# lamp as "cat", the holder's hair and face as "cat"/"dog"). Nobody holds up
# an animal to be sorted, so all of them go.
IGNORE = {"", "person", "chair", "couch", "bed", "dining table", "toilet", "tv",
          "refrigerator", "oven", "sink", "bench", "potted plant", "microwave",
          "toaster", "clock", "laptop", "keyboard",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
          "giraffe"}
FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake"}

DEFAULT_MODEL = "/opt/model_zoo/TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320"


class ObjectFinder:
    def __init__(self, model_dir=DEFAULT_MODEL, threshold=0.35, max_area=0.7, use_tidl=True,
                 fixture_frames=135):
        from tflite_runtime.interpreter import Interpreter, load_delegate

        self.threshold = threshold
        self.max_area = max_area
        # A box that sits still this many frames (~5 s) is part of the room:
        # a held item always jitters. Remembered until restart.
        self.fixture_frames = fixture_frames
        self.fixtures, self.still = [], (None, 0)
        model = glob.glob(os.path.join(model_dir, "model", "*.tflite"))[0]
        delegates = []
        if use_tidl:
            delegates = [load_delegate("/usr/lib/libtidl_tfl_delegate.so", {
                "artifacts_folder": os.path.join(model_dir, "artifacts"),
                "platform": "J7", "version": "7.2", "debug_level": 0})]
        self.it = Interpreter(model_path=model, experimental_delegates=delegates)
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        self.input_index = inp["index"]
        self.size = (int(inp["shape"][2]), int(inp["shape"][1]))
        self.outputs = [o["index"] for o in self.it.get_output_details()]

    def detect(self, frame_bgr):
        """All detections as (name, score, (x0, y0, x1, y1)) in pixels."""
        h, w = frame_bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB)[None]
        self.it.set_tensor(self.input_index, x)
        self.it.invoke()
        boxes, classes, scores, _ = (self.it.get_tensor(i) for i in self.outputs)
        out = []
        for b, c, s in zip(boxes[0], classes[0], scores[0]):
            if s < self.threshold:
                continue
            c = int(c)
            name = COCO90[c] if 0 <= c < len(COCO90) else ""
            y0, x0, y1, x1 = np.clip(b, 0, 1)
            out.append((name, float(s), (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))))
        return out

    def pick(self, dets, w, h):
        """The held item: confident, reasonably big, near the centre."""
        best, best_score = None, 0.0
        for name, s, (x0, y0, x1, y1) in dets:
            area = (x1 - x0) * (y1 - y0) / float(w * h)
            if name in IGNORE or area > self.max_area or area < 0.005:
                continue
            # Cut off by two or more frame edges: part of the room, not a held item.
            mx, my = 0.02 * w, 0.02 * h
            edges = (x0 < mx) + (y0 < my) + (x1 > w - mx) + (y1 > h - my)
            if edges >= 2:
                continue
            if any(iou((x0, y0, x1, y1), f) > 0.7 for f in self.fixtures):
                continue
            cx, cy = (x0 + x1) / 2 / w - 0.5, (y0 + y1) / 2 / h - 0.5
            score = s * np.sqrt(area) * (1.0 - min(1.0, np.hypot(cx, cy)))
            if score > best_score:
                best, best_score = (name, s, (x0, y0, x1, y1)), score
        if best:
            box, n = self.still
            n = n + 1 if box and iou(best[2], box) > 0.9 else 1
            self.still = (best[2] if n == 1 else box, n)
            if n >= self.fixture_frames:
                self.fixtures.append(box)
                self.still = (None, 0)
                return None
        return best


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop(frame, box, margin=0.15):
    """Square crop around the box with some context, clipped to the frame."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    a1, b1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return frame[b0:b1, a0:a1]
