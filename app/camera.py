"""Camera input as BGR frames, straight from a GStreamer appsink.

GStreamer is driven through PyGObject rather than cv2.VideoCapture: the OpenCV
that t3-gem-o1-edgeai installs from PyPI is built without GStreamer.

For the IMX219 the Bayer frames go through the VPAC ISP (tiovxisp), the
hardware scaler and the hardware colour converter, so the A53 only receives
the final 1280x720 RGB picture.
"""

from threading import Lock, Thread

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

SINK = "appsink name=sink drop=true max-buffers=1 sync=false"


def pipeline(cfg):
    src, w, h = cfg["source"], cfg["width"], cfg["height"]
    if src == "imx219":
        return (
            f"v4l2src device={cfg['device']} ! "
            "video/x-bayer,width=1920,height=1080,format=rggb ! "
            f"tiovxisp sink_0::device={cfg['subdev']} sensor-name=SENSOR_SONY_IMX219_RPI "
            f"dcc-isp-file={cfg['dcc_viss']} sink_0::dcc-2a-file={cfg['dcc_2a']} format-msb=7 ! "
            f"tiovxmultiscaler ! video/x-raw,format=NV12,width={w},height={h} ! "
            f"tiovxdlcolorconvert ! video/x-raw,format=RGB ! {SINK}"
        )
    if src.startswith("usb:"):
        return (
            f"v4l2src device={src[4:]} ! videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={w},height={h} ! {SINK}"
        )
    if src.startswith("file:"):
        return (
            f"filesrc location={src[5:]} ! decodebin ! videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={w},height={h} ! "
            "appsink name=sink max-buffers=2 sync=true"
        )
    raise ValueError(f"bilinmeyen kamera kaynağı: {src}")


class Camera:
    """Reads in a thread and keeps only the newest frame."""

    def __init__(self, cfg):
        Gst.init(None)
        p = pipeline(cfg)
        self.pipe = Gst.parse_launch(p)
        self.sink = self.pipe.get_by_name("sink")
        self.pipe.set_state(Gst.State.PLAYING)
        ok, _, _ = self.pipe.get_state(10 * Gst.SECOND)
        if ok == Gst.StateChangeReturn.FAILURE:
            msg = self.pipe.get_bus().pop_filtered(Gst.MessageType.ERROR)
            err = msg.parse_error()[0].message if msg else "bilinmiyor"
            self.pipe.set_state(Gst.State.NULL)
            raise RuntimeError(f"GStreamer hattı açılamadı ({err}). Kamera bağlı mı, "
                               "setup_cameras.sh çalıştırıldı mı?\n  " + p)
        self._frame = None
        self._seq = 0
        self._lock = Lock()
        self.running = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self.running:
            sample = self.sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                if self.sink.is_eos():
                    self.running = False
                continue
            s = sample.get_caps().get_structure(0)
            w, h = s.get_value("width"), s.get_value("height")
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                frame = np.frombuffer(info.data, np.uint8, count=w * h * 3).reshape(h, w, 3)
                if s.get_value("format") == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = frame.copy()
            finally:
                buf.unmap(info)
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, last_seq=0):
        """Returns (seq, frame); frame is None until a newer one arrives."""
        with self._lock:
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._frame

    def close(self):
        # Stop the reader before tearing the pipeline down: pulling from an
        # appsink that is going to NULL segfaults now and then.
        self.running = False
        self._thread.join(timeout=3)
        self.pipe.set_state(Gst.State.NULL)
