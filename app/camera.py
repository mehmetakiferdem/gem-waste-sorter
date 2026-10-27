"""Camera input as BGR frames through OpenCV's GStreamer backend.

For the IMX219 the Bayer frames go through the VPAC ISP (tiovxisp) and the
hardware scaler, so the A53 only sees the final 1280x720 picture.
"""

from threading import Lock, Thread

import cv2


def pipeline(cfg):
    src, w, h = cfg["source"], cfg["width"], cfg["height"]
    if src == "imx219":
        return (
            f"v4l2src device={cfg['device']} ! "
            "video/x-bayer,width=1920,height=1080,format=rggb ! "
            f"tiovxisp sink_0::device={cfg['subdev']} sensor-name=SENSOR_SONY_IMX219_RPI "
            f"dcc-isp-file={cfg['dcc_viss']} sink_0::dcc-2a-file={cfg['dcc_2a']} format-msb=7 ! "
            f"tiovxmultiscaler ! video/x-raw,format=NV12,width={w},height={h} ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )
    if src.startswith("usb:"):
        return (
            f"v4l2src device={src[4:]} ! videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={w},height={h} ! appsink drop=true max-buffers=1 sync=false"
        )
    if src.startswith("file:"):
        return (
            f"filesrc location={src[5:]} ! decodebin ! videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={w},height={h} ! appsink sync=true"
        )
    raise ValueError(f"bilinmeyen kamera kaynağı: {src}")


class Camera:
    """Reads in a thread and keeps only the newest frame."""

    def __init__(self, cfg):
        p = pipeline(cfg)
        self.cap = cv2.VideoCapture(p, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("GStreamer hattı açılamadı. Kamera bağlı mı, "
                               "setup_cameras.sh çalıştırıldı mı?\n  " + p)
        self._frame = None
        self._seq = 0
        self._lock = Lock()
        self.running = True
        Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.running = False
                break
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
        self.running = False
        self.cap.release()
