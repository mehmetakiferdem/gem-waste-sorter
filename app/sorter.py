"""Turns the sorting flap to the decided bin, then back home."""

import glob
import os
import threading
import time


class SysfsPwmServo:
    """One servo on a hardware PWM channel, via /sys/class/pwm."""

    PERIOD_NS = 20000000  # 50 Hz

    def __init__(self, device, channel):
        # pwmchipN numbers follow probe order, so find the chip by the
        # address of the peripheral instead.
        chips = [c for c in glob.glob("/sys/class/pwm/pwmchip*")
                 if os.path.basename(os.path.realpath(c + "/device")) == device]
        if not chips:
            raise RuntimeError(f"{device} için pwmchip yok (overlay yüklü mü?)")
        self.path = f"{chips[0]}/pwm{channel}"
        if not os.path.isdir(self.path):
            self._put(chips[0] + "/export", channel)
            for _ in range(50):
                if os.access(self.path + "/period", os.W_OK):
                    break
                time.sleep(0.01)
        self._put("enable", 0)
        # duty_cycle may not exceed period, so clear it before the period.
        self._put("duty_cycle", 0)
        self._put("period", self.PERIOD_NS)
        self._put("enable", 1)

    def _put(self, name, value):
        path = name if name.startswith("/") else f"{self.path}/{name}"
        with open(path, "w") as f:
            f.write(str(value))

    def angle(self, deg):
        deg = max(-90.0, min(90.0, float(deg)))
        self._put("duty_cycle", int((1500 + deg * 1000 / 90) * 1000))

    def close(self):
        self._put("enable", 0)


class Sorter:
    def __init__(self, cfg, bins):
        self.bins = bins
        self.home = cfg.get("home_angle", 0)
        self.hold_s = cfg.get("hold_s", 1.5)
        self.servo = None
        if cfg.get("backend", "none") == "hwpwm":
            self.servo = SysfsPwmServo(cfg["pwm_device"], cfg["pwm_channel"])
            self.servo.angle(self.home)
        self._timer = None

    def route(self, label):
        b = self.bins.get(label)
        if self.servo is None or b is None:
            return
        if self._timer:
            self._timer.cancel()
        self.servo.angle(b.get("angle", self.home))
        self._timer = threading.Timer(self.hold_s, self.servo.angle, (self.home,))
        self._timer.start()

    def close(self):
        if self._timer:
            self._timer.cancel()
        if self.servo:
            self.servo.angle(self.home)
            time.sleep(0.3)
            self.servo.close()
