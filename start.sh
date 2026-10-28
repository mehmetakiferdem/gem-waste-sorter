#!/bin/bash
# Kamerayı hazırlar ve uygulamayı başlatır (systemd de bunu çağırır).
set -e
# A previous run still holding the camera makes this one fail with "busy".
for i in $(seq 20); do pgrep -f "^python3 app/(main|collect).py" >/dev/null || break; sleep 1; done
if pgrep -f "^python3 app/(main|collect).py" >/dev/null; then
    echo "[HATA] önceki uygulama hâlâ çalışıyor: sudo pkill -INT -f app/main.py"
    exit 1
fi
source /opt/t3-edgeai-env
bash /opt/edgeai-gst-apps/scripts/setup_cameras.sh
if [ ! -e /dev/video-imx219-cam0 ]; then
    echo "[HATA] /dev/video-imx219-cam0 yok: kamera CSI0'a takılı mı, overlay yüklü mü?"
    exit 1
fi
cd "$(dirname "$(readlink -f "$0")")"
# exec: python ana süreç olur ve SIGINT'i kendisi alır, TIOVX düzgün kapanır.
exec python3 app/main.py configs/atik.yaml
