#!/bin/bash
# Kamerayı hazırlar ve uygulamayı başlatır (systemd de bunu çağırır).
set -e
source /opt/t3-edgeai-env
bash /opt/edgeai-gst-apps/scripts/setup_cameras.sh
if [ ! -e /dev/video-imx219-cam0 ]; then
    echo "[HATA] /dev/video-imx219-cam0 yok: kamera CSI0'a takılı mı, overlay yüklü mü?"
    exit 1
fi
cd "$(dirname "$(readlink -f "$0")")"
# exec: python ana süreç olur ve SIGINT'i kendisi alır, TIOVX düzgün kapanır.
exec python3 app/main.py configs/atik.yaml
