#!/usr/bin/env python3
"""Compiles model.onnx into TIDL artifacts for the AM67A C7x (8-bit).

Runs on an x86 PC inside TI's edgeai-tidl-tools environment, whose version
MUST match the Processor SDK on the board (11_00_xx_xx for SDK 11.00):

    git clone -b 11_00_06_00 https://github.com/TexasInstruments/edgeai-tidl-tools
    cd edgeai-tidl-tools && export SOC=am67a && source ./setup.sh
    export TIDL_TOOLS_PATH=$PWD/tools/AM67A/tidl_tools
    export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
    python3 compile_tidl.py --model ../models/waste_mnv2 --calib dataset/val

(Önerilen yol: Dockerfile.tidl. Sürüm notu için README.)

Option names follow examples/osrt_python/common_utils.py of that release.
"""

import argparse
import glob
import os
import random
import shutil

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="../models/waste_mnv2")
    ap.add_argument("--calib", default="dataset/val")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    import onnx
    import onnxruntime as rt

    tools = os.environ.get("TIDL_TOOLS_PATH")
    if not tools:
        raise SystemExit("TIDL_TOOLS_PATH ayarlı değil (yukarıdaki kuruluma bakın)")

    onnx_path = os.path.join(args.model, "model.onnx")
    artifacts = os.path.join(args.model, "artifacts")
    shutil.rmtree(artifacts, ignore_errors=True)
    os.makedirs(artifacts)
    onnx.shape_inference.infer_shapes_path(onnx_path, onnx_path)

    # Calibration images from every class, so each one shapes the ranges.
    imgs = sorted(glob.glob(os.path.join(args.calib, "*", "*")))
    random.Random(0).shuffle(imgs)
    imgs = imgs[:args.frames]

    opts = {
        "tidl_tools_path": tools,
        "artifacts_folder": artifacts,
        "platform": "J7",
        "tensor_bits": 8,
        "debug_level": 0,
        "accuracy_level": 1,
        "advanced_options:calibration_frames": len(imgs),
        "advanced_options:calibration_iterations": args.iterations,
    }
    so = rt.SessionOptions()
    sess = rt.InferenceSession(onnx_path, sess_options=so,
                               providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
                               provider_options=[opts, {}])
    name = sess.get_inputs()[0].name
    for i, f in enumerate(imgs):
        x = np.asarray(Image.open(f).convert("RGB").resize((args.size, args.size)), np.float32)
        sess.run(None, {name: x.transpose(2, 0, 1)[None]})
        print(f"\rkalibrasyon {i + 1}/{len(imgs)}", end="", flush=True)
    print(f"\n{artifacts} hazır. Karta kopyalayın:\n"
          f"  scp -r {args.model} gemstone@192.168.7.2:~/gem-waste-sorter/models/")


if __name__ == "__main__":
    main()
