"""Grab one 640x480 frame from the front and side cameras into logs/station/lerobot/.

    .venv-lerobot/bin/python tools/capture_cams.py --front 1 --side 0
"""

import argparse
import os

import cv2
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

parser = argparse.ArgumentParser()
parser.add_argument("--front", type=int, default=1)
parser.add_argument("--side", type=int, default=0)
args = parser.parse_args()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "station", "lerobot")
os.makedirs(out_dir, exist_ok=True)
for name, idx in (("front", args.front), ("side", args.side)):
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=idx, fps=30, width=640, height=480))
    cam.connect()
    for _ in range(15):  # let exposure settle
        img = cam.read()
    path = os.path.join(out_dir, f"{name}_640x480.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"{name}: {img.shape} -> {path}")
    cam.disconnect()
