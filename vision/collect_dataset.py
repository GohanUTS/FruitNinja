#!/usr/bin/env python3
"""
collect_dataset.py — Dataset collection tool for FruitNinja vision pipeline.

Streams frames from a RealSense D435i (1280×720 @ 30 fps, BGR8).
If pyrealsense2 is not installed, falls back to the first USB webcam (cv2.VideoCapture(0)).

Controls
--------
  S  — save the current frame to dataset/raw/frame_{timestamp}_{count:04d}.jpg
  Q  — quit and clean up

Saved frames are intended as raw material for Roboflow labelling.
"""

import os
import time
import cv2
import numpy as np

# ── Attempt RealSense import ─────────────────────────────────────────────────

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False
    print(
        "[WARN] pyrealsense2 not found — falling back to webcam (cv2.VideoCapture(0)).\n"
        "       Install with:  pip install pyrealsense2"
    )

# ── Constants ────────────────────────────────────────────────────────────────

SAVE_DIR   = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
WIDTH      = 1280
HEIGHT     = 720
FPS        = 30
WINDOW     = "FruitNinja Dataset Collector  |  S=save  Q=quit"


def _ensure_save_dir() -> None:
    """Create the save directory if it does not already exist."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[INFO] Saving frames to: {os.path.abspath(SAVE_DIR)}")


# ── RealSense capture ────────────────────────────────────────────────────────

def _run_realsense() -> None:
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    print(f"[INFO] RealSense pipeline started at {WIDTH}×{HEIGHT} @ {FPS} fps.")

    count = 0
    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            cv2.imshow(WINDOW, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                count    += 1
                timestamp = int(time.time())
                filename  = os.path.join(SAVE_DIR, f"frame_{timestamp}_{count:04d}.jpg")
                cv2.imwrite(filename, img)
                print(f"[SAVE] {count:4d}  →  {filename}")
            elif key == ord('q') or key == ord('Q'):
                print("[INFO] Quitting.")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Done. {count} frame(s) saved.")


# ── Webcam fallback capture ──────────────────────────────────────────────────

def _run_webcam() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open any camera (cv2.VideoCapture(0) failed).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    print(f"[INFO] Webcam opened. Requested {WIDTH}×{HEIGHT} @ {FPS} fps.")

    count = 0
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("[WARN] Frame grab failed — retrying…")
                continue

            cv2.imshow(WINDOW, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                count    += 1
                timestamp = int(time.time())
                filename  = os.path.join(SAVE_DIR, f"frame_{timestamp}_{count:04d}.jpg")
                cv2.imwrite(filename, img)
                print(f"[SAVE] {count:4d}  →  {filename}")
            elif key == ord('q') or key == ord('Q'):
                print("[INFO] Quitting.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Done. {count} frame(s) saved.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    _ensure_save_dir()
    if _HAS_RS:
        _run_realsense()
    else:
        _run_webcam()


if __name__ == "__main__":
    main()
