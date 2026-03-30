#!/usr/bin/env python3
"""
Fruit colour detection using HSV thresholding.
Detects Apple (red) and Lettuce (green), draws bounding boxes,
and estimates distance from camera using apparent object size.

Returns: (annotated_frame, detections)
  detections = [{'label': str, 'distance_m': float}, ...]
"""

import cv2
import numpy as np

# ── Colour profiles (label, bgr_colour, hsv_ranges, real_width_m) ─────────────
#
# real_width_m: approximate real-world width of the object used for
# distance estimation via the pinhole model.
#   Apple  ≈ 8 cm wide
#   Lettuce ≈ 25 cm wide
#
COLOUR_PROFILES = [
    ('Apple',   (0, 50, 255),  [
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([160, 120, 70]),  np.array([180, 255, 255])),
    ], 0.08),
    ('Lettuce', (0, 200, 50),  [
        (np.array([36,  60, 60]),   np.array([85,  255, 255])),
    ], 0.25),
]

# Minimum bounding-box size to count as a real detection
MIN_W = 60
MIN_H = 45

# Approximate focal length in pixels for a typical 640-wide webcam.
# Tune this value if your camera differs (higher = wider angle lens).
FOCAL_LENGTH_PX = 700.0

_KERNEL = np.ones((5, 5), np.uint8)


def detect_fruits(frame):
    """
    Detect Apple (red) and Lettuce (green) regions in a BGR frame.
    Annotates the frame with bounding boxes, labels, and distance.

    Returns
    -------
    annotated_frame : np.ndarray
    detections      : list of {'label': str, 'distance_m': float}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for label, bgr, ranges, real_width_m in COLOUR_PROFILES:
        # Combine HSV ranges into one mask
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # Clean noise and fill small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_W or h < MIN_H:
                continue

            # Pinhole distance estimate: D = (W_real * f) / W_pixels
            distance_m = (real_width_m * FOCAL_LENGTH_PX) / w

            text = f'{label}  {distance_m:.2f} m'
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

            detections.append({'label': label, 'distance_m': distance_m})

    return frame, detections
