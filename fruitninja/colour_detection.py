#!/usr/bin/env python3
"""
Fruit colour detection using HSV thresholding.
Detects solid colour regions larger than 20x15px and draws bounding boxes.
"""

import cv2
import numpy as np

# (label, bgr_draw_colour, [hsv_ranges]) — each range is (lower, upper)
COLOUR_PROFILES = [
    ('Red',    (0,   50,  255), [
        (np.array([0,   100, 80]),  np.array([10,  255, 255])),
        (np.array([160, 100, 80]),  np.array([180, 255, 255])),
    ]),
    ('Orange', (0,   120, 255), [
        (np.array([11, 100, 80]),   np.array([20,  255, 255])),
    ]),
    ('Yellow', (0,   220, 255), [
        (np.array([21, 100, 80]),   np.array([35,  255, 255])),
    ]),
    ('Green',  (0,   200,  50), [
        (np.array([36,  60, 60]),   np.array([85,  255, 255])),
    ]),
    ('Cyan',   (200, 200,   0), [
        (np.array([86,  60, 60]),   np.array([100, 255, 255])),
    ]),
    ('Blue',     (220,  80,   0), [
        (np.array([101, 80, 60]),   np.array([115, 255, 255])),
    ]),
    ('Dark Blue', (139,  0,   0), [
        (np.array([116, 80, 30]),   np.array([130, 255, 130])),
    ]),
    ('Purple', (180,  30, 180), [
        (np.array([131, 60, 60]),   np.array([155, 255, 255])),
    ]),
    ('Pink',   (160,  80, 255), [
        (np.array([156, 50, 100]),  np.array([169, 255, 255])),
    ]),
    ('White',  (240, 240, 240), [
        (np.array([0,   0,  200]),  np.array([180, 40,  255])),
    ]),
]

# Minimum bounding-box size to keep a detection
MIN_W = 60
MIN_H = 45

_KERNEL = np.ones((5, 5), np.uint8)


def detect_fruits(frame):
    """
    Detect solid colour regions in a BGR frame.
    Draws labelled bounding boxes for every region >= MIN_W x MIN_H pixels.
    Returns the annotated frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for label, bgr, ranges in COLOUR_PROFILES:
        # combine all HSV ranges for this colour
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # clean up noise and fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_W or h < MIN_H:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

    return frame
