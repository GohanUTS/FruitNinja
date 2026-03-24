#!/usr/bin/env python3
"""
Fruit colour detection using HSV thresholding.
Detects red, yellow/orange, and green fruits and draws bounding boxes.
"""

import cv2
import numpy as np


def detect_fruits(frame):
    """
    Detect coloured fruits in a BGR frame.
    Returns the frame with bounding boxes and labels drawn.
    """
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    # Red wraps around 0/180 in HSV
    m_r = (
        cv2.inRange(hsv, np.array([0,   80, 80]), np.array([12,  255, 255]))
        | cv2.inRange(hsv, np.array([158, 80, 80]), np.array([180, 255, 255]))
    )
    # Orange / yellow
    m_y = cv2.inRange(hsv, np.array([13, 80, 80]), np.array([38, 255, 255]))
    # Green
    m_g = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([85, 255, 255]))

    colours = [
        (m_r, (0, 80,  255), 'Red'),
        (m_y, (0, 200, 255), 'Yellow/Orange'),
        (m_g, (0, 200,  50), 'Green'),
    ]

    for mask, bgr, label in colours:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 600:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

    return frame
