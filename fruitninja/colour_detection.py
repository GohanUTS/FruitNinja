#!/usr/bin/env python3
"""
Fruit colour detection + whiteboard grid overlay.

  Apple   (red)  : bounding box + distance in mm
  Lettuce (green): bounding box + distance in mm
  Blue corners   : 4 blue markers on a whiteboard → square grid drawn inside

Returns: (annotated_frame, detections)
  detections = [{'label': str, 'distance_mm': float}, ...]
"""

import cv2
import numpy as np

# ── Apple / Lettuce profiles (label, bgr, hsv_ranges, real_width_m) ──────────
COLOUR_PROFILES = [
    ('Apple',   (0, 50, 255),  [
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([160, 120, 70]),  np.array([180, 255, 255])),
    ], 0.08),
    ('Lettuce', (0, 200, 50),  [
        (np.array([36,  60, 60]),   np.array([85,  255, 255])),
    ], 0.25),
]

# ── Blue corner marker config ─────────────────────────────────────────────────
BLUE_HSV_RANGES = [
    (np.array([100, 80, 60]),  np.array([130, 255, 255])),
]
BLUE_BGR        = (220, 80, 0)     # colour for marker dots / corner labels
GRID_BGR        = (200, 200, 0)    # cyan grid lines
GRID_N          = 5                # draws an N×N cell grid (N+1 lines each way)
MIN_MARKER_AREA = 300              # px² — ignore tiny blue blobs

# ── General detection thresholds ──────────────────────────────────────────────
MIN_W = 60
MIN_H = 45

# Tune FOCAL_LENGTH_PX: if distance reads too HIGH → lower it, too LOW → raise it
FOCAL_LENGTH_PX = 150.0

_KERNEL      = np.ones((5, 5), np.uint8)
_KERNEL_SM   = np.ones((3, 3), np.uint8)


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t)


def _ipt(pt):
    return (int(round(pt[0])), int(round(pt[1])))


def _sort_corners(pts):
    """
    Sort 4 (x, y) points into [TL, TR, BR, BL] order.
    Uses the sum/diff trick for robustness against non-rectangular quads.
    """
    pts = np.array(pts, dtype=np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).ravel()
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]
    return tl, tr, br, bl


def _draw_grid(frame, tl, tr, br, bl, n):
    """
    Draw a bilinear n×n grid inside the quad TL→TR→BR→BL.
    Uses many sub-segments so lines curve naturally with perspective.
    """
    steps = max(n * 6, 24)

    for i in range(n + 1):
        t = i / n

        # lines running top→bottom  (vary left↔right position)
        left  = _lerp(tl, bl, t)
        right = _lerp(tr, br, t)
        prev  = _ipt(_lerp(left, right, 0))
        for j in range(1, steps + 1):
            cur = _ipt(_lerp(left, right, j / steps))
            cv2.line(frame, prev, cur, GRID_BGR, 1, cv2.LINE_AA)
            prev = cur

        # lines running left→right  (vary top↔bottom position)
        top = _lerp(tl, tr, t)
        bot = _lerp(bl, br, t)
        prev = _ipt(_lerp(top, bot, 0))
        for j in range(1, steps + 1):
            cur = _ipt(_lerp(top, bot, j / steps))
            cv2.line(frame, prev, cur, GRID_BGR, 1, cv2.LINE_AA)
            prev = cur


def _detect_grid(frame, hsv):
    """
    Find the 4 largest blue blobs (corner markers), draw marker dots,
    corner labels, and the grid inside the bounded area.
    Returns the number of blue markers found (0–4+).
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in BLUE_HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL_SM)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Collect centroid of each blob large enough to be a marker
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_MARKER_AREA:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        blobs.append((area, cx, cy))

    # Keep the 4 biggest
    blobs.sort(reverse=True)
    blobs = blobs[:4]

    # Draw a filled dot + white ring at each detected marker
    for _, cx, cy in blobs:
        cv2.circle(frame, (cx, cy), 9,  BLUE_BGR,      -1)
        cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 1)

    if len(blobs) == 4:
        pts      = [(cx, cy) for _, cx, cy in blobs]
        tl, tr, br, bl = _sort_corners(pts)

        # Draw the grid first (so labels sit on top)
        _draw_grid(frame, tl, tr, br, bl, GRID_N)

        # Draw the outer quad border
        quad = np.array([_ipt(tl), _ipt(tr), _ipt(br), _ipt(bl)], np.int32)
        cv2.polylines(frame, [quad], isClosed=True,
                      color=GRID_BGR, thickness=2, lineType=cv2.LINE_AA)

        # Corner labels
        for name, pt in [('TL', tl), ('TR', tr), ('BR', br), ('BL', bl)]:
            cv2.putText(frame, name,
                        (_ipt(pt)[0] + 13, _ipt(pt)[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, name,
                        (_ipt(pt)[0] + 13, _ipt(pt)[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE_BGR, 1)

    return len(blobs)


# ── Main detection entry point ────────────────────────────────────────────────

def detect_fruits(frame):
    """
    Detect Apple/Lettuce (with distance) and draw the blue corner grid.

    Returns
    -------
    annotated_frame : np.ndarray
    detections      : list of {'label': str, 'distance_mm': float}
    """
    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    # ── Apple / Lettuce ───────────────────────────────────────────────────────
    for label, bgr, ranges, real_width_m in COLOUR_PROFILES:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_W or h < MIN_H:
                continue
            distance_mm = (real_width_m * FOCAL_LENGTH_PX / w) * 1000.0
            text = f'{label}  {distance_mm:.0f} mm'
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)
            detections.append({'label': label, 'distance_mm': distance_mm})

    # ── Blue grid ─────────────────────────────────────────────────────────────
    _detect_grid(frame, hsv)

    return frame, detections
