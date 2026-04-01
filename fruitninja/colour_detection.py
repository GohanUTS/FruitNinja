#!/usr/bin/env python3
"""
Fruit colour detection + whiteboard grid overlay.

  Apple   (red)   : bounding box + distance in mm + grid cell (if grid active)
  Lettuce (green) : bounding box + distance in mm + grid cell (if grid active)
  Banana  (yellow): bounding box + distance in mm + grid cell (if grid active)
  Orange  (orange): bounding box + distance in mm + grid cell (if grid active)
  Blue corners    : 4 blue markers → labelled A1-E5 grid drawn inside

Returns: (annotated_frame, detections)
  detections = [{'label': str, 'distance_mm': float, 'cell': str|None}, ...]
"""

import cv2
import numpy as np

# ── Fruit profiles (label, bgr, hsv_ranges, real_width_m) ────────────────────
COLOUR_PROFILES = [
    ('Apple',   (0, 50, 255),  [
        (np.array([0,   60, 40]),  np.array([10,  255, 255])),
        (np.array([160, 60, 40]),  np.array([180, 255, 255])),
    ], 0.08),
    ('Lettuce', (0, 200, 50),  [
        (np.array([36,  45, 40]),   np.array([85,  255, 255])),
    ], 0.25),
    ('Banana',  (0, 220, 255), [
        (np.array([20,  80, 80]),  np.array([35, 255, 255])),
    ], 0.18),
    ('Orange',  (0, 130, 255), [
        (np.array([10, 120, 80]),  np.array([20, 255, 255])),
    ], 0.07),
]

# ── Blue corner marker config ─────────────────────────────────────────────────
BLUE_HSV_RANGES = [
    (np.array([100, 120, 60]),  np.array([120, 255, 255])),
]
BLUE_BGR        = (220, 80,  0)
GRID_BGR        = (200, 200, 0)
CELL_LABEL_BGR  = (255, 255, 255)
GRID_N          = 4
MIN_MARKER_AREA = 300

# ── Detection size thresholds — small for drawn marker dots ──────────────────
MIN_W = 12
MIN_H = 12

# Tune FOCAL_LENGTH_PX: too HIGH → reading too far, too LOW → reading too close
FOCAL_LENGTH_PX = 100.0

_KERNEL    = np.ones((5, 5), np.uint8)
_KERNEL_SM = np.ones((3, 3), np.uint8)

# ── Grid stability state ──────────────────────────────────────────────────────
# EMA smoothing so small tilts don't cause flicker.
# If corners are lost, keep showing the last known grid for MAX_LOST_FRAMES.
_SMOOTH_ALPHA   = 0.25   # lower = smoother but slower to track intentional moves
_MAX_LOST_FRAMES = 20    # frames to hold grid after corners disappear (~0.6 s)
_smooth_corners  = None  # (tl, tr, br, bl) as np.float32 arrays
_lost_frames     = 0


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t)


def _ipt(pt):
    return (int(round(pt[0])), int(round(pt[1])))


def _sort_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).ravel()
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]
    return tl, tr, br, bl


def _cell_name(col, row):
    """Convert 0-based (col, row) to chess-style label e.g. (0,0) → 'A1'."""
    return f"{chr(ord('A') + col)}{row + 1}"


def _draw_grid(frame, tl, tr, br, bl, n):
    """Draw bilinear n×n grid and label every cell center."""
    steps = max(n * 6, 24)

    for i in range(n + 1):
        t = i / n
        # vertical lines
        left  = _lerp(tl, bl, t)
        right = _lerp(tr, br, t)
        prev  = _ipt(_lerp(left, right, 0))
        for j in range(1, steps + 1):
            cur = _ipt(_lerp(left, right, j / steps))
            cv2.line(frame, prev, cur, GRID_BGR, 1, cv2.LINE_AA)
            prev = cur
        # horizontal lines
        top  = _lerp(tl, tr, t)
        bot  = _lerp(bl, br, t)
        prev = _ipt(_lerp(top, bot, 0))
        for j in range(1, steps + 1):
            cur = _ipt(_lerp(top, bot, j / steps))
            cv2.line(frame, prev, cur, GRID_BGR, 1, cv2.LINE_AA)
            prev = cur

    # Label each cell at its centre
    for row in range(n):
        for col in range(n):
            u   = (col + 0.5) / n
            v   = (row + 0.5) / n
            top = _lerp(tl, tr, u)
            bot = _lerp(bl, br, u)
            ctr = _ipt(_lerp(top, bot, v))
            lbl = _cell_name(col, row)
            # shadow then label
            cv2.putText(frame, lbl, (ctr[0] - 8, ctr[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2)
            cv2.putText(frame, lbl, (ctr[0] - 8, ctr[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, CELL_LABEL_BGR, 1)


def _build_transform(tl, tr, br, bl, n):
    """Return perspective transform H mapping image→grid coords (0..n)."""
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    return cv2.getPerspectiveTransform(src, dst)


def _point_to_cell(H, px, py, n):
    """Return cell name for image point (px,py), or None if outside grid."""
    pt  = np.float32([[[px, py]]])
    uv  = cv2.perspectiveTransform(pt, H)[0][0]
    col = int(uv[0])
    row = int(uv[1])
    if 0 <= col < n and 0 <= row < n:
        return _cell_name(col, row)
    return None


# ── Grid detection ────────────────────────────────────────────────────────────

def _detect_grid(frame, hsv):
    """
    Find 4 blue corner markers, apply EMA smoothing, draw grid with cell labels.
    Holds the last known grid for _MAX_LOST_FRAMES if corners temporarily vanish.
    Returns perspective transform H, or None if no grid has ever been seen.
    """
    global _smooth_corners, _lost_frames

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in BLUE_HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL_SM)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
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

    blobs.sort(reverse=True)
    blobs = blobs[:4]

    if len(blobs) == 4:
        pts          = [(cx, cy) for _, cx, cy in blobs]
        raw_corners  = _sort_corners(pts)   # (tl, tr, br, bl)

        if _smooth_corners is None:
            # First detection — initialise directly, no smoothing yet
            _smooth_corners = raw_corners
        else:
            # EMA: blend new corners toward current smooth estimate
            _smooth_corners = tuple(
                _SMOOTH_ALPHA * np.array(new, dtype=np.float32)
                + (1.0 - _SMOOTH_ALPHA) * np.array(old, dtype=np.float32)
                for new, old in zip(raw_corners, _smooth_corners)
            )
        _lost_frames = 0

        # Draw detected marker dots
        for _, cx, cy in blobs:
            cv2.circle(frame, (cx, cy), 9,  BLUE_BGR,       -1)
            cv2.circle(frame, (cx, cy), 11, (255, 255, 255),  1)
    else:
        _lost_frames += 1

    # Use smoothed corners as long as we haven't lost them too long
    if _smooth_corners is None or _lost_frames > _MAX_LOST_FRAMES:
        return None

    tl, tr, br, bl = _smooth_corners

    _draw_grid(frame, tl, tr, br, bl, GRID_N)

    quad = np.array([_ipt(tl), _ipt(tr), _ipt(br), _ipt(bl)], np.int32)
    cv2.polylines(frame, [quad], isClosed=True,
                  color=GRID_BGR, thickness=2, lineType=cv2.LINE_AA)

    for name, pt in [('TL', tl), ('TR', tr), ('BR', br), ('BL', bl)]:
        cv2.putText(frame, name, (_ipt(pt)[0] + 13, _ipt(pt)[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(frame, name, (_ipt(pt)[0] + 13, _ipt(pt)[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE_BGR, 1)

    return _build_transform(tl, tr, br, bl, GRID_N)


# ── Main entry point ──────────────────────────────────────────────────────────

def detect_fruits(frame):
    """
    Detect Apple/Lettuce with distance + grid cell, and draw the blue grid.

    Returns
    -------
    annotated_frame : np.ndarray
    detections      : list of {'label': str, 'distance_mm': float, 'cell': str|None}
    """
    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    # Grid first so fruit labels draw on top
    H = _detect_grid(frame, hsv)

    for label, bgr, ranges, real_width_m in COLOUR_PROFILES:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL_SM)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_SM)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_W or h < MIN_H:
                continue

            cx = x + w // 2
            cy = y + h // 2
            distance_mm = (real_width_m * FOCAL_LENGTH_PX / w) * 1000.0 

            cell = _point_to_cell(H, cx, cy, GRID_N) if H is not None else None

            # When grid is active, ignore anything outside it
            if H is not None and cell is None:
                continue

            cell_tag = f'  [{cell}]' if cell else ''
            text = f'{label}  {distance_mm:.0f} mm{cell_tag}'
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, bgr, 2)

            # Draw crosshair at centroid
            cv2.drawMarker(frame, (cx, cy), bgr,
                           cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

            detections.append({'label': label,
                                'distance_mm': distance_mm,
                                'cell': cell})

    return frame, detections
