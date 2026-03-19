"""
OpenCV Webcam Explorer
======================
Left panel  — raw webcam feed
Right panel — detection overlay

Controls (press key while window is focused):
  Q        quit
  F        toggle face detection
  E        toggle edge detection (Canny)
  M        toggle motion detection
  C        toggle contour detection
  H        toggle hue/colour tracking (green by default)
  I        show/hide info overlay
  S        save snapshot
"""

import cv2
import numpy as np
import time
import os

# ── Configuration ──────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0          # change if your webcam isn't device 0
WINDOW_NAME    = "OpenCV Explorer  |  Q=quit  F=face  E=edges  M=motion  C=contours  H=hue  I=info  S=snap"
PANEL_WIDTH    = 640        # width of each panel
PANEL_HEIGHT   = 480

# Colour palette (BGR)
CLR_GREEN  = (0, 255, 120)
CLR_CYAN   = (255, 220, 0)
CLR_RED    = (60, 60, 255)
CLR_ORANGE = (0, 165, 255)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (20, 20, 20)

# ── Detector state ─────────────────────────────────────────────────────────────
state = {
    "face":    False,
    "edges":   False,
    "motion":  False,
    "contour": False,
    "hue":     False,
    "info":    True,
}

prev_gray      = None
face_cascade   = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade    = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

snapshot_dir = os.path.expanduser("~/Desktop")
fps_history  = []


# ── Helper: labelled toggle pill ───────────────────────────────────────────────
def draw_badge(img, label, active, x, y):
    colour = CLR_GREEN if active else (80, 80, 80)
    dot    = CLR_GREEN if active else (60, 60, 60)
    cv2.rectangle(img, (x, y), (x + 110, y + 22), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + 110, y + 22), colour, 1)
    cv2.circle(img, (x + 14, y + 11), 5, dot, -1)
    cv2.putText(img, label, (x + 24, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


# ── Detectors ──────────────────────────────────────────────────────────────────
def detect_faces(frame):
    out    = frame.copy()
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray   = cv2.equalizeHist(gray)
    faces  = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    count  = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x+w, y+h), CLR_GREEN, 2)
        # corner ticks
        tl, tc = 12, 2
        for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(out, (px, py), (px+dx*tl, py), CLR_GREEN, tc)
            cv2.line(out, (px, py), (px, py+dy*tl), CLR_GREEN, tc)
        label = f"face {count+1}"
        cv2.putText(out, label, (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREEN, 1, cv2.LINE_AA)
        # eyes inside face ROI
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cx, cy = x+ex+ew//2, y+ey+eh//2
            cv2.circle(out, (cx, cy), ew//2, CLR_CYAN, 1)
        count += 1
    return out, count


def detect_edges(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # tint edges cyan on dark background
    tinted = np.zeros_like(frame)
    tinted[edges > 0] = CLR_CYAN
    out = cv2.addWeighted(frame, 0.3, tinted, 0.9, 0)
    edge_px = int(np.sum(edges > 0))
    return out, edge_px


def detect_motion(frame, prev):
    global prev_gray
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray   = cv2.GaussianBlur(gray, (21, 21), 0)
    out    = frame.copy()
    blobs  = 0
    if prev is not None:
        delta = cv2.absdiff(prev, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 800:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x+w, y+h), CLR_ORANGE, 2)
            blobs += 1
        # motion heatmap overlay
        heat = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        heat[:,:,0] = 0; heat[:,:,1] = 0   # keep only red channel
        out = cv2.addWeighted(out, 1.0, heat, 0.25, 0)
    prev_gray = gray
    return out, blobs


def detect_contours(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out    = frame.copy()
    big    = [c for c in cnts if cv2.contourArea(c) > 500]
    cv2.drawContours(out, big, -1, CLR_ORANGE, 1)
    return out, len(big)


def detect_hue(frame, lower=(35,80,50), upper=(85,255,255)):
    """Detect green-ish objects by default."""
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo     = np.array(lower, dtype=np.uint8)
    hi     = np.array(upper, dtype=np.uint8)
    mask   = cv2.inRange(hsv, lo, hi)
    mask   = cv2.erode(mask,  None, iterations=2)
    mask   = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out    = frame.copy()
    count  = 0
    for c in cnts:
        if cv2.contourArea(c) < 600:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x+w, y+h), CLR_RED, 2)
        count += 1
    # tint detected region
    tint = cv2.bitwise_and(frame, frame, mask=mask)
    out  = cv2.addWeighted(out, 0.8, tint, 0.4, 0)
    return out, count


# ── Info overlay ───────────────────────────────────────────────────────────────
def draw_info(img, fps, detection_label, detection_count):
    h, w = img.shape[:2]
    # semi-transparent bar at bottom
    bar = img.copy()
    cv2.rectangle(bar, (0, h-34), (w, h), CLR_DARK, -1)
    img = cv2.addWeighted(img, 0.6, bar, 0.4, 0)

    fps_text   = f"FPS: {fps:.1f}"
    det_text   = f"{detection_label}: {detection_count}"
    ts_text    = time.strftime("%H:%M:%S")

    cv2.putText(img, fps_text, (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, det_text, (w//2 - 60, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GREEN, 1, cv2.LINE_AA)
    cv2.putText(img, ts_text, (w-80, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)
    return img


def draw_badges(img):
    pairs = [
        ("F  Face",    state["face"]),
        ("E  Edges",   state["edges"]),
        ("M  Motion",  state["motion"]),
        ("C  Contour", state["contour"]),
        ("H  Hue",     state["hue"]),
    ]
    for i, (lbl, active) in enumerate(pairs):
        draw_badge(img, lbl, active, 8, 8 + i * 28)
    return img


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    global prev_gray

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}. Try a different index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PANEL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_HEIGHT)

    print("OpenCV Explorer running — press Q in the window to quit.")

    t_prev     = time.time()
    snap_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying…")
            continue

        frame = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT))
        raw   = frame.copy()

        # ── Build detection panel ──────────────────────────────────────────────
        det_frame  = frame.copy()
        det_label  = "detections"
        det_count  = 0

        if state["face"]:
            det_frame, det_count = detect_faces(det_frame)
            det_label = "faces"
        if state["edges"]:
            det_frame, det_count = detect_edges(det_frame)
            det_label = "edge px"
        if state["motion"]:
            det_frame, det_count = detect_motion(det_frame, prev_gray)
            det_label = "motion blobs"
        if state["contour"]:
            det_frame, det_count = detect_contours(det_frame)
            det_label = "contours"
        if state["hue"]:
            det_frame, det_count = detect_hue(det_frame)
            det_label = "green blobs"

        # ── FPS ────────────────────────────────────────────────────────────────
        now  = time.time()
        fps  = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # ── Labels on raw panel ────────────────────────────────────────────────
        cv2.putText(raw, "RAW", (PANEL_WIDTH - 50, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(det_frame, "DETECTION", (PANEL_WIDTH - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_GREEN, 1, cv2.LINE_AA)

        # ── Info & badges ──────────────────────────────────────────────────────
        if state["info"]:
            raw       = draw_info(raw, avg_fps, det_label, det_count)
            det_frame = draw_badges(det_frame)

        # ── Divider line ───────────────────────────────────────────────────────
        divider = np.zeros((PANEL_HEIGHT, 4, 3), dtype=np.uint8)
        divider[:] = (60, 60, 60)

        combined = np.hstack([raw, divider, det_frame])
        cv2.imshow(WINDOW_NAME, combined)

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            state["face"]    = not state["face"]
        elif key == ord('e'):
            state["edges"]   = not state["edges"]
        elif key == ord('m'):
            state["motion"]  = not state["motion"]
            prev_gray = None   # reset baseline on toggle
        elif key == ord('c'):
            state["contour"] = not state["contour"]
        elif key == ord('h'):
            state["hue"]     = not state["hue"]
        elif key == ord('i'):
            state["info"]    = not state["info"]
        elif key == ord('s'):
            snap_count += 1
            filename = os.path.join(snapshot_dir, f"snap_{snap_count:03d}.jpg")
            cv2.imwrite(filename, combined)
            print(f"[SNAP] Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()