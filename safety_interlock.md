# UR3e Hand Detection Safety Interlock

A real-time camera-based safety system that pauses a UR3e robot when a hand is detected inside or near the designated cutting zone.

---

## Requirements

```bash
pip install mediapipe opencv-python
```

---

## Configuration

Before running, update these values at the top of the script to match your setup:

| Variable | Description |
|---|---|
| `ZONE` | Pixel coordinates of your cutting zone (from camera calibration) |
| `TOLERANCE_PX` | Extra pixels outside the zone that still trigger a stop |
| `ROBOT_IP` | IP address of your UR3e controller |

---

## Full Implementation

```python
import cv2
import mediapipe as mp
import socket
import time

# ---- Zone config (set these to your camera-calibrated values) ----
ZONE = {"x1": 150, "y1": 100, "x2": 450, "y2": 380}  # pixel coords
TOLERANCE_PX = 40  # how many pixels outside the zone still triggers a stop

# ---- UR3e connection ----
ROBOT_IP = "192.168.1.100"
DASHBOARD_PORT = 29999

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

def expanded_zone(zone, tolerance):
    return {
        "x1": zone["x1"] - tolerance,
        "y1": zone["y1"] - tolerance,
        "x2": zone["x2"] + tolerance,
        "y2": zone["y2"] + tolerance,
    }

def hand_in_zone(landmarks, frame_w, frame_h, zone):
    ez = expanded_zone(zone, TOLERANCE_PX)
    for lm in landmarks.landmark:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        if ez["x1"] <= px <= ez["x2"] and ez["y1"] <= py <= ez["y2"]:
            return True
    return False

class UR3eDashboard:
    def __init__(self, ip, port=29999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.recv(1024)  # welcome message
        self._stopped = False

    def _send(self, cmd):
        self.sock.sendall((cmd + "\n").encode())
        return self.sock.recv(1024).decode().strip()

    def pause(self):
        if not self._stopped:
            self._send("pause")
            self._stopped = True
            print("[SAFETY] Robot PAUSED - hand detected in zone")

    def resume(self):
        if self._stopped:
            self._send("play")
            self._stopped = False
            print("[SAFETY] Robot RESUMED - zone clear")

    def close(self):
        self.sock.close()

def draw_zones(frame, zone):
    ez = expanded_zone(zone, TOLERANCE_PX)
    # tolerance zone (orange)
    cv2.rectangle(frame,
        (ez["x1"], ez["y1"]), (ez["x2"], ez["y2"]),
        (0, 140, 255), 1)
    cv2.putText(frame, "tolerance", (ez["x1"] + 4, ez["y1"] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
    # cutting zone (green)
    cv2.rectangle(frame,
        (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]),
        (0, 220, 80), 2)
    cv2.putText(frame, "cutting zone", (zone["x1"] + 4, zone["y1"] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)

def run():
    cap = cv2.VideoCapture(0)
    robot = UR3eDashboard(ROBOT_IP)

    print("[SAFETY] System active. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_detected = False
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    if hand_in_zone(hand_landmarks, w, h, ZONE):
                        hand_detected = True
                    # draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if hand_detected:
                robot.pause()
                cv2.putText(frame, "HAND DETECTED - ROBOT STOPPED",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
            else:
                robot.resume()
                cv2.putText(frame, "Zone clear",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 80), 2)

            draw_zones(frame, ZONE)
            cv2.imshow("Safety Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        robot.resume()  # always restore motion on exit
        robot.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
```

---

## How It Works

### System loop (per frame)

1. Camera captures a frame
2. MediaPipe processes the frame and returns hand landmarks
3. Each landmark is checked against the expanded zone (zone + tolerance)
4. If any landmark falls inside the expanded zone, the UR3e is paused via the Dashboard Server
5. Once all landmarks are outside, the robot resumes

### Zone checking

The cutting zone is defined in pixel coordinates from your camera calibration. The `expanded_zone()` function pads all four edges by `TOLERANCE_PX` before the check runs. This means the robot stops slightly before a hand fully enters the zone, giving extra reaction margin.

```
Original zone:   x1=150, y1=100, x2=450, y2=380
With 40px pad:   x1=110, y1=60,  x2=490, y2=420
```

### Robot communication

The `UR3eDashboard` class connects to the UR3e Dashboard Server over TCP on port `29999`. Two commands are used:

- `pause` - suspends the running URScript program, holds position, preserves state
- `play` - resumes from where it paused, no re-homing needed

Using `pause/play` rather than `stop` means the cutting sequence continues from the exact point it was interrupted once the zone is clear.

### Why all 21 landmarks

MediaPipe tracks 21 keypoints per hand (wrist, knuckles, fingertips). Checking all of them means a single fingertip entering the zone is enough to trigger a stop, rather than requiring the whole palm to be inside. This is intentional for safety.

---

## Tuning guide

| Parameter | Effect | Suggested range |
|---|---|---|
| `TOLERANCE_PX` | Larger = stops sooner, more margin | 20 to 80px |
| `min_detection_confidence` | Lower = catches more hands, more false positives | 0.5 to 0.8 |
| `min_tracking_confidence` | Lower = more responsive tracking | 0.5 to 0.7 |
| `max_num_hands` | Set to 1 if only one operator, improves speed | 1 or 2 |

---

## Performance notes

At 30fps, each frame budget is ~33ms. MediaPipe Hands on CPU typically takes 20-40ms. To keep the loop real-time:

- Set `max_num_hands=1` if only one operator is present
- Run on a machine with a dedicated GPU for faster inference
- Reduce frame resolution if latency is too high (`cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`)

---

## Safety notice

This is a **software safety layer**. For production use with a cutting tool, it must be paired with a hardware-level safeguard such as a safety-rated light curtain or physical e-stop. Software interlocks can fail due to process crashes, camera occlusion, or detection misses. Hardware interlocks operate independently of the software stack and are the true last line of defence.
