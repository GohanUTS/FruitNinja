#!/usr/bin/env python3
"""
vision_node.py — ROS 2 vision pipeline for the FruitNinja UR3e cutting robot.

WHAT IT DOES
============
1. Subscribes to the RealSense D435i colour and depth streams plus camera_info.
2. Runs a YOLOv8 instance-segmentation model (fruit detector) to locate fruit.
3. Runs a second YOLOv8 model (hand/person/tool detector) as a safety E-stop layer.
4. Projects the depth-backed 3-D centroid of each detected fruit from the camera
   frame into the robot base_link frame via TF2, then maps it to a board grid cell
   (A1–N4) on the 14-col × 4-row cutting board.
5. Falls back to HSV-based red-blob detection when neither YOLO model is available,
   so the node is still useful during development before training is complete.

TOPICS PUBLISHED
================
  /fruit_target          (geometry_msgs/PointStamped)  — 3-D centroid in base_link
  /estop                 (std_msgs/Bool)               — True when hand/person seen
  /fruit_detections_debug (sensor_msgs/Image)          — annotated BGR debug image

TOPICS SUBSCRIBED
=================
  /camera/color/image_raw          (sensor_msgs/Image)
  /camera/depth/image_rect_raw     (sensor_msgs/Image)
  /camera/color/camera_info        (sensor_msgs/CameraInfo)

DEPLOYMENT NOTES
================
* Place trained fruit model at:  vision/runs/fruitninja_seg/weights/best.pt
* Place hand detector model at:  vision/models/hand_detector.pt
  (Both paths are relative to the package root — run from ros2_ws/src/FruitNinja.)
* TF2 transform camera_link → base_link must be broadcast by the UR driver.
* Build: add 'vision_node' to setup.py console_scripts entry points.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool

from cv_bridge import CvBridge

import tf2_ros

# ── YOLO (optional at import time — node degrades gracefully) ─────────────────
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

# ── Board geometry constants ──────────────────────────────────────────────────
BOARD_X_ORIGIN  = -0.031          # A1 far-left corner, x in robot base frame (m)
BOARD_Y_ORIGIN  =  0.10           # A1 far-left corner, y in robot base frame (m)
BOARD_WIDTH_M   =  0.73           # total board width  (m)
BOARD_HEIGHT_M  =  0.22           # total board height (m)
GRID_COLS       =  14             # columns A … N
GRID_ROWS       =  4              # rows 1 … 4
CELL_WIDTH_M    =  BOARD_WIDTH_M  / GRID_COLS
CELL_HEIGHT_M   =  BOARD_HEIGHT_M / GRID_ROWS

# Safety class names that trigger E-stop
_ESTOP_CLASSES = {"hand", "person", "tool", "ruler"}

# Annotaion colours (BGR)
_BOX_BGR        = (0, 220, 50)
_LABEL_BGR      = (255, 255, 255)
_FALLBACK_BGR   = (0, 80, 255)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cell_name(col: int, row: int) -> str:
    """Convert 0-based (col, row) to chess-style label, e.g. (0,0) → 'A1'."""
    return f"{chr(ord('A') + col)}{row + 1}"


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _xyz_from_transform(tf_stamped) -> np.ndarray:
    """Extract the 4×4 homogeneous transform from a TransformStamped."""
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation

    # Quaternion → rotation matrix (standard formula)
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    translation = np.array([t.x, t.y, t.z], dtype=np.float64)
    return R, translation


# ── Main node ─────────────────────────────────────────────────────────────────

class VisionNode(Node):
    """Fruit-detection + safety-check vision pipeline for FruitNinja."""

    def __init__(self) -> None:
        super().__init__("vision_node")

        # ── CV bridge ────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Load YOLO models (graceful degradation) ───────────────────────────
        if _HAS_YOLO:
            try:
                self.fruit_model = YOLO("vision/runs/fruitninja_seg/weights/best.pt")
                self.get_logger().info("Fruit YOLO model loaded.")
            except FileNotFoundError:
                self.get_logger().warn(
                    "Fruit YOLO model not found at "
                    "'vision/runs/fruitninja_seg/weights/best.pt'. "
                    "Falling back to HSV detection."
                )
                self.fruit_model = None

            try:
                self.safety_model = YOLO("vision/models/hand_detector.pt")
                self.get_logger().info("Safety (hand) YOLO model loaded.")
            except FileNotFoundError:
                self.get_logger().warn(
                    "Safety YOLO model not found at "
                    "'vision/models/hand_detector.pt'. "
                    "Safety E-stop check is DISABLED."
                )
                self.safety_model = None
        else:
            self.get_logger().warn(
                "ultralytics not installed — both YOLO models unavailable. "
                "Falling back to HSV detection. Install with: pip install ultralytics"
            )
            self.fruit_model  = None
            self.safety_model = None

        # ── State ─────────────────────────────────────────────────────────────
        self.K            : np.ndarray | None = None   # 3×3 camera intrinsic matrix
        self._depth_frame : np.ndarray | None = None   # latest uint16 depth image

        # ── TF2 ───────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            Image, "/camera/color/image_raw",      self._image_cb,  10)
        self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self._depth_cb,  10)
        self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._info_cb,  1)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_target = self.create_publisher(PointStamped, "/fruit_target",           10)
        self._pub_estop  = self.create_publisher(Bool,         "/estop",                  10)
        self._pub_debug  = self.create_publisher(Image,        "/fruit_detections_debug", 10)

        self.get_logger().info("VisionNode ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo) -> None:
        """Store camera intrinsics once and unsubscribe."""
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.get_logger().info(
            f"Camera intrinsics loaded. fx={self.K[0,0]:.1f}  fy={self.K[1,1]:.1f}  "
            f"cx={self.K[0,2]:.1f}  cy={self.K[1,2]:.1f}"
        )

    def _depth_cb(self, msg: Image) -> None:
        """Cache the latest depth frame as a uint16 numpy array."""
        try:
            self._depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Depth frame conversion failed: {exc}")

    def _image_cb(self, msg: Image) -> None:
        """Main processing callback — runs every colour frame."""
        # Need intrinsics and depth before we can do anything useful
        if self.K is None or self._depth_frame is None:
            return

        # Convert ROS Image → OpenCV BGR
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Colour frame conversion failed: {exc}")
            return

        img_h, img_w = img.shape[:2]

        # ── Safety check ─────────────────────────────────────────────────────
        if self.safety_model is not None:
            results = self.safety_model(img, conf=0.7, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = self.safety_model.names.get(cls_id, "").lower()
                    if cls_name in _ESTOP_CLASSES:
                        self.get_logger().error(
                            f"[E-STOP] Detected '{cls_name}' near cutting board — "
                            "halting fruit targeting."
                        )
                        self._pub_estop.publish(Bool(data=True))
                        # Publish unannotated image so operator can see what triggered it
                        self._pub_debug.publish(
                            self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
                        return

        # ── Fruit detection — YOLO path ───────────────────────────────────────
        detection_made = False

        if self.fruit_model is not None:
            results = self.fruit_model(img, conf=0.5, verbose=False)
            for result in results:
                if result.masks is None or result.boxes is None:
                    continue

                # Iterate detections; stop after the first successful 3-D projection
                for mask_data, box in zip(result.masks.data, result.boxes):
                    # Convert mask tensor to uint8 numpy, resize to image size
                    mask_np = mask_data.cpu().numpy().astype(np.uint8)
                    if mask_np.shape[:2] != (img_h, img_w):
                        mask_np = cv2.resize(
                            mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                    # Centroid from moments
                    M = cv2.moments(mask_np, binaryImage=True)
                    if M["m00"] == 0:
                        continue
                    u = int(M["m10"] / M["m00"])
                    v = int(M["m01"] / M["m00"])

                    # Guard against out-of-bounds pixel access
                    u = _clamp(u, 0, img_w - 1)
                    v = _clamp(v, 0, img_h - 1)

                    # Depth in metres
                    depth_raw = self._depth_frame[v, u]
                    if depth_raw == 0:
                        continue          # no depth data at centroid — skip
                    depth_m = float(depth_raw) * 0.001

                    # Back-project to camera frame
                    fx, fy = self.K[0, 0], self.K[1, 1]
                    cx, cy = self.K[0, 2], self.K[1, 2]
                    p_cam = np.array([
                        (u - cx) * depth_m / fx,
                        (v - cy) * depth_m / fy,
                        depth_m,
                    ], dtype=np.float64)

                    # TF2: camera_link → base_link
                    try:
                        tf_stamped = self._tf_buffer.lookup_transform(
                            "base_link", "camera_link",
                            rclpy.time.Time(),
                            Duration(seconds=0.1),
                        )
                    except (tf2_ros.LookupException,
                            tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException) as exc:
                        self.get_logger().warn(f"TF lookup failed: {exc}")
                        continue

                    R, trans = _xyz_from_transform(tf_stamped)
                    p_base   = R @ p_cam + trans

                    # Map to grid cell
                    col = int((p_base[0] - BOARD_X_ORIGIN) / CELL_WIDTH_M)
                    row = int((p_base[1] - BOARD_Y_ORIGIN) / CELL_HEIGHT_M)
                    col = _clamp(col, 0, GRID_COLS - 1)
                    row = _clamp(row, 0, GRID_ROWS - 1)
                    cell_name = _cell_name(col, row)

                    self.get_logger().info(
                        f"Detected fruit at {cell_name}, depth={depth_m:.3f}m  "
                        f"p_base=({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f})"
                    )

                    # Publish 3-D point
                    pt_msg = PointStamped()
                    pt_msg.header.stamp    = self.get_clock().now().to_msg()
                    pt_msg.header.frame_id = "base_link"
                    pt_msg.point.x = float(p_base[0])
                    pt_msg.point.y = float(p_base[1])
                    pt_msg.point.z = float(p_base[2])
                    self._pub_target.publish(pt_msg)

                    # Annotate image: bounding box + cell label
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(img, (x1, y1), (x2, y2), _BOX_BGR, 2)
                    label_text = f"fruit [{cell_name}] {depth_m:.2f}m"
                    cv2.putText(
                        img, label_text, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                    )
                    cv2.putText(
                        img, label_text, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _LABEL_BGR, 1,
                    )
                    cv2.drawMarker(img, (u, v), _BOX_BGR,
                                   cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

                    detection_made = True
                    break   # first successful detection is sufficient
                if detection_made:
                    break

        # ── Fruit detection — HSV fallback ────────────────────────────────────
        if not detection_made:
            self._hsv_fallback(img, img_w, img_h)

        # ── Publish debug image ───────────────────────────────────────────────
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            self._pub_debug.publish(debug_msg)
        except Exception as exc:
            self.get_logger().warn(f"Debug image publish failed: {exc}")

    # ── HSV fallback ──────────────────────────────────────────────────────────

    def _hsv_fallback(self, img: np.ndarray, img_w: int, img_h: int) -> None:
        """
        Simple HSV red-blob detector used when YOLO models are unavailable.

        Projects the image-space centroid to a board cell using a linear
        approximation (camera frame ≈ board frame overhead view).  This is
        intentionally approximate — it is a development stand-in only.
        """
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red wraps around 0/180 in HSV
        mask_lo = cv2.inRange(hsv,
                              np.array([0,   100, 80]),
                              np.array([10,  255, 255]))
        mask_hi = cv2.inRange(hsv,
                              np.array([170, 100, 80]),
                              np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # Largest blob only
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 200:
            return

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        # Camera frame ≈ board frame: map pixel position to grid cell linearly
        col = int((cx / img_w) * GRID_COLS)
        row = int((cy / img_h) * GRID_ROWS)
        col = _clamp(col, 0, GRID_COLS - 1)
        row = _clamp(row, 0, GRID_ROWS - 1)
        cell_name = _cell_name(col, row)

        self.get_logger().info(
            f"[HSV fallback] Red blob at pixel ({cx},{cy}) → cell {cell_name}"
        )

        # Publish an approximate point (Z=0, XY from board geometry only)
        p_x = BOARD_X_ORIGIN + (col + 0.5) * CELL_WIDTH_M
        p_y = BOARD_Y_ORIGIN + (row + 0.5) * CELL_HEIGHT_M
        pt_msg = PointStamped()
        pt_msg.header.stamp    = self.get_clock().now().to_msg()
        pt_msg.header.frame_id = "base_link"
        pt_msg.point.x = p_x
        pt_msg.point.y = p_y
        pt_msg.point.z = 0.0
        self._pub_target.publish(pt_msg)

        # Annotate
        label_text = f"fruit(hsv) [{cell_name}]"
        cv2.rectangle(img, (x, y), (x + w, y + h), _FALLBACK_BGR, 2)
        cv2.putText(
            img, label_text, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
        )
        cv2.putText(
            img, label_text, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _LABEL_BGR, 1,
        )
        cv2.drawMarker(img, (cx, cy), _FALLBACK_BGR,
                       cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
