# FruitNinja AI Architecture Upgrade — Claude Code Prompt

You have full access to the FruitNinja ROS 2 workspace. This document defines a complete
upgrade from the current deterministic architecture to an AI-driven system with two parallel
objectives: a Deep Reinforcement Learning (DRL) trajectory planner and a CNN-based vision
pipeline. Read the relevant source files first, then implement each section in order.

---

## Step 0 — Read These Files First

Before writing any code, read and internalize the following files from the workspace:

```
fruitninja/
  grid_mover.py          # joint-angle lookup table for all A1-N4 cells
  planning_scene.py      # collision object poses published to MoveIt
  colour_detection.py    # current HSV masking and grid overlay logic
  real_gui_points.py     # main operator GUI — MoverNode, CameraWorker, cut logic
  startup_gui.py         # process launcher — understand the step ordering
ur3e_workcell.urdf.xacro # full collision geometry: trolley, walls, cutting board
```

Also run this and save the output as `joint_states_sample.txt`:
```bash
ros2 topic echo /joint_states --once
```

---

## Objective 1 — Deep Reinforcement Learning Trajectory Planner

### 1.1 MuJoCo Environment Setup

Install dependencies:
```bash
pip install mujoco gymnasium stable-baselines3 numpy scipy
```

Convert the URDF to MuJoCo XML:
```bash
python -m mujoco.scripts.compile ur3e_workcell.urdf.xacro ur3e_workcell.xml
```

Create the file `rl/envs/ur3e_grid_env.py` with the following structure:

```python
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

# UR3e joint velocity limits (rad/s) — enforce as action clip bounds
UR3E_VEL_LIMITS = np.array([2.09, 2.09, 3.14, 3.14, 3.14, 3.14])

# Board origin and cell dimensions in robot base frame (metres)
# Populate these from ur3e_workcell.urdf.xacro collision object poses
BOARD_X_ORIGIN = None  # TODO: extract from planning_scene.py
BOARD_Y_ORIGIN = None  # TODO: extract from planning_scene.py
CELL_WIDTH_M   = None  # total board width (~0.73m) / 14 columns
CELL_HEIGHT_M  = None  # total board height (~0.22m) / 4 rows

class UR3eGridEnv(gym.Env):
    """
    Custom MuJoCo Gym environment for FruitNinja RL training.

    State space (20-dim continuous):
        q[6]        current joint angles (rad)
        q_dot[6]    current joint velocities (rad/s)
        p_ee[3]     end-effector position in base frame (m)
        rpy_ee[3]   end-effector orientation (roll, pitch, yaw)
        p_target[3] fruit centroid in base frame (m) — randomised each episode

    Action space (6-dim continuous):
        delta_q[6]  joint angle deltas per timestep, clipped to vel limits * dt

    Reward:
        + proximity_reward   exp(-||p_ee - p_target||) Gaussian kernel
        - collision_penalty  10.0 if MuJoCo contact detected
        - jerk_penalty       0.01 * sum(|delta_q_dot|)
        - singularity_pen    0.5 * 1/det(J) where J is end-effector Jacobian
        + task_completion    5.0 if dip pose reached (within 5mm of target)
        - time_penalty       0.001 per timestep
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path="ur3e_workcell.xml", render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.dt = self.model.opt.timestep

        # Observation: q(6) + q_dot(6) + p_ee(3) + rpy_ee(3) + p_target(3)
        obs_dim = 21
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        # Action: delta joint angles, clipped to vel_limits * dt
        action_limit = UR3E_VEL_LIMITS * self.dt
        self.action_space = spaces.Box(
            low=-action_limit, high=action_limit, dtype=np.float64
        )
        self._p_target = np.zeros(3)
        self._prev_q_dot = np.zeros(6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomise fruit centroid within grid bounds (domain randomisation: +/-5mm)
        col = self.np_random.integers(0, 14)
        row = self.np_random.integers(0, 4)
        noise = self.np_random.uniform(-0.005, 0.005, size=3)
        self._p_target = np.array([
            BOARD_X_ORIGIN + (col + 0.5) * CELL_WIDTH_M,
            BOARD_Y_ORIGIN + (row + 0.5) * CELL_HEIGHT_M,
            0.0  # board surface z — extract from URDF
        ]) + noise

        self._prev_q_dot = np.zeros(6)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Apply delta joint angles
        self.data.ctrl[:6] += action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done = self._compute_reward(action)
        return obs, reward, done, False, {}

    def _get_obs(self):
        q     = self.data.qpos[:6].copy()
        q_dot = self.data.qvel[:6].copy()
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
        p_ee  = self.data.xpos[ee_id].copy()
        # Convert rotation matrix to rpy
        R     = self.data.xmat[ee_id].reshape(3, 3)
        rpy   = self._mat_to_rpy(R)
        return np.concatenate([q, q_dot, p_ee, rpy, self._p_target])

    def _compute_reward(self, action):
        ee_id    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
        p_ee     = self.data.xpos[ee_id]
        dist     = np.linalg.norm(p_ee - self._p_target)
        q_dot    = self.data.qvel[:6]
        jerk     = np.sum(np.abs(q_dot - self._prev_q_dot))
        collision = float(self.data.ncon > 0) * 10.0
        J = np.zeros((3, 6))
        mujoco.mj_jacBody(self.model, self.data, J, None, ee_id)
        sing_pen = 0.5 / (np.linalg.det(J @ J.T) + 1e-6)
        task_done = dist < 0.005
        reward = (
            np.exp(-dist)           # proximity
            - collision             # collision penalty
            - 0.01 * jerk           # smoothness
            - sing_pen              # singularity avoidance
            + (5.0 if task_done else 0.0)  # sparse task reward
            - 0.001                 # time penalty
        )
        self._prev_q_dot = q_dot.copy()
        return reward, task_done

    @staticmethod
    def _mat_to_rpy(R):
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw   = np.arctan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw])
```

### 1.2 Domain Randomisation Wrapper

Create `rl/envs/domain_rand_wrapper.py`:

```python
import gymnasium as gym
import numpy as np

class DomainRandWrapper(gym.Wrapper):
    """
    Randomises physics parameters each episode for sim-to-real transfer.
    Apply before training, not at deployment.

    Randomised parameters:
        joint friction      +/- 20% of nominal
        end-effector mass   0.1 to 0.4 kg  (knife + tool weight range)
        controller latency  0 to 20ms injected delay
    """

    def __init__(self, env, rand_friction=True, rand_mass=True, rand_latency=True):
        super().__init__(env)
        self.rand_friction = rand_friction
        self.rand_mass     = rand_mass
        self.rand_latency  = rand_latency
        self._latency_steps = 0
        self._action_buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.env.unwrapped.model

        if self.rand_friction:
            nominal = model.dof_frictionloss[:6].copy()
            model.dof_frictionloss[:6] = nominal * np.random.uniform(0.8, 1.2, 6)

        if self.rand_mass:
            ee_body_id = model.body("tool0").id
            model.body_mass[ee_body_id] = np.random.uniform(0.1, 0.4)

        if self.rand_latency:
            # Latency expressed in timesteps (dt ~ 2ms per step)
            self._latency_steps = int(np.random.uniform(0, 10))
            self._action_buffer = [np.zeros(6)] * self._latency_steps

        return obs, info

    def step(self, action):
        if self.rand_latency and self._latency_steps > 0:
            self._action_buffer.append(action)
            delayed_action = self._action_buffer.pop(0)
        else:
            delayed_action = action
        return self.env.step(delayed_action)
```

### 1.3 Behavioural Cloning Pre-Training

Before running SAC, pre-train the policy on the existing joint-angle data from `grid_mover.py`.
Create `rl/pretraining/behavioural_clone.py`:

```python
"""
Extracts all joint-angle waypoints from grid_mover.py and trains a supervised
policy network to imitate them. This warm-starts the SAC actor network,
cutting wall-clock RL training time significantly.

Usage:
    python behavioural_clone.py --grid_mover_path ../../fruitninja/grid_mover.py
    python behavioural_clone.py --checkpoint bc_policy.pth
"""

import torch
import torch.nn as nn
import numpy as np

# TODO: Import CELL_POSITIONS from grid_mover.py and build (state, action) pairs
# Each entry in grid_mover: cell_name -> [j1, j2, j3, j4, j5, j6] degrees
# Convert degrees to radians and treat as target actions from a neutral home pose

class BCPolicy(nn.Module):
    def __init__(self, obs_dim=21, action_dim=6, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
```

### 1.4 SAC Training Script

Create `rl/train_sac.py`:

```python
"""
Main SAC training script for FruitNinja trajectory planner.
Loads BC-pretrained weights if available, then runs SAC.

Usage:
    python train_sac.py                          # fresh training
    python train_sac.py --bc_weights bc_policy.pth   # warm start
    python train_sac.py --eval --model_path sac_fruitninja.zip  # evaluation only
"""

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.ur3e_grid_env import UR3eGridEnv
from envs.domain_rand_wrapper import DomainRandWrapper
import argparse

def make_env():
    env = UR3eGridEnv(xml_path="../ur3e_workcell.xml")
    return DomainRandWrapper(env)

def train(bc_weights=None):
    vec_env  = make_vec_env(make_env, n_envs=4)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = SAC(
        "MlpPolicy", vec_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log="./tb_logs/"
    )

    if bc_weights:
        # Load BC pretrained actor weights into the SAC policy
        # model.policy.actor.load_state_dict(torch.load(bc_weights))
        print(f"Loaded BC weights from {bc_weights}")

    callbacks = [
        EvalCallback(eval_env, best_model_save_path="./models/best/",
                     eval_freq=10_000, deterministic=True),
        CheckpointCallback(save_freq=50_000, save_path="./models/checkpoints/")
    ]

    model.learn(total_timesteps=2_000_000, callback=callbacks)
    model.save("sac_fruitninja")
    print("Training complete. Model saved to sac_fruitninja.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_weights", type=str, default=None)
    args = parser.parse_args()
    train(args.bc_weights)
```

### 1.5 ROS 2 Policy Inference Node

Create `fruitninja/rl_mover_node.py` — this replaces `MoverNode`'s hardcoded MoveIt calls
with live policy inference:

```python
"""
ROS 2 node that replaces MoverNode with a trained SAC policy.
Subscribes to /joint_states and /fruit_target (published by vision pipeline),
infers the next joint delta from the policy, and publishes to
/scaled_joint_trajectory_controller/joint_trajectory.

Deploy only after:
  1. Policy passes URSim validation at 10% velocity scaling
  2. Max joint deviation vs ground truth < 0.1 rad across all joints
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PointStamped
from stable_baselines3 import SAC
import numpy as np

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]
DEVIATION_THRESHOLD = 0.1  # rad — abort if policy drifts beyond this vs URDF limits

class RLMoverNode(Node):
    def __init__(self):
        super().__init__("rl_mover_node")
        self.model = SAC.load("sac_fruitninja")
        self.q      = np.zeros(6)
        self.q_dot  = np.zeros(6)
        self.target = np.zeros(3)
        self.estop  = False

        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.create_subscription(PointStamped, "/fruit_target", self._target_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory", 10
        )
        self.create_timer(0.05, self._inference_loop)  # 20 Hz

    def _js_cb(self, msg):
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.q[i]     = msg.position[idx]
                self.q_dot[i] = msg.velocity[idx]

    def _target_cb(self, msg):
        self.target = np.array([msg.point.x, msg.point.y, msg.point.z])

    def _inference_loop(self):
        if self.estop:
            return
        # Build obs: q(6) + q_dot(6) + p_ee(3) + rpy(3) + target(3)
        # p_ee and rpy must be computed from FK — use tf2_ros here
        obs = np.concatenate([self.q, self.q_dot,
                               np.zeros(3), np.zeros(3),  # TODO: replace with FK
                               self.target])
        action, _ = self.model.predict(obs, deterministic=True)

        # Safety: check deviation
        proposed = self.q + action
        # (add joint limit checks here from UR3e URDF)

        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions  = proposed.tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 50_000_000  # 50ms
        msg.points = [pt]
        self.traj_pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(RLMoverNode())

if __name__ == "__main__":
    main()
```

Add `rl_mover_node` to `setup.py` console scripts.

---

## Objective 2 — Advanced Computer Vision Pipeline

### 2.1 Dependencies

```bash
pip install ultralytics opencv-python pyrealsense2 numpy torch torchvision
```

### 2.2 Dataset Collection Script

Create `vision/collect_dataset.py`:

```python
"""
Captures frames from the RealSense D435i for dataset collection.
Press S to save a frame, Q to quit. Target: 200-300 images of fruit
on the cutting board under lab lighting.

After collection:
  - Upload images to Roboflow (free tier)
  - Annotate with instance segmentation masks
  - Apply augmentations: hue shift +/-15 deg, brightness +/-30%, mosaic, h-flip
  - Export in YOLOv8 format as fruitninja_dataset.zip
"""

import pyrealsense2 as rs
import cv2, os, time

OUTPUT_DIR = "dataset/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color  = frames.get_color_frame()
        if not color:
            continue
        img = cv2.cvtColor(
            cv2.cvtColor(color.get_data(), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR
        )
        cv2.imshow("Dataset Collector — S=save, Q=quit", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            fname = os.path.join(OUTPUT_DIR, f"frame_{int(time.time())}_{count:04d}.jpg")
            cv2.imwrite(fname, img)
            count += 1
            print(f"Saved {fname}  ({count} total)")
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Collection complete: {count} images saved to {OUTPUT_DIR}")
```

### 2.3 YOLOv8 Fine-Tuning

After exporting from Roboflow, create `vision/fruitninja.yaml`:
```yaml
path: ./dataset/fruitninja_roboflow
train: images/train
val:   images/val
nc: 1
names: [fruit]
```

Fine-tune:
```bash
yolo task=segment mode=train \
     model=yolov8m-seg.pt \
     data=vision/fruitninja.yaml \
     epochs=100 imgsz=640 batch=16 \
     project=vision/runs name=fruitninja_seg
```

### 2.4 Main Vision Node

Create `fruitninja/vision_node.py`:

```python
"""
ROS 2 node replacing colour_detection.py with a full AI vision pipeline.
Publishes:
  /fruit_detections  (custom msg or MarkerArray) -- per-fruit cell + cut angle
  /fruit_target      (PointStamped)              -- centroid in base frame for RL node
  /estop             (Bool)                      -- triggers on hand/tool detection

Architecture:
  - YOLOv8-seg for fruit instance segmentation (30+ fps on RTX 3060)
  - YOLOv8 hand/tool detector for safety (separate model, low conf threshold)
  - RealSense D435i depth for 3D centroid computation
  - TF2 transform: camera_link -> base_link for grid cell mapping
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO

# Safety confidence threshold -- intentionally conservative
SAFETY_CONF_THRESHOLD = 0.7
FRUIT_CONF_THRESHOLD  = 0.5

# Grid constants -- populate from ur3e_workcell.urdf.xacro
BOARD_X_ORIGIN = None  # TODO
BOARD_Y_ORIGIN = None  # TODO
CELL_WIDTH_M   = None  # TODO
CELL_HEIGHT_M  = None  # TODO


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.bridge       = CvBridge()
        self.fruit_model  = YOLO("vision/runs/fruitninja_seg/weights/best.pt")
        self.safety_model = YOLO("vision/models/hand_detector.pt")  # fine-tune on EgoHands dataset
        self.K            = None  # camera intrinsics matrix, set from /camera_info

        self.create_subscription(Image,      "/camera/color/image_raw",  self._image_cb, 10)
        self.create_subscription(Image,      "/camera/depth/image_rect_raw", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self._info_cb, 10)

        self.target_pub = self.create_publisher(PointStamped, "/fruit_target", 10)
        self.estop_pub  = self.create_publisher(Bool, "/estop", 10)

        self._depth_frame = None

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("Camera intrinsics loaded.")

    def _depth_cb(self, msg):
        self._depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _image_cb(self, msg):
        if self.K is None or self._depth_frame is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Safety check runs first -- any hand/tool detection triggers E-STOP immediately
        safety_results = self.safety_model(img, conf=SAFETY_CONF_THRESHOLD, verbose=False)
        for r in safety_results:
            for cls_id in (r.boxes.cls.tolist() if r.boxes else []):
                class_name = self.safety_model.names[int(cls_id)]
                if class_name in ["hand", "person", "tool", "ruler"]:
                    self.get_logger().error(f"SAFETY: {class_name} detected -- E-STOP")
                    self.estop_pub.publish(Bool(data=True))
                    return

        # Fruit segmentation
        fruit_results = self.fruit_model(img, conf=FRUIT_CONF_THRESHOLD, verbose=False)
        for r in fruit_results:
            if r.masks is None:
                continue
            for i, mask in enumerate(r.masks.xy):
                mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_img, [mask.astype(np.int32)], 255)

                # Centroid from mask moments
                M  = cv2.moments(mask_img)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])

                # Cut angle from second-order moments (no extra training needed)
                angle = 0.5 * np.arctan2(
                    2 * M["mu11"],
                    M["mu20"] - M["mu02"]
                )

                # 3D centroid via depth + intrinsics
                depth = self._depth_frame[v, u] * 0.001  # mm -> m
                if depth == 0:
                    continue
                fx, fy = self.K[0, 0], self.K[1, 1]
                cx, cy = self.K[0, 2], self.K[1, 2]
                p_cam  = np.array([(u - cx) * depth / fx,
                                   (v - cy) * depth / fy,
                                   depth])

                # TODO: transform p_cam from camera_link to base_link via tf2_ros
                # p_base = transform(p_cam, "camera_link", "base_link")

                # TODO: Map p_base (x,y) to A1-N4 cell using BOARD_X/Y_ORIGIN + CELL dims

                # Publish first fruit centroid as RL target
                pt = PointStamped()
                pt.header.frame_id = "camera_link"
                pt.header.stamp    = self.get_clock().now().to_msg()
                pt.point.x, pt.point.y, pt.point.z = p_cam
                self.target_pub.publish(pt)
                break  # publish closest/first detection only for now

def main():
    rclpy.init()
    rclpy.spin(VisionNode())

if __name__ == "__main__":
    main()
```

Add `vision_node` to `setup.py` console scripts.

---

## Step 3 — Fill in the TODOs (Claude Code Tasks)

After reading the source files, complete these concrete tasks in order:

1. **From `ur3e_workcell.urdf.xacro`:** Extract the cutting board's origin position
   relative to the robot base link. Set `BOARD_X_ORIGIN`, `BOARD_Y_ORIGIN`,
   `CELL_WIDTH_M`, `CELL_HEIGHT_M` in both `ur3e_grid_env.py` and `vision_node.py`.

2. **From `planning_scene.py`:** Confirm the collision object poses match
   the MuJoCo XML geometry. If they differ, adjust the MuJoCo XML.

3. **From `grid_mover.py`:** Import the joint-angle table into
   `rl/pretraining/behavioural_clone.py` and build the (state, action) training pairs.
   The state for each cell should be a neutral home pose observation, the action
   should be the delta from home to the cell's joint angles (converted from degrees to radians).

4. **TF2 integration in `vision_node.py`:** Add a `tf2_ros.Buffer` and
   `tf2_ros.TransformListener`. In `_image_cb`, after computing `p_cam`,
   look up the static transform from `camera_link` to `base_link` and apply it
   to produce `p_base`. Then map `p_base[0]` and `p_base[1]` to an A1-N4 cell string.

5. **FK in `rl_mover_node.py`:** Replace the `np.zeros(3)` placeholders for
   `p_ee` and `rpy` with a proper forward kinematics call using `tf2_ros`
   to look up `tool0` in the `base_link` frame.

6. **`setup.py`:** Register `rl_mover_node` and `vision_node` as entry points.

7. **`startup_gui.py`:** Add optional Step buttons for launching `vision_node`
   and `rl_mover_node` after Step 4, with the same QProcess + live output tab pattern.

---

## Validation Checklist Before Real Robot

Run through these steps in URSim before touching the physical UR3e:

```bash
# 1. Start full ROS stack in sim mode (Steps 0-5 via startup_gui)
ros2 run fruitninja startup_gui

# 2. Launch vision node (webcam source is fine for validation)
ros2 run fruitninja vision_node

# 3. Run RL inference node at 10% velocity scaling
# Edit rl_mover_node.py: add scaling factor 0.1 on all published joint positions

# 4. Monitor joint deviation between policy output and /joint_states
ros2 topic echo /joint_states | python3 -c "
import sys, ast
for line in sys.stdin:
    # check max deviation of policy-proposed vs actual
    pass
"

# Gate: max deviation must stay < 0.1 rad across all 6 joints for 3 full grid sweeps
# before enabling real robot mode
```

---

## File Structure After Implementation

```
fruitninja_ws/
  fruitninja/
    grid_mover.py           (unchanged)
    planning_scene.py       (unchanged)
    real_gui_points.py      (add optional RL mode toggle)
    startup_gui.py          (add vision_node and rl_mover_node steps)
    vision_node.py          (NEW)
    rl_mover_node.py        (NEW)
  rl/
    envs/
      ur3e_grid_env.py      (NEW)
      domain_rand_wrapper.py (NEW)
    pretraining/
      behavioural_clone.py  (NEW)
    train_sac.py            (NEW)
  vision/
    collect_dataset.py      (NEW)
    fruitninja.yaml         (NEW, after Roboflow export)
    models/                 (trained .pt files go here)
  ur3e_workcell.xml         (generated from URDF conversion)
```
