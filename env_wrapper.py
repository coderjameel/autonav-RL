"""
PerceptDrive - MetaDrive Environment Wrapper
Wraps MetaDrive with a structured perception observation space:
  [raw_frame_features (CNN), yolo_detections, per_object_depths, ego_speed]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# perception constants (must match perception.py)
MAX_OBJECTS    = 10
OBJECT_DIM     = 7
PERCEPTION_DIM = MAX_OBJECTS * OBJECT_DIM   # 70
CNN_DIM        = 256   # CNN feature vector from raw frame
EGO_DIM        = 3     # [speed, steering_angle, heading_diff]
OBS_DIM        = CNN_DIM + PERCEPTION_DIM + EGO_DIM   # 329


class PerceptDriveEnv(gym.Env):
    """
    Observation space (329-d):
        [0:256]   - CNN features from raw camera frame
        [256:326] - YOLO+ZoeDepth structured perception (70-d)
        [326:329] - Ego state: speed, steering, heading_diff

    Action space (3-d continuous):
        [0] steering   ∈ [-1, 1]
        [1] throttle   ∈ [ 0, 1]
        [2] brake      ∈ [ 0, 1]
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        use_perception: bool = True,
        num_scenarios: int = 1000,
        traffic_density: float = 0.1,
        device: str = "cuda",
        render_mode: str = None,
        image_size: tuple = (84, 84),
    ):
        super().__init__()

        self.use_perception = use_perception
        self.device = device
        self.image_size = image_size
        self.render_mode = render_mode
        self._step_count = 0

        # ── Observation & action spaces ────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([ 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # ── CNN feature extractor (small, trainable) ───────────────────────
        import torch
        import torch.nn as nn
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(CNN_DIM), nn.ReLU()
        ).to(device)
        self._cnn_built = False

        # ── Perception pipeline ────────────────────────────────────────────
        if use_perception:
            from perception import PerceptionPipeline
            self.perception = PerceptionPipeline(device=device)
        else:
            self.perception = None

        # ── MetaDrive env (lazy init) ──────────────────────────────────────
        self._metadrive_config = dict(
            use_render=False,
            image_observation=False,   # we handle images ourselves
            norm_pixel=True,
            num_scenarios=num_scenarios,
            traffic_density=traffic_density,
            map="SSSSSSSS",            # 8 straight segments to start
            random_traffic=True,
            accident_prob=0.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=10.0,
            crash_object_penalty=5.0,
            driving_reward=1.0,
            speed_reward=0.1,
            use_lateral_reward=True,
        )
        self._env = None

    def _init_env(self):
        """Lazy-init MetaDrive to avoid import issues at module level."""
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv
            self._env = MetaDriveEnv(self._metadrive_config)
        except Exception as e:
            print(f"[PerceptDrive] MetaDrive unavailable ({e}), using mock env.")
            self._env = None

    # ── Helpers ────────────────────────────────────────────────────────────
    def _get_frame(self) -> np.ndarray:
        """Render current frame as (H, W, 3) uint8."""
        if self._env is None:
            return np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        try:
            frame = self._env.render(mode="top_down", film_size=(200, 200))
            import cv2
            frame = cv2.resize(frame, self.image_size)
            return frame.astype(np.uint8)
        except Exception:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def _cnn_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract CNN features from frame. Returns (CNN_DIM,) float32."""
        import torch
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.cnn(tensor).squeeze(0).cpu().numpy()
        return feat   # (CNN_DIM,)

    def _build_obs(self, md_obs: dict, frame: np.ndarray) -> np.ndarray:
        """Construct full 329-d observation vector."""
        # 1. CNN features from raw frame
        cnn_feat = self._cnn_features(frame)   # (256,)

        # 2. YOLO + ZoeDepth structured perception
        if self.perception is not None:
            perc_feat = self.perception.process(frame)  # (70,)
        else:
            perc_feat = np.zeros(PERCEPTION_DIM, dtype=np.float32)

        # 3. Ego state
        if isinstance(md_obs, dict):
            speed = float(md_obs.get("speed", [0])[0] if hasattr(md_obs.get("speed", 0), "__len__") else md_obs.get("speed", 0))
            steering = float(md_obs.get("steering", [0])[0] if hasattr(md_obs.get("steering", 0), "__len__") else 0)
            heading  = float(md_obs.get("heading_diff", [0])[0] if hasattr(md_obs.get("heading_diff", 0), "__len__") else 0)
        else:
            speed, steering, heading = 0.0, 0.0, 0.0
        ego = np.array([speed, steering, heading], dtype=np.float32)

        return np.concatenate([cnn_feat, perc_feat, ego])   # (329,)

    # ── Gym API ────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._env is None:
            self._init_env()

        self._step_count = 0
        if self._env is not None:
            md_obs, info = self._env.reset()
        else:
            md_obs, info = {}, {}

        frame = self._get_frame()
        obs   = self._build_obs(md_obs, frame)
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # MetaDrive expects [steering, acceleration] where acceleration
        # encodes both throttle and brake sign
        steering  = float(action[0])
        throttle  = float(action[1])
        brake     = float(action[2])

        # Physics coupling: high steering → reduce throttle
        throttle *= (1.0 - 0.4 * abs(steering))
        # Mutual exclusivity
        brake = brake * (1.0 - throttle)

        if self._env is not None:
            md_action = np.array([steering, throttle - brake])
            md_obs, reward, terminated, truncated, info = self._env.step(md_action)
        else:
            # Mock env for testing without MetaDrive
            md_obs   = {}
            reward   = np.random.uniform(-0.1, 1.0)
            terminated = self._step_count >= 200
            truncated  = False
            info       = {}

        # Shaped reward
        reward = self._shape_reward(reward, action, info)

        frame = self._get_frame()
        obs   = self._build_obs(md_obs, frame)
        return obs, reward, terminated, truncated, info

    def _shape_reward(self, base_reward: float, action: np.ndarray, info: dict) -> float:
        r = base_reward
        # Penalise simultaneous throttle + brake
        r -= 0.5 * float(action[1]) * float(action[2])
        # Penalise jerk (large steering)
        r -= 0.3 * abs(float(action[0]))
        # Penalise collisions
        if info.get("crash", False):
            r -= 10.0
        return float(r)

    def render(self):
        return self._get_frame()

    def close(self):
        if self._env is not None:
            self._env.close()
