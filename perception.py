"""
PerceptDrive - Perception Module
YOLOv8 + ZoeDepth frozen front-end that converts raw camera frames
into structured semantic-spatial feature vectors for the PPO policy.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

# ── Constants ──────────────────────────────────────────────────────────────
MAX_OBJECTS   = 10        # max YOLO detections kept per frame
OBJECT_DIM    = 7         # [class_id, conf, cx, cy, w, h, depth_m]
PERCEPTION_DIM = MAX_OBJECTS * OBJECT_DIM  # 70-d structured perception vector


# ── YOLO Wrapper ───────────────────────────────────────────────────────────
class YOLOPerception:
    """
    Wraps YOLOv8n (nano) for fast inference.
    Returns top-K detections sorted by confidence.
    """
    COCO_DRIVE_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", 9: "traffic_light", 11: "stop_sign"
    }

    def __init__(self, model_size: str = "yolov8n", device: str = "cuda"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{model_size}.pt")
            self.model.to(device)
            self.available = True
        except Exception:
            self.available = False
        self.device = device

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_rgb: (H, W, 3) uint8 image
        Returns:
            detections: (N, 6) array [class_id, conf, cx_norm, cy_norm, w_norm, h_norm]
                        normalized to [0,1] by image dims, top-MAX_OBJECTS by conf
        """
        if not self.available:
            return np.zeros((MAX_OBJECTS, 6), dtype=np.float32)

        H, W = frame_rgb.shape[:2]
        results = self.model(frame_rgb, verbose=False, conf=0.25)[0]
        boxes   = results.boxes

        if boxes is None or len(boxes) == 0:
            return np.zeros((MAX_OBJECTS, 6), dtype=np.float32)

        dets = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            w  = (x2 - x1) / W
            h  = (y2 - y1) / H
            dets.append([cls_id, conf, cx, cy, w, h])

        dets = sorted(dets, key=lambda d: d[1], reverse=True)[:MAX_OBJECTS]
        out  = np.zeros((MAX_OBJECTS, 6), dtype=np.float32)
        for i, d in enumerate(dets):
            out[i] = d
        return out   # (MAX_OBJECTS, 6)


# ── ZoeDepth Wrapper ───────────────────────────────────────────────────────
class ZoeDepthPerception:
    """
    ZoeDepth metric monocular depth estimator.
    Returns per-detection depth in meters.
    """
    def __init__(self, device: str = "cuda"):
        try:
            self.model = torch.hub.load(
                "isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True
            ).to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.available = True
        except Exception:
            self.available = False
        self.device = device

    def get_depth_map(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Returns (H, W) metric depth map in metres."""
        if not self.available:
            return np.ones(frame_rgb.shape[:2], dtype=np.float32) * 10.0

        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((384, 512)),
        ])
        tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.model.infer(tensor).squeeze().cpu().numpy()
        # resize back to original
        import cv2
        depth = cv2.resize(depth, (frame_rgb.shape[1], frame_rgb.shape[0]))
        return depth   # (H, W) float32, metres

    def get_object_depths(
        self, depth_map: np.ndarray, detections: np.ndarray, img_shape: tuple
    ) -> np.ndarray:
        """
        For each detection, return median depth within bounding box.
        Args:
            depth_map:  (H, W)
            detections: (MAX_OBJECTS, 6) normalised
            img_shape:  (H, W)
        Returns:
            depths: (MAX_OBJECTS,) float32
        """
        H, W = img_shape
        depths = np.zeros(MAX_OBJECTS, dtype=np.float32)
        for i, det in enumerate(detections):
            if det[1] == 0:   # no detection
                depths[i] = 50.0
                continue
            cx, cy, bw, bh = det[2], det[3], det[4], det[5]
            x1 = max(0, int((cx - bw/2) * W))
            x2 = min(W, int((cx + bw/2) * W))
            y1 = max(0, int((cy - bh/2) * H))
            y2 = min(H, int((cy + bh/2) * H))
            region = depth_map[y1:y2, x1:x2]
            depths[i] = float(np.median(region)) if region.size > 0 else 50.0
        return depths


# ── Unified Perception Pipeline ────────────────────────────────────────────
class PerceptionPipeline:
    """
    Combines YOLOv8 + ZoeDepth into a single structured observation vector.

    Output shape: (PERCEPTION_DIM,) = (MAX_OBJECTS * OBJECT_DIM,) = (70,)
    Each object slot: [class_id_norm, conf, cx, cy, w, h, depth_norm]
    """

    def __init__(self, device: str = "cuda"):
        self.yolo  = YOLOPerception(device=device)
        self.depth = ZoeDepthPerception(device=device)
        self.device = device

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Returns structured perception vector (70,) float32.
        Safe to call even if models unavailable (returns zeros).
        """
        # 1. YOLO detections
        dets = self.yolo.detect(frame_rgb)              # (10, 6)

        # 2. Depth map + per-object depth
        depth_map = self.depth.get_depth_map(frame_rgb)
        obj_depths = self.depth.get_object_depths(
            depth_map, dets, frame_rgb.shape[:2]
        )                                               # (10,)

        # 3. Normalise depth to [0,1] (clip at 80 m)
        depth_norm = np.clip(obj_depths / 80.0, 0.0, 1.0)   # (10,)

        # 4. Normalise class_id to [0,1]  (80 COCO classes)
        class_norm = dets[:, 0:1] / 80.0               # (10, 1)

        # 5. Concatenate → [class_norm, conf, cx, cy, w, h, depth_norm]
        perception = np.concatenate([
            class_norm,                                 # (10,1)
            dets[:, 1:],                                # (10,5) conf,cx,cy,w,h
            depth_norm[:, None]                         # (10,1)
        ], axis=1)                                      # (10,7)

        return perception.flatten().astype(np.float32) # (70,)

    @property
    def output_dim(self) -> int:
        return PERCEPTION_DIM   # 70
