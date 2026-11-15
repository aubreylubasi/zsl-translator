"""Hand + body tracking: turns a camera frame into landmark numbers.

Only numbers come out of here. No image or video is ever saved, which keeps
the people who sign for the project anonymous.

Works with both MediaPipe styles:
  * new MediaPipe (0.10.30+ / 1.x): "tasks" API, downloads two small model
    files (~10 MB) into ./models the first time it runs
  * older MediaPipe (<= 0.10.21): built-in "solutions" API, nothing to download
"""
import os
import ssl
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_DRAW = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]  # shoulders + arms


@dataclass
class Landmarks:
    """What the tracker saw in one frame (all coordinates 0..1 of the image)."""
    hands: dict = field(default_factory=dict)   # "Right"/"Left" -> (21, 3) array
    pose: np.ndarray = None                      # (33, 3) array or None
    aspect: float = 16 / 9                       # image width / height


def _download(name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    url = MODEL_URLS[name]
    print(f"Downloading {name} (one time only)...")
    try:
        import certifi  # macOS Python often lacks root certificates
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    tmp = path + ".part"
    with urllib.request.urlopen(url, context=ctx) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, path)
    return path


class Tracker:
    def __init__(self, backend="auto", num_hands=2):
        if backend == "auto":
            backend = "legacy" if hasattr(mp, "solutions") else "tasks"
        self.backend = backend
        self._last_ts = -1
        if backend == "tasks":
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            self._vision = vision
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_download("hand_landmarker.task")),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                ))
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_download("pose_landmarker_lite.task")),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                ))
        elif backend == "legacy":
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=num_hands, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def process(self, frame_bgr, timestamp_ms=None):
        """Return Landmarks for one BGR frame (as read by OpenCV)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = Landmarks(aspect=w / h)
        raw = []  # (label, score, (21,3) array)

        if self.backend == "tasks":
            ts = int(timestamp_ms if timestamp_ms is not None else self._last_ts + 33)
            ts = max(ts, self._last_ts + 1)  # MediaPipe needs increasing timestamps
            self._last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hr = self._hands.detect_for_video(image, ts)
            for lms, cats in zip(hr.hand_landmarks, hr.handedness):
                arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                raw.append((cats[0].category_name, cats[0].score, arr))
            pr = self._pose.detect_for_video(image, ts)
            if pr.pose_landmarks:
                out.pose = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks[0]], dtype=np.float32)
        else:
            rgb.flags.writeable = False
            hr = self._hands.process(rgb)
            if hr.multi_hand_landmarks:
                for lms, hd in zip(hr.multi_hand_landmarks, hr.multi_handedness):
                    arr = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
                    c = hd.classification[0]
                    raw.append((c.label, c.score, arr))
            pr = self._pose.process(rgb)
            if pr.pose_landmarks:
                out.pose = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks.landmark], dtype=np.float32)

        # One hand per slot; if both got the same label, the weaker one takes the other slot.
        raw.sort(key=lambda r: -r[1])
        for label, _, arr in raw:
            if label not in out.hands:
                out.hands[label] = arr
            else:
                other = "Left" if label == "Right" else "Right"
                out.hands.setdefault(other, arr)
        return out

    def close(self):
        for m in (self._hands, self._pose):
            try:
                m.close()
            except Exception:
                pass


def draw(frame_bgr, lm):
    """Draw the tracked skeleton onto a frame (in place)."""
    h, w = frame_bgr.shape[:2]
    if lm.pose is not None:
        pts = [(int(x * w), int(y * h)) for x, y, _ in lm.pose]
        for a, b in POSE_DRAW:
            cv2.line(frame_bgr, pts[a], pts[b], (200, 200, 200), 2)
    colours = {"Right": (0, 200, 255), "Left": (255, 120, 0)}
    for label, arr in lm.hands.items():
        pts = [(int(x * w), int(y * h)) for x, y, _ in arr]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], colours.get(label, (0, 255, 0)), 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 3, (255, 255, 255), -1)
    return frame_bgr
