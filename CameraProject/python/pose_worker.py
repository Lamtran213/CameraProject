import argparse
import base64
import json
import math
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.25
TRACK_DISTANCE_THRESHOLD = 90.0


class SimpleTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks = {}

    def update(self, people):
        assigned = set()

        for person in people:
            center = person["center"]
            best_id = None
            best_distance = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in assigned:
                    continue

                prev_center = track["last_center"]
                distance = math.dist(center, prev_center)
                if distance < best_distance and distance <= TRACK_DISTANCE_THRESHOLD:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "last_center": center,
                    "center_history": deque(maxlen=12),
                    "wrist_left_history": deque(maxlen=12),
                    "wrist_right_history": deque(maxlen=12),
                }

            self.tracks[best_id]["last_center"] = center
            self.tracks[best_id]["center_history"].append(center)
            assigned.add(best_id)
            person["track_id"] = best_id

        stale_ids = [track_id for track_id in self.tracks if track_id not in assigned]
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)


def is_valid(kp):
    return kp[2] >= CONF_THRESHOLD


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle(a, b, c):
    ab = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    cb = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)

    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return None

    cosine = np.clip(np.dot(ab, cb) / (norm_ab * norm_cb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def movement_amplitude(history):
    if len(history) < 3:
        return 0.0

    xs = [item[0] for item in history]
    ys = [item[1] for item in history]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def classify_pose(keypoints, bbox, track):
    # COCO keypoint index
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    bbox_w = max(1.0, float(bbox[2] - bbox[0]))
    bbox_h = max(1.0, float(bbox[3] - bbox[1]))
    aspect_ratio = bbox_w / bbox_h

    shoulder_mid = None
    hip_mid = None
    if is_valid(left_shoulder) and is_valid(right_shoulder):
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
    if is_valid(left_hip) and is_valid(right_hip):
        hip_mid = midpoint(left_hip, right_hip)

    torso_horizontal = False
    torso_angle = None
    if shoulder_mid and hip_mid:
        dy = hip_mid[1] - shoulder_mid[1]
        dx = hip_mid[0] - shoulder_mid[0]
        torso_angle = abs(math.degrees(math.atan2(dy, dx)))
        torso_horizontal = torso_angle < 35.0 or torso_angle > 145.0

    knee_angles = []
    if is_valid(left_hip) and is_valid(left_knee) and is_valid(left_ankle):
        val = angle(left_hip, left_knee, left_ankle)
        if val is not None:
            knee_angles.append(val)
    if is_valid(right_hip) and is_valid(right_knee) and is_valid(right_ankle):
        val = angle(right_hip, right_knee, right_ankle)
        if val is not None:
            knee_angles.append(val)

    is_knee_bent = len(knee_angles) > 0 and np.mean(knee_angles) < 135.0

    center_history = track["center_history"]
    center_drop = 0.0
    center_movement = movement_amplitude(center_history)
    if len(center_history) >= 2:
        center_drop = center_history[-1][1] - center_history[-2][1]

    if is_valid(left_wrist):
        track["wrist_left_history"].append((left_wrist[0], left_wrist[1]))
    if is_valid(right_wrist):
        track["wrist_right_history"].append((right_wrist[0], right_wrist[1]))

    left_wrist_move = movement_amplitude(track["wrist_left_history"])
    right_wrist_move = movement_amplitude(track["wrist_right_history"])

    wrist_above_shoulder = False
    if shoulder_mid:
        if is_valid(left_wrist) and left_wrist[1] < shoulder_mid[1] - 10:
            wrist_above_shoulder = True
        if is_valid(right_wrist) and right_wrist[1] < shoulder_mid[1] - 10:
            wrist_above_shoulder = True

    is_waving = wrist_above_shoulder and (left_wrist_move > 30 or right_wrist_move > 30)

    is_sitting = False
    if shoulder_mid and hip_mid:
        torso_height = hip_mid[1] - shoulder_mid[1]
        is_sitting = torso_height > 25 and is_knee_bent and not torso_horizontal

    is_fall_like = torso_horizontal and aspect_ratio > 1.2

    # Te: dang nga nhanh xuong theo chieu doc + than nguoi nam ngang
    if is_fall_like and center_drop > 20:
        return "te"

    # Nga: than nguoi nam ngang, nhung khong phat hien roi nhanh
    if is_fall_like:
        return "nga"

    if is_sitting:
        return "ngoi"

    # Dung: mac dinh
    return "dung"


def decode_image(image_b64):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def init_model(model_path, conf, imgsz, device):
    # ONNX pose model is typically several MB. Very small files are likely broken.
    suspicious_onnx = (
        model_path.lower().endswith(".onnx")
        and os.path.exists(model_path)
        and os.path.getsize(model_path) < 1_000_000
    )

    if suspicious_onnx:
        print(
            f"warning: ONNX co kich thuoc bat thuong ({os.path.getsize(model_path)} bytes), se fallback sang .pt",
            file=sys.stderr,
            flush=True,
        )

    candidate_paths = [model_path]
    if model_path.lower().endswith(".onnx"):
        local_pt = os.path.splitext(model_path)[0] + ".pt"
        if os.path.exists(local_pt):
            candidate_paths.append(local_pt)
        candidate_paths.append("yolov8n-pose.pt")

    last_error = None

    for candidate in candidate_paths:
        try:
            model = YOLO(candidate)

            # Validate model early to avoid repeated runtime protobuf errors per frame.
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            _ = model.predict(
                source=dummy,
                conf=conf,
                imgsz=imgsz,
                device=device,
                classes=[0],
                verbose=False,
            )

            if candidate != model_path:
                print(
                    f"fallback model dang su dung: {candidate}",
                    file=sys.stderr,
                    flush=True,
                )

            return model
        except Exception as ex:
            last_error = ex
            print(
                f"model init failed with '{candidate}': {ex}",
                file=sys.stderr,
                flush=True,
            )

    raise RuntimeError(f"Khong the khoi tao model: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = init_model(args.model, args.conf, args.imgsz, args.device)
    tracker = SimpleTracker()

    print("YOLOv8 pose worker started", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
            frame_id = int(payload.get("frame_id", payload.get("FrameId", 0)))
            image_b64 = payload.get("image_b64", payload.get("ImageB64"))
            if not image_b64:
                continue

            image = decode_image(image_b64)
            if image is None:
                continue

            result = model.predict(
                source=image,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                classes=[0],
                verbose=False,
            )[0]

            response = {"frameId": frame_id, "people": []}

            if result.boxes is None or result.keypoints is None:
                print(json.dumps(response), flush=True)
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy()

            people = []
            for idx in range(len(boxes)):
                bbox = boxes[idx].astype(float)
                kps = keypoints[idx].astype(float)
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

                people.append(
                    {
                        "bbox": [float(v) for v in bbox.tolist()],
                        "confidence": float(confs[idx]),
                        "keypoints": [[float(p[0]), float(p[1]), float(p[2])] for p in kps.tolist()],
                        "center": center,
                    }
                )

            tracker.update(people)

            for person in people:
                track = tracker.tracks[person["track_id"]]
                label = classify_pose(np.array(person["keypoints"]), person["bbox"], track)
                response["people"].append(
                    {
                        "bbox": person["bbox"],
                        "confidence": person["confidence"],
                        "label": label,
                        "keypoints": person["keypoints"],
                    }
                )

            print(json.dumps(response), flush=True)

        except Exception as ex:
            print(f"worker error: {ex}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
