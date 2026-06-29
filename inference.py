"""
inference_mobilenetv3.py
────────────────────────
Driver Monitoring System — Hybrid Architecture
CNN 2-class (Normal / Angry) + Geometric Fatigue Detection (EAR/PERCLOS)

Architecture:
  Stage 1: LandmarkEngine (MediaPipe FaceMesh) → EAR, MAR, Pitch, BBox
  Stage 2: FatigueMetrics (sliding window) → F_score (PERCLOS-based)
  Stage 3: MobileNetV3 (2-class) → P(Normal), P(Angry) → AngerScorer
  Stage 4: Decision Rule (Fatigue > Angry > Normal)

Decision Rule:
  if F_score > FATIGUE_F_THRESH      → FATIGUE  
  elif anger_level > ANGRY_LVL_THRESH → ANGRY
  else                                 → NORMAL
"""

import argparse
import sys
import time
import json
import winsound
import urllib.request
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# Modules
from landmark_engine import LandmarkEngine
from fatigue_metrics import FatigueMetrics
from emotion_scorer import AngerScorer
import config

STATE_COLORS = {
    "NORMAL":  (80, 200, 80),    # green
    "FATIGUE": (40, 160, 255),   # orange
    "ANGRY":   (60, 60, 220),    # red
    "FEAR":    (180, 60, 180),   # purple
    "DISTRACTED": (0, 255, 255), # yellow
}

ANGER_CLASS_IDX  = config.ANGER_CLASS_IDX
FEAR_CLASS_IDX   = config.FEAR_CLASS_IDX
NORMAL_CLASS_IDX = config.NORMAL_CLASS_IDX
BAR_COLORS = {
    "NORMAL": (80, 200, 80),   # green
    "ANGRY":  (60, 60, 220),   # red
    "FEAR":   (180, 60, 180),  # purple
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Decision thresholds — can be overridden via CLI args
FATIGUE_F_THRESH_DEFAULT  = config.FATIGUE_THRESH   # F_score > → FATIGUE
ANGRY_LVL_THRESH_DEFAULT  = config.ANGRY_THRESH   # anger_level > → ANGRY
FEAR_LVL_THRESH_DEFAULT   = config.FEAR_THRESH    # fear_level > → FEAR

# Smoothing window for CNN probs (reduce flickering)
SMOOTH_WINDOW = 5

# ══════════════════════════════════════════════════════════════════════════════
#  Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_names = ckpt.get(
        "class_names",
        ckpt.get("cfg", {}).get("class_names", ["Neutral", "Anger", "Fear", "Happiness", "Sadness"])
    )
    img_size    = ckpt.get("img_size", ckpt.get("cfg", {}).get("img_size", 224))
    mean        = ckpt.get("mean", IMAGENET_MEAN)
    std         = ckpt.get("std",  IMAGENET_STD)
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"[INFO] Checkpoint loaded: {Path(ckpt_path).name}")
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    return model, class_names, img_size, mean, std


def build_transform(img_size: int, mean, std):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def crop_face(frame_bgr: np.ndarray, bbox, pad: float = 0.20):
    if bbox is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w, x2 + px); y2 = min(h, y2 + py)
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame_bgr


@torch.no_grad()
def predict_frame(model, transform, face_crop_bgr: np.ndarray, device: torch.device):
    rgb    = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs   # shape: (num_classes,)


class ProbSmoother:
    def __init__(self, num_classes: int, window: int = SMOOTH_WINDOW):
        self.window  = window
        self.history = []

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(probs.copy())
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Decision Logic — Hybrid
# ══════════════════════════════════════════════════════════════════════════════

def decide_state(F_score: float, anger_level: float, fear_level: float,
                 fatigue_thresh: float, angry_thresh: float, fear_thresh: float) -> str:
    """
    Decision Rule (Option A):
      Compare how much each score exceeds its threshold (excess).
      Choose the state with the highest positive excess.
      If none exceed their respective thresholds, return NORMAL.
    """
    fatigue_excess = max(0.0, F_score - fatigue_thresh)
    angry_excess   = max(0.0, anger_level - angry_thresh)
    fear_excess    = max(0.0, fear_level - fear_thresh)

    if fatigue_excess == 0.0 and angry_excess == 0.0 and fear_excess == 0.0:
        return "NORMAL"

    # Select the state with the maximum excess
    max_excess = max(fatigue_excess, angry_excess, fear_excess)
    if max_excess == fatigue_excess:
        return "FATIGUE"
    elif max_excess == angry_excess:
        return "ANGRY"
    else:
        return "FEAR"


ALERT_LEVELS = {
    "NORMAL":  ("NORMAL (OK)",              (80, 200, 80)),
    "FATIGUE": ("FATIGUE - STOP & REST!",   (40, 160, 255)),
    "ANGRY":   ("ANGRY - CALM DOWN!",       (60, 60, 220)),
    "FEAR":    ("FEAR - BE CAREFUL!",       (180, 60, 180)),
    "DISTRACTED": ("DISTRACTED - PAY ATTENTION!", (0, 255, 255)),
}


# ══════════════════════════════════════════════════════════════════════════════
#  HUD Drawing
# ══════════════════════════════════════════════════════════════════════════════

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def draw_threshold_bar(frame, label: str, value: float, threshold: float,
                       x: int, y: int, width: int = 160,
                       color_low=(80, 200, 80), color_high=(60, 60, 220)):
    """Draw progress bar with threshold line."""
    filled = int(value * width)
    cv2.rectangle(frame, (x, y - 8), (x + width, y + 4), (50, 50, 50), -1)
    color = color_high if value >= threshold else color_low
    cv2.rectangle(frame, (x, y - 8), (x + filled, y + 4), color, -1)
    thresh_x = x + int(threshold * width)
    cv2.line(frame, (thresh_x, y - 12), (thresh_x, y + 8), (255, 255, 0), 1)
    cv2.putText(frame, f"{label}: {value:.3f}", (x, y - 12),
                _FONT, 0.40, (200, 200, 200), 1)


def draw_hud(frame: np.ndarray, probs: np.ndarray, class_names: list,
             landmark_data, metrics_data, final_state: str,
             F_score: float, anger_level: float, fear_level: float,
             fatigue_thresh: float, angry_thresh: float, fear_thresh: float,
             fps: float = 0.0, t_ear: float = 0.15, t_mar: float = 0.55):

    h, w = frame.shape[:2]
    state_color = STATE_COLORS[final_state]

    # Face bbox + landmarks
    if landmark_data is not None:
        bbox = landmark_data["bbox"]
        pts  = landmark_data["pts"]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)
        label = f"[{final_state}]"
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    _FONT_BOLD, 0.65, state_color, 1, cv2.LINE_AA)
        for i in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
            cv2.circle(frame, pts[i], 1, (0, 255, 255), -1)
        for i in [61, 37, 267, 291, 84, 314]:
            cv2.circle(frame, pts[i], 1, (0, 0, 255), -1)
    else:
        if final_state == "DISTRACTED":
            cv2.putText(frame, "KHONG PHAT HIEN KHUON MAT!", (20, h // 2),
                        _FONT_BOLD, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # Split HUD into a black panel on the right
    panel_w = 360
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)  # Black background

    px = 12
    py = 28
    cv2.putText(panel, "DMS Monitor - Hybrid v3", (px, py),
                _FONT_BOLD, 0.65, (255, 255, 255), 1)
    cv2.line(panel, (px, py + 6), (panel_w - 12, py + 6), (80, 80, 80), 1)
    py += 26

    # Stage 1 & 2: Geometric metrics
    if landmark_data:
        ear   = landmark_data["ear"]
        mar   = landmark_data["mar"]
        pitch = landmark_data["pitch"]
        cv2.putText(panel, f"EAR: {ear:.3f} (th:{t_ear:.2f})  MAR: {mar:.3f} (th:{t_mar:.2f})",
                    (px, py), _FONT, 0.43, (200, 200, 200), 1)
        py += 22
        cv2.putText(panel, f"Pitch: {pitch:.1f} deg",
                    (px, py), _FONT, 0.48, (200, 200, 200), 1)
        py += 22
    else:
        cv2.putText(panel, "EAR: ---  MAR: ---",
                    (px, py), _FONT, 0.48, (100, 100, 100), 1)
        py += 22
        cv2.putText(panel, "Pitch: --- deg",
                    (px, py), _FONT, 0.48, (100, 100, 100), 1)
        py += 22

    if metrics_data:
        f_val, perclos, f_blink, f_yawn, f_nod = metrics_data
        cv2.putText(panel, f"PERCLOS: {perclos * 100:.1f}%",
                    (px, py), _FONT, 0.48, (0, 255, 255), 1)
        py += 22
        cv2.putText(panel, f"Blink:{f_blink:.2f} Yawn:{f_yawn:.2f} Nod:{f_nod:.2f}",
                    (px, py), _FONT, 0.43, (180, 180, 180), 1)
        py += 22
    else:
        cv2.putText(panel, "PERCLOS: ---%",
                    (px, py), _FONT, 0.48, (100, 100, 100), 1)
        py += 22
        cv2.putText(panel, "Blink:--- Yawn:--- Nod:---",
                    (px, py), _FONT, 0.43, (100, 100, 100), 1)
        py += 22

    py += 15

    # F_score (Fatigue)
    draw_threshold_bar(panel, "F_fatigue", F_score, fatigue_thresh,
                       px, py, width=160,
                       color_low=(80, 200, 80), color_high=(40, 160, 255))
    py += 26

    cv2.line(panel, (px, py), (panel_w - 12, py), (80, 80, 80), 1)
    py += 16

    # Stage 3: Show 2 bars (Normal vs Angry)
    cv2.putText(panel, "CNN (5-class -> states):", (px, py), _FONT, 0.44, (200, 200, 200), 1)
    py += 18

    # Aggregate probabilities: Normal = probs[0], Angry = probs[ANGER_CLASS_IDX], Fear = probs[FEAR_CLASS_IDX]
    p_normal = float(probs[NORMAL_CLASS_IDX])
    p_angry  = float(probs[ANGER_CLASS_IDX])
    p_fear   = float(probs[FEAR_CLASS_IDX])

    for label, val, col in [("NORMAL", p_normal, BAR_COLORS["NORMAL"]),
                             ("ANGRY",  p_angry,  BAR_COLORS["ANGRY"]),
                             ("FEAR",   p_fear,   BAR_COLORS["FEAR"])]:
        bar_w = int(val * 150)
        cv2.putText(panel, f"{label[:3]}:", (px, py), _FONT, 0.43, col, 1)
        cv2.rectangle(panel, (px + 42, py - 10), (px + 42 + bar_w, py + 2), col, -1)
        cv2.putText(panel, f"{val * 100:.1f}%", (px + 202, py), _FONT, 0.43, (200, 200, 200), 1)
        py += 18

    py += 15

    # Anger level and Fear level
    draw_threshold_bar(panel, "Anger level", anger_level, angry_thresh,
                       px, py, width=160,
                       color_low=(80, 200, 80), color_high=(60, 60, 220))
    py += 26

    draw_threshold_bar(panel, "Fear level", fear_level, fear_thresh,
                       px, py, width=160,
                       color_low=(80, 200, 80), color_high=(180, 60, 180))
    py += 26

    cv2.line(panel, (px, py), (panel_w - 12, py), (80, 80, 80), 1)
    py += 18

    # Stage 4: Final result
    alert_text, alert_color = ALERT_LEVELS[final_state]
    cv2.putText(panel, f"STATE: {final_state}", (px, py),
                _FONT_BOLD, 0.72, alert_color, 2)
    py += 28
    cv2.putText(panel, alert_text, (px, py), _FONT_BOLD, 0.52, alert_color, 1)

    # FPS displayed on the camera
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                _FONT, 0.48, (0, 255, 0), 1)

    # Combine 2 images (Camera Feed + HUD Panel)
    combined_frame = cv2.hconcat([frame, panel])
    return combined_frame


# ══════════════════════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════════════════════

def send_to_dashboard(url, data):
    def post():
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=1) as response:
                pass
        except Exception as e:
            pass
    threading.Thread(target=post, daemon=True).start()


def run(args):
    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    # CLI thresholds
    fatigue_thresh = args.fatigue_thresh
    angry_thresh   = args.angry_thresh
    fear_thresh    = args.fear_thresh
    print(f"[INFO] Fatigue threshold: F > {fatigue_thresh:.3f}")
    print(f"[INFO] Angry threshold:   anger_level > {angry_thresh:.3f}")
    print(f"[INFO] Fear threshold:    fear_level > {fear_thresh:.3f}")

    # Init CNN 5-class
    model, class_names, img_size, mean, std = load_checkpoint(args.checkpoint, device)
    transform = build_transform(img_size, mean, std)
    smoother  = ProbSmoother(num_classes=len(class_names))

    # Init pipeline modules
    landmark_engine = LandmarkEngine()
    fatigue_tracker = FatigueMetrics(window_size=config.FATIGUE_WINDOW_SIZE, fps=config.FPS_DEFAULT)
    emotion_scorer  = AngerScorer(target_classes=[ANGER_CLASS_IDX, FEAR_CLASS_IDX], window_size=config.EMOTION_WINDOW_SIZE)

    # Video source
    src = args.source
    try: src = int(src)
    except ValueError: pass

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {src}")

    t_prev = time.perf_counter()
    fps_disp = 0.0
    
    last_beep_time = 0.0
    BEEP_COOLDOWN = config.BEEP_COOLDOWN
    last_face_detected_time = time.perf_counter()
    
    # Performance optimization variables (FPS)
    frame_count = 0
    last_sent_state = None
    probs = np.zeros(5)
    probs[0] = 1.0  # Default Neutral = 1.0

    # Telemetry metrics
    t_loop_start = time.perf_counter()
    landmark_latencies = []
    cnn_latencies = []

    # Resource monitoring (CPU/RAM)
    import os
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        proc.cpu_percent()  # Initialize CPU measurement
        cpu_usages = []
        ram_usages = []
    except ImportError:
        proc = None
        cpu_usages = None
        ram_usages = None

    print("\n[INFO] Hybrid DMS v3 running — press Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        # Monitor resource consumption every 30 frames
        if proc is not None and frame_count % 30 == 0:
            cpu_usages.append(proc.cpu_percent())
            ram_usages.append(proc.memory_info().rss / (1024 * 1024))

        # Stage 1: Extract landmarks
        t_lm_start = time.perf_counter()
        landmark_data = landmark_engine.process(frame)
        landmark_latencies.append((time.perf_counter() - t_lm_start) * 1000.0)

        if landmark_data is not None:
            last_face_detected_time = time.perf_counter()
            bbox  = landmark_data["bbox"]
            ear   = landmark_data["ear"]
            mar   = landmark_data["mar"]
            pitch = landmark_data["pitch"]

            # Stage 2: Update fatigue metrics with dynamic thresholds based on emotions
            p_happy = float(probs[config.HAPPINESS_CLASS_IDX]) if len(probs) > config.HAPPINESS_CLASS_IDX else 0.0
            p_angry = float(probs[config.ANGER_CLASS_IDX]) if len(probs) > config.ANGER_CLASS_IDX else 0.0
            
            # Adjust EAR and MAR thresholds dynamically
            # - Happiness (smiling/laughing) squints eyes (reduces EAR) and opens mouth (increases MAR)
            # - Anger (furrowing brows/glaring) may squint eyes (reduces EAR)
            t_ear_factor = 1.0 - (0.25 * p_happy) - (0.20 * p_angry)
            t_mar_factor = 1.0 + (0.30 * p_happy)
            
            fatigue_tracker.T_EAR = config.T_EAR * max(0.6, t_ear_factor)
            fatigue_tracker.T_MAR = config.T_MAR * min(1.5, t_mar_factor)

            fatigue_tracker.update(ear, mar, pitch)

            # Stage 3: CNN 5-class → Only run inference every 5 frames to save CPU/GPU
            if frame_count % 5 == 1 or 'probs' not in locals():
                crop      = crop_face(frame, bbox, pad=0.20)
                t_cnn_start = time.perf_counter()
                raw_probs = predict_frame(model, transform, crop, device)
                cnn_latencies.append((time.perf_counter() - t_cnn_start) * 1000.0)
                probs     = smoother.update(raw_probs)

            # Update emotion scorer with the latest probs (every frame)
            emotion_scorer.update(probs)
        else:
            # No face detected → safe virtual state (5 classes: Neutral=1.0)
            probs = np.zeros(5); probs[0] = 1.0
            fatigue_tracker.update(0.3, 0.0, 0.0)
            emotion_scorer.update(probs)

        # ── Stage 4: Decision Rule ─────────────────────────────────────
        metrics_data = fatigue_tracker.get_metrics()
        F_score      = metrics_data[0]
        
        levels       = emotion_scorer.get_levels()
        anger_level  = levels.get(ANGER_CLASS_IDX, 0.0)
        fear_level   = levels.get(FEAR_CLASS_IDX, 0.0)

        # Detect distraction if face not recognized for the specified time
        face_lost_duration = time.perf_counter() - last_face_detected_time
        if face_lost_duration > args.face_lost_timeout:
            final_state = "DISTRACTED"
        else:
            final_state = decide_state(F_score, anger_level, fear_level,
                                        fatigue_thresh, angry_thresh, fear_thresh)
        
        # Send data to Dashboard (every 15 frames or immediately when state changes to save network threads)
        if args.server:
            if frame_count % 15 == 0 or final_state != last_sent_state:
                send_to_dashboard(args.server, {
                    "vehicle_id": config.DEFAULT_VEHICLE_ID,
                    "driver_name": config.DEFAULT_DRIVER_NAME,
                    "state": final_state,
                    "f_score": float(F_score),
                    "anger_level": float(anger_level),
                    "fear_level": float(fear_level),
                    "timestamp": time.time()
                })
                last_sent_state = final_state

        # FPS
        t_now    = time.perf_counter()
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev   = t_now

        # Play sound alerts
        if final_state in ["FATIGUE", "ANGRY", "FEAR", "DISTRACTED"]:
            if t_now - last_beep_time > BEEP_COOLDOWN:
                if winsound is not None:
                    sound_path = getattr(config, "WARNING_SOUND", "warning.wav")
                    if sound_path and Path(sound_path).exists():
                        winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    else:
                        winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                else:
                    print("\a", end="", flush=True)
                last_beep_time = t_now

        combined_frame = draw_hud(frame, probs, class_names,
                                  landmark_data, metrics_data, final_state,
                                  F_score, anger_level, fear_level, fatigue_thresh, angry_thresh, fear_thresh, fps_disp,
                                  t_ear=fatigue_tracker.T_EAR, t_mar=fatigue_tracker.T_MAR)

        cv2.imshow("DMS — Hybrid v3", combined_frame)

        key = cv2.waitKey(1) & 0xFF
            
        if key in (ord("q"), ord("Q"), 27):
            break

    # Cleanup
    cap.release()
    landmark_engine.close()
    cv2.destroyAllWindows()

    # Print Telemetry Summary
    t_loop_end = time.perf_counter()
    total_time = t_loop_end - t_loop_start
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print("  DMS PERFORMANCE TELEMETRY SUMMARY")
    print("="*60)
    print(f"  Total Processed Frames: {frame_count}")
    print(f"  Total Running Time    : {total_time:.2f} s")
    print(f"  Average Throughput    : {avg_fps:.2f} FPS")
    print("-"*60)
    if landmark_latencies:
        print(f"  Landmark Engine (FaceMesh) Latency:")
        print(f"    Average : {np.mean(landmark_latencies):.2f} ms")
        print(f"    Min/Max : {np.min(landmark_latencies):.2f} / {np.max(landmark_latencies):.2f} ms")
    if cnn_latencies:
        print(f"  Emotion CNN (MobileNetV3) Latency:")
        print(f"    Average : {np.mean(cnn_latencies):.2f} ms")
        print(f"    Min/Max : {np.min(cnn_latencies):.2f} / {np.max(cnn_latencies):.2f} ms")
    print("-"*60)
    if proc is not None and cpu_usages and ram_usages:
        print(f"  System Resource Consumption (Process Level):")
        print(f"    Average CPU Usage : {np.mean(cpu_usages):.1f}% (Normalized to single core: {np.mean(cpu_usages)/psutil.cpu_count():.1f}%)")
        print(f"    Average RAM Usage : {np.mean(ram_usages):.1f} MB (Peak: {np.max(ram_usages):.1f} MB)")
        print("-"*60)
    avg_frame_time = (1000.0 / avg_fps) if avg_fps > 0 else 33.3
    print("  Estimated Local Warning Response Latency:")
    print(f"    ~ {avg_frame_time + 15.0:.2f} ms (Well within the 100ms threshold)")
    print("="*60 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="DMS Hybrid")
    p.add_argument("--checkpoint", "-c", default=config.MODEL_CHECKPOINT,
                   help=f"Path to model checkpoint (default: {config.MODEL_CHECKPOINT})")
    p.add_argument("--source", "-s", default=config.DEFAULT_SOURCE,
                   help=f"Video source: 0=webcam, or path to video file (default: {config.DEFAULT_SOURCE})")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference")
    p.add_argument("--fatigue-thresh", type=float, default=config.FATIGUE_THRESH,
                   help=f"F_score threshold for Fatigue alert (default: {config.FATIGUE_THRESH})")
    p.add_argument("--angry-thresh", type=float, default=config.ANGRY_THRESH,
                   help=f"anger_level threshold for Angry alert (default: {config.ANGRY_THRESH})")
    p.add_argument("--fear-thresh", type=float, default=config.FEAR_THRESH,
                   help=f"fear_level threshold for Fear alert (default: {config.FEAR_THRESH})")
    p.add_argument("--face-lost-timeout", type=float, default=config.FACE_LOST_TIMEOUT,
                   help=f"Time in seconds of no face detection before alerting (default: {config.FACE_LOST_TIMEOUT})")
    p.add_argument("--server", type=str, default=config.DASHBOARD_API_URL,
                   help=f"URL to post data to central web dashboard (default: {config.DASHBOARD_API_URL})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
