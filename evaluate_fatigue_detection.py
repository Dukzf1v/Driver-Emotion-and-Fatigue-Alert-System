"""
evaluate_fatigue_detection.py
==============================
Đánh giá khả năng phát hiện mệt mỏi của hệ thống DMS trên Fatigue Dataset.

Dataset:
  - Active Subjects/  → ground truth: active (0)
  - Fatigue Subjects/ → ground truth: fatigue (1)

Phương pháp đánh giá:
  Với mỗi ảnh tĩnh, dùng LandmarkEngine để trích xuất EAR & MAR.
  Quyết định mệt mỏi dựa trên:
    - EAR < T_EAR  → mắt nhắm (drowsy eye indicator)
    - MAR > T_MAR  → ngáp (yawn indicator)
    - OR: nếu EAR thấp hơn ngưỡng đủ rõ → predict = fatigue

Vì ảnh tĩnh không có chuỗi thời gian nên PERCLOS / sliding window không dùng được.
Thay vào đó, ta đánh giá trực tiếp trên ngưỡng EAR và MAR per-frame:
  predict_fatigue = (ear < T_EAR) OR (mar > T_MAR)
"""

import sys
import io
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

sys.path.insert(0, str(Path(__file__).parent))
import config
from landmark_engine import LandmarkEngine

# ── Paths ─────────────────────────────────────────────────────────────────
FATIGUE_ROOT = Path(__file__).parent / "Fatigue Dataset" / "archive"
OUTPUT_DIR   = Path(__file__).parent / "SOICT_DATN" / "Hinhve"
OUTPUT_JSON  = Path(__file__).parent / "fatigue_detection_eval.json"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

print("="*60)
print("  DMS Fatigue Detection Evaluation — Fatigue Dataset")
print("="*60)
print(f"  T_EAR threshold: {config.T_EAR}")
print(f"  T_MAR threshold: {config.T_MAR}")
print()

# ── Init LandmarkEngine ───────────────────────────────────────────────────
engine = LandmarkEngine()
print("LandmarkEngine initialized.\n")

# ── Process each folder ───────────────────────────────────────────────────
def process_folder(folder: Path, true_label: int, label_name: str):
    """
    Returns list of dicts: {ear, mar, pred_label, true_label, detected_face}
    pred_label: 1=fatigue, 0=active
    predict_fatigue = (ear < T_EAR) OR (mar > T_MAR)
    """
    files = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    results = []
    no_face = 0

    print(f"Processing {label_name} ({len(files)} images)...")
    for fp in files:
        frame = cv2.imread(str(fp))
        if frame is None:
            no_face += 1
            continue
        out = engine.process(frame)
        if out is None:
            no_face += 1
            results.append({
                "ear": None, "mar": None,
                "pred_label": 0,       
                "true_label": true_label,
                "detected_face": False
            })
            continue

        ear = out["ear"]
        mar = out["mar"]
        pred = 1 if (ear < config.T_EAR or mar > config.T_MAR) else 0
        results.append({
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "pred_label": pred,
            "true_label": true_label,
            "detected_face": True
        })

    detected = sum(1 for r in results if r["detected_face"])
    print(f"  Done: {detected}/{len(files)} faces detected, {no_face} no-read errors.\n")
    return results

results_active  = process_folder(FATIGUE_ROOT / "Active Subjects",  0, "Active")
results_fatigue = process_folder(FATIGUE_ROOT / "Fatigue Subjects", 1, "Fatigue")

engine.close()

all_results = results_active + results_fatigue
y_true = [r["true_label"]  for r in all_results]
y_pred = [r["pred_label"]  for r in all_results]
ears_active  = [r["ear"] for r in results_active  if r["ear"] is not None]
ears_fatigue = [r["ear"] for r in results_fatigue if r["ear"] is not None]
mars_active  = [r["mar"] for r in results_active  if r["mar"] is not None]
mars_fatigue = [r["mar"] for r in results_fatigue if r["mar"] is not None]

# ── Classification Metrics ────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print("="*60)
print("  FATIGUE DETECTION RESULTS (EAR/MAR per-frame threshold)")
print("="*60)
print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")
print()
print("  Confusion Matrix:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
print()
print(classification_report(y_true, y_pred, target_names=["Active","Fatigue"]))

print(f"  EAR stats — Active : mean={np.mean(ears_active):.4f}  std={np.std(ears_active):.4f}  median={np.median(ears_active):.4f}")
print(f"  EAR stats — Fatigue: mean={np.mean(ears_fatigue):.4f}  std={np.std(ears_fatigue):.4f}  median={np.median(ears_fatigue):.4f}")
print(f"  MAR stats — Active : mean={np.mean(mars_active):.4f}  std={np.std(mars_active):.4f}  median={np.median(mars_active):.4f}")
print(f"  MAR stats — Fatigue: mean={np.mean(mars_fatigue):.4f}  std={np.std(mars_fatigue):.4f}  median={np.median(mars_fatigue):.4f}")

# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred: Active","Pred: Fatigue"],
            yticklabels=["True: Active","True: Fatigue"],
            ax=ax, cbar=False, linewidths=0.5)
ax.set_title(f"Confusion Matrix — Fatigue Detection\n(EAR < {config.T_EAR} OR MAR > {config.T_MAR})\nAcc={acc*100:.1f}%  F1={f1:.3f}", fontsize=11)
plt.tight_layout()
out_cm = OUTPUT_DIR / "fatigue_detection_confusion_matrix.png"
plt.savefig(out_cm, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_cm}")
plt.close()

# ── Plot 2: EAR distribution Active vs Fatigue ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("EAR & MAR Distributions: Active vs Fatigue Subjects\n(MediaPipe FaceMesh — Fatigue Dataset)", fontsize=13, fontweight='bold')

axes[0].hist(ears_active,  bins=50, alpha=0.65, color="#2196F3", label=f"Active  (n={len(ears_active)})", density=True)
axes[0].hist(ears_fatigue, bins=50, alpha=0.65, color="#F44336", label=f"Fatigue (n={len(ears_fatigue)})", density=True)
axes[0].axvline(x=config.T_EAR, color='black', linestyle='--', linewidth=1.5, label=f"T_EAR={config.T_EAR}")
axes[0].set_xlabel("Eye Aspect Ratio (EAR)", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("EAR Distribution", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].text(0.02, 0.95, f"Active  μ={np.mean(ears_active):.3f}\nFatigue μ={np.mean(ears_fatigue):.3f}",
             transform=axes[0].transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[1].hist(mars_active,  bins=50, alpha=0.65, color="#2196F3", label=f"Active  (n={len(mars_active)})", density=True)
axes[1].hist(mars_fatigue, bins=50, alpha=0.65, color="#F44336", label=f"Fatigue (n={len(mars_fatigue)})", density=True)
axes[1].axvline(x=config.T_MAR, color='black', linestyle='--', linewidth=1.5, label=f"T_MAR={config.T_MAR}")
axes[1].set_xlabel("Mouth Aspect Ratio (MAR)", fontsize=11)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_title("MAR Distribution", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].text(0.55, 0.95, f"Active  μ={np.mean(mars_active):.3f}\nFatigue μ={np.mean(mars_fatigue):.3f}",
             transform=axes[1].transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
out_dist = OUTPUT_DIR / "fatigue_ear_mar_distribution.png"
plt.savefig(out_dist, dpi=150, bbox_inches='tight')
print(f"Saved: {out_dist}")
plt.close()

# ── Plot 3: Bar summary ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
metrics_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [acc, prec, rec, f1]
colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', alpha=0.85)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"Fatigue Detection Metrics\n(EAR < {config.T_EAR} OR MAR > {config.T_MAR}, n=9,120 images)", fontsize=12)
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
            f"{val:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
out_bar = OUTPUT_DIR / "fatigue_detection_metrics_bar.png"
plt.savefig(out_bar, dpi=150, bbox_inches='tight')
print(f"Saved: {out_bar}")
plt.close()

# ── Save JSON ─────────────────────────────────────────────────────────────
summary = {
    "n_total": len(all_results),
    "n_active": len(results_active),
    "n_fatigue": len(results_fatigue),
    "face_detection_rate": round(sum(1 for r in all_results if r["detected_face"]) / len(all_results), 4),
    "threshold_EAR": config.T_EAR,
    "threshold_MAR": config.T_MAR,
    "accuracy":  round(acc,  4),
    "precision": round(prec, 4),
    "recall":    round(rec,  4),
    "f1_score":  round(f1,   4),
    "confusion_matrix": cm.tolist(),
    "ear_stats": {
        "active":  {"mean": round(float(np.mean(ears_active)),  4), "std": round(float(np.std(ears_active)),  4), "median": round(float(np.median(ears_active)),  4)},
        "fatigue": {"mean": round(float(np.mean(ears_fatigue)), 4), "std": round(float(np.std(ears_fatigue)), 4), "median": round(float(np.median(ears_fatigue)), 4)},
    },
    "mar_stats": {
        "active":  {"mean": round(float(np.mean(mars_active)),  4), "std": round(float(np.std(mars_active)),  4), "median": round(float(np.median(mars_active)),  4)},
        "fatigue": {"mean": round(float(np.mean(mars_fatigue)), 4), "std": round(float(np.std(mars_fatigue)), 4), "median": round(float(np.median(mars_fatigue)), 4)},
    },
}
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Saved: {OUTPUT_JSON}")
print("\n=== DONE ===")
