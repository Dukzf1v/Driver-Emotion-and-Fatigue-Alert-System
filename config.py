import os

# Model path
MODEL_CHECKPOINT = "models/best_mobilenetv3_5class.pth"
LANDMARK_MODEL_PATH = "models/face_landmarker.task"

# Camera & Processing Config
DEFAULT_SOURCE = "0"
FACE_LOST_TIMEOUT = 2.0  # s

# Geometric Thresholds
T_EAR = 0.20      # Eye Aspect Ratio threshold
T_MAR = 0.50      # Mouth Aspect Ratio threshold
T_PITCH_UP = -18.0    # Head pitch up threshold
T_PITCH_DOWN = 18.0   # Head pitch down threshold
FPS_DEFAULT = 30
FATIGUE_WINDOW_SIZE = 90
EMOTION_WINDOW_SIZE = 90

# Decision Thresholds
FATIGUE_THRESH = 0.25 
ANGRY_THRESH = 0.40
FEAR_THRESH = 0.35

# Emotion Classification
ANGER_CLASS_IDX = 1
FEAR_CLASS_IDX = 2
NORMAL_CLASS_IDX = 0
HAPPINESS_CLASS_IDX = 3
SADNESS_CLASS_IDX = 4

# Beep cooldown (seconds)
BEEP_COOLDOWN = 1.5

# Flask Server
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DASHBOARD_API_URL = "https://dms-dashboard-udqh.onrender.com/api/update"

DEFAULT_VEHICLE_ID = "36A-120.36"
DEFAULT_DRIVER_NAME = "Phùng Thanh Độ"

WARNING_SOUND = "sound\warning.wav"
