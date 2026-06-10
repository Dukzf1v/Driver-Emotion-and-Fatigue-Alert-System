import os

# Model path
MODEL_CHECKPOINT = "best_mobilenetv3_5class.pth"
LANDMARK_MODEL_PATH = "models/face_landmarker.task"

# Cấu hình Camera & Xử lý
DEFAULT_SOURCE = "0"
FACE_LOST_TIMEOUT = 2.0  # s

# Ngưỡng hình học (Geometric Thresholds)
T_EAR = 0.15      # Ngưỡng nhắm mắt 
T_MAR = 0.55      # Ngưỡng ngáp 
T_PITCH = -18.0   # Ngưỡng gật đầu
FPS_DEFAULT = 30
FATIGUE_WINDOW_SIZE = 90
EMOTION_WINDOW_SIZE = 90

# Decision Thresholds
FATIGUE_THRESH = 0.25 
ANGRY_THRESH = 0.25
FEAR_THRESH = 0.25

# Emotion Classification
ANGER_CLASS_IDX = 1
FEAR_CLASS_IDX = 2
NORMAL_CLASS_IDX = 0
HAPPINESS_CLASS_IDX = 3
SADNESS_CLASS_IDX = 4

# Tần suất phát tiếng beep cảnh báo (giây)
BEEP_COOLDOWN = 1.5

# Flask Server
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DASHBOARD_API_URL = "http://localhost:5000/api/update"

DEFAULT_VEHICLE_ID = "36A-120.36"
DEFAULT_DRIVER_NAME = "Phùng Thanh Độ"

WARNING_SOUND = "sound\warning.wav"
