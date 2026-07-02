# Driver Emotion and Fatigue Alert System (DEFAS)

The Driver Emotion and Fatigue Alert System (**DEFAS**) is a comprehensive real-time driver monitoring solution optimized for hardware-constrained vehicle cabin environments. The system implements a **hybrid approach** combining **3D geometric facial landmark analysis** for biological states (blinking, yawning, head nodding) with a **fine-tuned deep convolutional neural network (MobileNetV3)** to classify negative driver emotional states (anger, panic).

---

## Simulation Architecture

The project operates under a simulated client-central architecture, enabling local monitoring programs to send telemetry to a centralized simulation application:

```
                  ┌──────────────────────────────┐
                  │    Camera / Video Stream     │
                  └──────────────┬───────────────┘
                                 │ RGB Frames
                                 ▼
          ┌──────────────────────────────────────────────┐
          │           Local Monitoring Apps              │
          │  - PC App (Python, MediaPipe, PyTorch)       │
          │  - Mobile App (Flutter, ML Kit, PyTorch)     │
          └───────────────┬──────────────────────┬───────┘
                          │                      │
                          │ HTTP POST            │ Audio Alarms
                          │ (Telemetry payload)  ▼
                          ▼             ┌─────────────────┐
         ┌────────────────────────────┐ │   Local Alert   │
         │   Centralized Simulation   │ └─────────────────┘
         │    Management Program      │
         └──────────────┬─────────────┘
                        ├────────────────────────┐
                        ▼                        ▼
               ┌─────────────────┐      ┌─────────────────┐
               │ SQLite Database │      │   Monitoring    │
               │  (Warning Logs) │      │  Simulation UI  │
               └─────────────────┘      └─────────────────┘
```

1. **Local Monitoring (PC/Mobile Apps):** Performs real-time geometric landmark extraction, calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head rotation angles (Pitch, Roll, Yaw) using PnP. It runs priority decision logic and triggers audio warnings locally.
2. **Simulation Logger (Flask):** Receives telemetry data from local apps and logs warning histories to a local SQLite database.
3. **Simulation UI:** A simple web interface that visualizes real-time driver metrics and allows querying warning logs.

---

## System Preview

| **PC App HUD** | **Simulation UI** | **Mobile App (Flutter)** |
| :---: | :---: | :---: |
| ![PC App HUD](docs/images/pc_client_hud.png) | ![Web Dashboard](docs/images/web_dashboard.png) | ![Mobile App](docs/images/mobile_client_app.jpg) |

### Real-Time Detection Scenarios

The system runs real-time hybrid decision logic to classify the driver's state into five distinct categories:

| **Normal State** | **Fatigue State** |
| :---: | :---: |
| ![Normal](docs/images/detect_normal.png) | ![Fatigue](docs/images/detect_fatigue.png) |

| **Distracted State** | **Angry State** |
| :---: | :---: |
| ![Distracted](docs/images/detect_distracted.png) | ![Angry](docs/images/detect_angry.png) |

| **Fear State** |
| :---: |
| ![Fear](docs/images/detect_fear.png) |

---

## Key Features

- **Hybrid Decision Logic:** Integrates fast geometric calculations (EAR for eye closure, MAR for yawning, Pitch for head nodding) with MobileNetV3 deep learning emotion scores.
- **Dynamic Threshold Adjustments:** EAR and MAR alarm thresholds are dynamically adjusted in real-time based on the driver's current happiness/anger probabilities to reduce false alerts due to speaking or smiling.
- **Optimized Data Syncing:** Local monitoring apps execute data transmission asynchronously on background threads. Telemetry is sent every 15 frames or immediately upon a state change to optimize device CPU performance.
- **Cross-Platform Support:** Fully functional implementations on PC (Python, OpenCV, MediaPipe) and Mobile (Flutter/Dart, Google ML Kit Face Mesh, PyTorch Mobile Lite `.ptl` format).
- **Offline Evaluation Pipeline:** Integrated classifier evaluation script generating accuracy reports, confusion matrices, and ROC curves on public datasets.

---

## Directory Structure

```
├── Fatigue Dataset/             # Fatigue evaluation dataset (Active/Fatigue subjects)
├── Emotion Dataset/             # Emotion classification datasets (RAF-DB, FER-2013, etc.)
├── dms_mobile_app/              # Flutter Mobile App Source
│   ├── lib/
│   │   ├── main.dart            # Visual display interface & Camera capture
│   │   └── dms_processor.dart   # ML Kit & PyTorch Lite inference logic
│   └── assets/                  # PyTorch Mobile Lite model & labels
├── models/                      # Model binaries (face landmarker, task files)
├── templates/                   # HTML templates for the simulation UI
├── static/                      # Static assets for the simulation UI (CSS, JS, icons)
├── config.py                    # Global system thresholds and settings
├── landmark_engine.py           # MediaPipe facial landmark extractor (PC)
├── fatigue_metrics.py           # Sliding window PERCLOS & geometric calculations
├── emotion_scorer.py            # Average emotion accumulator
├── inference.py                 # PC monitoring execution script (Visual display interface)
├── dashboard_server.py          # Script to run the simulation UI & database logger
├── data_preprocessing.py        # Dataset preprocessing and augmentation
├── evaluate_fatigue_detection.py# Dataset evaluation and metrics plotting
├── train.ipynb                  # MobileNetV3 emotion model training notebook
├── requirements.txt             # Dependencies for local monitoring & evaluation
└── requirements_server.txt      # Dependencies for the simulation UI
```


---

## Dataset Download & Directory Structure

Since the datasets are extremely large, they are not stored in this Git repository. You can download the pre-packaged datasets from Google Drive:

* **Fatigue Dataset:** [Google Drive Link](https://drive.google.com/file/d/1dxvNPF7RBDpI1MblxNtbSge-sOl-wy2_/view?usp=sharing)
* **Emotion Dataset:** [Google Drive Link](https://drive.google.com/file/d/1oarn9ye_6l4jEuAHNChA2E6iEwizqrX4/view?usp=sharing)

To run the data preprocessing (`data_preprocessing.py`) or fatigue evaluation (`evaluate_fatigue_detection.py`) scripts, download and extract the files, then organize them in the root directory as follows:

```text
├── Fatigue Dataset/
│   └── archive/
│       ├── Active Subjects/     # Active driver images (ground truth: 0)
│       └── Fatigue Subjects/    # Fatigued/drowsy driver images (ground truth: 1)
└── Emotion Dataset/
    ├── RAF-DB/                  # RAF-DB dataset
    │   ├── train/               # Folders 1 to 7
    │   └── test/                # Folders 1 to 7
    ├── MLI-DER/                 # MLI-DER dataset
    │   └── image data/          # Normal, littleBright, littleDark, veryBright folders
    ├── KMU-FED/                 # KMU-FED flat folder of images
    ├── AffectNet/               # AffectNet dataset
    │   ├── Train/               # Folders (neutral, anger, disgust, etc.)
    │   └── Test/                # Folders (neutral, anger, disgust, etc.)
    ├── FER-2013/                # FER-2013 dataset
    │   ├── train/               # Folders (neutral, angry, etc.)
    │   └── test/                # Folders (neutral, angry, etc.)
    ├── KDEF/                    # KDEF dataset (folders: neutral, angry, etc.)
    └── SFEW/                    # SFEW dataset
        ├── Train/
        ├── Val/
        └── Test/
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Flutter SDK & Dart (for the mobile app)
- Web camera or video files for testing

### 2. Setting Up Python Virtual Environment
Since virtual environment folders are not uploaded to the repository, you must create and activate a new virtual environment manually:

```bash
# Create a new virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate

# Install all required dependencies
pip install -r requirements.txt
```

### 3. Running the System

#### Step A: Start the Simulation Logger & UI
Run the simulation program first. It initializes the local SQLite database (`dms_database.db`) and opens a web interface at port `5000`:
```bash
python dashboard_server.py
```
Open `http://localhost:5000` in your web browser to view the **Simulation UI**.

#### Step B: Start the PC App
Open a new terminal window (activate the virtual environment first) and run the local monitoring program:
```bash
# Run using default webcam (source 0)
python inference.py --source 0

# Run using a test video file
python inference.py --source video_test/test_driver.mp4
```
The program will stream telemetry to the simulation database logger automatically. Press `Q` or `ESC` on the camera window to close the visual display interface. Upon closing, the terminal will output execution metrics including FPS and CPU/RAM resources.

#### Step C: Run the Mobile App (Flutter)
Ensure you have a mobile device (with USB debugging enabled) connected or an emulator open:
```bash
cd dms_mobile_app
flutter pub get
flutter run
```

---

## Performance Evaluation

The evaluation of the system is split into two parts: geometric fatigue detection and deep-learning-based emotion classification.

### 1. Fatigue Detection (EAR/MAR)

To evaluate the fatigue detection algorithm on the static `Fatigue Dataset` and generate performance plots:
```bash
python evaluate_fatigue_detection.py
```

| **Confusion Matrix** | **EAR & MAR Distribution** |
| :---: | :---: |
| ![Fatigue Confusion Matrix](docs/images/fatigue_detection_confusion_matrix.png) | ![EAR MAR Distribution](docs/images/fatigue_ear_mar_distribution.png) |

### 2. Emotion Classification (MobileNetV3)

The MobileNetV3 5-class model was trained on facial expression datasets. The training history curves and validation confusion matrices are shown below:

| **Training Curves** | **Emotion Confusion Matrices** |
| :---: | :---: |
| ![Training Curves](docs/images/training_curves.png) | ![Emotion Confusion Matrices](docs/images/confusion_matrices.png) |