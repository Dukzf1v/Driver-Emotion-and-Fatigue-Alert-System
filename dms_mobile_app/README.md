# DMS Mobile Application Client (Flutter Native)

This directory contains a complete cross-platform **Flutter** native mobile application for the Driver Monitoring System (DMS). 

It performs camera streaming, detects face landmarks using Google ML Kit (for EAR/MAR calculations), runs emotion classification using PyTorch Mobile Lite (`pytorch_lite`), and posts real-time statuses to the Flask central dashboard over Wi-Fi.

## Directory Structure

```
dms_mobile_app/
├── assets/
│   ├── emotion_model.ptl   # Converted PyTorch Mobile model checkpoint
│   └── warning.wav         # Alarm sound asset
├── lib/
│   ├── dms_processor.dart  # AI frame processing, math, and network synchronization
│   └── main.dart           # Application entrypoint, camera view, and UI dashboards
└── pubspec.yaml            # Project configuration and plugin dependencies
```

## Setup Instructions

### 1. Prerequisite: Install Flutter SDK
Make sure you have Flutter installed on your machine.
* Follow the official [Flutter installation guide](https://docs.flutter.dev/get-started/install) for your OS (Windows/macOS/Linux).
* Verify installation:
  ```bash
  flutter doctor
  ```

### 2. Prepare the Project
Navigate to this folder and fetch dependencies:
```bash
cd dms_mobile_app
flutter pub get
```

### 3. Add Android/iOS Native Permissions
Before running, you must grant Camera and Audio permissions in the native config files:

#### For Android:
Open `android/app/src/main/AndroidManifest.xml` and add these permissions inside the `<manifest>` tag:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

#### For iOS:
Open `ios/Runner/Info.plist` and add these descriptions inside the `<dict>` tag:
```xml
<key>NSCameraUsageDescription</key>
<string>This app requires camera access to monitor driver fatigue and emotions.</string>
<key>NSMicrophoneUsageDescription</key>
<string>This app requires microphone access to record audio (if needed).</string>
```

### 4. Run the Application
Connect your physical phone (with USB Debugging enabled) and run:
```bash
flutter run
```
Input your computer's local IP address (e.g. `http://192.168.1.15:5000`) and the vehicle credentials, then press **Start Monitoring** to begin streaming live.
