import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'dart:ui' show Size, Rect;
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_mesh_detection/google_mlkit_face_mesh_detection.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'package:image/image.dart' as img;

class DmsProcessor {
  // Configs
  static const double tEar = 0.15;
  static const double tMar = 0.55;
  static const double tPitch = -18.0;

  static const double fatigueThresh = 0.25;
  static const double angryThresh = 0.25;
  static const double fearThresh = 0.25;

  static const int fatigueWindowSize = 90;
  static const int emotionWindowSize = 90;

  // Indices
  static const int angerClassIdx = 1;
  static const int fearClassIdx = 2;
  static const int normalClassIdx = 0;
  static const int happinessClassIdx = 3;
  static const int sadnessClassIdx = 4;
  static const int faceLostTimeoutMs = 2000;

  // State
  ClassificationModel? _model;
  final FaceMeshDetector _faceDetector = FaceMeshDetector(
    option: FaceMeshDetectorOptions.faceMesh,
  );

  bool _isProcessing = false;
  int _frameCount = 0;

  // History lists
  final List<Map<String, dynamic>> _fatigueHistory = [];
  final List<double> _angerHistory = [];
  final List<double> _fearHistory = [];

  final List<List<double>> _cnnRawHistory = [];
  final int cnnSmoothWindow = 5;

  List<double> _currentProbs = [1.0, 0.0, 0.0, 0.0, 0.0];

  // ignore: unused_field
  int _blinkCount = 0; // Tracked in window; read value not needed for current f-score formula
  int _yawnCount = 0;
  int _nodCount = 0;

  bool _isBlinking = false;
  bool _isYawning = false;
  bool _isNodding = false;

  DateTime? _faceLostTimestamp;
  String _lastSentState = "NORMAL";

  // Callback to notify UI
  final Function(String state, double fScore, double anger, double fear) onMetricsUpdated;
  final Function() onTriggerBeep;

  DmsProcessor({
    required this.onMetricsUpdated,
    required this.onTriggerBeep,
  });

  Future<void> init() async {
    // Load PyTorch Mobile Lite model
    print('Attempting to load model from assets/emotion_model.ptl');
    try {
      _model = await PytorchLite.loadClassificationModel(
        "assets/emotion_model.ptl",
        224,
        224,
        5, // numClasses: normal, anger, fear, happiness, sadness
        labelPath: "assets/labels.txt", // explicit label path to prevent asset lookup error
      );
      // ignore: avoid_print
      print("PyTorch Mobile PTL model loaded successfully.");
    } catch (e) {
      // ignore: avoid_print
      print("Error loading PyTorch PTL model: $e");
    }
  }

  void close() {
    _faceDetector.close();
  }

  // Math Helper for FaceMeshPoint
  double _distance(FaceMeshPoint p1, FaceMeshPoint p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
  }

  // Compute Eye Aspect Ratio using 6 specific points
  double _computeEar(List<FaceMeshPoint> pts, List<int> indices) {
    if (pts.length < 468) return 0.3;
    final p1 = pts[indices[0]];
    final p2 = pts[indices[1]];
    final p3 = pts[indices[2]];
    final p4 = pts[indices[3]];
    final p5 = pts[indices[4]];
    final p6 = pts[indices[5]];

    final vert1 = _distance(p2, p6);
    final vert2 = _distance(p3, p5);
    final horiz = _distance(p1, p4);

    return (vert1 + vert2) / (2.0 * horiz + 1e-6);
  }

  // Compute Mouth Aspect Ratio using specific points
  double _computeMar(List<FaceMeshPoint> pts) {
    if (pts.length < 468) return 0.0;
    final p49 = pts[61];
    final p55 = pts[291];
    final p51 = pts[37];
    final p59 = pts[84];
    final p53 = pts[267];
    final p57 = pts[314];

    final vert1 = _distance(p51, p59);
    final vert2 = _distance(p53, p57);
    final horiz = _distance(p49, p55);

    return (vert1 + vert2) / (2.0 * horiz + 1e-6);
  }

  // Process a Camera Frame
  Future<void> processImage(CameraImage image, int sensorOrientation, String serverUrl, String vehicleId, String driverName) async {
    if (_isProcessing || _model == null) return;
    _isProcessing = true;
    _frameCount++;

    try {
      // 1. Convert CameraImage to InputImage for ML Kit
      // Concatenate all plane bytes using typed_data
      int totalBytes = 0;
      for (final Plane plane in image.planes) {
        totalBytes += plane.bytes.length;
      }
      final bytes = Uint8List(totalBytes);
      int offset = 0;
      for (final Plane plane in image.planes) {
        bytes.setRange(offset, offset + plane.bytes.length, plane.bytes);
        offset += plane.bytes.length;
      }

      final Size imageSize = Size(image.width.toDouble(), image.height.toDouble());
      final InputImageRotation imageRotation = InputImageRotationValue.fromRawValue(sensorOrientation) ?? InputImageRotation.rotation0deg;
      final InputImageFormat inputImageFormat = InputImageFormatValue.fromRawValue(image.format.raw) ?? InputImageFormat.nv21;

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: imageSize,
          rotation: imageRotation,
          format: inputImageFormat,
          bytesPerRow: image.planes[0].bytesPerRow,
        ),
      );

      // 2. Detect Faces
      final List<FaceMesh> faces = await _faceDetector.processImage(inputImage);

      if (faces.isNotEmpty) {
        _faceLostTimestamp = null;
        final FaceMesh face = faces.first;
        final points = face.points;

        // Calculate EAR using exact Python indices
        final rEar = _computeEar(points, [33, 160, 158, 133, 153, 144]);
        final lEar = _computeEar(points, [362, 385, 387, 263, 373, 380]);
        final ear = (lEar + rEar) / 2.0;

        // Calculate MAR using exact Python indices
        final mar = _computeMar(points);

        // Approximate Pitch (Nodding) using Z distance or just use 0
        final pitch = 0.0; // MediaPipe face mesh doesn't give euler angles directly

        // Calculate precise Bounding Box from points (like MediaPipe python)
        double minX = image.width.toDouble();
        double minY = image.height.toDouble();
        double maxX = 0;
        double maxY = 0;
        for (final p in points) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        final preciseBoundingBox = Rect.fromLTRB(minX, minY, maxX, maxY);

        // Running Emotion Model every 5 frames
        if (_frameCount % 5 == 1) {
          final faceJpegBytes = _cropFaceYuv(image, preciseBoundingBox, sensorOrientation);
          if (faceJpegBytes != null) {
            final List<double>? probs = await _model?.getImagePredictionList(
              faceJpegBytes,
              mean: [0.485, 0.456, 0.406],
              std: [0.229, 0.224, 0.225],
            );
            if (probs != null && probs.length >= 5) {
              _cnnRawHistory.add(probs);
              if (_cnnRawHistory.length > cnnSmoothWindow) {
                _cnnRawHistory.removeAt(0);
              }
              // Average raw probabilities (ProbSmoother)
              final smoothedProbs = List<double>.filled(5, 0.0);
              for (final raw in _cnnRawHistory) {
                for (int i = 0; i < 5; i++) {
                  smoothedProbs[i] += raw[i];
                }
              }
              for (int i = 0; i < 5; i++) {
                smoothedProbs[i] /= _cnnRawHistory.length;
              }
              _currentProbs = smoothedProbs;
            }
          }
        }

        // Dynamic thresholds based on Happiness/Sadness
        final pHappy = _currentProbs[happinessClassIdx];
        final pSad = _currentProbs[sadnessClassIdx];
        final dynamicTEar = tEar * max(0.6, 1.0 - (0.20 * pHappy) - (0.15 * pSad));
        final dynamicTMar = tMar * min(1.5, 1.0 + (0.30 * pHappy));

        // Update fatigue state machine
        Map<String, dynamic> state = {
          "ear": ear,
          "mar": mar,
          "pitch": pitch,
          "blink_event": 0,
          "yawn_event": 0,
          "nod_event": 0
        };

        if (ear < dynamicTEar) {
          _isBlinking = true;
        } else if (ear >= dynamicTEar && _isBlinking) {
          _isBlinking = false;
          state["blink_event"] = 1;
          _blinkCount++;
        }

        if (mar > dynamicTMar) {
          _isYawning = true;
        } else if (mar <= dynamicTMar && _isYawning) {
          _isYawning = false;
          state["yawn_event"] = 1;
          _yawnCount++;
        }

        if (pitch < tPitch) {
          _isNodding = true;
        } else if (pitch >= tPitch && _isNodding) {
          _isNodding = false;
          state["nod_event"] = 1;
          _nodCount++;
        }

        _fatigueHistory.add(state);
        if (_fatigueHistory.length > fatigueWindowSize) {
          final removed = _fatigueHistory.removeAt(0);
          _blinkCount -= removed["blink_event"] as int;
          _yawnCount -= removed["yawn_event"] as int;
          _nodCount -= removed["nod_event"] as int;
        }

        // Calculate F_score
        final closedFrames = _fatigueHistory.where((s) => s["ear"] < dynamicTEar).length;
        final perclos = closedFrames / (_fatigueHistory.length);
        final fYawn = min(1.0, _yawnCount / 2.0);
        final fNod = min(1.0, _nodCount / 2.0);
        final fScore = (0.7 * perclos) + (0.15 * fYawn) + (0.15 * fNod);

        // Smoothed emotions
        _angerHistory.add(_currentProbs[angerClassIdx]);
        if (_angerHistory.length > emotionWindowSize) _angerHistory.removeAt(0);
        _fearHistory.add(_currentProbs[fearClassIdx]);
        if (_fearHistory.length > emotionWindowSize) _fearHistory.removeAt(0);

        final avgAnger = _angerHistory.reduce((a, b) => a + b) / _angerHistory.length;
        final avgFear = _fearHistory.reduce((a, b) => a + b) / _fearHistory.length;

        // Decision rule
        final fatigueExcess = max(0.0, fScore - fatigueThresh);
        final angryExcess = max(0.0, avgAnger - angryThresh);
        final fearExcess = max(0.0, avgFear - fearThresh);

        String finalState = "NORMAL";
        if (fatigueExcess > 0.0 || angryExcess > 0.0 || fearExcess > 0.0) {
          final maxExcess = max(fatigueExcess, max(angryExcess, fearExcess));
          if (maxExcess == fatigueExcess) {
            finalState = "FATIGUE";
          } else if (maxExcess == angryExcess) {
            finalState = "ANGRY";
          } else {
            finalState = "FEAR";
          }
        }

        // Notify UI
        onMetricsUpdated(finalState, fScore, avgAnger, avgFear);

        // Sound alert
        if (finalState != "NORMAL") {
          onTriggerBeep();
        }

        // Sync with dashboard API
        if (_frameCount % 15 == 0 || finalState != _lastSentState) {
          _sendToDashboard(serverUrl, vehicleId, driverName, finalState, fScore, avgAnger, avgFear);
          _lastSentState = finalState;
        }

      } else {
        // Face Lost
        _faceLostTimestamp ??= DateTime.now();
        final lostDuration = DateTime.now().difference(_faceLostTimestamp!).inMilliseconds;
        String finalState = "NORMAL";
        if (lostDuration > faceLostTimeoutMs) {
          finalState = "DISTRACTED";
        }

        // UI update
        onMetricsUpdated(finalState, 0.0, 0.0, 0.0);

        if (finalState == "DISTRACTED") {
          onTriggerBeep();
        }

        if (_frameCount % 15 == 0 || finalState != _lastSentState) {
          _sendToDashboard(serverUrl, vehicleId, driverName, finalState, 0.0, 0.0, 0.0);
          _lastSentState = finalState;
        }
      }

    } catch (e) {
      // ignore: avoid_print
      print("DMS processing frame error: $e");
    } finally {
      _isProcessing = false;
    }
  }

  // Network POST
  void _sendToDashboard(String serverUrl, String vehicleId, String driverName, String state, double fScore, double anger, double fear) {
    final url = Uri.parse('$serverUrl/api/update');
    http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: json.encode({
        "vehicle_id": vehicleId,
        "driver_name": driverName,
        "state": state,
        "f_score": fScore,
        "anger_level": anger,
        "fear_level": fear,
      }),
    ).then((res) {
      // success
    }).catchError((err) {
      // ignore: avoid_print
      print("Failed to sync status with server: $err");
    });
  }

  Uint8List? _cropFaceYuv(CameraImage image, Rect boundingBox, int sensorOrientation) {
    try {
      final originalWidth = image.width;
      final originalHeight = image.height;

      // Rotated sizes
      final int rotatedWidth = (sensorOrientation == 90 || sensorOrientation == 270) ? originalHeight : originalWidth;
      final int rotatedHeight = (sensorOrientation == 90 || sensorOrientation == 270) ? originalWidth : originalHeight;

      // Add 20% padding
      const double pad = 0.20;
      final double bw = boundingBox.width;
      final double bh = boundingBox.height;

      final int rx1 = max(0, (boundingBox.left - bw * pad).toInt());
      final int ry1 = max(0, (boundingBox.top - bh * pad).toInt());
      final int rx2 = min(rotatedWidth - 1, (boundingBox.right + bw * pad).toInt());
      final int ry2 = min(rotatedHeight - 1, (boundingBox.bottom + bh * pad).toInt());

      final int cropWidth = rx2 - rx1 + 1;
      final int cropHeight = ry2 - ry1 + 1;

      if (cropWidth <= 0 || cropHeight <= 0) return null;

      // Create destination image
      final croppedImg = img.Image(width: cropWidth, height: cropHeight);

      // YUV planes
      final yPlane = image.planes[0];
      final yBytes = yPlane.bytes;
      final yRowStride = yPlane.bytesPerRow;
      final yPixelStride = yPlane.bytesPerPixel ?? 1;

      final bool hasUVPlanes = image.planes.length >= 3;
      final bool has2Planes = image.planes.length == 2;
      
      final uPlane = hasUVPlanes || has2Planes ? image.planes[1] : null;
      final vPlane = hasUVPlanes ? image.planes[2] : null;

      final uBytes = uPlane?.bytes;
      final vBytes = vPlane?.bytes;

      final uvRowStride = uPlane?.bytesPerRow ?? originalWidth;
      final uvPixelStride = uPlane?.bytesPerPixel ?? 2;

      for (int ry = ry1; ry <= ry2; ry++) {
        for (int rx = rx1; rx <= rx2; rx++) {
          // Map rotated coordinates (rx, ry) to original coordinates (ux, uy)
          int ux = rx;
          int uy = ry;

          if (sensorOrientation == 90) {
            ux = ry;
            uy = originalHeight - 1 - rx;
          } else if (sensorOrientation == 270) {
            ux = originalWidth - 1 - ry;
            uy = rx;
          } else if (sensorOrientation == 180) {
            ux = originalWidth - 1 - rx;
            uy = originalHeight - 1 - ry;
          }

          if (ux < 0 || ux >= originalWidth || uy < 0 || uy >= originalHeight) continue;

          // Read Y
          int yIdx = uy * yRowStride + ux * yPixelStride;
          if (yIdx >= yBytes.length) continue;
          final int yVal = yBytes[yIdx];

          // Read U & V
          int u = 128;
          int v = 128;

          if (image.planes.length == 1) {
            // Packed NV21 (1 plane)
            final uvOffset = originalWidth * originalHeight;
            final uvx = ux >> 1;
            final uvy = uy >> 1;
            final uvIdx = uvOffset + (uvy * originalWidth) + (uvx * 2);
            if (uvIdx < yBytes.length - 1) {
              v = yBytes[uvIdx];
              u = yBytes[uvIdx + 1];
            }
          } else {
            final int uvx = ux >> 1;
            final int uvy = uy >> 1;
            int uvIdx = uvy * uvRowStride + uvx * uvPixelStride;

            if (vPlane != null) {
              // YUV 420 3 planes
              if (uvIdx < uBytes!.length) u = uBytes[uvIdx];
              int vIdx = uvy * vPlane.bytesPerRow + uvx * (vPlane.bytesPerPixel ?? 1);
              if (vIdx < vBytes!.length) v = vBytes[vIdx];
            } else if (uPlane != null) {
              // NV21 or NV12 2 planes
              if (uvIdx < uBytes!.length) {
                if (image.format.raw == 35 || image.format.raw == 842094169) {
                  // NV21: VU interleaved
                  v = uBytes[uvIdx];
                  if (uvIdx + 1 < uBytes.length) {
                    u = uBytes[uvIdx + 1];
                  }
                } else {
                  // NV12: UV interleaved
                  u = uBytes[uvIdx];
                  if (uvIdx + 1 < uBytes.length) {
                    v = uBytes[uvIdx + 1];
                  }
                }
              }
            }
          }

          // Convert YUV to RGB
          final int r = (yVal + 1.402 * (v - 128)).round().clamp(0, 255);
          final int g = (yVal - 0.344136 * (u - 128) - 0.714136 * (v - 128)).round().clamp(0, 255);
          final int b = (yVal + 1.772 * (u - 128)).round().clamp(0, 255);

          croppedImg.setPixelRgb(rx - rx1, ry - ry1, r, g, b);
        }
      }

      // Encode image to JPEG
      return Uint8List.fromList(img.encodeJpg(croppedImg));
    } catch (e) {
      // ignore: avoid_print
      print("Error in _cropFaceYuv: $e");
      return null;
    }
  }
}
