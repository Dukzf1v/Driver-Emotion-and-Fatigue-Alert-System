import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:audioplayers/audioplayers.dart';
import 'dms_processor.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final List<CameraDescription> cameras = await availableCameras();
  runApp(DmsApp(cameras: cameras));
}

class DmsApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const DmsApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DMS Mobile Client',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        primaryColor: const Color(0xFF4A9EE8),
        scaffoldBackgroundColor: const Color(0xFF0F1319),
        cardColor: const Color(0xFF141920),
        colorScheme: const ColorScheme.dark().copyWith(
          secondary: const Color(0xFF4A9EE8),
          surface: const Color(0xFF141920),
        ),
      ),
      home: HomeScreen(cameras: cameras),
    );
  }
}

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const HomeScreen({super.key, required this.cameras});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // deployed server URL
  static const String _serverUrl = "https://dms-dashboard-udqh.onrender.com";

  // Form controllers (vehicle info only)
  final TextEditingController _vehicleIdController = TextEditingController(text: "36A-120.36");
  final TextEditingController _driverNameController = TextEditingController(text: "Phùng Thanh Độ");

  // Camera & Audio
  CameraController? _cameraController;
  final AudioPlayer _audioPlayer = AudioPlayer();
  DateTime _lastBeepTime = DateTime.now();

  // DMS Processor
  DmsProcessor? _dmsProcessor;

  // UI state
  bool _isMonitoring = false;
  bool _modelsLoaded = false;
  String _dmsState = "UNKNOWN";
  double _fScore = 0.0;
  double _angerLevel = 0.0;
  double _fearLevel = 0.0;

  @override
  void initState() {
    super.initState();
    _initDms();
  }

  Future<void> _initDms() async {
    _dmsProcessor = DmsProcessor(
      onMetricsUpdated: (state, fScore, anger, fear) {
        if (!mounted) return;
        setState(() {
          _dmsState = state;
          _fScore = fScore;
          _angerLevel = anger;
          _fearLevel = fear;
        });
      },
      onTriggerBeep: _playWarningBeep,
    );
    await _dmsProcessor!.init();
    if (!mounted) return;
    setState(() {
      _modelsLoaded = true;
    });
  }

  Future<void> _startMonitoring() async {
    if (_cameraController != null) return;
    if (widget.cameras.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No cameras available on this device.")),
      );
      return;
    }

    // Use front camera (normally the secondary camera on phones)
    final frontCamera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21,
    );

    try {
      await _cameraController!.initialize();
      
      // Start processing images
      _cameraController!.startImageStream((CameraImage image) {
        if (!_isMonitoring) return;
        final sensorOrientation = frontCamera.sensorOrientation;
        _dmsProcessor?.processImage(
          image,
          sensorOrientation,
          _serverUrl,
          _vehicleIdController.text.trim(),
          _driverNameController.text.trim(),
        );
      });

      if (mounted) {
        setState(() {
          _isMonitoring = true;
          _dmsState = "NORMAL";
        });
      }
    } catch (e) {
      print("Camera init error: $e");
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to initialize camera: $e")),
      );
    }
  }

  void _stopMonitoring() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _cameraController = null;
    if (mounted) {
      setState(() {
        _isMonitoring = false;
        _dmsState = "UNKNOWN";
        _fScore = 0.0;
        _angerLevel = 0.0;
        _fearLevel = 0.0;
      });
    }
  }

  Future<void> _playWarningBeep() async {
    final now = DateTime.now();
    if (now.difference(_lastBeepTime).inMilliseconds < 1500) return;
    _lastBeepTime = now;

    try {
      await _audioPlayer.play(AssetSource('warning.wav'));
    } catch (e) {
      print("Error playing warning sound: $e");
    }
  }

  @override
  void dispose() {
    _stopMonitoring();
    _audioPlayer.dispose();
    _dmsProcessor?.close();
    super.dispose();
  }

  // Get color depending on state
  Color _getStateColor() {
    switch (_dmsState) {
      case "NORMAL":
        return const Color(0xFF3DCC8E);
      case "FATIGUE":
        return const Color(0xFFF0A030);
      case "ANGRY":
        return const Color(0xFFE85A5A);
      case "FEAR":
        return const Color(0xFF8888F8);
      case "DISTRACTED":
        return const Color(0xFFF0D030);
      default:
        return const Color(0xFF8B949E);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Driver Monitor', style: TextStyle(fontFamily: 'Outfit', fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF141920),
        elevation: 0,
        actions: [
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: _getStateColor()),
                color: _getStateColor().withValues(alpha: 0.1),
              ),
              child: Row(
                children: [
                  Icon(Icons.wifi, size: 14, color: _getStateColor()),
                  const SizedBox(width: 4),
                  Text(_isMonitoring ? 'Active' : 'Ready', style: TextStyle(color: _getStateColor(), fontSize: 12, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          )
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 1. Loading status
              if (!_modelsLoaded)
                const Card(
                  child: Padding(
                    padding: EdgeInsets.all(16.0),
                    child: Row(
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(width: 16),
                        Text("Loading AI models (PyTorch Lite)..."),
                      ],
                    ),
                  ),
                ),

              // 2. Camera Preview Box
              if (_isMonitoring && _cameraController != null && _cameraController!.value.isInitialized)
                AspectRatio(
                  aspectRatio: 4 / 3,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        FittedBox(
                          fit: BoxFit.cover,
                          child: SizedBox(
                            width: 100,
                            height: 100 * _cameraController!.value.aspectRatio,
                            child: CameraPreview(_cameraController!),
                          ),
                        ),
                        // Mirrored effect for driver camera
                        Positioned.fill(
                          child: Align(
                            alignment: Alignment.center,
                            child: Container(),
                          ),
                        ),
                      ],
                    ),
                  ),
                )
              else if (_modelsLoaded)
                Container(
                  height: 200,
                  decoration: BoxDecoration(
                    color: const Color(0xFF141920),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: const Color(0xFF1E3050)),
                  ),
                  child: const Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.video_camera_front, size: 50, color: Color(0xFF8B949E)),
                      SizedBox(height: 8),
                      Text("Camera offline. Press start to monitor.", style: TextStyle(color: Color(0xFF8B949E))),
                    ],
                  ),
                ),

              const SizedBox(height: 16),

              // 3. State Banner
              if (_isMonitoring)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _getStateColor().withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: _getStateColor(), width: 2),
                  ),
                  child: Column(
                    children: [
                      Text(
                        _dmsState,
                        style: TextStyle(
                          color: _getStateColor(),
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'Outfit',
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _dmsState == "NORMAL"
                            ? "Driver is focused. Keep driving safe."
                            : "WARNING: Abnormal driver state detected!",
                        style: const TextStyle(fontSize: 13, color: Colors.white70),
                        textAlign: TextAlign.center,
                      )
                    ],
                  ),
                ),

              const SizedBox(height: 16),

              // 4. Live indicators progress bars
              if (_isMonitoring)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const Text("Live Parameters", style: TextStyle(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 12),
                        // Fatigue
                        _buildIndicatorRow("Fatigue", _fScore, const Color(0xFFF0A030)),
                        const SizedBox(height: 16),
                        // Anger
                        _buildIndicatorRow("Anger", _angerLevel, const Color(0xFFE85A5A)),
                        const SizedBox(height: 16),
                        // Fear
                        _buildIndicatorRow("Fear", _fearLevel, const Color(0xFF8888F8)),
                      ],
                    ),
                  ),
                ),

              const SizedBox(height: 16),

              // 5. Configurations Card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text("Driver Info", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                      const SizedBox(height: 4),
                      // Show server URL as read-only info
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1E2838),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.cloud_done, size: 14, color: Color(0xFF3DCC8E)),
                            const SizedBox(width: 8),
                            const Expanded(
                              child: Text(
                                "Connected to deployed server",
                                style: TextStyle(fontSize: 12, color: Color(0xFF3DCC8E)),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _vehicleIdController,
                        decoration: const InputDecoration(
                          labelText: "Vehicle ID (License Plate)",
                          border: OutlineInputBorder(),
                        ),
                        enabled: !_isMonitoring,
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _driverNameController,
                        decoration: const InputDecoration(
                          labelText: "Driver Name",
                          border: OutlineInputBorder(),
                        ),
                        enabled: !_isMonitoring,
                      ),
                      const SizedBox(height: 16),
                      if (!_isMonitoring)
                        ElevatedButton.icon(
                          onPressed: _modelsLoaded ? _startMonitoring : null,
                          icon: const Icon(Icons.play_arrow),
                          label: const Text("Start Monitoring"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF4A9EE8),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        )
                      else
                        ElevatedButton.icon(
                          onPressed: _stopMonitoring,
                          icon: const Icon(Icons.stop),
                          label: const Text("Stop Monitoring"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFE85A5A),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildIndicatorRow(String label, double value, Color barColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
            Text('${(value * 100).toStringAsFixed(1)}%', style: const TextStyle(fontFamily: 'Outfit', fontWeight: FontWeight.bold, fontSize: 14)),
          ],
        ),
        const SizedBox(height: 6),
        Stack(
          children: [
            Container(
              height: 10,
              decoration: BoxDecoration(
                color: const Color(0xFF1E2838),
                borderRadius: BorderRadius.circular(5),
              ),
            ),
            FractionallySizedBox(
              widthFactor: value.clamp(0.0, 1.0),
              child: Container(
                height: 10,
                decoration: BoxDecoration(
                  color: barColor,
                  borderRadius: BorderRadius.circular(5),
                ),
              ),
            ),
          ],
        )
      ],
    );
  }
}
