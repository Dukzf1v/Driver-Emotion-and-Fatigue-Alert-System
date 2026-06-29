import cv2
import numpy as np
import mediapipe as mp
import config

class LandmarkEngine:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = mp.tasks.BaseOptions(model_asset_path=config.LANDMARK_MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        
        # 3D generic model (Anthropometric) in mm.
        # Origin (0,0,0) at Nose Tip (MP: 1).
        # OpenCV coordinate: X right, Y down, Z from back to front camera.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (MP: 1)
            (0.0, 95.0, -45.0),          # Chin (MP: 152) -> below nose (+Y) and behind (-Z)
            (-65.0, -50.0, -45.0),       # Outer right eye corner (MP: 33)
            (65.0, -50.0, -45.0),        # Outer left eye corner (MP: 263)
            (-50.0, 45.0, -35.0),        # Right mouth corner (MP: 61)
            (50.0, 45.0, -35.0)          # Left mouth corner (MP: 291)
        ], dtype=np.float64)

    def _euclidean(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def process(self, frame_bgr):
        """
        Receive BGR frame, convert to RGB.
        Return result dict or None if face not detected.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        
        # Keep float32 for smooth Pose and EAR/MAR calculation
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

        # 1. Bounding Box using Numpy
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        bbox = (max(0, int(x_min)), max(0, int(y_min)), min(w, int(x_max)), min(h, int(y_max)))

        # 2. EAR (Eye Aspect Ratio)
        r_ear = self._compute_ear(pts, [33, 160, 158, 133, 153, 144])
        l_ear = self._compute_ear(pts, [362, 385, 387, 263, 373, 380])
        ear = (r_ear + l_ear) / 2.0

        # 3. MAR (Mouth Aspect Ratio) - Using standard inner lip landmarks
        mar = self._compute_mar(pts)

        # 4. Head Pose (Calculate Pitch, Roll, Yaw)
        pitch, roll, yaw = self._compute_head_pose(pts, w, h)

        return {
            "bbox": bbox,
            "ear": ear,
            "mar": mar,
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "pts": pts.astype(np.int32).tolist() # Convert to int for OpenCV drawing
        }

    def _compute_ear(self, pts, indices):
        p1, p2, p3, p4, p5, p6 = pts[indices]
        vert1 = self._euclidean(p2, p6)
        vert2 = self._euclidean(p3, p5)
        horiz = self._euclidean(p1, p4)
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _compute_mar(self, pts):
        # Standard inner lip landmarks
        p_left, p_right = pts[78], pts[308]
        p_top1, p_bottom1 = pts[81], pts[178]
        p_top2, p_bottom2 = pts[311], pts[402]

        vert1 = self._euclidean(p_top1, p_bottom1)
        vert2 = self._euclidean(p_top2, p_bottom2)
        horiz = self._euclidean(p_left, p_right)
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _compute_head_pose(self, pts, w, h):
        image_points = np.array([
            pts[1],     # Nose tip (MediaPipe index: 1)
            pts[152],   # Chin
            pts[33],    # Outer right eye corner (MP: 33)
            pts[263],   # Outer left eye corner (MP: 263)
            pts[61],    # Right mouth corner (MP: 61)
            pts[291]    # Left mouth corner (MP: 291)
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180

        yaw = -yaw
        roll = -roll

        return pitch, roll, yaw

    def close(self):
        self.face_landmarker.close()
