import cv2
import numpy as np
import mediapipe as mp
import config

class LandmarkEngine:
    """
    Trích xuất đặc trưng khuôn mặt (Landmarks) bằng MediaPipe FaceMesh
    Tính toán các chỉ số hình học: EAR, MAR, Head Pose
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Sử dụng MediaPipe Tasks API thay vì legacy solutions
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
        
        # 3D generic model cho solvePnP (OpenCV uses Y-down coordinate system)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (MP: 1)
            (0.0, 330.0, -65.0),         # Chin (MP: 152) -> +Y (below nose)
            (-225.0, -170.0, -135.0),    # Left eye left corner (MP: 33) -> -Y (above nose)
            (225.0, -170.0, -135.0),     # Right eye right corner (MP: 263) -> -Y (above nose)
            (-150.0, 150.0, -125.0),     # Left Mouth corner (MP: 61) -> +Y (below nose)
            (150.0, 150.0, -125.0)       # Right mouth corner (MP: 291) -> +Y (below nose)
        ])

    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def process(self, frame_bgr):
        """
        Nhận frame BGR (cv2 format), tự convert sang RGB nội bộ.
        Trả về dict gồm: EAR, MAR, Pitch, Roll, Yaw, BBox, và landmarks pts.
        """
        h, w = frame_bgr.shape[:2]
        # MediaPipe Tasks API cần mp.Image định dạng RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # 1. Bounding Box (dành cho face crop)
        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        x_max = max(p[0] for p in pts)
        y_max = max(p[1] for p in pts)
        bbox = (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max))

        # 2. EAR (Eye Aspect Ratio)
        # Right eye: 33, 160, 158, 133, 153, 144
        r_ear = self._compute_ear(pts, [33, 160, 158, 133, 153, 144])
        # Left eye: 362, 385, 387, 263, 373, 380
        l_ear = self._compute_ear(pts, [362, 385, 387, 263, 373, 380])
        ear = (r_ear + l_ear) / 2.0

        # 3. MAR (Mouth Aspect Ratio)
        # MP equivalents for Dlib P49-P55
        mar = self._compute_mar(pts)

        # 4. Head Pose (Pitch)
        pitch, roll, yaw = self._compute_head_pose(pts, w, h)

        return {
            "bbox": bbox,
            "ear": ear,
            "mar": mar,
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "pts": pts
        }

    def _compute_ear(self, pts, indices):
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in indices]
        vert1 = self._euclidean(p2, p6)
        vert2 = self._euclidean(p3, p5)
        horiz = self._euclidean(p1, p4)
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _compute_mar(self, pts):
        # Dlib P49=61, P51=37, P53=267, P55=291, P57=314, P59=84
        p49, p55 = pts[61], pts[291]
        p51, p59 = pts[37], pts[84]
        p53, p57 = pts[267], pts[314]

        vert1 = self._euclidean(p51, p59)
        vert2 = self._euclidean(p53, p57)
        horiz = self._euclidean(p49, p55)
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _compute_head_pose(self, pts, w, h):
        image_points = np.array([
            pts[1],     # Nose tip
            pts[152],   # Chin
            pts[33],    # Left eye corner
            pts[263],   # Right eye corner
            pts[61],    # Left mouth corner
            pts[291]    # Right mouth corner
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return pitch, roll, yaw

    def close(self):
        self.face_landmarker.close()
