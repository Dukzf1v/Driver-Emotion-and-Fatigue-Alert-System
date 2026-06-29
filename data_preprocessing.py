import os
import re
import cv2
import sys
import shutil
import random
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from typing import Optional, Tuple, List

# CONFIGURATION
BASE_DIR     = Path(__file__).parent
EMOTION_DIR  = BASE_DIR / "Emotion Dataset"

RAF_DB_DIR   = EMOTION_DIR / "RAF-DB"
MLIDER_DIR   = EMOTION_DIR / "MLI-DER"
KMUFED_DIR   = EMOTION_DIR / "KMU-FED"
AFFECTNET_DIR= EMOTION_DIR / "AffectNet"
FER2013_DIR  = EMOTION_DIR / "FER-2013"
SFEW_DIR     = EMOTION_DIR / "SFEW"
KDEF_DIR     = EMOTION_DIR / "KDEF"

# Output directory
OUTPUT_DIR   = BASE_DIR / "Unified_DMS_Dataset_v4"

# Processing parameters
IMG_SIZE     = (224, 224)      
TRAIN_RATIO  = 0.80            
RANDOM_SEED  = 42              

FACE_MODEL_PATH = str(BASE_DIR / "models" / "blaze_face_short_range.tflite")
MIN_DETECTION_CONFIDENCE = 0.5
BBOX_PADDING = 0.25

CLASS_FOLDERS = {
    0: "0_Neutral",
    1: "1_Anger",
    2: "2_Fear",
    3: "3_Happiness",
    4: "4_Sadness",
}

# SETUP LOGGING
import io

_stdout_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_stdout_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 
_log_file = str(OUTPUT_DIR / "preprocessing_v4.log")
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_stdout_handler, _file_handler]
)
logger = logging.getLogger(__name__)


# STATISTICS
class Stats:
    """Simple counter to track dataset processing statistics."""
    def __init__(self):
        self.processed    = 0   # images successfully saved
        self.no_face      = 0   # images dropped: no face detected
        self.skipped      = 0   # images skipped: wrong/ignored class
        self.errors       = 0   # images dropped: read / decode errors

    def report(self, dataset_name: str):
        total_attempted = self.processed + self.no_face + self.errors
        logger.info(
            f"  [{dataset_name}] Processed={self.processed:,}  |  "
            f"No-face dropped={self.no_face:,}  |  "
            f"Skipped (ignored class)={self.skipped:,}  |  "
            f"Errors={self.errors:,}  |  "
            f"Attempted={total_attempted:,}"
        )


# FACE DETECTION HELPER
class FaceDetector:
    def __init__(self,
                 model_path: str = FACE_MODEL_PATH,
                 min_confidence: float = MIN_DETECTION_CONFIDENCE):
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect_and_crop(self, img_bgr: np.ndarray,
                        padding: float = BBOX_PADDING
                        ) -> Optional[np.ndarray]:
        """
        Detect the most-confident face in the image and return a padded BGR crop.
        Returns None if no face is detected.

        Args:
            img_bgr : Input image in BGR format (as loaded by cv2).
            padding : Fractional padding added around the detected bounding box.

        Returns:
            Cropped + padded face region (BGR) or None.
        """
        h, w = img_bgr.shape[:2]

        # Tasks API requires an mp.Image in RGB format
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        # Pick the detection with the highest score (most confident)
        best = max(result.detections, key=lambda d: d.categories[0].score)

        # BoundingBox in Tasks API uses absolute pixel coordinates
        bbox = best.bounding_box
        x1_abs = bbox.origin_x
        y1_abs = bbox.origin_y
        bw_abs = bbox.width
        bh_abs = bbox.height

        # Apply padding
        pad_x = int(bw_abs * padding)
        pad_y = int(bh_abs * padding)

        x1 = max(0, x1_abs - pad_x)
        y1 = max(0, y1_abs - pad_y)
        x2 = min(w, x1_abs + bw_abs + pad_x)
        y2 = min(h, y1_abs + bh_abs + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return img_bgr[y1:y2, x1:x2]

    def close(self):
        """Release native resources held by the detector."""
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()





# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def make_output_dirs(output_root: Path):
    """Create the full output folder tree."""
    for split in ["train", "val"]:
        for cls_folder in CLASS_FOLDERS.values():
            (output_root / split / cls_folder).mkdir(parents=True, exist_ok=True)
    for test_dataset in ["affectnet", "fer2013", "rafdb"]:
        for cls_folder in CLASS_FOLDERS.values():
            (output_root / "test" / test_dataset / cls_folder).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory tree created at: {output_root}")


def load_and_crop_face(image_path: Path,
                       detector: FaceDetector) -> Optional[np.ndarray]:
    """
    Load an image, detect + crop the face, and resize to IMG_SIZE.

    Returns:
        Processed numpy array (H, W, 3) BGR, or None if face not found.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    face_crop = detector.detect_and_crop(img)
    if face_crop is None:
        return None

    resized = cv2.resize(face_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def save_image(img: np.ndarray, dst_path: Path) -> bool:
    """Save a numpy BGR image to disk. Returns True on success."""
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), img)
        return True
    except Exception as e:
        logger.warning(f"Failed to save {dst_path}: {e}")
        return False


def train_val_split(file_list: List[Path],
                    train_ratio: float = TRAIN_RATIO,
                    seed: int = RANDOM_SEED) -> Tuple[List[Path], List[Path]]:
    """Randomly split a file list into train/val sets."""
    rng = random.Random(seed)
    shuffled = file_list.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def get_unique_filename(dst_dir: Path, stem: str, ext: str = ".jpg") -> Path:
    """
    Generate a unique file path to avoid name collisions when merging
    multiple datasets into the same directory.
    """
    candidate = dst_dir / f"{stem}{ext}"
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = dst_dir / f"{stem}_{counter:04d}{ext}"
    return candidate


def process_and_save(src_path: Path,
                     dst_dir: Path,
                     detector: FaceDetector,
                     stats: Stats,
                     dataset_prefix: str = "",
                     dms_cls: int = -1) -> bool:
    """Full pipeline: load → detect face → crop → resize → save.

    Returns True if the image was successfully saved.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        stats.errors += 1
        logger.debug(f"Cannot read: {src_path}")
        return False

    face_crop = detector.detect_and_crop(img)
    if face_crop is None:
        stats.no_face += 1
        logger.debug(f"No face detected: {src_path}")
        return False

    resized = cv2.resize(face_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Build unique output filename preserving source stem
    stem = f"{dataset_prefix}_{src_path.stem}" if dataset_prefix else src_path.stem
    dst_path = get_unique_filename(dst_dir, stem)

    if save_image(resized, dst_path):
        stats.processed += 1
        return True
    else:
        stats.errors += 1
        return False


# ---------------------------------------------------------------------------
# DATASET PROCESSOR: AffectNet
# ---------------------------------------------------------------------------

AFFECTNET_CLASS_MAP = {
    "neutral":  0,
    "anger":    1,
    "disgust":  1,  # Merged with Anger
    "fear":     2,
    "happy":    3,
    "sad":      4,
}

def process_affectnet(output_root: Path, detector: FaceDetector):
    """
    Process AffectNet dataset.
    - 'Train' folder is pooled and split into train/val (80/20).
    - 'Test' folder goes entirely to the 'test' split (replacing FED-RO).
    """
    logger.info("=" * 60)
    logger.info("Processing: AffectNet")
    logger.info("=" * 60)

    stats = Stats()

    # ── 1. Process Train folder (split into train/val) ──
    train_src = AFFECTNET_DIR / "Train"
    if train_src.exists():
        class_files: dict = {i: [] for i in range(5)}
        for folder_name, dms_cls in AFFECTNET_CLASS_MAP.items():
            cls_dir = train_src / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            class_files[dms_cls].extend(img_files)

        for dms_cls, files in class_files.items():
            if not files:
                continue

            train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
            cls_folder = CLASS_FOLDERS[dms_cls]

            logger.info(f"  [Train Split] Class {dms_cls} ({cls_folder}): "
                        f"{len(train_files)} train + {len(val_files)} val")

            for split, file_list in [("train", train_files), ("val", val_files)]:
                dst_dir = output_root / split / cls_folder
                for src in file_list:
                    process_and_save(src, dst_dir, detector, stats,
                                     dataset_prefix="affectnet_train",
                                     dms_cls=dms_cls)
    else:
        logger.warning(f"AffectNet Train folder not found at {train_src}")

    # ── 2. Process Test folder (100% to test split) ──
    test_src = AFFECTNET_DIR / "Test"
    if test_src.exists():
        for folder_name, dms_cls in AFFECTNET_CLASS_MAP.items():
            cls_dir = test_src / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            
            cls_folder = CLASS_FOLDERS[dms_cls]
            logger.info(f"  [Test Split] Class {dms_cls} ({cls_folder}): {len(img_files)} images -> test/affectnet/")
            dst_dir = output_root / "test" / "affectnet" / cls_folder

            for src in img_files:
                process_and_save(src, dst_dir, detector, stats,
                                 dataset_prefix="affectnet_test",
                                 dms_cls=dms_cls)
    else:
        logger.warning(f"AffectNet Test folder not found at {test_src}")

    stats.report("AffectNet")


# ---------------------------------------------------------------------------
# DATASET PROCESSOR: RAF-DB
# ---------------------------------------------------------------------------
# Structure:
#   RAF-DB/
#   ├── train/
#   │   ├── 1/   <- Surprise 
#   │   ├── 2/   <- Fear     
#   │   ├── 3/   <- Disgust   
#   │   ├── 4/   <- Happiness
#   │   ├── 5/   <- Sadness 
#   │   ├── 6/   <- Anger    
#   │   └── 7/   <- Neutral   
#   └── test/
#       └── (same structure)
# ---------------------------------------------------------------------------

# RAF-DB folder number -> DMS class (5-class)
RAFDB_CLASS_MAP = {
    "2":  2,   # Fear      -> Class 2
    "3":  1,   # Disgust   -> Class 1 (Merged with Anger)
    "4":  3,   # Happiness -> Class 3
    "5":  4,   # Sadness   -> Class 4
    "6":  1,   # Anger     -> Class 1
    "7":  0,   # Neutral   -> Class 0
}


def process_rafdb(output_root: Path, detector: FaceDetector):
    """
    Process RAF-DB dataset.
    - 'train' folder is split 80/20 into train/val splits.
    - 'test' folder is sent 100% to 'test/rafdb'.
    """
    logger.info("=" * 60)
    logger.info("Processing: RAF-DB")
    logger.info("=" * 60)

    stats = Stats()

    # ── 1. Process Train folder (split into train/val) ──
    train_src = RAF_DB_DIR / "train"
    if train_src.exists():
        class_files: dict = {i: [] for i in range(5)}
        for folder_name, dms_cls in RAFDB_CLASS_MAP.items():
            cls_dir = train_src / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            class_files[dms_cls].extend(img_files)

        for dms_cls, files in class_files.items():
            if not files:
                continue

            train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
            cls_folder = CLASS_FOLDERS[dms_cls]

            logger.info(f"  [Train Split] Class {dms_cls} ({cls_folder}): "
                        f"{len(train_files)} train + {len(val_files)} val")

            for split, file_list in [("train", train_files), ("val", val_files)]:
                dst_dir = output_root / split / cls_folder
                for src in file_list:
                    process_and_save(src, dst_dir, detector, stats,
                                     dataset_prefix="rafdb_train",
                                     dms_cls=dms_cls)
    else:
        logger.warning(f"RAF-DB train folder not found at {train_src}")

    # ── 2. Process Test folder (100% to test/rafdb split) ──
    test_src = RAF_DB_DIR / "test"
    if test_src.exists():
        for folder_name, dms_cls in RAFDB_CLASS_MAP.items():
            cls_dir = test_src / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            
            cls_folder = CLASS_FOLDERS[dms_cls]
            logger.info(f"  [Test Split] Class {dms_cls} ({cls_folder}): {len(img_files)} images -> test/rafdb/")
            dst_dir = output_root / "test" / "rafdb" / cls_folder

            for src in img_files:
                process_and_save(src, dst_dir, detector, stats,
                                 dataset_prefix="rafdb_test",
                                 dms_cls=dms_cls)
    else:
        logger.warning(f"RAF-DB test folder not found at {test_src}")

    stats.report("RAF-DB")


# ---------------------------------------------------------------------------
# DATASET PROCESSOR: MLI-DER
# ---------------------------------------------------------------------------
# Structure:
#   MLI-DER dataset/
#   └── image data/
#       ├── Normal/       <- various lighting conditions
#       ├── littleBright/
#       ├── littleDark/
#       └── veryBright/
#           Files: subject{N}-{emotion}-{idx}.jpg
#           Emotions: angry, happy, normal
#
# MAPPING (2-class):
# MAPPING (5-class):
#   angry, disgust -> Class 1 (TỨC GIẬN)
#   happy          -> Class 3 (HẠNH PHÚC)
#   normal         -> Class 0 (TRUNG TÍNH)
# ---------------------------------------------------------------------------

# MLI-DER emotion keyword -> DMS class (5-class)
MLIDER_EMOTION_MAP = {
    "angry":  1,   # Anger     -> Class 1
    "happy":  3,   # Happiness -> Class 3
    "normal": 0,   # Neutral   -> Class 0
}


def parse_mlider_emotion(filename: str) -> Optional[int]:
    """
    Parse the DMS class from an MLI-DER filename.
    Format: subject{N}-{emotion}-{idx}.jpg
    Example: subject11-angry-5.jpg
    """
    parts = filename.lower().split("-")
    if len(parts) >= 2:
        emotion_keyword = parts[1]
        return MLIDER_EMOTION_MAP.get(emotion_keyword, None)
    return None


def process_mlider(output_root: Path, detector: FaceDetector):
    """
    Process MLI-DER dataset across all lighting condition subfolders.
    """
    logger.info("=" * 60)
    logger.info("Processing: MLI-DER Dataset")
    logger.info("=" * 60)

    stats = Stats()
    image_data_dir = MLIDER_DIR / "image data"

    if not image_data_dir.exists():
        logger.error(f"MLI-DER image data directory not found: {image_data_dir}")
        return

    # Lighting condition subfolders - all treated equally
    lighting_folders = [d for d in image_data_dir.iterdir() if d.is_dir()]
    if not lighting_folders:
        logger.warning("No lighting subfolders found in MLI-DER")
        return

    # Collect all images per DMS class (pooling across lighting conditions)
    class_files: dict = {i: [] for i in range(5)}

    for light_folder in sorted(lighting_folders):
        img_files = sorted([f for f in light_folder.iterdir()
                            if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for f in img_files:
            dms_cls = parse_mlider_emotion(f.name)
            if dms_cls is None:
                stats.skipped += 1
            elif dms_cls in class_files:
                class_files[dms_cls].append(f)

    # Split and process per class
    for dms_cls, files in class_files.items():
        if not files:
            continue

        train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
        cls_folder = CLASS_FOLDERS[dms_cls]

        logger.info(f"  Class {dms_cls} ({cls_folder}): "
                    f"{len(train_files)} train + {len(val_files)} val "
                    f"= {len(files)} total source images")

        for split, file_list in [("train", train_files), ("val", val_files)]:
            dst_dir = output_root / split / cls_folder
            for src in file_list:
                process_and_save(src, dst_dir, detector, stats,
                                 dataset_prefix="mlider",
                                 dms_cls=dms_cls)

    stats.report("MLI-DER")


# ---------------------------------------------------------------------------
# DATASET PROCESSOR: KMU-FED
# ---------------------------------------------------------------------------
# Structure:
#   KMU-FED/  (flat directory)
#     Files: {subject_id}_{EMOTION_CODE}_{session_id}_{frame_num}.jpg
#     Example: 01_AN_mr_001.jpg
#
#     Emotion codes (5-class mapping):
#       AN = Anger   -> Class 1
#       DI = Disgust -> Class 1 (Merged with Anger)
#       FE = Fear    -> Class 2
#       HA = Happy   -> Class 3
#       SA = Sadness -> Class 4
# ---------------------------------------------------------------------------

# KMU-FED emotion code -> DMS class (5-class)
KMUFED_EMOTION_MAP = {
    "AN":  1,    # Anger   -> Class 1
    "DI":  1,    # Disgust -> Class 1 (Merged with Anger)
    "FE":  2,    # Fear    -> Class 2
    "HA":  3,    # Happy   -> Class 3
    "SA":  4,    # Sadness -> Class 4
}


def parse_kmufed_emotion(filename: str) -> Optional[int]:
    """
    Parse DMS class from KMU-FED filename.
    Format: {id}_{EMOTION_CODE}_{session}_{frame}.jpg
    Example: 01_AN_mr_001.jpg
    """
    # Remove extension, split on underscore
    parts = Path(filename).stem.split("_")
    if len(parts) >= 2:
        emotion_code = parts[1].upper()
        return KMUFED_EMOTION_MAP.get(emotion_code, None)  # None = unknown code -> skip
    return None


def process_kmufed(output_root: Path, detector: FaceDetector):
    """
    Process KMU-FED dataset (flat directory of images).
    """
    logger.info("=" * 60)
    logger.info("Processing: KMU-FED")
    logger.info("=" * 60)

    stats = Stats()

    if not KMUFED_DIR.exists():
        logger.error(f"KMU-FED directory not found: {KMUFED_DIR}")
        return

    img_files = sorted([f for f in KMUFED_DIR.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

    if not img_files:
        logger.warning("No images found in KMU-FED directory")
        return

    # Group by DMS class
    class_files: dict = {i: [] for i in range(5)}

    for f in img_files:
        dms_cls = parse_kmufed_emotion(f.name)
        if dms_cls is None or dms_cls == -1:
            stats.skipped += 1
        elif dms_cls in class_files:
            class_files[dms_cls].append(f)

    # Split and process
    for dms_cls, files in class_files.items():
        if not files:
            continue

        train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
        cls_folder = CLASS_FOLDERS[dms_cls]

        logger.info(f"  Class {dms_cls} ({cls_folder}): "
                    f"{len(train_files)} train + {len(val_files)} val "
                    f"= {len(files)} total source images")

        for split, file_list in [("train", train_files), ("val", val_files)]:
            dst_dir = output_root / split / cls_folder
            for src in file_list:
                process_and_save(src, dst_dir, detector, stats,
                                 dataset_prefix="kmufed",
                                 dms_cls=dms_cls)

    stats.report("KMU-FED")




# ---------------------------------------------------------------------------
# DATASET PROCESSOR: FER-2013
# ---------------------------------------------------------------------------
FER2013_CLASS_MAP = {
    "neutral":  0,
    "angry":    1,
    "disgust":  1,  # Merged with Anger
    "fear":     2,
    "happy":    3,
    "sad":      4,
}

def process_fer2013(output_root: Path, detector: FaceDetector):
    """
    Process FER-2013 dataset.
    - 'Train' folder is pooled and split into train/val (80/20).
    - 'Test' folder goes entirely to the 'test' split.
    """
    logger.info("=" * 60)
    logger.info("Processing: FER-2013")
    logger.info("=" * 60)

    stats = Stats()

    # ── 1. Process Train folder (split into train/val) ──
    train_src = FER2013_DIR / "train"
    if train_src.exists():
        class_files: dict = {i: [] for i in range(5)}
        for folder_name, dms_cls in FER2013_CLASS_MAP.items():
            cls_dir = train_src / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            class_files[dms_cls].extend(img_files)

        for dms_cls, files in class_files.items():
            if not files:
                continue

            train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
            cls_folder = CLASS_FOLDERS[dms_cls]

            logger.info(f"  [Train Split] Class {dms_cls} ({cls_folder}): "
                        f"{len(train_files)} train + {len(val_files)} val")

            for split, file_list in [("train", train_files), ("val", val_files)]:
                dst_dir = output_root / split / cls_folder
                for src in file_list:
                    process_and_save(src, dst_dir, detector, stats,
                                     dataset_prefix="fer2013_train",
                                     dms_cls=dms_cls)
    else:
        logger.warning(f"FER-2013 train folder not found at {train_src}")

    # ── 2. Process Test folder (100% to test split) ──
    test_src = FER2013_DIR / "test"
    if test_src.exists():
        for folder_name, dms_cls in FER2013_CLASS_MAP.items():
            cls_dir = test_src / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir()
                                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            
            cls_folder = CLASS_FOLDERS[dms_cls]
            logger.info(f"  [Test Split] Class {dms_cls} ({cls_folder}): {len(img_files)} images -> test/fer2013/")
            dst_dir = output_root / "test" / "fer2013" / cls_folder

            for src in img_files:
                process_and_save(src, dst_dir, detector, stats,
                                 dataset_prefix="fer2013_test",
                                 dms_cls=dms_cls)
    else:
        logger.warning(f"FER-2013 test folder not found at {test_src}")

    stats.report("FER-2013")


# ---------------------------------------------------------------------------
# DATASET PROCESSOR: KDEF
# ---------------------------------------------------------------------------

KDEF_CLASS_MAP = {
    "neutral":  0,
    "angry":    1,
    "disgust":  1,  # Merged
    "fear":     2,
    "happy":    3,
    "sad":      4,
}

def process_kdef(output_root: Path, detector: FaceDetector):
    """
    Process KDEF dataset.
    Split 80/20 into train/val.
    """
    logger.info("=" * 60)
    logger.info("Processing: KDEF")
    logger.info("=" * 60)
    
    stats = Stats()
    class_files: dict = {i: [] for i in range(5)}
    
    for folder_name, dms_cls in KDEF_CLASS_MAP.items():
        cls_dir = KDEF_DIR / folder_name
        if not cls_dir.exists():
            continue
        img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        class_files[dms_cls].extend(img_files)

    for dms_cls, files in class_files.items():
        if not files:
            continue
        train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
        cls_folder = CLASS_FOLDERS[dms_cls]
        
        logger.info(f"  Class {dms_cls} ({cls_folder}): {len(train_files)} train + {len(val_files)} val = {len(files)} total source images")
        
        for split, file_list in [("train", train_files), ("val", val_files)]:
            dst_dir = output_root / split / cls_folder
            for src in file_list:
                process_and_save(src, dst_dir, detector, stats, dataset_prefix="kdef", dms_cls=dms_cls)

    stats.report("KDEF")

# ---------------------------------------------------------------------------
# DATASET PROCESSOR: SFEW
# ---------------------------------------------------------------------------

SFEW_CLASS_MAP = {
    "Neutral":  0,
    "Angry":    1,
    "Disgust":  1,  # Merged
    "Fear":     2,
    "Happy":    3,
    "Sad":      4,
}

def process_sfew(output_root: Path, detector: FaceDetector):
    """
    Process SFEW dataset.
    Pool Train/Val/Test subsets and re-split 80/20 into train/val.
    """
    logger.info("=" * 60)
    logger.info("Processing: SFEW")
    logger.info("=" * 60)
    
    stats = Stats()
    class_files: dict = {i: [] for i in range(5)}
    
    for subset in ["Train", "Val", "Test"]:
        subset_dir = SFEW_DIR / subset
        if not subset_dir.exists():
            continue
            
        for folder_name, dms_cls in SFEW_CLASS_MAP.items():
            cls_dir = subset_dir / folder_name
            if not cls_dir.exists():
                continue
            img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif")])
            class_files[dms_cls].extend(img_files)

    for dms_cls, files in class_files.items():
        if not files:
            continue
        train_files, val_files = train_val_split(files, TRAIN_RATIO, seed=RANDOM_SEED + dms_cls)
        cls_folder = CLASS_FOLDERS[dms_cls]
        
        logger.info(f"  Class {dms_cls} ({cls_folder}): {len(train_files)} train + {len(val_files)} val = {len(files)} total source images")
        
        for split, file_list in [("train", train_files), ("val", val_files)]:
            dst_dir = output_root / split / cls_folder
            for src in file_list:
                process_and_save(src, dst_dir, detector, stats, dataset_prefix="sfew", dms_cls=dms_cls)

    stats.report("SFEW")


# ---------------------------------------------------------------------------
# FINAL STATISTICS REPORTER
# ---------------------------------------------------------------------------

def report_final_stats(output_root: Path):
    """Count and display the final number of images in each split/class."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("=" * 60)

    total_all = 0
    
    # Train and Val splits
    for split in ["train", "val"]:
        split_dir = output_root / split
        split_total = 0
        logger.info(f"\n  Split: {split.upper()}")
        for dms_cls, cls_folder in CLASS_FOLDERS.items():
            cls_dir = split_dir / cls_folder
            if cls_dir.exists():
                count = len(list(cls_dir.glob("*.jpg"))) + \
                        len(list(cls_dir.glob("*.jpeg"))) + \
                        len(list(cls_dir.glob("*.png")))
                logger.info(f"    Class {dms_cls} ({cls_folder:12s}): {count:6,} images")
                split_total += count
            else:
                logger.info(f"    Class {dms_cls} ({cls_folder:12s}):       0 images")

        logger.info(f"    {'SPLIT TOTAL':16s}: {split_total:6,} images")
        total_all += split_total

    # Separate Test datasets
    for test_dataset in ["affectnet", "fer2013", "rafdb"]:
        split_dir = output_root / "test" / test_dataset
        split_total = 0
        logger.info(f"\n  Split: TEST ({test_dataset.upper()})")
        for dms_cls, cls_folder in CLASS_FOLDERS.items():
            cls_dir = split_dir / cls_folder
            if cls_dir.exists():
                count = len(list(cls_dir.glob("*.jpg"))) + \
                        len(list(cls_dir.glob("*.jpeg"))) + \
                        len(list(cls_dir.glob("*.png")))
                logger.info(f"    Class {dms_cls} ({cls_folder:12s}): {count:6,} images")
                split_total += count
            else:
                logger.info(f"    Class {dms_cls} ({cls_folder:12s}):       0 images")

        logger.info(f"    {'SPLIT TOTAL':16s}: {split_total:6,} images")
        total_all += split_total

    logger.info(f"\n  GRAND TOTAL: {total_all:,} images")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# MAIN PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("  DMS Preprocessing Pipeline [v4 — 5-class Emotions]")
    logger.info("=" * 60)
    logger.info(f"  Output directory : {OUTPUT_DIR}")
    logger.info(f"  Classes          : 0=Neutral, 1=Anger, 2=Fear, 3=Happiness, 4=Sadness")
    logger.info(f"  Image size       : {IMG_SIZE}")
    logger.info(f"  Train/Val ratio  : {TRAIN_RATIO:.0%} / {1 - TRAIN_RATIO:.0%}")
    logger.info(f"  Random seed      : {RANDOM_SEED}")
    logger.info("")

    # Step 1: Create output directory structure (5-class)
    make_output_dirs(OUTPUT_DIR)

    # Step 2: Initialize shared MediaPipe face detector
    with FaceDetector(min_confidence=MIN_DETECTION_CONFIDENCE) as detector:

        # Step 3: Process trainable datasets
        process_affectnet(OUTPUT_DIR, detector)
        process_rafdb(OUTPUT_DIR,    detector)
        process_mlider(OUTPUT_DIR,   detector)
        process_kmufed(OUTPUT_DIR,   detector)
        process_fer2013(OUTPUT_DIR,  detector)
        process_kdef(OUTPUT_DIR,     detector)
        process_sfew(OUTPUT_DIR,     detector)

    # Step 4: Print final statistics
    report_final_stats(OUTPUT_DIR)

    logger.info("")
    logger.info("Preprocessing complete!")
    logger.info(f"Dataset ready for PyTorch ImageFolder at: {OUTPUT_DIR}")
    logger.info("")
    logger.info("  Example PyTorch usage:")
    logger.info("  train_ds      = datasets.ImageFolder('Unified_DMS_Dataset_v4/train', transform)")
    logger.info("  val_ds        = datasets.ImageFolder('Unified_DMS_Dataset_v4/val',   transform)")
    logger.info("  test_ds_aff   = datasets.ImageFolder('Unified_DMS_Dataset_v4/test/affectnet', transform)")
    logger.info("  test_ds_fer   = datasets.ImageFolder('Unified_DMS_Dataset_v4/test/fer2013',   transform)")
    logger.info("  test_ds_raf   = datasets.ImageFolder('Unified_DMS_Dataset_v4/test/rafdb',     transform)")


if __name__ == "__main__":
    main()
