"""
BUSI Ground-Truth Segmentation Mask Loader & Matching Utility.

Provides:
1. find_busi_masks: Matches classification image filenames (e.g., benign (100).png) to their
   corresponding ground-truth mask files in raw BUSI archives or directories.
2. load_and_preprocess_mask: Reads binary mask image(s), merges multi-lesion masks via bitwise OR,
   applies explicit binarization (mask > 0), and resizes using NEAREST-NEIGHBOR interpolation.
3. get_busi_test_masks_manifest: Scans test set and returns matching breakdown dictionary.
"""

import os
import sys
import io
import zipfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_DIR, TEST_DIR


def get_default_busi_archive_path():
    """
    Dynamically resolves the location of the BUSI mask zip archive.
    Priority:
    1. Environment variable: BUSI_MASK_ARCHIVE
    2. Project dataset directory: DATASET_DIR / "archive.zip"
    3. User home Downloads directory: Path.home() / "Downloads" / "archive.zip"
    4. Fallback to DATASET_DIR / "archive.zip"
    """
    env_path = os.getenv("BUSI_MASK_ARCHIVE")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    proj_path = DATASET_DIR / "archive.zip"
    if proj_path.exists():
        return proj_path

    home_downloads = Path.home() / "Downloads" / "archive.zip"
    if home_downloads.exists():
        return home_downloads

    zips = list(DATASET_DIR.glob("*.zip"))
    if zips:
        return zips[0]

    return proj_path


DEFAULT_ARCHIVE_PATH = get_default_busi_archive_path()


class BUSIMaskLoader:
    """
    Robust Ground-Truth Mask Loader for BUSI dataset scans.
    Handles single mask, multi-mask combinations, missing masks, and zip archive loading.
    """

    def __init__(self, archive_path=None):
        if archive_path is None:
            archive_path = get_default_busi_archive_path()
        self.archive_path = Path(archive_path)
        self.zip_ref = None
        self.mask_index = {}  # stem.lower() -> list of archive/file paths

        self._build_index()

    def _build_index(self):
        """Builds index of mask paths keyed by base scan stem (e.g. 'benign (100)')."""
        if self.archive_path.exists() and self.archive_path.is_file():
            try:
                with zipfile.ZipFile(self.archive_path, 'r') as z:
                    for name in z.namelist():
                        if "_mask" in name.lower() and name.lower().endswith(".png"):
                            filename = Path(name).name
                            base_stem = filename.split("_mask")[0].strip().lower()
                            self.mask_index.setdefault(base_stem, []).append(("zip", name))
            except Exception as e:
                print(f"[WARNING] MaskLoader failed to index zip archive '{self.archive_path}': {str(e)}")
        else:
            print(f"[INFO] BUSI mask archive not found at '{self.archive_path}'. Searching local dataset directories...")

        # Also scan local dataset/ directory for extracted masks if present
        for mask_file in DATASET_DIR.rglob("*_mask*.png"):
            filename = mask_file.name
            base_stem = filename.split("_mask")[0].strip().lower()
            if base_stem not in self.mask_index:
                self.mask_index.setdefault(base_stem, []).append(("local", mask_file))

    def get_mask_info(self, image_path):
        """
        Retrieves matching mask records for a given classification image path.

        Returns:
            list: List of tuples (source_type, path_or_zip_name)
        """
        image_path = Path(image_path)
        stem_key = image_path.stem.lower()
        return self.mask_index.get(stem_key, [])

    def load_mask(self, image_path, target_shape=None):
        """
        Loads and preprocesses the ground-truth binary mask for an image.

        Args:
            image_path (str or Path): Classification image file path.
            target_shape (tuple, optional): Target (H, W) to resize mask using NEAREST-NEIGHBOR.

        Returns:
            tuple: (binary_mask_np, is_valid_lesion, num_masks_found)
                - binary_mask_np: uint8 numpy array [H, W] with values 0 or 255.
                - is_valid_lesion: bool (True if mask contains >0 lesion pixels, False if Normal/empty).
                - num_masks_found: int (count of matching mask files found).
        """
        mask_records = self.get_mask_info(image_path)
        num_masks = len(mask_records)

        if num_masks == 0:
            # No mask found
            if target_shape:
                blank = np.zeros(target_shape, dtype=np.uint8)
            else:
                blank = np.zeros((224, 224), dtype=np.uint8)
            return blank, False, 0

        merged_mask = None

        for src_type, location in mask_records:
            try:
                if src_type == "zip":
                    with zipfile.ZipFile(self.archive_path, 'r') as z:
                        mask_bytes = z.read(location)
                        pil_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                else:
                    pil_img = Image.open(location).convert("L")

                mask_np = np.array(pil_img, dtype=np.uint8)

                # Binarize explicitly: mask > 0
                binary_single = np.where(mask_np > 0, 255, 0).astype(np.uint8)

                if merged_mask is None:
                    merged_mask = binary_single
                else:
                    # If multiple masks exist (e.g. benign (100)_mask.png & benign (100)_mask_1.png), merge via bitwise OR
                    if merged_mask.shape != binary_single.shape:
                        binary_single = cv2.resize(binary_single, (merged_mask.shape[1], merged_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    merged_mask = cv2.bitwise_or(merged_mask, binary_single)

            except Exception as e:
                print(f"[WARNING] Failed to load mask record '{location}': {str(e)}")

        if merged_mask is None:
            if target_shape:
                blank = np.zeros(target_shape, dtype=np.uint8)
            else:
                blank = np.zeros((224, 224), dtype=np.uint8)
            return blank, False, num_masks

        # Resize to target (H, W) using NEAREST-NEIGHBOR interpolation to preserve binary mask boundary integrity
        if target_shape and (merged_mask.shape[0] != target_shape[0] or merged_mask.shape[1] != target_shape[1]):
            # cv2.resize takes (width, height)
            target_w, target_h = target_shape[1], target_shape[0]
            merged_mask = cv2.resize(merged_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Final explicit binarization check
        binary_mask = np.where(merged_mask > 127, 255, 0).astype(np.uint8)
        has_lesion_pixels = bool(np.sum(binary_mask > 0) > 0)

        return binary_mask, has_lesion_pixels, num_masks


def get_test_set_mask_manifest(test_dir=TEST_DIR, archive_path=DEFAULT_ARCHIVE_PATH):
    """
    Scans the test split and returns a comprehensive manifest of mask matching counts.
    """
    loader = BUSIMaskLoader(archive_path=archive_path)
    test_dir = Path(test_dir)

    manifest = {
        "total_test_images": 0,
        "matched_exactly_1_mask": 0,
        "matched_multiple_masks": 0,
        "matched_0_masks": 0,
        "by_class": {}
    }

    for class_name in ["Normal", "Benign", "Malignant"]:
        cls_dir = test_dir / class_name
        manifest["by_class"][class_name] = {
            "total": 0,
            "exactly_1": 0,
            "multiple": 0,
            "zero": 0,
            "valid_lesion_scans": 0
        }

        if not cls_dir.exists():
            continue

        for img_file in cls_dir.iterdir():
            if not img_file.is_file() or "_mask" in img_file.stem.lower():
                continue

            manifest["total_test_images"] += 1
            manifest["by_class"][class_name]["total"] += 1

            records = loader.get_mask_info(img_file)
            n_records = len(records)

            if n_records == 1:
                manifest["matched_exactly_1_mask"] += 1
                manifest["by_class"][class_name]["exactly_1"] += 1
            elif n_records > 1:
                manifest["matched_multiple_masks"] += 1
                manifest["by_class"][class_name]["multiple"] += 1
            else:
                manifest["matched_0_masks"] += 1
                manifest["by_class"][class_name]["zero"] += 1

            _, has_lesion, _ = loader.load_mask(img_file)
            if has_lesion:
                manifest["by_class"][class_name]["valid_lesion_scans"] += 1

    return manifest
