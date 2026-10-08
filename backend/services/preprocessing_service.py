"""
Preprocessing and Image Validation Service for OncoVision FastAPI Backend.

Provides:
1. validate_image_file: Checks file format, extension, and file size boundaries.
2. load_and_preprocess_image: Converts raw upload bytes to PIL RGB Image,
   applies standard evaluation torchvision transforms, and outputs PyTorch tensors.
"""

import sys
import io
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
from fastapi import UploadFile, HTTPException

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VALID_IMAGE_EXTENSIONS
from src.preprocessing.transforms import get_val_test_transforms

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit


def validate_image_file(file: UploadFile, file_bytes: bytes):
    """
    Validates uploaded file extension, MIME type, and size constraints.
    """
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit of 15 MB.")

    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext and file_ext not in VALID_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{file_ext}'. Supported extensions: {list(VALID_IMAGE_EXTENSIONS)}"
        )


def load_and_preprocess_image(file_bytes: bytes):
    """
    Converts image bytes to PIL RGB image and returns PyTorch preprocessed tensor.

    Returns:
        tuple: (pil_image, img_tensor)
    """
    try:
        img_stream = io.BytesIO(file_bytes)
        pil_img = Image.open(img_stream)
        pil_img.verify()  # Header verification

        # Re-open after verify() (PIL requirement)
        img_stream.seek(0)
        pil_img = Image.open(img_stream).convert("RGB")

    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file is corrupted or not a valid readable image format. Details: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded image: {str(e)}")

    transform = get_val_test_transforms()
    img_tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    return pil_img, img_tensor
