import os
from pathlib import Path
import torch

# Fix OpenMP duplicate library load issue on Windows environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Base Directory of the Project

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Directories
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Models and Outputs Directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create required output and model directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Target Classes for Breast Ultrasound Classification
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Supported Image File Formats
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Image Preprocessing Configuration
IMAGE_SIZE = (256, 256)  # Target dimensions (Height, Width) for Experiment 9 model (256x256)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet RGB mean values
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet RGB standard deviation values

# Default DataLoader & Training Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 0 if os.name == "nt" else 2  # Set to 0 on Windows by default to avoid multiprocessing issues
LEARNING_RATE = 1e-4
EPOCHS = 20

# Hardware Device Selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random Seed for Reproducibility
SEED = 42

