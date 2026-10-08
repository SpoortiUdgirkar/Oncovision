"""
Grad-CAM Test Script for OncoVision.

Tests generate_gradcam on sample images from dataset/test across Normal, Benign,
and Malignant classes, and saves side-by-side heatmaps to outputs/.

Usage:
    python scripts/test_gradcam.py
"""

import argparse
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    TEST_DIR,
    OUTPUTS_DIR,
    CLASS_NAMES,
    VALID_IMAGE_EXTENSIONS,
    DEVICE
)
from src.gradcam.explain import generate_gradcam
from src.models.resnet import build_resnet50
import torch


def main():
    parser = argparse.ArgumentParser(description="Test Grad-CAM heatmap generation on test set images.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODELS_DIR / "resnet50_best.pth"),
        help="Path to trained model checkpoint."
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at '{model_path}'. Run training first.")

    print("=" * 70)
    print("        ONCOVISION - GRAD-CAM EXPLAINABILITY TEST RUNNER        ")
    print("=" * 70)

    # Load model once to share across test evaluations
    print(f"Loading ResNet50 model weights from: {model_path} ...")
    model = build_resnet50(pretrained=False, freeze_backbone=False)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    test_dir = Path(TEST_DIR)
    results = []

    # Iterate over classes and select 1 sample image per class
    for class_name in CLASS_NAMES:
        class_folder = None
        for child in test_dir.iterdir():
            if child.is_dir() and child.name.lower() == class_name.lower():
                class_folder = child
                break

        if class_folder is None or not class_folder.exists():
            print(f"[SKIP] Class folder '{class_name}' not found in {test_dir}")
            continue

        # Find first valid non-mask test image
        sample_img_path = None
        for f in class_folder.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS and "_mask" not in f.stem.lower():
                sample_img_path = f
                break

        if sample_img_path is None:
            print(f"[SKIP] No valid sample image found in '{class_folder}'")
            continue

        print(f"\n[TESTING GRAD-CAM] Processing Class '{class_name}' Sample: {sample_img_path.name}")
        out_file = OUTPUTS_DIR / f"gradcam_test_{class_name.lower()}.png"

        res = generate_gradcam(
            image_path=sample_img_path,
            model=model,
            output_path=out_file,
            device=DEVICE
        )

        # Assertions to verify return values
        assert "predicted_class" in res, "Missing 'predicted_class' in return dict"
        assert "confidence" in res, "Missing 'confidence' in return dict"
        assert res["heatmap"].ndim == 3, "Heatmap must be 3-channel RGB image"
        assert res["overlay"].ndim == 3, "Overlay must be 3-channel RGB image"

        results.append((class_name, sample_img_path.name, res["predicted_class"], res["confidence"], res["output_path"]))

    print("\n" + "=" * 70)
    print("                    GRAD-CAM TEST SUMMARY RESULTS                       ")
    print("=" * 70)
    print(f"{'Target Class':<12} | {'File Name':<25} | {'Predicted Class':<15} | {'Confidence':<10}")
    print("-" * 70)
    for target_cls, fname, pred_cls, conf, out_p in results:
        print(f"{target_cls:<12} | {fname:<25} | {pred_cls:<15} | {conf*100:6.2f}%")
        print(f"  -> Visualization Saved To: {out_p}")

    print("\n[PASSED] Grad-CAM explainability pipeline verified successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
