"""
Quantitative Grad-CAM Localization Evaluation Module for OncoVision.

Computes:
1. Grad-CAM Localization IoU (Intersection over Union) against BUSI ground-truth lesion masks.
2. Thresholding: Fixed threshold (heatmap >= 0.5) to convert continuous heatmaps to binary attention masks.
3. Normal Class Handling: Excludes non-lesion (Normal) scans from IoU calculation (reported as N/A).
4. Side-by-side localization visualization generator (original, mask, heatmap, overlay).
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, DEVICE, CLASS_NAMES, IDX_TO_CLASS
from src.preprocessing.mask_loader import BUSIMaskLoader
from src.gradcam.explain import generate_gradcam, GradCAM


def compute_iou(attention_mask: np.ndarray, ground_truth_mask: np.ndarray, eps=1e-8):
    """
    Computes Intersection over Union (IoU) between binary attention mask and binary ground-truth mask.

    Args:
        attention_mask (np.ndarray): Binary boolean/uint8 mask [H, W] (True/1 where attention >= threshold).
        ground_truth_mask (np.ndarray): Binary boolean/uint8 mask [H, W] (True/1 where lesion exists).

    Returns:
        float: IoU score in range [0.0, 1.0].
    """
    att = (attention_mask > 0)
    gt = (ground_truth_mask > 0)

    intersection = np.logical_and(att, gt).sum()
    union = np.logical_or(att, gt).sum()

    if union == 0:
        return 0.0

    return float(intersection / (union + eps))


def evaluate_gradcam_localization(
    model,
    test_dir=None,
    mask_loader=None,
    threshold=0.5,
    device=DEVICE,
    output_dir=OUTPUTS_DIR
):
    """
    Evaluates Grad-CAM localization accuracy (Mean/Median IoU) across test set images.

    Args:
        model (nn.Module): Trained model (e.g. EfficientNet-B0).
        test_dir (str or Path): Path to test dataset split directory.
        mask_loader (BUSIMaskLoader): Pre-initialized BUSIMaskLoader instance.
        threshold (float): Threshold to binarize continuous Grad-CAM heatmap [0.0, 1.0]. Default: 0.5.
        device (str): Compute device ('cuda' or 'cpu').
        output_dir (str or Path): Directory to save output plots and figures.

    Returns:
        dict: Summary metrics for Benign, Malignant, and Overall lesion scans.
    """
    if test_dir is None:
        from src.config import TEST_DIR
        test_dir = TEST_DIR
    test_dir = Path(test_dir)

    if mask_loader is None:
        mask_loader = BUSIMaskLoader()

    output_dir = Path(output_dir)
    loc_vis_dir = output_dir / "gradcam" / "localization"
    loc_vis_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.to(device)
    gradcam_engine = GradCAM(model)

    results = {
        "Benign": {"ious": [], "count": 0},
        "Malignant": {"ious": [], "count": 0},
        "Normal": {"count": 0, "status": "N/A — no lesion ground truth expected for Normal class"}
    }

    eval_records = []

    print("=" * 70)
    print("      ONCOVISION - GRAD-CAM LOCALIZATION IOU EVALUATION      ")
    print("=" * 70)
    print(f"Fixed Binary Threshold : heatmap >= {threshold}")
    print(f"Evaluation Split       : {test_dir}\n")

    for class_name in ["Normal", "Benign", "Malignant"]:
        cls_dir = test_dir / class_name
        if not cls_dir.exists():
            continue

        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file() or "_mask" in img_path.stem.lower():
                continue

            # 1. Load original image & preprocess tensor once
            orig_pil = Image.open(img_path).convert("RGB")
            orig_np = np.array(orig_pil)
            orig_h, orig_w = orig_np.shape[:2]

            img_tensor = from_transforms(orig_pil, device)

            # 2. Single Grad-CAM generation for predicted class
            cam_map_2d, pred_idx, conf, probs = gradcam_engine.generate_map(img_tensor)
            pred_class = IDX_TO_CLASS[pred_idx]

            # 3. Resize continuous 2D heatmap to original image dimensions
            cam_resized = cv2.resize(cam_map_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # 4. Generate color heatmap and overlay for visualization
            heatmap_uint8 = np.uint8(255 * cam_resized)
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)

            # 5. Load ground-truth mask resized to original image dimensions using NEAREST-NEIGHBOR
            gt_mask, has_lesion, num_masks = mask_loader.load_mask(img_path, target_shape=(orig_h, orig_w))

            if class_name == "Normal" or not has_lesion:
                results["Normal"]["count"] += 1
                continue

            # 6. Threshold continuous heatmap -> binary attention mask (heatmap >= threshold)
            att_mask_binary = (cam_resized >= threshold).astype(np.uint8) * 255

            # 7. Compute IoU
            iou_score = compute_iou(att_mask_binary, gt_mask)
            results[class_name]["ious"].append(iou_score)
            results[class_name]["count"] += 1

            eval_records.append({
                "class_name": class_name,
                "image_name": img_path.name,
                "image_path": img_path,
                "pred_class": pred_class,
                "confidence": conf,
                "iou": iou_score,
                "orig_np": orig_np,
                "gt_mask": gt_mask,
                "heatmap_rgb": heatmap_rgb,
                "overlay_rgb": overlay_rgb
            })

    # Save representative localization visualizations for Benign and Malignant
    for cls in ["Benign", "Malignant"]:
        cls_records = [r for r in eval_records if r["class_name"] == cls]
        if not cls_records:
            continue

        # Sort by IoU to get high, median, and low IoU examples
        cls_records.sort(key=lambda x: x["iou"], reverse=True)
        selected_samples = [cls_records[0], cls_records[len(cls_records)//2], cls_records[-1]]

        for idx, rec in enumerate(selected_samples):
            save_localization_figure(
                rec=rec,
                save_path=loc_vis_dir / f"efficientnet_{cls.lower()}_sample_{idx+1}.png"
            )

    # Compute Summary Statistics
    benign_ious = results["Benign"]["ious"]
    malig_ious = results["Malignant"]["ious"]
    all_lesion_ious = benign_ious + malig_ious

    benign_mean = float(np.mean(benign_ious)) if benign_ious else 0.0
    benign_median = float(np.median(benign_ious)) if benign_ious else 0.0

    malig_mean = float(np.mean(malig_ious)) if malig_ious else 0.0
    malig_median = float(np.median(malig_ious)) if malig_ious else 0.0

    overall_mean = float(np.mean(all_lesion_ious)) if all_lesion_ious else 0.0
    overall_median = float(np.median(all_lesion_ious)) if all_lesion_ious else 0.0

    print("--- GRAD-CAM LOCALIZATION IOU SUMMARY ---")
    print(f"Benign Scans (N={len(benign_ious)})    : Mean IoU = {benign_mean:.4f} ({benign_mean*100:.2f}%) | Median IoU = {benign_median:.4f}")
    print(f"Malignant Scans (N={len(malig_ious)}) : Mean IoU = {malig_mean:.4f} ({malig_mean*100:.2f}%) | Median IoU = {malig_median:.4f}")
    print(f"Normal Scans (N={results['Normal']['count']})   : N/A — no lesion ground truth expected")
    print(f"Overall Lesion IoU (N={len(all_lesion_ious)})  : Mean IoU = {overall_mean:.4f} ({overall_mean*100:.2f}%) | Median IoU = {overall_median:.4f}")
    print("=" * 70 + "\n")

    summary = {
        "threshold": threshold,
        "benign": {"count": len(benign_ious), "mean_iou": benign_mean, "median_iou": benign_median},
        "malignant": {"count": len(malig_ious), "mean_iou": malig_mean, "median_iou": malig_median},
        "normal": {"count": results["Normal"]["count"], "status": "N/A — no lesion ground truth expected for Normal class"},
        "overall": {"count": len(all_lesion_ious), "mean_iou": overall_mean, "median_iou": overall_median}
    }

    return summary


def from_transforms(pil_img, device):
    """Helper to convert PIL Image to preprocessed FloatTensor."""
    from src.preprocessing.transforms import get_val_test_transforms
    transform = get_val_test_transforms()
    return transform(pil_img).unsqueeze(0).to(device)


def save_localization_figure(rec, save_path):
    """Saves a 4-panel localization figure (Original, Ground Truth Mask, Grad-CAM Heatmap, Overlay)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"EfficientNet-B0 Grad-CAM Localization — Image: {rec['image_name']} | Pred: {rec['pred_class']} | IoU: {rec['iou']*100:.1f}%",
        fontsize=12,
        fontweight="bold",
        color="darkblue"
    )

    axes[0].imshow(rec["orig_np"])
    axes[0].set_title("Original Scan", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(rec["gt_mask"], cmap="gray")
    axes[1].set_title("Ground-Truth BUSI Mask", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(rec["heatmap_rgb"])
    axes[2].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(rec["overlay_rgb"])
    axes[3].set_title(f"Overlay (IoU: {rec['iou']*100:.1f}%)", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
