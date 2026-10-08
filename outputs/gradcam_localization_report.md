# EfficientNet-B0 Grad-CAM Localization Evaluation Report

**Generated Date**: October 8, 2026  
**Primary Model**: EfficientNet-B0  
**Model Checkpoint**: [`models/efficientnet_b0_best.pth`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Selected Model Benchmark Performance

Evaluating the selected primary model (**EfficientNet-B0**) on the independent held-out BUSI test split ($N=120$):

* **Test Accuracy**: **72.50%**
* **Macro Precision**: **70.70%**
* **Macro Recall**: **68.88%**
* **Macro F1-Score**: **69.47%**
* **Weighted F1-Score**: **72.43%**

*(Note: These figures reflect measured classification performance on the BUSI test split for academic research demonstration purposes).*

---

## 2. Grad-CAM Target Layer Specification

* **Model Architecture**: EfficientNet-B0
* **Discovered Target Layer**: `model.features[-1][0]` (The final `Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)` layer inside the final `Conv2dNormActivation` block).
* **Spatial Feature Map Shape**: $[1, 1280, 7, 7]$ for input tensor $[1, 3, 224, 224]$.

---

## 3. Dataset & Image-to-Mask Matching Manifest

* **Total Test Classification Images**: **120**
* **Total Matched Mask Records**:
  * **Exactly 1 Mask Matched**: **115** images ($21$ Normal, $62$ Benign, $32$ Malignant).
  * **Multiple Masks Matched**: **5** images (all in Benign class, e.g. `benign (100)_mask.png` and `benign (100)_mask_1.png`). Multi-lesion masks were merged via bitwise OR.
  * **Zero Masks Matched**: **0** images.
* **Lesion Ground-Truth Validity**:
  * **Benign**: $67 / 67$ scans contain non-zero lesion ground-truth annotations ($N=67$ evaluated).
  * **Malignant**: $32 / 32$ scans contain non-zero lesion ground-truth annotations ($N=32$ evaluated).
  * **Normal**: $21 / 21$ scans contain completely black/empty masks ($N=21$, reported as **N/A**).

---

## 4. Mask Preprocessing & Format Specification

* **Source Mask Format**: Single-channel grayscale / RGB PNG mask files.
* **Binarization Rule**: Explicit thresholding $\text{mask} > 0 \rightarrow 255$, $\text{else} \rightarrow 0$.
* **Multi-Mask Merging**: Merged using bitwise OR (`cv2.bitwise_or`) to combine separate lesion masks.
* **Resizing & Interpolation**: Resized to original scan spatial resolution using **NEAREST-NEIGHBOR** interpolation (`cv2.INTER_NEAREST`) to preserve binary boundary integrity without introducing anti-aliased interpolation artifacts.
* **Configurable Mask Archive Path**: Resolved dynamically via environment variable `BUSI_MASK_ARCHIVE`, `DATASET_DIR / "archive.zip"`, or `Path.home() / "Downloads" / "archive.zip"`, eliminating hardcoded Windows user paths.

---

## 5. Grad-CAM Explainability Pipeline

1. **Target Class Selection**: Dynamically selects the predicted class logit score ($\hat{y} = \text{argmax}(\mathbf{p})$) predicted by EfficientNet-B0.
2. **Activation & Gradient Extraction**: PyTorch forward and backward hooks capture activations $A^k$ and gradients $A^k$ from `model.features[-1][0]`.
3. **Single-Pass Heatmap Generation**: Heatmap generated once per image to eliminate redundant computation and guarantee exact alignment between visualization overlays and IoU thresholding.
4. **Gradient Pooling**: Global Average Pooling over spatial dimensions: $w_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$.
5. **Weighted Combination & ReLU**: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k w_k^c A^k \right)$.
6. **Normalization & Resizing**: Scaled to $[0.0, 1.0]$ and resized to original scan dimensions via bilinear interpolation.

---

## 6. Binary Attention Thresholding

* **Binarization Threshold**: Fixed threshold $\text{heatmap} \ge 0.5 \rightarrow 1$, $\text{else} \rightarrow 0$.
* **Policy Compliance**: Fixed threshold selected without tuning on test set labels to prevent evaluation bias.

---

## 7. Quantitative Grad-CAM Localization IoU Results

Grad-CAM Localization IoU measures the spatial overlap between the model's high-attention region ($\text{heatmap} \ge 0.5$) and the ground-truth BUSI lesion segmentation mask:

$$\text{IoU} = \frac{\text{Intersection}(\text{Attention\_Mask}, \text{Ground\_Truth\_Mask})}{\text{Union}(\text{Attention\_Mask}, \text{Ground\_Truth\_Mask})}$$

### Measured Localization Scores ($N=99$ Lesion-Containing Scans)

| Target Class Category | Evaluated Scans ($N$) | Mean Grad-CAM IoU (%) | Median Grad-CAM IoU |
| :--- | :---: | :---: | :---: |
| **Benign** | 67 | **8.10%** ($0.0810$) | **0.0375** |
| **Malignant** | 32 | **13.32%** ($0.1332$) | **0.0695** |
| **Normal** | 21 | **N/A** | **N/A** |
| **Overall Lesion Scans** | **99** | **9.78%** ($0.0978$) | **0.0519** |

*(Note: Normal class scans do not contain lesion ground truth; reported as N/A per protocol).*

---

## 8. Interpretation & Clinical Safety Notice

1. **Interpretation**: Grad-CAM Localization IoU measures visual feature alignment between coarse network attention maps ($7 \times 7$ receptive fields) and fine-grained anatomical lesion annotations.
2. **What IoU Does NOT Prove**:
   * It does **NOT** prove clinical diagnostic accuracy.
   * It does **NOT** prove causal medical reasoning or diagnostic safety.
   * It does **NOT** prove the model is operating as a clinical radiologist.
3. **Coarse Attention Feature Extraction**: Because classification CNNs downsample spatial features by $32\times$, activation heatmaps cover broad acoustic regions surrounding the lesion rather than exact tumor boundaries.

---

## 9. Limitations

* **Test Set Size**: Evaluation conducted on 120 test scans ($99$ lesion-containing scans).
* **Fixed Threshold Sensitivity**: Binary attention mapping using fixed threshold $0.5$ provides conservative attention boundaries; adaptive thresholding may yield different IoU distributions.
* **Coarse Receptive Fields**: $7 \times 7$ feature maps inherently limit boundary precision compared to dense segmentation models (e.g. U-Net).
