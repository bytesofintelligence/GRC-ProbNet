"""
Visualise the MM-WHS centroid-crop preprocessing pipeline for representative samples.

Produces:
  plots/preproc_pipeline_<fname>.png  — 5-stage strip for N_SAMPLES samples

Run from project root (conda env grcnet):
  python visualise_preprocessing.py
"""

import os, sys, json, warnings
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from monai.inferers import sliding_window_inference

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("anatomix")  # expose the anatomix submodule package

from img.datasets import (
    load_sitk_array, clean_segmentation_mask, get_bbox_size, centroid_crop_fixed,
)
from img.processing import resample_image_d
from anatomix.segmentation.segmentation_utils import load_model

# ── Config ─────────────────────────────────────────────────────────────────────
N_SAMPLES      = 3
CSV_PATH       = "data/config/train.csv"
TARGET_SPACING = (2.0, 2.0, 2.0)
NUM_CLASSES    = 8
SPATIAL_SIZE   = (96, 96, 96)
CROP_JSON      = "output/mm-whs/full-stn/train/model/crop_size.json"
MODEL_DIR      = "saved_models/segmentation"
PRETRAINED     = "anatomix/model-weights/anatomix.pth"
OUT_DIR        = "plots"
DEVICE         = "cpu"
DPI            = 150

CLASS_NAMES = [
    "Background", "Myocardium", "L. Atrium", "L. Ventricle",
    "R. Atrium", "R. Ventricle", "Aorta", "Pulm. Artery",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 9,
})
SEG_CMAP = plt.cm.get_cmap("tab10")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_best_checkpoint():
    candidates = [
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("finetuned_MM-WHS") and f.endswith(".pth")
    ]
    if not candidates:
        return os.path.join(MODEL_DIR, "anatomix_trained_MM-WHS.pth")
    best = min(candidates, key=lambda n: float(n[len("finetuned_MM-WHS"):-4]))
    return os.path.join(MODEL_DIR, best)


def ct_window(vol, lo=2, hi=98):
    fg = vol[vol > -900]
    if len(fg) == 0:
        return float(vol.min()), float(vol.max())
    return float(np.percentile(fg, lo)), float(np.percentile(fg, hi))


def make_seg_overlay(label_2d, alpha=0.55):
    """RGBA overlay; background (class 0) is fully transparent."""
    out = np.zeros((*label_2d.shape, 4), dtype=np.float32)
    for cls in range(1, NUM_CLASSES):
        c = SEG_CMAP(cls / 9.0)
        m = label_2d == cls
        out[m, :3] = c[:3]
        out[m, 3]  = alpha
    return out


def mask_centroid_z(mask3d):
    idx = np.where(mask3d > 0)
    return int(round(idx[0].mean())) if len(idx[0]) > 0 else mask3d.shape[0] // 2


def bbox_extents_3d(mask3d):
    """(z0, z1, y0, y1, x0, x1) of the non-zero region, or None."""
    idx = np.where(mask3d > 0)
    if len(idx[0]) == 0:
        return None
    return (
        int(idx[0].min()), int(idx[0].max()),
        int(idx[1].min()), int(idx[1].max()),
        int(idx[2].min()), int(idx[2].max()),
    )


def style_spine(ax, color, lw=2.0):
    for s in ax.spines.values():
        s.set_edgecolor(color)
        s.set_linewidth(lw)
        s.set_visible(True)


def show_ct_ax(ax, img_2d, vmin, vmax, title=""):
    ax.set_facecolor("black")
    ax.imshow(img_2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_title(title, fontsize=8, pad=5)
    ax.set_xticks([]); ax.set_yticks([])

# ── Load model and config ───────────────────────────────────────────────────────

ckpt_path = find_best_checkpoint()
print(f"[Anatomix] Loading checkpoint: {ckpt_path}")
model = load_model(pretrained_ckpt=PRETRAINED, n_classes=NUM_CLASSES - 1, device=DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

with open(CROP_JSON) as f:
    crop_size = tuple(json.load(f)["crop_size"])
print(f"[Config]   crop_size = {crop_size}")

df        = pd.read_csv(CSV_PATH, comment="#")
img_paths = df["img"].tolist()[:N_SAMPLES]

# ── Per-sample processing ───────────────────────────────────────────────────────

for s_idx, img_path in enumerate(tqdm(img_paths, desc="Processing samples")):
    fname = os.path.basename(img_path).replace(".nii.gz", "")
    print(f"\n{'='*62}")
    print(f"[Sample {s_idx}] {fname}")

    # Stage 1: load native ─────────────────────────────────────────────────────
    native    = load_sitk_array(img_path)     # MetaTensor (D, H, W)
    native_np = native.numpy()
    D0, H0, W0 = native.shape
    sx, sy, sz  = native.meta["spacing"]      # SimpleITK (x,y,z) = (W,H,D) spacing
    print(f"  Native shape   (D,H,W) : ({D0:4d}, {H0:4d}, {W0:4d})")
    print(f"  Native spacing (D,H,W) : ({sz:.3f}, {sy:.3f}, {sx:.3f}) mm")

    # Stage 2: resample to 2 mm isotropic ──────────────────────────────────────
    native_ch  = native.unsqueeze(0)           # (1, D, H, W)  [channel dim for resample]
    res_ch     = resample_image_d(native_ch, out_spacing=TARGET_SPACING)
    resampled  = res_ch.squeeze(0)             # (D', H', W')
    res_np     = resampled.numpy()
    D1, H1, W1 = resampled.shape
    print(f"  Resampled shape (D,H,W): ({D1:4d}, {H1:4d}, {W1:4d})")

    # Stage 3: Anatomix inference ──────────────────────────────────────────────
    # res_ch is (1, D, H, W); sliding_window_inference needs (N, C, D, H, W)
    img_t = res_ch.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = sliding_window_inference(
            img_t, roi_size=SPATIAL_SIZE, sw_batch_size=1, predictor=model, overlap=0.35
        )
    label_vol = torch.argmax(logits, dim=1)[0].cpu()  # (D', H', W'), values 0-7

    # Stage 4: clean mask + bounding box ──────────────────────────────────────
    cleaned_mask = clean_segmentation_mask(label_vol, dilation_iters=3)
    bbox         = get_bbox_size(cleaned_mask)
    bbox_ext     = bbox_extents_3d(cleaned_mask)
    print(f"  Bbox (D,H,W)           : {bbox}")

    # Stage 5: centroid crop ───────────────────────────────────────────────────
    cropped_img, _ = centroid_crop_fixed(
        resampled, label_vol.float(), cleaned_mask, crop_size
    )
    crop_np = cropped_img.numpy()
    D2, H2, W2 = crop_np.shape
    print(f"  Crop shape  (D,H,W)    : ({D2:4d}, {H2:4d}, {W2:4d})")

    n_res = int(np.prod([D1, H1, W1]))
    n_crp = int(np.prod([D2, H2, W2]))
    pct   = (1.0 - n_crp / n_res) * 100.0
    print(f"  Voxel count            : {n_res:,} -> {n_crp:,}  ({pct:.1f}% removed)")

    # Axial slice selection ────────────────────────────────────────────────────
    heart_z  = mask_centroid_z(cleaned_mask)              # heart centroid in resampled vol
    native_z = int(np.clip(heart_z / D1 * D0, 0, D0 - 1))
    crop_z   = D2 // 2

    # CT display windows
    vn0, vn1 = ct_window(native_np)
    vr0, vr1 = ct_window(res_np)
    vc0, vc1 = ct_window(crop_np)

    # ── Figure A: 5-stage pipeline strip ──────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.8), constrained_layout=True)
    fig.suptitle(
        f"Preprocessing Pipeline  ·  {fname}",
        fontsize=12, fontweight="bold"
    )

    # Panel ①: native
    show_ct_ax(
        axes[0], native_np[native_z], vn0, vn1,
        title=f"① Native image\n({D0} × {H0} × {W0}) voxels\n"
              f"spacing {sz:.2f} × {sy:.2f} × {sx:.2f} mm (D,H,W)"
    )

    # Panel ②: resampled
    show_ct_ax(
        axes[1], res_np[heart_z], vr0, vr1,
        title=f"② Resampled (2 mm iso)\n({D1} × {H1} × {W1}) voxels\n"
              f"axial slice z = {heart_z}"
    )

    # Panel ③: resampled + segmentation overlay
    axes[2].set_facecolor("black")
    axes[2].imshow(res_np[heart_z], cmap="gray", vmin=vr0, vmax=vr1, interpolation="bilinear")
    axes[2].imshow(make_seg_overlay(label_vol[heart_z].numpy()), interpolation="nearest")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title("③ Anatomix Segmentation\n(7 cardiac structures)", fontsize=8, pad=5)
    seg_patches = [
        mpatches.Patch(facecolor=SEG_CMAP(c / 9.0), label=CLASS_NAMES[c])
        for c in range(1, NUM_CLASSES)
    ]
    axes[2].legend(
        handles=seg_patches, loc="lower right", fontsize=6,
        framealpha=0.85, handlelength=1.2, borderpad=0.5, labelspacing=0.3,
    )

    # Panel ④: resampled + bounding box
    axes[3].set_facecolor("black")
    axes[3].imshow(res_np[heart_z], cmap="gray", vmin=vr0, vmax=vr1, interpolation="bilinear")
    if bbox_ext is not None:
        _, _, y0b, y1b, x0b, x1b = bbox_ext
        rect = mpatches.Rectangle(
            (x0b, y0b), x1b - x0b, y1b - y0b,
            linewidth=2.5, edgecolor="red", facecolor="none", zorder=5,
        )
        axes[3].add_patch(rect)
        axes[3].text(
            x0b, y0b - 3, "bbox", color="red", fontsize=7, va="bottom",
            fontweight="bold",
        )
    axes[3].set_xticks([]); axes[3].set_yticks([])
    axes[3].set_title(
        f"④ Cleaned Mask  +  Bbox\nbbox (D,H,W) = {bbox[0]} × {bbox[1]} × {bbox[2]}",
        fontsize=8, pad=5,
    )

    # Panel ⑤: final crop
    show_ct_ax(
        axes[4], crop_np[crop_z], vc0, vc1,
        title=f"⑤ Centroid Crop\n({D2} × {H2} × {W2}) voxels\n"
              f"{pct:.1f}% background removed"
    )

    out_a = os.path.join(OUT_DIR, f"preproc_pipeline_{fname}.png")
    fig.savefig(out_a, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figure A saved: {out_a}")

print("\n[Done] All figures written to plots/")
