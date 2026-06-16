"""
Visualise pseudo-segmentation predictions from 5 fine-tuned Anatomix models
on a single MM-WHS CT scan, alongside the ground-truth label map.

Output: plots/anatomix_5seeds_predictions.png
Layout: 7 columns — CT | seed_42 | seed_123 | seed_456 | seed_789 | seed_999 | GT
"""
import sys, os

PROJECT_ROOT = "/vol/biomedic2/bglocker_studproj/<USERNAME>/grc-net"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "anatomix"))

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from monai.inferers import sliding_window_inference
from monai.data import MetaTensor
from anatomix.segmentation.segmentation_utils import load_model

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SEEDS      = [42, 123, 456, 789, 999]
MODEL_TMPL = "saved_models/segmentation/seed_{}/anatomix_trained_MM-WHS.pth"
PRETRAIN   = "anatomix/model-weights/anatomix.pth"
N_CLASSES  = 7       # foreground classes (background is class 0)
CROP_SIZE  = 96
SPACING    = 2.0
SW_OVERLAP = 0.5
SW_BATCH   = 2

# Val image to visualise (from data/config/val.csv)
IMG_PATH = "data/MM-WHS/images/ct_train_1004_image.nii.gz"
SEG_PATH = "data/MM-WHS/labels/ct_train_1004_label.nii.gz"

OUT_PATH  = "plots/anatomix_5seeds_predictions.png"

# MM-WHS label remapping: raw value -> class index 1-7
REMAP = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}

# Structure names for legend
STRUCTURE_NAMES = [
    "Background",
    "Myocardium (LV)",
    "Left Atrium",
    "Left Ventricle",
    "Right Atrium",
    "Right Ventricle",
    "Aorta",
    "Pulmonary Artery",
]

# Visually distinct colours per class (background transparent/black)
COLORS = [
    "#000000",  # 0 background — black
    "#E6194B",  # 1 MYO        — red
    "#3CB44B",  # 2 LA         — green
    "#4363D8",  # 3 LV         — blue
    "#F58231",  # 4 RA         — orange
    "#911EB4",  # 5 RV         — purple
    "#42D4F4",  # 6 AA         — cyan
    "#F032E6",  # 7 PA         — magenta
]

SEG_CMAP = ListedColormap(COLORS)

os.makedirs("plots", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_and_resample(path, is_label=False, target_spacing=SPACING):
    """Load a NIfTI/ITK image, orient to RAS, resample to target_spacing."""
    img_sitk = sitk.ReadImage(path)
    img_sitk = sitk.DICOMOrient(img_sitk, "RAS")

    orig_spacing = np.array(img_sitk.GetSpacing())   # (x, y, z)
    orig_size    = np.array(img_sitk.GetSize())       # (x, y, z)
    new_size     = np.round(orig_size * orig_spacing / target_spacing).astype(int)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([target_spacing] * 3)
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(img_sitk.GetDirection())
    resample.SetOutputOrigin(img_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampled = resample.Execute(img_sitk)

    arr = sitk.GetArrayFromImage(resampled).astype(np.float32)  # (z, y, x)
    return arr                                                   # keep (z, y, x)


def remap_labels(label_arr):
    out = np.zeros_like(label_arr, dtype=np.int64)
    for raw, cls in REMAP.items():
        out[label_arr == raw] = cls
    return out


def run_inference(model, img_np):
    """img_np: float32 array (z, y, x).  Returns argmax label (z, y, x)."""
    # Add batch + channel dims -> (1, 1, z, y, x)
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    roi   = (CROP_SIZE, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        logits = sliding_window_inference(img_t, roi, SW_BATCH, model, overlap=SW_OVERLAP)
    # logits: (1, N_CLASSES+1, z, y, x) — background + 7 fg
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (z, y, x)
    return pred.astype(np.int64)


def load_model_from_ckpt(seed):
    m = load_model(
        pretrained_ckpt=PRETRAIN,
        n_classes=N_CLASSES,
        device=device,
    )
    ckpt_path = MODEL_TMPL.format(seed)
    print(f"  Loading {ckpt_path} …")
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Load image + GT
# ──────────────────────────────────────────────────────────────────────────────
print("Loading and resampling CT image …")
img_arr = load_and_resample(IMG_PATH, is_label=False)     # (z, y, x) float32

print("Loading and resampling GT label …")
seg_arr_raw = load_and_resample(SEG_PATH, is_label=True)  # (z, y, x) float32
gt_arr  = remap_labels(seg_arr_raw.astype(np.int32))       # (z, y, x) int64

print(f"  Image shape: {img_arr.shape},  GT unique: {np.unique(gt_arr)}")

# ──────────────────────────────────────────────────────────────────────────────
# Run inference with each of the 5 models
# ──────────────────────────────────────────────────────────────────────────────
preds = {}
for seed in SEEDS:
    print(f"\nSeed {seed}:")
    m = load_model_from_ckpt(seed)
    preds[seed] = run_inference(m, img_arr)
    del m
    torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────────────
# Pick a representative axial slice (centre of the GT bounding box)
# ──────────────────────────────────────────────────────────────────────────────
fg_z = np.where(gt_arr > 0)[0]
if len(fg_z):
    z_slice = (fg_z.min() + fg_z.max()) // 2
else:
    z_slice = img_arr.shape[0] // 2
print(f"\nVisualising axial slice z={z_slice}")

# ──────────────────────────────────────────────────────────────────────────────
# Build figure  (1 row × 7 cols)
# ──────────────────────────────────────────────────────────────────────────────
n_cols   = 7
fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.2))
fig.patch.set_facecolor("#0d0d0d")

col_titles = [f"Seed {s}" for s in SEEDS] + ["Ground Truth"]
col_data   = [preds[s][z_slice] for s in SEEDS] + [gt_arr[z_slice]]

img_slice  = img_arr[z_slice]

# Intensity normalise for display
p2, p98 = np.percentile(img_slice, 2), np.percentile(img_slice, 98)
img_disp = np.clip((img_slice - p2) / (p98 - p2 + 1e-8), 0, 1)

norm_kwargs = dict(vmin=0, vmax=N_CLASSES, cmap=SEG_CMAP)

# ── Column 0: raw CT ────────────────────────────────────────────────────────
ax = axes[0]
ax.imshow(img_disp, cmap="gray", origin="lower", aspect="equal")
ax.set_title("CT Image", color="white", fontsize=11, fontweight="bold", pad=6)
ax.axis("off")

# ── Columns 1–6: predictions + GT ───────────────────────────────────────────
for i, (title, label_slice) in enumerate(zip(col_titles, col_data)):
    ax = axes[i + 1]
    ax.imshow(img_disp, cmap="gray", origin="lower", aspect="equal")
    mask = np.ma.masked_where(label_slice == 0, label_slice)
    ax.imshow(mask, origin="lower", aspect="equal",
              vmin=0, vmax=N_CLASSES, cmap=SEG_CMAP, alpha=0.70)
    is_gt = (i == len(SEEDS))
    colour = "gold" if is_gt else "white"
    weight = "bold"
    ax.set_title(title, color=colour, fontsize=11, fontweight=weight, pad=6)
    ax.axis("off")

# ── Legend ───────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=COLORS[c], label=STRUCTURE_NAMES[c])
    for c in range(1, N_CLASSES + 1)
]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=N_CLASSES,
    frameon=False,
    fontsize=9,
    labelcolor="white",
    bbox_to_anchor=(0.5, -0.03),
)

plt.suptitle(
    f"MM-WHS  ct_train_1004  |  axial slice z={z_slice}  |  5 Anatomix seeds vs. Ground Truth",
    color="white", fontsize=12, fontweight="bold", y=1.01,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved -> {OUT_PATH}")
