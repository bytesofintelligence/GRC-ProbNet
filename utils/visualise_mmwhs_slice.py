import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk

IMAGE_PATH = "data/MM-WHS/images/ct_train_1001_image.nii.gz"
LABEL_PATH = "data/MM-WHS/labels/ct_train_1001_label.nii.gz"
OUTPUT_PATH = "plots/mmwhs_slice.png"

# MM-WHS raw label values -> class indices 0-7
LABEL_MAP = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}
CLASS_NAMES = [
    "Background",
    "Myocardium",
    "Left Atrium",
    "Left Ventricle",
    "Right Atrium",
    "Right Ventricle",
    "Aorta",
    "Pulmonary Artery",
]
COLORS = [
    "#000000",  # background — black
    "#e41a1c",  # myocardium — red
    "#377eb8",  # left atrium — blue
    "#4daf4a",  # left ventricle — green
    "#984ea3",  # right atrium — purple
    "#ff7f00",  # right ventricle — orange
    "#a65628",  # aorta — brown
    "#f781bf",  # pulmonary artery — pink
]

img_sitk = sitk.ReadImage(IMAGE_PATH)
lab_sitk = sitk.ReadImage(LABEL_PATH)

img = sitk.GetArrayFromImage(img_sitk)   # (Z, Y, X)
lab = sitk.GetArrayFromImage(lab_sitk)

# Remap non-contiguous label values to 0-7
lab_remapped = np.zeros_like(lab)
for raw_val, idx in LABEL_MAP.items():
    lab_remapped[lab == raw_val] = idx

# Pick the axial slice with the most labelled voxels
foreground_per_slice = (lab_remapped > 0).sum(axis=(1, 2))
slice_idx = int(np.argmax(foreground_per_slice))
print(f"Using axial slice {slice_idx} of {img.shape[0]}")

ct_slice = img[slice_idx]
lab_slice = lab_remapped[slice_idx]

# Build an RGB image for the label overlay
from matplotlib.colors import ListedColormap
cmap = ListedColormap(COLORS)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e")

axes[0].imshow(ct_slice, cmap="gray", vmin=-200, vmax=600)
axes[0].set_title("Input CT", color="white", fontsize=14, pad=10)
axes[0].axis("off")

axes[1].imshow(ct_slice, cmap="gray", vmin=-200, vmax=600)
axes[1].imshow(lab_slice, cmap=cmap, vmin=0, vmax=7, alpha=0.6)
axes[1].set_title("Ground Truth Labels", color="white", fontsize=14, pad=10)
axes[1].axis("off")

# Legend — skip background
patches = [
    mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i])
    for i in range(1, len(CLASS_NAMES))
]
axes[1].legend(
    handles=patches,
    loc="lower right",
    fontsize=7,
    framealpha=0.7,
    facecolor="#1a1a2e",
    labelcolor="white",
)

plt.tight_layout()
import os
os.makedirs("plots", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved to {OUTPUT_PATH}")
