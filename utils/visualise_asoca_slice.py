import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk

IMAGE_PATH = "data/ASOCA/images/Normal_1.nii.gz"
LABEL_PATH = "data/ASOCA/labels/Normal_1.nii.gz"
OUTPUT_PATH = "plots/asoca_slice.png"

img_sitk = sitk.ReadImage(IMAGE_PATH)
lab_sitk = sitk.ReadImage(LABEL_PATH)

img     = sitk.GetArrayFromImage(img_sitk)   # (Z, Y, X)
lab_arr = sitk.GetArrayFromImage(lab_sitk)

# Pick the axial slice with the most coronary artery voxels
foreground_per_slice = (lab_arr > 0).sum(axis=(1, 2))
slice_idx = int(np.argmax(foreground_per_slice))
print(f"Using axial slice {slice_idx} of {img.shape[0]} "
      f"({int(foreground_per_slice[slice_idx])} foreground voxels)")

ct_slice  = img[slice_idx]
lab_slice = lab_arr[slice_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e")

axes[0].imshow(ct_slice, cmap="gray", vmin=-200, vmax=600)
axes[0].set_title("Input CT", color="white", fontsize=14, pad=10)
axes[0].axis("off")

axes[1].imshow(ct_slice, cmap="gray", vmin=-200, vmax=600)
axes[1].imshow(np.ma.masked_equal(lab_slice, 0), cmap="Reds", vmin=0, vmax=1, alpha=0.7)
axes[1].set_title("Ground Truth Labels", color="white", fontsize=14, pad=10)
axes[1].axis("off")

patch = mpatches.Patch(color="#e41a1c", label="Coronary Artery")
axes[1].legend(
    handles=[patch],
    loc="lower right",
    fontsize=9,
    framealpha=0.7,
    facecolor="#1a1a2e",
    labelcolor="white",
)

plt.tight_layout()
import os
os.makedirs("plots", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved to {OUTPUT_PATH}")
