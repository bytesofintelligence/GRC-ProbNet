"""
Visualisation grid: error maps vs uncertainty maps for one MM-WHS test sample.

Grid layout (5 rows × 8 cols):
  Row 0:   Input context — CT | Seed 1..5 (Anatomix) | GT | (blank)
  Row 1-4: Error | Entropy | KL Divergence | Aleatoric  (Multiclass + 7 structures)

Saves to: output/mm-whs/full-stn/uncertainty_analysis/ncc_results/uncertainty_error_grid.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_ID = 4
SLICE_Z   = 50
SEEDS     = [42, 123, 456, 789, 999]

BASE     = Path("output/mm-whs/full-stn")
OUT_PATH = BASE / "uncertainty_analysis" / "ncc_results" / "uncertainty_error_grid.png"

STRUCT_NAMES = [
    "Myocardium", "Left_Atrium", "Left_Ventricle",
    "Right_Atrium", "Right_Ventricle", "Aorta", "Pulmonary_Artery",
]
COL_LABELS = [
    "Multiclass", "Myocardium", "Left Atrium", "Left Ventricle",
    "Right Atrium", "Right Ventricle", "Aorta", "Pulmonary Artery",
]
ROW_LABELS = ["Input", "Error", "Entropy", "KL Divergence", "Aleatoric"]

SEG_CMAP = mcolors.ListedColormap(
    ["black", "#E6194B", "#3CB44B", "#4363D8", "#F58231",
     "#911EB4", "#42D4F4", "#F032E6"]
)
SEG_NORM = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 8.5), ncolors=8)

FONT_COL     = 18   # column header font size
FONT_ROW     = 20   # row label font size
FONT_TOP     = 15   # seed cell label font size
FONT_CT_GT   = 19   # CT and GT label font size

# Gridspec margins
GS_LEFT   = 0.055
GS_RIGHT  = 0.98
GS_TOP    = 0.97
GS_BOTTOM = 0.02
GS_HSPACE = 0.14
GS_WSPACE = 0.04

N_ROWS = 5
N_COLS = 8

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

s = SAMPLE_ID
err_dir  = BASE / "uncertainty_analysis" / "error_maps"
ent_dir  = BASE / "uncertainty_analysis" / "entropy_maps"
ent_pc   = BASE / "uncertainty_analysis" / "per_class_entropy_maps"
kl_dir   = BASE / "uncertainty_analysis_kl" / "kl_divergence_maps"
kl_pc    = BASE / "uncertainty_analysis_kl" / "per_class_kl_maps"
al_dir   = BASE / "uncertainty_analysis_aleatoric" / "kl_and_aleatoric_divergence_maps"
al_pc    = BASE / "uncertainty_analysis_aleatoric" / "per_class_aleatoric_maps"
test_dir = BASE / "test"
unc_dir  = BASE / "uncertainty"

# Top row: (path, cell_label, cmap_hint)
context_row = (
    [(test_dir / f"sample_{s}_image.nii.gz", "CT", "gray")]
    + [(unc_dir / f"seed_{seed}" / f"sample_{s}_warped_atlas_labelmap_argmax.nii.gz",
        f"Seed {i+1}", "seg") for i, seed in enumerate(SEEDS)]
    + [(test_dir / f"sample_{s}_labelmap_argmax.nii.gz", "GT", "seg")]
)  # 7 items; col 7 left blank

map_rows = [
    ([err_dir / f"sample_{s}_error_multiclass.nii.gz"]
     + [err_dir / f"sample_{s}_error_class_{k}_{STRUCT_NAMES[k-1]}.nii.gz" for k in range(1, 8)],
     "Reds"),
    ([ent_dir / f"sample_{s}_uncertainty_entropy.nii.gz"]
     + [ent_pc / f"sample_{s}_class_{k}_entropy.nii.gz" for k in range(1, 8)],
     "hot"),
    ([kl_dir / f"sample_{s}_avg_kl_divergence.nii.gz"]
     + [kl_pc / f"sample_{s}_class_{k}_kl_divergence.nii.gz" for k in range(1, 8)],
     "hot"),
    ([al_dir / f"sample_{s}_aleatoric_map.nii.gz"]
     + [al_pc / f"sample_{s}_class_{k}_aleatoric.nii.gz" for k in range(1, 8)],
     "hot"),
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_slice(path: Path, z: int) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)[:, :, z].T


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(N_COLS * 2.8, N_ROWS * 2.8))

gs = gridspec.GridSpec(
    N_ROWS, N_COLS,
    figure=fig,
    left=GS_LEFT, right=GS_RIGHT,
    top=GS_TOP,   bottom=GS_BOTTOM,
    hspace=GS_HSPACE, wspace=GS_WSPACE,
)

# ---- Row 0: context images ------------------------------------------------

for col_idx, (path, label, cmap_hint) in enumerate(context_row):
    ax = fig.add_subplot(gs[0, col_idx])
    sl = load_slice(path, SLICE_Z)
    if cmap_hint == "gray":
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal",
                  vmin=np.percentile(sl, 2), vmax=np.percentile(sl, 98))
    else:
        ax.imshow(sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower", aspect="equal")
    fs = FONT_CT_GT if label in ("CT", "GT") else FONT_TOP
    ax.set_title(label, fontsize=fs, fontweight="bold", pad=4)
    ax.axis("off")

fig.add_subplot(gs[0, 7]).axis("off")  # blank col 7

# "Anatomix Segmentations" group label above seed columns (cols 1-5)
# Compute approximate x-centre of cols 1-5 in figure coordinates
col_w  = (GS_RIGHT - GS_LEFT) / (N_COLS + (N_COLS - 1) * GS_WSPACE)
col_g  = GS_WSPACE * col_w
seed_col_left  = GS_LEFT + 1 * (col_w + col_g)
seed_col_right = GS_LEFT + 5 * (col_w + col_g) + col_w
seed_center_x  = (seed_col_left + seed_col_right) / 2

fig.text(
    seed_center_x, GS_TOP + 0.015,
    "Anatomix Segmentations",
    ha="center", va="bottom",
    fontsize=FONT_TOP + 1, fontweight="bold", fontstyle="italic", color="#111111",
)

# ---- Rows 1-4: map rows ---------------------------------------------------

for row_idx, (row_paths, cmap) in enumerate(map_rows):
    slices = [load_slice(p, SLICE_Z) for p in row_paths]
    row_max = max(sl.max() for sl in slices)
    row_max = row_max if row_max > 0 else 1.0

    for col_idx, sl in enumerate(slices):
        ax = fig.add_subplot(gs[row_idx + 1, col_idx])
        ax.imshow(sl, cmap=cmap, vmin=0, vmax=row_max, origin="lower", aspect="equal")
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(COL_LABELS[col_idx], fontsize=FONT_COL, fontweight="bold", pad=5)

# ---- Horizontal separator after row 0 ------------------------------------

# Compute figure-coord y of the gap between row 0 and row 1
row_h = (GS_TOP - GS_BOTTOM) / (N_ROWS + (N_ROWS - 1) * GS_HSPACE)
row_g = GS_HSPACE * row_h
row0_bottom = GS_TOP - row_h
row1_top    = row0_bottom - row_g
# Place separator just below row 0 — above the column header text that sits above row 1
sep_y = row0_bottom - 0.004

line = mlines.Line2D(
    [GS_LEFT - 0.01, GS_RIGHT],
    [sep_y, sep_y],
    transform=fig.transFigure,
    color="#666666", linewidth=1.8, linestyle="--",
    clip_on=False,
)
fig.add_artist(line)

# ---- Row labels on the left -----------------------------------------------

label_x = GS_LEFT / 2   # centre of the left-margin strip

for row_idx, label in enumerate(ROW_LABELS):
    row_center_y = GS_TOP - (row_idx + 0.5) * (row_h + row_g) + row_g / 2
    fig.text(
        label_x, row_center_y,
        label,
        va="center", ha="center",
        fontsize=FONT_ROW, fontweight="bold",
        rotation=90,
    )

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
