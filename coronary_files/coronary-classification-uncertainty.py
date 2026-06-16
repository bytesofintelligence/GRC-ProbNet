#!/usr/bin/env python
# coding: utf-8

# # Coronary Geo-Radio Classification per-fold
#
# Pipeline prerequisites (must run in order before this script):
#   Stage 1 — coronary-seg-finetune.py
#             Anatomix coronary segmentation checkpoints:
#             saved_models/segmentation_asoca/fold_k/*.pth
#   Stage 2 — coronary-atlas-istn.py
#             Fold-specific Atlas / STN models:
#             output/asoca-coronary/fold_k/full-stn/train/model/
#             {stn.pt, atlas_labelmap_final.nii.gz, crop_size.json}
#
# This script (Stage 3):
#   - Loads the frozen fold-k STN (NOT retrained here).
#   - Runs forward passes on ALL 40 ASOCA patients to extract deformation fields.
#   - Extracts geometric (SVD) + PyRadiomics features from the coronary ROI.
#   - Trains MLP on fold-k train patients only; evaluates on fold-k test patients.
#   - Saves per-fold outputs for later aggregation.
#
# CV design: the fold-k STN was trained exclusively on fold-k train patients.
#   Applying it to all 40 patients during feature extraction is NOT leakage
#   (analogous to extracting pretrained-CNN features).  The MLP classifier
#   trains only on fold-k train features and tests only on fold-k test features.
#
# GT coronary masks are NEVER used — all segmentations are Anatomix predictions.
#
# Usage:
#   python coronary-classification.py --fold fold_1

# # Imports and Global Config


import sys
import os
import json
import argparse
sys.path.insert(0, "/vol/biomedic2/bglocker_studproj/<INSERT WHERE ANATOMIX IS FOR YOU>/anatomix/")

import torch
from monai.data import ThreadDataLoader, CacheDataset
from monai.transforms import Lambdad
from nets.stn import FullSTN3D
from img.datasets import ImageSegmentationOneHotDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

_parser = argparse.ArgumentParser(description="Coronary Geo-Radio Classification")
# Coronary: --fold selects the cross-validation fold and derives all paths automatically.
_parser.add_argument(
    "--fold", default="fold_1",
    help="Cross-validation fold identifier, e.g. fold_1 … fold_5 (default: fold_1)."
)
_parser.add_argument(
    "--use-uncertainty", action="store_true", default=False,
    help="Append per-structure uncertainty entropy features to the classifier input. "
)
_parser.add_argument(
    "--uncertainty-csv",
    # Coronary uncertainty CSV path
    default="output/asoca-coronary/uncertainty_analysis/metrics/per_structure_uncertainty.csv",
    help="Path to per_structure_uncertainty.csv produced by compute_uncertainty.py."
)
_parser.add_argument(
    "--run-resnet", action="store_true", default=False,
    help="Run the ResNet-50 3D image-only baseline after the MLP experiment. "
         "Disabled by default"
)
_args = _parser.parse_args()

# Accept bare digit: --fold 1 -> fold_1
if _args.fold.isdigit():
    _args.fold = f"fold_{_args.fold}"

fold = _args.fold

# Derive fold-specific paths from --fold.
# All Stage-2 outputs live under output/asoca-coronary/<fold>/full-stn/train/model/
_base_cfg = "data/config/asoca"
_base_out = f"output/asoca-coronary/{fold}"

spacing = (2.0, 2.0, 2.0)
anatomix_roi_size = (96, 96, 96)
crop_size = (96, 96, 96)
# Coronary: binary segmentation — background (0) + coronary_artery (1).
# MM-WHS uses num_classes=8 with 7 cardiac structures.
num_classes = 2

# Experiment switch — controlled by --use-uncertainty CLI flag.
USE_UNCERTAINTY_FEATURES = _args.use_uncertainty
UNCERTAINTY_CSV = _args.uncertainty_csv
RUN_RESNET = _args.run_resnet

# # Class Mapping
# Coronary: single foreground class.  MM-WHS had 7 structures (1–7).
class_mapping = {
        1: "coronary_artery",
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# # Load STN

# Coronary: fold-specific STN from Stage-2 outputs.  The STN is loaded frozen
# and is NOT retrained during classification.
stn_path = f"{_base_out}/full-stn/train/model/stn.pt"
# Since the trained STN may have been trained with a different crop_size,
# we check if there is a crop_size.json in the same directory to load the correct crop_size.
# ensures preprocessing and model dimensions are consistent
crop_size_path = os.path.join(os.path.dirname(stn_path), "crop_size.json")
if os.path.exists(crop_size_path):
    with open(crop_size_path, "r") as f:
        crop_meta = json.load(f)
    crop_size = tuple(int(v) for v in crop_meta.get("crop_size", crop_size))
    print(f"Loaded crop_size from {crop_size_path}: {crop_size}")
else:
    print(f"crop_size.json not found at {crop_size_path}; using fallback crop_size={crop_size}")

# input_channels = 2*(num_classes-1) = 2*(2-1) = 2  (binary: bg + coronary_artery)
# MM-WHS had 2*(8-1) = 14 channels for 7 cardiac structures.
stn = FullSTN3D(input_size=crop_size, input_channels=2*(num_classes-1), device=device).to(device)
stn.load_state_dict(torch.load(stn_path))
stn.eval()


# # Load dataset

# function to auto-discover the lowest-loss finetuned Anatomix checkpoint for a fold.
# same as _find_best_asoca_fold_checkpoint in coronary-atlas-istn.py
def _find_best_asoca_fold_checkpoint(fold_id, base_dir="saved_models/segmentation_asoca"):
    fold_dir = os.path.join(base_dir, fold_id)
    prefix   = f"finetuned_asoca_{fold_id}_"
    candidates = [f for f in os.listdir(fold_dir) if f.startswith(prefix) and f.endswith(".pth")]
    if not candidates:
        fallback = os.path.join(fold_dir, f"anatomix_trained_asoca_{fold_id}.pth")
        print(f"[Anatomix] No finetuned checkpoints for {fold_id}, using {fallback}")
        return fallback
    def _loss(name):
        try:
            return float(name[len(prefix):-len(".pth")])
        except ValueError:
            return float("inf")
    best = min(candidates, key=_loss)
    ckpt = os.path.join(fold_dir, best)
    print(f"[Anatomix] Auto-selected checkpoint for {fold_id}: {ckpt}")
    return ckpt

anatomix_ckpt = _find_best_asoca_fold_checkpoint(fold)

# Coronary: load ALL 40 ASOCA patients (32 train + 8 test) so that features
# are extracted for every patient using the frozen fold-k STN.
# The MLP classifier later trains only on train patients and tests on test patients.
# skip_lcc=True: coronary arteries have multiple legitimate connected components
_train_csv = f"{_base_cfg}/{fold}/train.csv"
_test_csv  = f"{_base_cfg}/{fold}/test.csv"

dataset_train_base = ImageSegmentationOneHotDataset(_train_csv,
                                            num_classes, anatomix_roi_size, spacing,
                                            normalizer=Lambdad(keys=["image"], func=lambda x: x),
                                            binarize=0, augmentation=False,
                                            fixed_crop_size=crop_size,
                                            model_checkpoint=anatomix_ckpt,
                                            skip_lcc=True)
dataset_train = CacheDataset(data=dataset_train_base, transform=None, cache_rate=1.0, num_workers=4)
dataset_train.get_sample = dataset_train_base.get_sample
dataloader_train = ThreadDataLoader(dataset_train, batch_size=1, shuffle=False)

dataset_test_base = ImageSegmentationOneHotDataset(_test_csv,
                                            num_classes, anatomix_roi_size, spacing,
                                            normalizer=Lambdad(keys=["image"], func=lambda x: x),
                                            binarize=0, augmentation=False,
                                            fixed_crop_size=crop_size,
                                            model_checkpoint=anatomix_ckpt,
                                            skip_lcc=True)
dataset_test = CacheDataset(data=dataset_test_base, transform=None, cache_rate=1.0, num_workers=4)
dataset_test.get_sample = dataset_test_base.get_sample
dataloader_test = ThreadDataLoader(dataset_test, batch_size=1, shuffle=False)


# # Extract radiomic and deformation data
# This loop will iterate over each CT test volume in `inference.csv`, creating features per volume in a list.
# For each test volume, we store the following features for downstream classification:
# - **label**, described as either "Diseased" or "Healthy" (obtained by parsing the file name).
# - **struct_disp**, a dictionary keyed per substructure storing the respective deformation displacement field.
# - **radiomics**, a dictionary keyed per substructure storing the respective radiomics features.
import torch
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


radiomics_settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkLinear,
    'verbose': False
}
extractor = featureextractor.RadiomicsFeatureExtractor(**radiomics_settings)

# Coronary: fold-specific atlas from previous stage outputs
atlas_label_itk = sitk.ReadImage(f"{_base_out}/full-stn/train/model/atlas_labelmap_final.nii.gz")

arr_lab = sitk.GetArrayFromImage(atlas_label_itk)
if arr_lab.ndim == 4:  # already one‐hot in last dim
    atlas_label = (
        torch.from_numpy(arr_lab)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        .to(device)
    )
else:
    labels_int = torch.from_numpy(arr_lab).long()
    one_hot    = torch.nn.functional.one_hot(labels_int, num_classes=num_classes)
    atlas_label = one_hot.permute(3, 0, 1, 2).unsqueeze(0).float().to(device)

# Coronary: batch_size is always 1 (one patient at a time); no need to repeat atlas.
batch_size = 1
atlas_label = atlas_label.repeat(batch_size, 1, 1, 1, 1)

# Precompute identity grid once (for displacement = T – grid)
identity_grid = stn.grid.unsqueeze(0)
identity_grid = stn.move_grid_dims(identity_grid)
identity_grid = identity_grid.repeat(batch_size, 1, 1, 1, 1).to(device)

# SEMANTIC_FEATURES: initialised empty and is populated on the first successful 
# radiomics call. previously, for MM-WHS this variable was defined inside the else-branch, which
# would crash if the very first patient had an empty mask (rare for cardiac structures
# but more likely for coronary arteries with their sparse Anatomix predictions)
SEMANTIC_FEATURES = []

subjects = []

# Coronary: iterate over BOTH train and test patients so all 40 ASOCA scans get features.
# Each subject dict carries a "split" key ("train" or "test") used later to partition
# the MLP inputs without mixing train and test during classifier training.
for _dataloader, _split in [(dataloader_train, "train"), (dataloader_test, "test")]:
    for sample_idx, batch in enumerate(tqdm(_dataloader, desc=f"extracting features [{_split}]")):
        image_tensor = batch["image"].to(device)
        label_onehot  = batch["labelmap"].to(device)
        fname         = batch["fname"][0]

        img_type = "Diseased" if "Diseased" in fname else "Normal"

        # Run STN to get full warp grid T
        src = label_onehot[:, 1:, ...]
        tgt = atlas_label[:, 1:, ...]
        _   = stn(torch.cat((src, tgt), dim=1))
        # used to get geometric features
        T = stn.get_T()
        full_disp = T - identity_grid # deformation field
        disp_np = full_disp[0].detach().cpu().numpy()

        # per‐structure displacement
        struct_disp = {}
        for L in class_mapping.keys():
            maskL = label_onehot[0, L].bool().cpu().numpy()
            disp_vox = disp_np[maskL]
            struct_disp[L] = disp_vox

        # per‐structure radiomics
        img_np = image_tensor[0, 0].detach().cpu().numpy()
        sitk_img = sitk.GetImageFromArray(img_np)
        sitk_img.SetSpacing(spacing)

        radiomics = {}
        for L, name in class_mapping.items():
            mask_np = label_onehot[0, L].cpu().numpy().astype(np.uint8)
            if mask_np.sum() == 0:
                # Coronary masks can be empty if Anatomix prediction is all-background
                # Store None here; backfilled after the loop once SEMANTIC_FEATURES length is known.
                radiomics[L] = None
            else:
                sitk_mask = sitk.GetImageFromArray(mask_np)
                sitk_mask.CopyInformation(sitk_img)
                result = extractor.execute(sitk_img, sitk_mask)
                # get only original features (features derived from the unfiltered CT image)
                # of the original subset, the radiomic features are:
                # – First-order statistics
                # – Shape descriptors (3D)
                # – GLCM (Gray Level Co-occurrence Matrix)
                # – GLRLM (Gray Level Run Length Matrix)
                # – GLSZM (Gray Level Size Zone Matrix)
                # – NGTDM (Neighbouring Gray Tone Difference Matrix)
                # – GLDM (Gray Level Dependence Matrix)
                if not SEMANTIC_FEATURES:
                    SEMANTIC_FEATURES = sorted([k for k in result.keys() if k.startswith("original_")])

                feats = []
                for feat_name in SEMANTIC_FEATURES:
                    val = result.get(feat_name, float("nan"))
                    feats.append(float(val))
                radiomics[L] = np.array(feats, dtype=float)

        # Collect everything into a single dict for this subject
        subject_data = {
            "sample_idx": sample_idx,   # matches sample_id in uncertainty CSV
            "fname":     fname,
            "label":     img_type,
            "split":     _split,        # "train" or "test" — used to partition MLP inputs
            "full_disp": disp_np,       # [D,H,W,3]
            "struct_disp": struct_disp, # dict L->(n_vox_L,3)
            "radiomics":   radiomics    # dict L->(len(SEMANTIC_FEATURES),) or None
        }
        subjects.append(subject_data)

# Backfill any deferred None radiomics (empty-mask patients) with NaN arrays now
# that SEMANTIC_FEATURES is known.
for _subj in subjects:
    for L in class_mapping.keys():
        if _subj["radiomics"][L] is None:
            _subj["radiomics"][L] = np.full(len(SEMANTIC_FEATURES), np.nan, dtype=float)

# After this loop, `subjects` is a list of length N (all patients in this fold),
# and each `subjects[i]` contains all the deformation + radiomics for that case.


# # MLP classification
# Here, we perform hyperparameter optimisation using `optuna` to find the best classification model for the diseased data.
# We perform 5-fold stratified cross-validation over 3 seeds to achieve confidence in the low volume of data.

import torch
import numpy as np
import pandas as pd
import optuna
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torchvision.ops import MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

MAX_DEF_PC = 3

# Load per-structure uncertainty CSVs (split-aware)
# Two separate CSVs: test split (8 rows, sample_ids 0-7) and train split (32 rows, sample_ids 0-31).
# USE_UNCERTAINTY_FEATURES controls only whether unc_* columns enter selected_cols in objective().
_unc_test_csv  = f"{_base_out}/full-stn/uncertainty_analysis/metrics/per_structure_uncertainty.csv"
_unc_train_csv = f"{_base_out}/full-stn-train-unc/uncertainty_analysis/metrics/per_structure_uncertainty.csv"

_unc_test_df   = pd.read_csv(_unc_test_csv)
_unc_train_df  = pd.read_csv(_unc_train_csv)

print("\n=== Uncertainty CSVs — Step 1 verification ===")
print(f"  Test  CSV : {_unc_test_csv}  shape={_unc_test_df.shape}")
print(f"  Train CSV : {_unc_train_csv}  shape={_unc_train_df.shape}")

_unc_test_pivot  = _unc_test_df.pivot(index="sample_id", columns="class_id", values="mean_entropy")
_unc_test_pivot.columns.name = None
_unc_train_pivot = _unc_train_df.pivot(index="sample_id", columns="class_id", values="mean_entropy")
_unc_train_pivot.columns.name = None

print(f"\n=== Uncertainty pivots — Step 2 verification ===")
print(f"  Test  pivot shape : {_unc_test_pivot.shape}   cols={list(_unc_test_pivot.columns)}")
print(f"  Train pivot shape : {_unc_train_pivot.shape}   cols={list(_unc_train_pivot.columns)}")

_missing_test  = [s["sample_idx"] for s in subjects if s["split"] == "test"  and s["sample_idx"] not in _unc_test_pivot.index]
_missing_train = [s["sample_idx"] for s in subjects if s["split"] == "train" and s["sample_idx"] not in _unc_train_pivot.index]
if _missing_test or _missing_train:
    print(f"\n  [WARN] missing test sample_ids : {_missing_test}")
    print(f"  [WARN] missing train sample_ids: {_missing_train}")
else:
    print(f"\n  All uncertainty entries present. ✓")

rows = []
for subj in subjects:
    row = {}
    
    for L, name in class_mapping.items():
        disp_vox = subj["struct_disp"][L]
        n_vox = disp_vox.shape[0]
        
        if n_vox < 1:
            # no voxels -> all zeros
            evr = np.zeros(MAX_DEF_PC, dtype=float)
        else:
            u, s, vh = np.linalg.svd(disp_vox, full_matrices=False)

            # Store top MAX_DEF_PC singular values, pad with zeros if needed
            evr = np.zeros(MAX_DEF_PC, dtype=float)
            n_comp = min(len(s), MAX_DEF_PC)
            evr[:n_comp] = s[:n_comp]
        
        # store under def_pc1_<name> … def_pc5_<name>
        for pc_idx in range(MAX_DEF_PC):
            col = f"def_pc{pc_idx+1}_{name}"
            row[col] = float(evr[pc_idx])
    
    #Radiomics (semantic features) per structure
    for L, name in class_mapping.items():
        rad_vec = subj["radiomics"][L]  # shape = (len(SEMANTIC_FEATURES),) or all-nan
        for idx, feat_name in enumerate(SEMANTIC_FEATURES):
            col = f"{feat_name}_{name}"
            row[col] = float(rad_vec[idx])

    # Uncertainty features: masked-mean binary entropy per foreground structure.
    # Split-conditional lookup: test subjects use _unc_test_pivot; train subjects use _unc_train_pivot.
    sid    = subj["sample_idx"]
    _pivot = _unc_test_pivot if subj["split"] == "test" else _unc_train_pivot
    for L, name in class_mapping.items():
        row[f"unc_{name}"] = (
            float(_pivot.loc[sid, L])
            if sid in _pivot.index
            else float("nan")
        )

    row["fname"] = subj["fname"]  # full path — used to derive patient_id in outputs
    row["label"] = subj["label"]  # "Normal" or "Diseased"
    row["split"] = subj["split"]  # "train" or "test" — needed for fold-based MLP split
    rows.append(row)

df = pd.DataFrame(rows)

df_full = pd.DataFrame.from_records(rows)
print(df_full.isna().any()[lambda x: x])
print(df_full.head(4))
print("Shape of df_full:", df_full.shape)

# Confirm uncertainty columns are present
_unc_col_names = [f"unc_{name}" for _, name in class_mapping.items()]
print(f"\n=== Uncertainty columns in df_full (Step 4 verification) ===")
print(f"  {_unc_col_names}")
print(f"  Example subject 0:\n{df_full[_unc_col_names].iloc[0].to_string()}")

# Useful to have a feature dimensionality summary
_n_structs   = len(class_mapping)                          # 1 (coronary_artery only)
_n_rad_feats = len(SEMANTIC_FEATURES) * _n_structs         # e.g. 107 × 1 = 107
_n_geo_feats = MAX_DEF_PC * _n_structs                     # 3 × 1 = 3
_n_unc_feats = _n_structs                                  # 1 (not yet in df_full for coronary)

print(f"\n=== Feature dimensionality (Step 6) ===")
print(f"  Radiomic features : {len(SEMANTIC_FEATURES)} per structure × {_n_structs} = {_n_rad_feats}")
print(f"  Geometric features: {MAX_DEF_PC} PCs × {_n_structs} = {_n_geo_feats}  "
      f"(Optuna searches def_pc_amt in 1–{MAX_DEF_PC})")
print(f"  Uncertainty feats : {_n_unc_feats}  (unc_coronary_artery)")
print(f"  ── df_full total cols (excl. label/split/fname): {df_full.shape[1] - 3}")
print(f"  ── Baseline  MLP input (def_pc_amt=MAX_DEF_PC): "
      f"{_n_geo_feats} + {_n_rad_feats} = {_n_geo_feats + _n_rad_feats}")
print(f"  ── USE_UNCERTAINTY_FEATURES = {USE_UNCERTAINTY_FEATURES}")

y_global = np.array(df_full["label"])
N = len(y_global)

seeds = [10, 101, 202]

# Coronary: fold split is predefined by the ASOCA fold CSVs, not random StratifiedKFold.
# train_mask / val_mask are boolean arrays derived from the "split" column in df_full.
# The MLP trains on train patients only and is evaluated on test patients only.
# 3 seeds control MLP weight initialisation / optimiser randomness.
_train_mask = (df_full["split"] == "train").to_numpy()
_val_mask   = (df_full["split"] == "test").to_numpy()

def objective(trial):
    hidden_units = trial.suggest_int("hidden_units", 8, 512, step=8)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)
    num_layers   = trial.suggest_int("num_layers", 1, 12)
    num_epochs   = trial.suggest_int("num_epochs", 100, 400, step=25)

    def_pc_amt = trial.suggest_int("def_pc_amt", 1, MAX_DEF_PC)

    selected_cols = []
    for L, name in class_mapping.items():
        for i in range(def_pc_amt):
            selected_cols.append(f"def_pc{i+1}_{name}")
        for i, feat_name in enumerate(SEMANTIC_FEATURES):
            selected_cols.append(f"{feat_name}_{name}")

    # Append uncertainty columns to the classifier input
    if USE_UNCERTAINTY_FEATURES:
        for _, name in class_mapping.items():
            selected_cols.append(f"unc_{name}")

    # If we're on trial 0, just confirm uncertainty columns are in X_np
    if trial.number == 0:
        unc_in_sel = [c for c in selected_cols if c.startswith("unc_")]
        print(f"\n[Trial 0] USE_UNCERTAINTY_FEATURES={USE_UNCERTAINTY_FEATURES}")
        print(f"[Trial 0] selected_cols count : {len(selected_cols)}")
        print(f"[Trial 0] unc_* cols present  : {unc_in_sel}")

    X_df = df_full[selected_cols]                   # shape (N_samples, def_pc_amt*1 + F_sem*1 [+ 1])
    X_np = X_df.to_numpy(dtype=np.float32)           # numpy array (N_samples, D_trial)

    if trial.number == 0:
        print(f"[Trial 0] X_np.shape          : {X_np.shape}  <- uncertainty features {'included' if USE_UNCERTAINTY_FEATURES else 'excluded'}")
    y_np = y_global.copy()                           # numpy array (N_samples,), strings "Normal"/"Diseased"

    X_all = torch.from_numpy(X_np).float()
    y_all = torch.from_numpy((y_np == "Diseased").astype(np.float32)).to(device)

    seed_means = []
    per_seed_fold_metrics = []

    # HPO uses inner 4-fold CV on the 32 training patients only.
    # The held-out test fold (_val_mask) is never seen during hyperparameter search.
    _tr_global = _train_mask.nonzero()[0]   # 32 training-patient global indices
    _y_inner   = y_np[_tr_global]           # string labels for stratified splitting

    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

        fold_accuracy_list = []
        fold_metrics_list = []

        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=s)
        for inner_tr, inner_val in inner_skf.split(np.zeros((len(_tr_global), 1)), _y_inner):
            train_idx = _tr_global[inner_tr]
            val_idx   = _tr_global[inner_val]
            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_val   = X_all[val_idx]
            y_val   = y_all[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled   = scaler.transform(X_val)

            X_train = torch.from_numpy(X_train_scaled).float().to(device)
            X_val   = torch.from_numpy(X_val_scaled).float().to(device)

            # MLP
            layers = [hidden_units] * num_layers + [1]
            model = MLP(X_np.shape[1], layers, dropout=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            # Training loop
            for epoch in range(1, num_epochs + 1):
                model.train()
                logits = model(X_train).squeeze(1)
                loss = criterion(logits, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val).squeeze(1)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int)

                y_true = y_val.cpu().numpy().astype(int)
                y_pred = val_preds

                acc  = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec  = recall_score(y_true, y_pred, zero_division=0)
                f1   = f1_score(y_true, y_pred, zero_division=0)
                cm   = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                # AUC uses the raw probability score (not the binary prediction).
                # try/except guards against the rare fold where only one class appears.
                try:
                    auc = roc_auc_score(y_true, val_probs)
                except ValueError:
                    auc = 0.5

            fold_accuracy_list.append(acc)
            fold_metrics_list.append({
                "accuracy":        acc,
                "precision":       prec,
                "recall":          rec,
                "f1":              f1,
                "sensitivity":     sensitivity,
                "specificity":     specificity,
                "auc":             auc,
                "confusion_matrix": cm
            })

        seed_mean_acc = np.mean(fold_accuracy_list)
        seed_means.append(seed_mean_acc)
        per_seed_fold_metrics.append(fold_metrics_list)

    trial.set_user_attr("seed_means", seed_means)
    trial.set_user_attr("per_seed_fold_metrics", per_seed_fold_metrics)
    trial.set_user_attr("hyperparams", {
        "hidden_units": hidden_units,
        "lr":            lr,
        "dropout":       dropout,
        "num_layers":    num_layers,
        "num_epochs":    num_epochs,
        "def_pc_amt":    def_pc_amt
    })

    return np.mean(seed_means)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, show_progress_bar=True)

# Tie-break on stddev
trials = study.trials
best_trial = max(trials, key=lambda t: (t.value, -np.std(t.user_attrs["seed_means"])))

best_score       = best_trial.value
best_std_seed    = np.std(best_trial.user_attrs["seed_means"])
best_hyperparams = best_trial.user_attrs["hyperparams"]
best_fold_info   = {
    'seeds': seeds,
    'seed_means': best_trial.user_attrs["seed_means"],
    'per_seed_fold_metrics': best_trial.user_attrs["per_seed_fold_metrics"]
}

# Aggregate all metrics over seeds & folds
all_accuracies    = []
all_precisions    = []
all_recalls       = []
all_f1s           = []
all_sensitivities = []
all_specificities = []
all_aucs          = []

for seed_metrics in best_fold_info['per_seed_fold_metrics']:
    for m in seed_metrics:
        all_accuracies.append(m['accuracy'])
        all_precisions.append(m['precision'])
        all_recalls.append(m['recall'])
        all_f1s.append(m['f1'])
        all_sensitivities.append(m['sensitivity'])
        all_specificities.append(m['specificity'])
        all_aucs.append(m['auc'])

acc_mean,  acc_std  = np.mean(all_accuracies),    np.std(all_accuracies)
prec_mean, prec_std = np.mean(all_precisions),   np.std(all_precisions)
rec_mean,  rec_std  = np.mean(all_recalls),      np.std(all_recalls)
f1_mean,   f1_std   = np.mean(all_f1s),          np.std(all_f1s)
sens_mean, sens_std = np.mean(all_sensitivities), np.std(all_sensitivities)
spec_mean, spec_std = np.mean(all_specificities), np.std(all_specificities)
auc_mean,  auc_std  = np.mean(all_aucs),          np.std(all_aucs)

print("\n=== Best Hyperparameters (by avg-seed, tie-break on lowest std) ===")
for k, v in best_hyperparams.items():
    print(f"{k:<12}: {v}")
print(f"Avg of seed-means = {best_score:.4f}")
print(f"Std of seed-means = {best_std_seed:.4f}\n")

print("=== Aggregate Metrics over all seeds & folds (mean ± std) ===")
print(f"Accuracy    : {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Precision   : {prec_mean:.4f} ± {prec_std:.4f}")
print(f"Recall      : {rec_mean:.4f} ± {rec_std:.4f}")
print(f"F1 Score    : {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Sensitivity : {sens_mean:.4f} ± {sens_std:.4f}")
print(f"Specificity : {spec_mean:.4f} ± {spec_std:.4f}")
print(f"AUC-ROC     : {auc_mean:.4f} ± {auc_std:.4f}\n")

for seed_idx, s in enumerate(best_fold_info['seeds']):
    print(f"--- Seed {s} (mean CV acc = {best_fold_info['seed_means'][seed_idx]:.4f}) ---")
    for fold_idx, m in enumerate(best_fold_info['per_seed_fold_metrics'][seed_idx], start=1):
        print(f"Fold {fold_idx}:")
        print(f"  Accuracy    = {m['accuracy']:.4f}")
        print(f"  Precision   = {m['precision']:.4f}")
        print(f"  Recall      = {m['recall']:.4f}")
        print(f"  F1 Score    = {m['f1']:.4f}")
        print(f"  Sensitivity = {m['sensitivity']:.4f}")
        print(f"  Specificity = {m['specificity']:.4f}")
        print(f"  AUC-ROC     = {m['auc']:.4f}")
        print(f"  Confusion Matrix:\n{m['confusion_matrix']}\n")


# Save per-fold results to a labelled CSV - useful for later analysis
_exp_tag = "uncertainty" if USE_UNCERTAINTY_FEATURES else "baseline"
_out_dir = os.path.join(_base_out, f"classification_{_exp_tag}")
os.makedirs(_out_dir, exist_ok=True)

# features.csv: raw feature vectors for all patients
_meta_cols = ["fname", "label", "split"]
_feat_cols = [c for c in df_full.columns if c not in _meta_cols]
_feats_df  = df_full[_meta_cols + _feat_cols].copy()
_feats_df.insert(0, "patient_id",
    _feats_df["fname"].apply(lambda p: os.path.basename(p).replace(".nii.gz", "")))
_feats_df  = _feats_df.drop(columns=["fname"])
_feats_path = os.path.join(_out_dir, "features.csv")
_feats_df.to_csv(_feats_path, index=False)
print(f"\nFeatures saved  -> {_feats_path}  (shape: {_feats_df.shape})")

# Retrain best MLP (all seeds) on train split; ensemble predictions on test split
_hp       = best_hyperparams
_lyrs_out = [_hp["hidden_units"]] * _hp["num_layers"] + [1]

_sel_out = []
for _L, _name in class_mapping.items():
    for _i in range(_hp["def_pc_amt"]):
        _sel_out.append(f"def_pc{_i+1}_{_name}")
    for _fn in SEMANTIC_FEATURES:
        _sel_out.append(f"{_fn}_{_name}")
if USE_UNCERTAINTY_FEATURES:
    for _, _name in class_mapping.items():
        _sel_out.append(f"unc_{_name}")

_X_out  = df_full[_sel_out].to_numpy(dtype=np.float32)
_tr_idx = _train_mask.nonzero()[0]
_te_idx = _val_mask.nonzero()[0]

_scl_out = StandardScaler()
_Xtr_out = _scl_out.fit_transform(_X_out[_tr_idx])
_Xte_out = _scl_out.transform(_X_out[_te_idx])

_ytr_out = (y_global[_tr_idx] == "Diseased").astype(np.float32)
_yte_out = (y_global[_te_idx] == "Diseased").astype(int)

_seed_probs = []
_first_mlp  = None
for _s in seeds:
    random.seed(_s); np.random.seed(_s); torch.manual_seed(_s)
    _mdl_out  = MLP(len(_sel_out), _lyrs_out, dropout=_hp["dropout"]).to(device)
    _opt_out  = torch.optim.AdamW(_mdl_out.parameters(), lr=_hp["lr"])
    _crit_out = torch.nn.BCEWithLogitsLoss()
    _Xt_out   = torch.from_numpy(_Xtr_out).float().to(device)
    _yt_out   = torch.from_numpy(_ytr_out).to(device)
    for _ in range(_hp["num_epochs"]):
        _mdl_out.train()
        _l = _crit_out(_mdl_out(_Xt_out).squeeze(1), _yt_out)
        _opt_out.zero_grad(); _l.backward(); _opt_out.step()
    _mdl_out.eval()
    with torch.no_grad():
        _p = torch.sigmoid(
            _mdl_out(torch.from_numpy(_Xte_out).float().to(device)).squeeze(1)
        ).cpu().numpy()
    _seed_probs.append(_p)
    if _first_mlp is None:
        _first_mlp = _mdl_out  # save first-seed model for mlp.pt

_ens_probs = np.mean(_seed_probs, axis=0)
_ens_preds = (_ens_probs >= 0.5).astype(int)

# predictions.csv: test patients only
_pred_rows = []
for _i, _fi in enumerate(_te_idx):
    _pfname = df_full["fname"].iloc[_fi]
    _pred_rows.append({
        "patient_id": os.path.basename(_pfname).replace(".nii.gz", ""),
        "label":      y_global[_fi],
        "pred_label": "Diseased" if _ens_preds[_i] else "Normal",
        "pred_prob":  float(_ens_probs[_i]),
    })
_pred_path = os.path.join(_out_dir, "predictions.csv")
pd.DataFrame(_pred_rows).to_csv(_pred_path, index=False)
print(f"Predictions saved -> {_pred_path}")

# mlp.pt: MLP weights from the first seed (deterministic reference model)
_mlp_path = os.path.join(_out_dir, "mlp.pt")
torch.save(_first_mlp.state_dict(), _mlp_path)
print(f"MLP model saved  -> {_mlp_path}  (seed={seeds[0]})")

# metrics.json: aggregate Optuna metrics + ensemble test metrics for this fold
_metrics_out = {
    "fold":        fold,
    "experiment":  _exp_tag,
    "n_train":     int(_tr_idx.shape[0]),
    "n_test":      int(_te_idx.shape[0]),
    "seeds":       seeds,
    "hyperparams": _hp,
    "aggregate": {
        "accuracy":    {"mean": float(acc_mean),  "std": float(acc_std)},
        "precision":   {"mean": float(prec_mean), "std": float(prec_std)},
        "recall":      {"mean": float(rec_mean),  "std": float(rec_std)},
        "f1":          {"mean": float(f1_mean),   "std": float(f1_std)},
        "sensitivity": {"mean": float(sens_mean), "std": float(sens_std)},
        "specificity": {"mean": float(spec_mean), "std": float(spec_std)},
        "auc":         {"mean": float(auc_mean),  "std": float(auc_std)},
    },
    "ensemble_test": {
        "accuracy":    float(accuracy_score(_yte_out, _ens_preds)),
        "precision":   float(precision_score(_yte_out, _ens_preds, zero_division=0)),
        "recall":      float(recall_score(_yte_out, _ens_preds, zero_division=0)),
        "f1":          float(f1_score(_yte_out, _ens_preds, zero_division=0)),
    },
}
_json_path = os.path.join(_out_dir, "metrics.json")
with open(_json_path, "w") as _jf:
    json.dump(_metrics_out, _jf, indent=2)
print(f"Metrics saved    -> {_json_path}")

# results csv
_result_rows = []
for _si, _s in enumerate(best_fold_info['seeds']):
    for _fi, _m in enumerate(best_fold_info['per_seed_fold_metrics'][_si], start=1):
        _result_rows.append({
            "experiment":  _exp_tag,
            "seed":        _s,
            "fold":        _fi,
            "accuracy":    _m["accuracy"],
            "precision":   _m["precision"],
            "recall":      _m["recall"],
            "f1":          _m["f1"],
            "sensitivity": _m["sensitivity"],
            "specificity": _m["specificity"],
            "auc":         _m["auc"],
        })
_results_csv = os.path.join(_out_dir, f"results_{_exp_tag}.csv")
pd.DataFrame(_result_rows).to_csv(_results_csv, index=False)
print(f"Results saved    -> {_results_csv}")


# Feature importance analysis (uncertainty experiment only) 
# Retrains with best hyperparameters on one seed (5 folds) to measure:
#   (a) First-layer weight magnitudes — proxy for how much each input feature is used.
#   (b) Permutation importance for unc_* features — direct accuracy-drop measure.
# Only runs when USE_UNCERTAINTY_FEATURES=True 

if USE_UNCERTAINTY_FEATURES:
    def analyse_uncertainty_importance():
        hp          = best_trial.user_attrs["hyperparams"]
        unc_cols    = [f"unc_{name}" for _, name in class_mapping.items()]

        # Reconstruct selected_cols exactly as the best trial used them
        sel = []
        for L, name in class_mapping.items():
            for i in range(hp["def_pc_amt"]):
                sel.append(f"def_pc{i+1}_{name}")
            for feat_name in SEMANTIC_FEATURES:
                sel.append(f"{feat_name}_{name}")
        for _, name in class_mapping.items():
            sel.append(f"unc_{name}")

        X     = df_full[sel].to_numpy(dtype=np.float32)
        y     = y_global.copy()
        n_in  = X.shape[1]
        unc_indices = [sel.index(c) for c in unc_cols]
        layers_fi   = [hp["hidden_units"]] * hp["num_layers"] + [1]

        w_imp   = np.zeros(n_in)
        p_drops = {c: [] for c in unc_cols}
        n_folds = 0

        s_fi = seeds[0]
        random.seed(s_fi); np.random.seed(s_fi); torch.manual_seed(s_fi)
        skf_fi = StratifiedKFold(n_splits=5, shuffle=True, random_state=s_fi)

        print(f"\n=== Feature Importance Analysis (seed={s_fi}, best HP={hp}) ===")
        print(f"  X shape           : {X.shape}")
        print(f"  Uncertainty cols  : {unc_cols}")
        print(f"  Uncertainty indices in X: {unc_indices}")

        for train_i, val_i in skf_fi.split(np.zeros((len(y), 1)), y):
            Xtr, Xvl = X[train_i], X[val_i]
            ytr = (y[train_i] == "Diseased").astype(np.float32)
            yvl = (y[val_i]   == "Diseased").astype(np.float32)

            sc   = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xvl_s = sc.transform(Xvl)

            Xtr_t = torch.from_numpy(Xtr_s).float().to(device)
            ytr_t = torch.from_numpy(ytr).to(device)

            mdl  = MLP(n_in, layers_fi, dropout=hp["dropout"]).to(device)
            opt  = torch.optim.AdamW(mdl.parameters(), lr=hp["lr"])
            crit = torch.nn.BCEWithLogitsLoss()

            for _ in range(hp["num_epochs"]):
                mdl.train()
                loss = crit(mdl(Xtr_t).squeeze(1), ytr_t)
                opt.zero_grad(); loss.backward(); opt.step()

            mdl.eval()

            # (a) First-layer weight magnitude per input feature
            first_lin = next(m for m in mdl.modules() if isinstance(m, torch.nn.Linear))
            w_imp += first_lin.weight.detach().abs().mean(dim=0).cpu().numpy()

            # Baseline accuracy on the validation fold
            with torch.no_grad():
                Xvl_t  = torch.from_numpy(Xvl_s).float().to(device)
                b_pred = (torch.sigmoid(mdl(Xvl_t).squeeze(1)).cpu().numpy() >= 0.5).astype(int)
            base_acc = accuracy_score(yvl.astype(int), b_pred)

            # (b) Permutation importance — only for the uncertainty feature
            rng = np.random.RandomState(42)
            for col in unc_cols:
                fi        = sel.index(col)
                Xvl_perm  = Xvl_s.copy()
                Xvl_perm[:, fi] = rng.permutation(Xvl_perm[:, fi])
                with torch.no_grad():
                    p_pred = (torch.sigmoid(
                        mdl(torch.from_numpy(Xvl_perm).float().to(device)).squeeze(1)
                    ).cpu().numpy() >= 0.5).astype(int)
                p_drops[col].append(base_acc - accuracy_score(yvl.astype(int), p_pred))

            n_folds += 1

        w_imp /= n_folds

        # Report
        print(f"\n--- (a) First-layer weight magnitude (avg over {n_folds} folds) ---")
        print(f"  {'Feature':<50}  {'Weight Mag':>10}")
        print(f"  {'-'*62}")
        unc_mags = []
        for col in unc_cols:
            fi  = sel.index(col)
            mag = float(w_imp[fi])
            unc_mags.append(mag)
            print(f"  {col:<50}  {mag:>10.6f}")

        print(f"\n  Uncertainty features — mean: {np.mean(unc_mags):.6f}  "
              f"max: {np.max(unc_mags):.6f}  min: {np.min(unc_mags):.6f}")
        print(f"  All features         — mean: {float(w_imp.mean()):.6f}  "
              f"max: {float(w_imp.max()):.6f}")

        ranked = sorted(enumerate(w_imp), key=lambda x: x[1], reverse=True)
        print(f"\n  Top-10 features by first-layer weight magnitude:")
        for rank, (fi, mag) in enumerate(ranked[:10], 1):
            tag = "  ← UNCERTAINTY" if sel[fi].startswith("unc_") else ""
            print(f"    #{rank:>2}: {sel[fi]:<58}  {mag:.6f}{tag}")

        print(f"\n--- (b) Permutation importance for uncertainty features ---")
        print(f"  Positive drop = accuracy falls when feature permuted -> feature is informative")
        print(f"  {'Feature':<50}  {'Mean drop':>10}  {'Std':>8}")
        print(f"  {'-'*72}")
        for col in unc_cols:
            drops = p_drops[col]
            print(f"  {col:<50}  {np.mean(drops):>+10.4f}  {np.std(drops):>8.4f}")

    analyse_uncertainty_importance()


# ResNet Baseline
# We provide a Resnet-50 model as an Image-only baseline to compare our model against.
# Disabled by default (--run-resnet flag required).

if RUN_RESNET:
    import torch
    import numpy as np
    import random
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    from monai.networks.nets import resnet
    from torch.utils.data import DataLoader, SubsetRandomSampler

    dataset = dataset_test
    all_indices = list(range(len(dataset)))

    valid_indices = []
    labels = []
    for i in all_indices:
        fname = dataset[i]['fname']
        if "Diseased" in fname:
            valid_indices.append(i)
            labels.append(1)
        elif "Normal" in fname:
            valid_indices.append(i)
            labels.append(0)
    labels = np.array(labels)
    N = len(valid_indices)

    seeds = [10, 101, 202]

    def make_resnet50_3d():
        return resnet.resnet50(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1
        )

    def get_folds(seed):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros((N, 1)), labels))

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        num_epochs = trial.suggest_categorical("num_epochs", [25, 50, 100])

        seed_means = []
        per_seed_fold_metrics = []

        for s in seeds:
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)

            folds = get_folds(s)
            fold_accs = []
            fold_metrics = []

            for train_idx, val_idx in folds:
                train_dataset_indices = [valid_indices[i] for i in train_idx]
                val_dataset_indices = [valid_indices[i] for i in val_idx]

                train_loader = DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=SubsetRandomSampler(train_dataset_indices),
                    num_workers=2,
                    pin_memory=torch.cuda.is_available()
                )
                val_loader = DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=SubsetRandomSampler(val_dataset_indices),
                    num_workers=2,
                    pin_memory=torch.cuda.is_available()
                )

                model = make_resnet50_3d().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                criterion = torch.nn.BCEWithLogitsLoss()

                for epoch in range(1, num_epochs + 1):
                    model.train()
                    for batch in train_loader:
                        imgs = batch['image'].float().to(device)
                        fname = batch['fname'][0]
                        lbl = torch.tensor(
                            [1.0 if "Diseased" in fname else 0.0],
                            device=device
                        ).view(-1)
                        optimizer.zero_grad()
                        logits = model(imgs).view(-1)
                        loss = criterion(logits, lbl)
                        loss.backward()
                        optimizer.step()

                model.eval()
                all_preds, all_trues = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch['image'].float().to(device)
                        fname = batch['fname'][0]
                        true_lbl = 1 if "Diseased" in fname else 0
                        logits = model(imgs).view(-1)
                        prob = torch.sigmoid(logits).cpu().item()
                        pred = 1 if prob >= 0.5 else 0
                        all_preds.append(pred)
                        all_trues.append(true_lbl)

                y_true = np.array(all_trues)
                y_pred = np.array(all_preds)
                acc  = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec  = recall_score(y_true, y_pred, zero_division=0)
                f1   = f1_score(y_true, y_pred, zero_division=0)
                cm   = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp/(tp+fn) if (tp+fn)>0 else 0.0
                specificity = tn/(tn+fp) if (tn+fp)>0 else 0.0

                fold_accs.append(acc)
                fold_metrics.append({
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'confusion_matrix': cm
                })

            seed_means.append(np.mean(fold_accs))
            per_seed_fold_metrics.append(fold_metrics)

        trial.set_user_attr("seed_means", seed_means)
        trial.set_user_attr("per_seed_fold_metrics", per_seed_fold_metrics)
        trial.set_user_attr("hyperparams", {
            'lr': lr,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs
        })

        return np.mean(seed_means)

    study = optuna.create_study(direction="maximize")
    # We do 3 trials for brevity, but can be increased further if you have more compute/time.
    # Else, you can reduce the number of seeds (from 3) or reduce the number of folds (from 5).
    study.optimize(objective, n_trials=3)

    trials = study.trials
    best_trial = max(trials, key=lambda t: (t.value, -np.std(t.user_attrs["seed_means"])))

    best_score       = best_trial.value
    best_std_seed    = np.std(best_trial.user_attrs["seed_means"])
    best_hyperparams = best_trial.user_attrs["hyperparams"]
    best_fold_info   = {
        'seeds': seeds,
        'seed_means': best_trial.user_attrs["seed_means"],
        'per_seed_fold_metrics': best_trial.user_attrs["per_seed_fold_metrics"]
    }

    # Print results
    print("\n=== Best Hyperparameters (by avg-seed, tie-break on lowest std) ===")
    for k, v in best_hyperparams.items():
        print(f"{k:<12}: {v}")
    print(f"Avg of seed-means = {best_score:.4f}")
    print(f"Std of seed-means = {best_std_seed:.4f}\n")

    # Aggregate all metrics over seeds & folds
    all_accuracies = []
    all_precisions = []
    all_recalls    = []
    all_f1s        = []
    all_sensitivities = []
    all_specificities = []

    for seed_metrics in best_fold_info['per_seed_fold_metrics']:
        for m in seed_metrics:
            all_accuracies.append(m['accuracy'])
            all_precisions.append(m['precision'])
            all_recalls.append(m['recall'])
            all_f1s.append(m['f1'])
            all_sensitivities.append(m['sensitivity'])
            all_specificities.append(m['specificity'])

    acc_mean,  acc_std  = np.mean(all_accuracies),    np.std(all_accuracies)
    prec_mean, prec_std = np.mean(all_precisions),   np.std(all_precisions)
    rec_mean,  rec_std  = np.mean(all_recalls),      np.std(all_recalls)
    f1_mean,   f1_std   = np.mean(all_f1s),          np.std(all_f1s)
    sens_mean, sens_std = np.mean(all_sensitivities), np.std(all_sensitivities)
    spec_mean, spec_std = np.mean(all_specificities), np.std(all_specificities)

    print("=== Aggregate Metrics over all seeds & folds (mean ± std) ===")
    print(f"Accuracy    : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Precision   : {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"Recall      : {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"F1 Score    : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Sensitivity : {sens_mean:.4f} ± {sens_std:.4f}")
    print(f"Specificity : {spec_mean:.4f} ± {spec_std:.4f}\n")

    for seed_idx, s in enumerate(best_fold_info['seeds']):
        print(f"--- Seed {s} (mean CV acc = {best_fold_info['seed_means'][seed_idx]:.4f}) ---")
        for fold_idx, m in enumerate(best_fold_info['per_seed_fold_metrics'][seed_idx], start=1):
            print(f"Fold {fold_idx}:")
            print(f"  Accuracy    = {m['accuracy']:.4f}")
            print(f"  Precision   = {m['precision']:.4f}")
            print(f"  Recall      = {m['recall']:.4f}")
            print(f"  F1 Score    = {m['f1']:.4f}")
            print(f"  Sensitivity = {m['sensitivity']:.4f}")
            print(f"  Specificity = {m['specificity']:.4f}")
            print(f"  Confusion Matrix:\n{m['confusion_matrix']}\n")

