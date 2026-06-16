#!/usr/bin/env python
# coding: utf-8

# Geo-Radio Classification — Multi-Seed Radiomic Uncertainty (Experiment 2)
#
# This is a fork of Geo-Radio-Classification.py
#
# Idea is: instead of running PyRadiomics once on a single consensus segmentation,
# this script runs PyRadiomics independently on each of the 5 per-seed
# warped-atlas segmentation maps already saved under:
#     output/asoca/uncertainty/seed_<S>/sample_<N>_warped_atlas_labelmap_argmax.nii.gz
#
# For every (feature, structure) pair we store:
#     mean_{feature}_{structure}  — mean across 5 seeds
#     var_{feature}_{structure}   — variance across 5 seeds
#
# optional --use-multiseed-geometry does the same aggregation for the
# 3 deformation-PC features derived from per-seed displacement fields.
#
# All outputs are written to separate files from Experiments 1 and baseline.

# Imports and Global Config

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

_parser = argparse.ArgumentParser(description="Geo-Radio Classification - Multi-Seed Experiment")
_parser.add_argument(
    "--use-multiseed-geometry", action="store_true", default=False,
    help="Aggregate the 3 deformation-PC geometric features across seeds "
         "(mean + variance).  Disabled by default as the 2(a) variant is only radiomic aggregation."
)
_parser.add_argument(
    "--uncertainty-root",
    default="output/asoca/uncertainty",
    help="Root directory containing seed_<S>/ subdirectories with per-seed warped-atlas outputs."
)
_parser.add_argument(
    "--seeds", nargs="+", type=int,
    default=[42, 123, 456, 789, 999],
    help="Anatomix segmentation seeds to aggregate over."
)
_parser.add_argument(
    "--results-csv",
    default=None,
    help="Output results CSV path.  Auto-derived from geometry flag if omitted."
)
_parser.add_argument(
    "--use-ivw", action="store_true", default=False,
    help="Replace the mean & variance feature pairs with a single inverse variance weighted "
         "mean per (radiomic/geometric, structure) pair. Halves the feature count back "
         "to the original (e.g. ~749 features) and ensures that high-uncertainty features are suppressed rather than "
         "inflated by StandardScaler normalisation."
)
_parser.add_argument(
    "--run-resnet", action="store_true", default=False,
    help="Run the ResNet 3D image-only baseline after the MLP experiment."
)
_args = _parser.parse_args()

spacing           = (2.0, 2.0, 2.0)
anatomix_roi_size = (96, 96, 96)
crop_size         = (96, 96, 96)
num_classes       = 8

# flags to control the four different variants within Experiment 2
USE_MULTISEED_GEOMETRY = _args.use_multiseed_geometry
USE_IVW                = _args.use_ivw
UNCERTAINTY_ROOT       = _args.uncertainty_root
SEEDS                  = _args.seeds
RUN_RESNET             = _args.run_resnet

# Small constant added to variance before dividing to prevent division by zero
# when all 5 seeds produce identical segmentations for a feature.
_IVW_EPS = 1e-8

# Auto-derive output CSV name and Optuna study name from active flags
_geo_tag = "geo_radio" if USE_MULTISEED_GEOMETRY else "radio"
_ivw_tag = "_ivw"      if USE_IVW               else ""

if _args.results_csv:
    RESULTS_CSV = _args.results_csv
else:
    RESULTS_CSV = f"results_multiseed_{_geo_tag}{_ivw_tag}.csv"

OPTUNA_STUDY_NAME = f"multiseed_{_geo_tag}{_ivw_tag}"

# Useful reporting for record-keeping 
print(f"=== Experiment 2 — Multi-Seed Radiomic Uncertainty ===")
print(f"  Seeds              : {SEEDS}")
print(f"  Uncertainty root   : {UNCERTAINTY_ROOT}")
print(f"  Geometry features  : {USE_MULTISEED_GEOMETRY}")
print(f"  Inverse-var weight : {USE_IVW}")
print(f"  Results CSV        : {RESULTS_CSV}")
print(f"  Optuna study       : {OPTUNA_STUDY_NAME}")

class_mapping = {
    1: "myocardium",
    2: "left atrium",
    3: "left ventricle",
    4: "right atrium",
    5: "right ventricle",
    6: "aorta",
    7: "pulmonary artery",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load STN
# Since the trained STN may have been trained with a different crop_size, 
# we check if there is a crop_size.json in the same directory to load the correct crop_size.
# ensures preprocessing and model dimensions are consistent
stn_path = "output/mm-whs/full-stn/train/model/stn.pt"
crop_size_path = os.path.join(os.path.dirname(stn_path), "crop_size.json")
if os.path.exists(crop_size_path):
    with open(crop_size_path, "r") as f:
        crop_meta = json.load(f)
    crop_size = tuple(int(v) for v in crop_meta.get("crop_size", crop_size))
    print(f"Loaded crop_size from {crop_size_path}: {crop_size}")
else:
    print(f"crop_size.json not found at {crop_size_path}; using fallback crop_size={crop_size}")

stn = FullSTN3D(input_size=crop_size, input_channels=2*(num_classes-1), device=device).to(device)
stn.load_state_dict(torch.load(stn_path))
stn.eval()

# Precompute identity grid in numpy for displacement = T - identity.
# Notes: "stn.grid shape: (3, W, H, D) before move_grid_dims.
# After move_grid_dims + [0]: (W, H, D, 3)  — same shape as loaded _transform.nii.gz arrays.
# (W==H==D==96 here so spatial ordering is irrelevant for the SVD-based geometric features.)"
identity_grid_np = (
    stn.move_grid_dims(stn.grid.unsqueeze(0))[0]
    .detach().cpu().numpy()
)  # (D, H, W, 3) effectively

dataset_test_base = ImageSegmentationOneHotDataset(
    "data/config/inference.csv",
    num_classes, anatomix_roi_size, spacing,
    normalizer=Lambdad(keys=["image"], func=lambda x: x),
    binarize=0, augmentation=False,
    fixed_crop_size=crop_size,
)
dataset_test = CacheDataset(data=dataset_test_base, transform=None, cache_rate=1.0, num_workers=4)
dataset_test.get_sample = dataset_test_base.get_sample
dataloader_test = ThreadDataLoader(dataset_test, batch_size=1, shuffle=False)

# Extract radiomic and deformation data
# This loop will iterate over each CT test volume in `inference.csv`, creating features per volume in a list.
# For each test volume, we store the following features for downstream classification:
# - **label**, described as either "Diseased" or "Healthy" (obtained by parsing the file name).
# - **struct_disp**, a dictionary keyed per substructure storing the respective deformation displacement field.
# - **radiomics**, a dictionary keyed per substructure storing the respective radiomics features.

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from radiomics import featureextractor
from tqdm import tqdm

radiomics_settings = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": sitk.sitkLinear,
    "verbose": False,
}
extractor = featureextractor.RadiomicsFeatureExtractor(**radiomics_settings)

# Helpers
# Extract radiomics from just one (image, mask) pair
def _extract_radiomics(sitk_img, mask_np):
    """Run PyRadiomics on one binary mask.
    Returns (feature_names, values) on success, or (None, None) if mask empty.
    """
    if mask_np.sum() == 0:
        return None, None
    sitk_mask = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    sitk_mask.CopyInformation(sitk_img)
    result = extractor.execute(sitk_img, sitk_mask)
    names = sorted(k for k in result.keys() if k.startswith("original_"))
    vals  = np.array([float(result.get(n, np.nan)) for n in names], dtype=float)
    return names, vals


# Load per-seed argmax segmentation from disk 
def _load_argmax(uncertainty_root, seed, sample_idx):
    """Load the hard-label warped-atlas segmentation for one seed
    Returns integer numpy array of shape (D, H, W).
    """
    path = Path(uncertainty_root) / f"seed_{seed}" / \
           f"sample_{sample_idx}_warped_atlas_labelmap_argmax.nii.gz"
    # debug: missing path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing per-seed segmentation: {path}\n"
            "  Run run_asoca_uncertainty_seeds.sh first."
        )
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)


# Load per-seed displacement from saved transform
def _load_displacement(uncertainty_root, seed, sample_idx, identity_np):
    """Load the STN grid T and return the displacement T − identity.
    The _transform.nii.gz file stores stn.get_T() saved with isVector=True.
    Loaded shape via SimpleITK: same as identity_np, i.e. (D, H, W, 3).
    """
    path = Path(uncertainty_root) / f"seed_{seed}" / \
           f"sample_{sample_idx}_transform.nii.gz"
    # Debug: missing path
    if not path.exists():
        raise FileNotFoundError(f"Missing per-seed transform: {path}")
    T_np = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # (D, H, W, 3)
    return T_np - identity_np  # displacement field, same shape


# Parallel seed worker 
# required to send work to child processes. Need function here (not inside
# the loop) at module level for pickling. 
# Pass numpy array not SimpleITK image objects, since latter is not picklable, 
# so we pass the raw numpy array and reconstruct sitk_img inside the worker
def _seed_worker(args):
    """Extract per-structure radiomics (+ optional geometry) for one seed."""
    seed, sample_idx, img_np = args

    sitk_img = sitk.GetImageFromArray(img_np)
    sitk_img.SetSpacing(spacing)

    argmax_np = _load_argmax(UNCERTAINTY_ROOT, seed, sample_idx)
    disp_np   = (
        _load_displacement(UNCERTAINTY_ROOT, seed, sample_idx, identity_grid_np)
        if USE_MULTISEED_GEOMETRY else None
    )

    rad_feats      = {}
    geo_feats      = {}
    feat_names_out = None
    warnings_out   = []

    for L, name in class_mapping.items():
        mask_np = (argmax_np == L).astype(np.uint8)
        feat_names, feat_vals = _extract_radiomics(sitk_img, mask_np)
        
        # warning if mask empty for this structure
        if feat_names is None:
            rad_feats[L] = None
            warnings_out.append(
                f"  [WARN] sample {sample_idx} seed {seed}: empty mask for '{name}'"
            )
        else:
            rad_feats[L]   = feat_vals
            feat_names_out = feat_names_out or feat_names

        # geometry features from SVD of displacement vectors within the structure mask
        if USE_MULTISEED_GEOMETRY:
            mask_bool = (argmax_np == L)
            disp_vox  = disp_np[mask_bool]
            evr = np.zeros(MAX_DEF_PC, dtype=float)
            if disp_vox.shape[0] >= 1:
                _, s, _ = np.linalg.svd(disp_vox, full_matrices=False)
                n_comp       = min(len(s), MAX_DEF_PC)
                evr[:n_comp] = s[:n_comp]
            geo_feats[L] = evr

    return seed, feat_names_out, rad_feats, geo_feats, warnings_out


# Main feature extraction loop
#
# idea is that for each subject we do:
#   FOR EACH seed:
#     FOR EACH structure:
#       extract 107 radiomics features from that seed's binary mask
#       optionally compute those 3 def pc features from that seed's disp field
#   THEN aggregate mean + variance across seeds per feature

MAX_DEF_PC        = 3
SEMANTIC_FEATURES = None 
N_RAD_FEATS       = None

subjects_rows = []

for sample_idx, batch in enumerate(tqdm(dataloader_test, desc="Extracting multi-seed features")):
    image_tensor = batch["image"].to(device)
    fname        = batch["fname"][0]
    img_type     = "Diseased" if "Diseased" in fname else "Normal"

    img_np = image_tensor[0, 0].detach().cpu().numpy()

    # Cannot safely use multiprocessing here as cuda is already active
    # run sequentially to avoid crashes due to too many interactions at once 
    results = [_seed_worker((seed, sample_idx, img_np)) for seed in SEEDS]

    # Collect per-seed outputs into position-indexed lists (index = seed order)
    seed_rad = {L: [None] * len(SEEDS) for L in class_mapping}
    seed_geo = {L: [None] * len(SEEDS) for L in class_mapping} if USE_MULTISEED_GEOMETRY else None

    for i, (seed, feat_names, rad_feats, geo_feats, warnings) in enumerate(results):
        for w in warnings:
            print(w)
        if feat_names is not None and SEMANTIC_FEATURES is None:
            SEMANTIC_FEATURES = feat_names
            N_RAD_FEATS       = len(feat_names)
        for L in class_mapping:
            seed_rad[L][i] = rad_feats[L]
            if USE_MULTISEED_GEOMETRY:
                seed_geo[L][i] = geo_feats[L]

    # Feature names must be known by now
    if SEMANTIC_FEATURES is None:
        raise RuntimeError(
            f"Could not determine PyRadiomics feature names for sample {sample_idx}: "
            "all seeds and all structures had empty masks."
        )

    # Replace None, empty masks, with NaN arrays
    for L in class_mapping:
        seed_rad[L] = [
            np.full(N_RAD_FEATS, np.nan, dtype=float) if v is None else v
            for v in seed_rad[L]
        ]

    # Aggregate across seeds - mean and variance 
    row = {}
    for L, name in class_mapping.items():
        rad_stack = np.stack(seed_rad[L], axis=0)   # (n_seeds, N_RAD_FEATS)
        mean_rad  = np.nanmean(rad_stack, axis=0)    # (N_RAD_FEATS,)
        var_rad   = np.nanvar(rad_stack, axis=0)     # (N_RAD_FEATS,)

        for i, fn in enumerate(SEMANTIC_FEATURES):
            if USE_IVW:
                # Inverse-variance weighting: divide the mean by (variance + eps).
                # features where the seeds agree has low variance, and these keep their mean
                # value; but features where seeds disagree heavily (high variance) will
                # shrunk near zero.  StandardScaler then operates on values that
                # are already ordered by reliability.
                row[f"ivw_{fn}_{name}"] = float(mean_rad[i] / (var_rad[i] + _IVW_EPS))
            else:
                row[f"mean_{fn}_{name}"] = float(mean_rad[i])
                row[f"var_{fn}_{name}"]  = float(var_rad[i])

        if USE_MULTISEED_GEOMETRY:
            geo_stack = np.stack(seed_geo[L], axis=0)  # (n_seeds, MAX_DEF_PC)
            mean_geo  = np.nanmean(geo_stack, axis=0)
            var_geo   = np.nanvar(geo_stack, axis=0)
            for pc_idx in range(MAX_DEF_PC):
                if USE_IVW:
                    row[f"ivw_def_pc{pc_idx+1}_{name}"] = float(
                        mean_geo[pc_idx] / (var_geo[pc_idx] + _IVW_EPS)
                    )
                else:
                    row[f"mean_def_pc{pc_idx+1}_{name}"] = float(mean_geo[pc_idx])
                    row[f"var_def_pc{pc_idx+1}_{name}"]  = float(var_geo[pc_idx])

    row["label"] = img_type
    subjects_rows.append(row)


# Build df_full 
import pandas as pd

df_full = pd.DataFrame(subjects_rows)

# debug: sanity check df_full 
print("\n=== df_full sanity check ===")
print(f"  Shape : {df_full.shape}")
nan_cols = df_full.isna().any()
if nan_cols.any():
    print(f"  NaN columns:\n{nan_cols[nan_cols].to_string()}")
else:
    print("  No NaN values.")
print(df_full.head(3))

# Useful to have a feature dimensionality summary
_n_structs = len(class_mapping)   # 7

# some useful prints to confirm feature counts 
print(f"\n=== Feature Dimensionality (Experiment 2, USE_IVW={USE_IVW}) ===")
print(f"  PyRadiomics features per structure : {N_RAD_FEATS}")
print(f"  Structures                         : {_n_structs}")
if USE_IVW:
    _n_rad = N_RAD_FEATS * _n_structs
    _n_geo = MAX_DEF_PC * _n_structs if USE_MULTISEED_GEOMETRY else 0
    print(f"  Radiomic IVW features              : {_n_rad}  (1 per feature×structure, mean÷var)")
    if USE_MULTISEED_GEOMETRY:
        print(f"  Geometric IVW features (≤{MAX_DEF_PC} PCs)  : {_n_geo}  (1 per PC×structure, mean÷var)")
    print(f"  Total features (excl. label)       : {_n_rad + _n_geo}")
else:
    _n_rad_mean = N_RAD_FEATS * _n_structs
    _n_rad_var  = N_RAD_FEATS * _n_structs
    _n_geo_mean = MAX_DEF_PC * _n_structs if USE_MULTISEED_GEOMETRY else 0
    _n_geo_var  = MAX_DEF_PC * _n_structs if USE_MULTISEED_GEOMETRY else 0
    print(f"  Radiomic mean features             : {_n_rad_mean}")
    print(f"  Radiomic variance features         : {_n_rad_var}")
    if USE_MULTISEED_GEOMETRY:
        print(f"  Geometric mean features (≤{MAX_DEF_PC} PCs)  : {_n_geo_mean}")
        print(f"  Geometric variance features        : {_n_geo_var}")
    _total_fixed = _n_rad_mean + _n_rad_var + _n_geo_mean + _n_geo_var
    print(f"  Total features (excl. label)       : {_total_fixed}")

y_global = np.array(df_full["label"])
N        = len(y_global)

# MLP Classification (Optuna hyperparameter optimisation)
# identical to previous Geo-Radio-Classification.py except selected_cols uses mean_* / var_* column names 

import optuna
import random
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score,
)
from torchvision.ops import MLP

seeds = [10, 101, 202]

def objective(trial):
    hidden_units = trial.suggest_int("hidden_units", 8, 512, step=8)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)
    num_layers   = trial.suggest_int("num_layers", 1, 12)
    num_epochs   = trial.suggest_int("num_epochs", 100, 400, step=25)

    # build full feature column list. 
    # USE_IVW=True then one ivw_* column per (feature, structure): mean ÷ (var + eps)
    # USE_IVW=False then two columns per (feature, structure): mean_* and var_*
    selected_cols = []
    for L, name in class_mapping.items():
        for fn in SEMANTIC_FEATURES:
            if USE_IVW:
                selected_cols.append(f"ivw_{fn}_{name}")
            else:
                selected_cols.append(f"mean_{fn}_{name}")
                selected_cols.append(f"var_{fn}_{name}")

    # option to include geometry features (Optuna searches def_pc_amt)
    if USE_MULTISEED_GEOMETRY:
        def_pc_amt = trial.suggest_int("def_pc_amt", 1, MAX_DEF_PC)
        for L, name in class_mapping.items():
            for pc_idx in range(def_pc_amt):
                if USE_IVW:
                    selected_cols.append(f"ivw_def_pc{pc_idx+1}_{name}")
                else:
                    selected_cols.append(f"mean_def_pc{pc_idx+1}_{name}")
                    selected_cols.append(f"var_def_pc{pc_idx+1}_{name}")

    if trial.number == 0:
        print(f"\n[Trial 0] selected_cols count : {len(selected_cols)}")
        print(f"[Trial 0] example cols (first 4): {selected_cols[:4]}")

    X_df = df_full[selected_cols]
    X_np = X_df.to_numpy(dtype=np.float32)

    if trial.number == 0:
        print(f"[Trial 0] X_np.shape : {X_np.shape} <- uncertainty features {'included' if USE_UNCERTAINTY_FEATURES else 'excluded'}")

    y_np      = y_global.copy()
    N_samples = len(y_np)

    X_all = torch.from_numpy(X_np).float()
    y_all = torch.from_numpy((y_np == "Diseased").astype(np.float32)).to(device)

    seed_means             = []
    per_seed_fold_metrics  = []

    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        folds = list(skf.split(np.zeros((N_samples, 1)), y_np))

        fold_accuracy_list = []
        fold_metrics_list  = []

        for train_idx, val_idx in folds:
            X_train_raw = X_np[train_idx]
            X_val_raw   = X_np[val_idx]
            y_train     = y_all[train_idx]
            y_val       = y_all[val_idx]

            # fill in missing values with estimated values 
            # for each feature, compute the mean using only the training data 
            imputer       = SimpleImputer(strategy="mean")
            X_train_imp   = imputer.fit_transform(X_train_raw)
            X_val_imp     = imputer.transform(X_val_raw)

            scaler        = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_val_scaled   = scaler.transform(X_val_imp)

            X_train = torch.from_numpy(X_train_scaled).float().to(device)
            X_val   = torch.from_numpy(X_val_scaled).float().to(device)

            # MLP
            layers    = [hidden_units] * num_layers + [1]
            model     = MLP(X_np.shape[1], layers, dropout=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            # Training loop
            for epoch in range(1, num_epochs + 1):
                model.train()
                logits = model(X_train).squeeze(1)
                loss   = criterion(logits, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val).squeeze(1)
                val_probs  = torch.sigmoid(val_logits).cpu().numpy()
                val_preds  = (val_probs >= 0.5).astype(int)

                y_true = y_val.cpu().numpy().astype(int)
                y_pred = val_preds

                acc  = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec  = recall_score(y_true, y_pred, zero_division=0)
                f1   = f1_score(y_true, y_pred, zero_division=0)
                cm   = confusion_matrix(y_true, y_pred)
                tn, fp, fn_count, tp = cm.ravel()
                sensitivity = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
                specificity = tn / (tn + fp)       if (tn + fp) > 0       else 0.0
                # AUC is added. It uses raw probability scores, so reflects the
                # model's confidence in its predictions, not just the binary outcome
                # try/except guards against the rare fold where only one class appears
                try:
                    auc = roc_auc_score(y_true, val_probs)
                except ValueError:
                    auc = 0.5

            fold_accuracy_list.append(acc)
            fold_metrics_list.append({
                "accuracy":         acc,
                "precision":        prec,
                "recall":           rec,
                "f1":               f1,
                "sensitivity":      sensitivity,
                "specificity":      specificity,
                "auc":              auc,
                "confusion_matrix": cm,
            })

        seed_mean_acc = np.mean(fold_accuracy_list)
        seed_means.append(seed_mean_acc)
        per_seed_fold_metrics.append(fold_metrics_list)

    trial.set_user_attr("seed_means",             seed_means)
    trial.set_user_attr("per_seed_fold_metrics",  per_seed_fold_metrics)
    trial.set_user_attr("hyperparams", {
        "hidden_units": hidden_units,
        "lr":            lr,
        "dropout":       dropout,
        "num_layers":    num_layers,
        "num_epochs":    num_epochs,
        **({"def_pc_amt": trial.params.get("def_pc_amt")} if USE_MULTISEED_GEOMETRY else {}),
    })

    return np.mean(seed_means)


# Run Optuna

study = optuna.create_study(study_name=OPTUNA_STUDY_NAME, direction="maximize")
study.optimize(objective, n_trials=500, show_progress_bar=True)

# Tie-break on stddev
trials     = study.trials
best_trial = max(trials, key=lambda t: (t.value, -np.std(t.user_attrs["seed_means"])))

best_score       = best_trial.value
best_std_seed    = np.std(best_trial.user_attrs["seed_means"])
best_hyperparams = best_trial.user_attrs["hyperparams"]
best_fold_info   = {
    "seeds":                  seeds,
    "seed_means":             best_trial.user_attrs["seed_means"],
    "per_seed_fold_metrics":  best_trial.user_attrs["per_seed_fold_metrics"],
}

# Aggregate all metrics over seeds & folds
all_accuracies    = []
all_precisions    = []
all_recalls       = []
all_f1s           = []
all_sensitivities = []
all_specificities = []
all_aucs          = []

for seed_metrics in best_fold_info["per_seed_fold_metrics"]:
    for m in seed_metrics:
        all_accuracies.append(m["accuracy"])
        all_precisions.append(m["precision"])
        all_recalls.append(m["recall"])
        all_f1s.append(m["f1"])
        all_sensitivities.append(m["sensitivity"])
        all_specificities.append(m["specificity"])
        all_aucs.append(m["auc"])

acc_mean,  acc_std  = np.mean(all_accuracies),    np.std(all_accuracies)
prec_mean, prec_std = np.mean(all_precisions),    np.std(all_precisions)
rec_mean,  rec_std  = np.mean(all_recalls),       np.std(all_recalls)
f1_mean,   f1_std   = np.mean(all_f1s),           np.std(all_f1s)
sens_mean, sens_std = np.mean(all_sensitivities),  np.std(all_sensitivities)
spec_mean, spec_std = np.mean(all_specificities),  np.std(all_specificities)
auc_mean,  auc_std  = np.mean(all_aucs),           np.std(all_aucs)

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

for seed_idx, s in enumerate(best_fold_info["seeds"]):
    print(f"--- Seed {s} (mean CV acc = {best_fold_info['seed_means'][seed_idx]:.4f}) ---")
    for fold_idx, m in enumerate(best_fold_info["per_seed_fold_metrics"][seed_idx], start=1):
        print(f"Fold {fold_idx}:")
        print(f"  Accuracy    = {m['accuracy']:.4f}")
        print(f"  Precision   = {m['precision']:.4f}")
        print(f"  Recall      = {m['recall']:.4f}")
        print(f"  F1 Score    = {m['f1']:.4f}")
        print(f"  Sensitivity = {m['sensitivity']:.4f}")
        print(f"  Specificity = {m['specificity']:.4f}")
        print(f"  AUC-ROC     = {m['auc']:.4f}")
        print(f"  Confusion Matrix:\n{m['confusion_matrix']}\n")

# Save results to a labelled CSV - useful for later analysis 
# does not overwrite previous experiments one
_result_rows = []
for _si, _s in enumerate(best_fold_info["seeds"]):
    for _fi, _m in enumerate(best_fold_info["per_seed_fold_metrics"][_si], start=1):
        _result_rows.append({
            "experiment":  OPTUNA_STUDY_NAME,
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
pd.DataFrame(_result_rows).to_csv(RESULTS_CSV, index=False)
print(f"\nResults saved -> {RESULTS_CSV}")


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