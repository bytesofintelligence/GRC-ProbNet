#!/usr/bin/env python
# coding: utf-8

# # Geo-Radio Classification — Prediction-Space Ensemble (Experiment 3)
#
# A fork of Geo-Radio-Classification.py
#
# Instead of evaluating on a single consensus segmentation at inference, this script
# runs the trained MLP independently on features extracted from each of the 5
# per-seed warped-atlas segmentations, then averages the resulting probs before thresholding.
# Training always uses consensus features (df_full) and only the validation forward pass
# is replaced with the 5-seed ensemble.
#
# Per-subject ensemble probabilities and uncertainty decomposition are saved
# to a separate CSV after the best trial is identified.

# # Imports and Global Config

import sys
import os
import json
import argparse
sys.path.insert(0, "/vol/biomedic2/bglocker_studproj/yrs23/grc-net/anatomix")

import torch
from monai.data import ThreadDataLoader, CacheDataset
from monai.transforms import Lambdad
from nets.stn import FullSTN3D
from img.datasets import ImageSegmentationOneHotDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

_parser = argparse.ArgumentParser(description="Geo-Radio Classification — Prediction Ensemble (Experiment 3)")
_parser.add_argument(
    "--use-prediction-ensemble", action="store_true", default=True,
    help="Enable 5-seed prediction ensemble at inference (default: True — always on for this script)."
)
_parser.add_argument(
    "--num-seeds", type=int, default=5,
    help="Number of segmentation seeds to use at inference (default: 5)."
)
_parser.add_argument(
    "--seg-seeds", nargs="+", type=int,
    default=[42, 123, 456, 789, 999],
    help="Segmentation seeds to aggregate over at inference."
)
_parser.add_argument(
    "--uncertainty-root",
    default="output/asoca/uncertainty",
    help="Root dir containing seed_<S>/ subdirs with per-seed warped-atlas outputs."
)
_parser.add_argument(
    "--subjects-csv",
    default="results_prediction_ensemble_subjects.csv",
    help="Per-subject probability output CSV (Experiment 3 only)."
)
_parser.add_argument(
    "--run-resnet", action="store_true", default=False,
    help="Run the ResNet-50 3D image-only baseline after the MLP experiment. "
         "Disabled by default"
)
_args = _parser.parse_args()

SEG_SEEDS        = _args.seg_seeds[:_args.num_seeds]   # segmentation seeds for inference
UNCERTAINTY_ROOT = _args.uncertainty_root
SUBJECTS_CSV     = _args.subjects_csv
RUN_RESNET       = _args.run_resnet

# debug: print
print(f"=== Experiment 3 — Prediction-Space Ensemble ===")
print(f"  Segmentation seeds : {SEG_SEEDS}")
print(f"  Uncertainty root   : {UNCERTAINTY_ROOT}")
print(f"  Subjects CSV       : {SUBJECTS_CSV}")

spacing = (2.0, 2.0, 2.0)
anatomix_roi_size = (96, 96, 96)
crop_size = (96, 96, 96)
num_classes = 8

# Class Mapping

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
# Ensures preprocessing and model dimensions are consistent.
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

# Load dataset

dataset_test_base = ImageSegmentationOneHotDataset("data/config/inference.csv",
                                            num_classes, anatomix_roi_size, spacing,
                                            normalizer=Lambdad(keys=["image"], func=lambda x: x),
                                            binarize=0, augmentation=False,
                                            fixed_crop_size=crop_size)
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
from pathlib import Path
from radiomics import featureextractor
from tqdm import tqdm


radiomics_settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkLinear,
    'verbose': False
}
extractor = featureextractor.RadiomicsFeatureExtractor(**radiomics_settings)

atlas_label_itk = sitk.ReadImage("output/mm-whs/full-stn/train/model/atlas_labelmap_final.nii.gz")

arr_lab = sitk.GetArrayFromImage(atlas_label_itk)
if arr_lab.ndim == 4: # already one‐hot in last dim
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

example_batch = next(iter(dataloader_test))
batch_size = example_batch["image"].size(0)
atlas_label = atlas_label.repeat(batch_size, 1, 1, 1, 1)

# Precompute identity grid once (for displacement = T – grid)
identity_grid = stn.grid.unsqueeze(0)
identity_grid = stn.move_grid_dims(identity_grid)
identity_grid = identity_grid.repeat(batch_size, 1, 1, 1, 1).to(device)

# numpy version of identity grid, used to compute displacements from saved transforms
identity_grid_np = (
    stn.move_grid_dims(stn.grid.unsqueeze(0))[0]
    .detach().cpu().numpy()
)  # (D, H, W, 3)

subjects = []

# changed to enumerate to get sample_idx for matching with uncertainty CSV
for sample_idx, batch in enumerate(tqdm(dataloader_test, desc="extracting consensus features")):
    image_tensor = batch["image"].to(device)
    label_onehot  = batch["labelmap"].to(device)
    fname         = batch["fname"][0]

    img_type = "Diseased" if "Diseased" in fname else "Normal"

    # Run STN to get full warp grid T
    src = label_onehot[:, 1:, ...]
    tgt = atlas_label[:, 1:, ...]
    _   = stn(torch.cat((src, tgt), dim=1))

    T = stn.get_T()
    full_disp = T - identity_grid
    disp_np = full_disp[0].detach().cpu().numpy()

    # per‐structure displacement
    struct_disp = {}
    for L in class_mapping.keys():
        maskL = label_onehot[0, L].bool().cpu().numpy()
        disp_vox = disp_np[maskL]
        struct_disp[L] = disp_vox

    img_np = image_tensor[0, 0].detach().cpu().numpy()
    sitk_img = sitk.GetImageFromArray(img_np)
    sitk_img.SetSpacing(spacing)

    # per‐structure radiomics
    radiomics = {}
    for L, name in class_mapping.items():
        mask_np = label_onehot[0, L].cpu().numpy().astype(np.uint8)
        if mask_np.sum() == 0:
            # No voxels, so store an array of nans with length = len(SEMANTIC_FEATURES)
            radiomics[L] = np.full((len(SEMANTIC_FEATURES),), np.nan, dtype=float)
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
        "full_disp": disp_np,       # [D,H,W,3]
        "struct_disp": struct_disp, # dict L->(n_vox_L,3)
        "radiomics":   radiomics    # dict L->(len(SEMANTIC_FEATURES),)
    }
    subjects.append(subject_data)

# After this loop, `subjects` is a list of length N (test cases),
# and each `subjects[i]` contains all the deformation + radiomics for that case.

# # MLP classification
# Here, we perform hyperparameter optimisation using `optuna` to find the best classification model for the diseased data.
# We perform 5-fold stratified cross-validation over 3 seeds to achieve confidence in the low volume of data.

import pandas as pd
import optuna
import random
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from torchvision.ops import MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

MAX_DEF_PC = 3

# removed uncertainty calculation here

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

    # Radiomics (semantic features) per structure
    for L, name in class_mapping.items():
        rad_vec = subj["radiomics"][L]  # shape = (len(SEMANTIC_FEATURES),) or all-nan
        for idx, feat_name in enumerate(SEMANTIC_FEATURES):
            col = f"{feat_name}_{name}"
            row[col] = float(rad_vec[idx])

    # The unc_* columns are removed, not used in Experiment 3

    row["label"] = subj["label"]  # "Normal" or "Diseased"
    rows.append(row)

df_full = pd.DataFrame(rows)
print(df_full.isna().any()[lambda x: x])
print(df_full.head(4))
print("Shape of df_full:", df_full.shape)

# Per-seed feature extraction
#
# For each segmentation seed and each subject, load the pre-saved argmax labelmap
# and displacement transform, then extract the same radiomics and geometric features
# as the consensus loop above.
#
# These per-seed features are used only at inference time (validation fold).
# The classifier is always trained on consensus features (df_full)
#
# Result: per_seed_dfs - list of DataFrames, one per seed, with identical columns to df_full.

def _load_argmax(uncertainty_root, seed, sample_idx):
    """Load integer argmax labelmap (D, H, W) for one seed/subject."""
    path = Path(uncertainty_root) / f"seed_{seed}" / \
           f"sample_{sample_idx}_warped_atlas_labelmap_argmax.nii.gz"
    # debug 
    if not path.exists():
        raise FileNotFoundError(
            f"Missing per-seed segmentation: {path}\n"
            "  Run run_asoca_uncertainty_seeds.sh first."
        )
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)


def _load_displacement(uncertainty_root, seed, sample_idx):
    """Load STN grid T from disk and return displacement T − identity."""
    path = Path(uncertainty_root) / f"seed_{seed}" / \
           f"sample_{sample_idx}_transform.nii.gz"
    # debug
    if not path.exists():
        raise FileNotFoundError(f"Missing per-seed transform: {path}")
    T_np = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # (D, H, W, 3)
    return T_np - identity_grid_np


per_seed_dfs = []  # one DataFrame per seed, columns identical to df_full (excluding label)

for seed in tqdm(SEG_SEEDS, desc="Per-seed feature extraction (seeds)"):
    seed_rows = []

    for sample_idx, batch in enumerate(tqdm(dataloader_test, desc=f"  seed_{seed}", leave=False)):
        image_tensor = batch["image"].to(device)
        fname        = batch["fname"][0]
        img_type     = "Diseased" if "Diseased" in fname else "Normal"

        img_np   = image_tensor[0, 0].detach().cpu().numpy()
        sitk_img = sitk.GetImageFromArray(img_np)
        sitk_img.SetSpacing(spacing)

        argmax_np = _load_argmax(UNCERTAINTY_ROOT, seed, sample_idx)
        disp_np   = _load_displacement(UNCERTAINTY_ROOT, seed, sample_idx)

        row = {}

        # Geometric features, just mask from argmax instead of one-hot
        for L, name in class_mapping.items():
            maskL    = (argmax_np == L)
            disp_vox = disp_np[maskL]
            evr      = np.zeros(MAX_DEF_PC, dtype=float)
            if disp_vox.shape[0] >= 1:
                _, sv, _ = np.linalg.svd(disp_vox, full_matrices=False)
                n_comp = min(len(sv), MAX_DEF_PC)
                evr[:n_comp] = sv[:n_comp]
            for pc_idx in range(MAX_DEF_PC):
                row[f"def_pc{pc_idx+1}_{name}"] = float(evr[pc_idx])

        # Radiomic features, mask from argmax
        for L, name in class_mapping.items():
            mask_np = (argmax_np == L).astype(np.uint8)
            if mask_np.sum() == 0:
                rad_vec = np.full(len(SEMANTIC_FEATURES), np.nan, dtype=float)
            else:
                sitk_mask = sitk.GetImageFromArray(mask_np)
                sitk_mask.CopyInformation(sitk_img)
                result  = extractor.execute(sitk_img, sitk_mask)
                rad_vec = np.array(
                    [float(result.get(fn, np.nan)) for fn in SEMANTIC_FEATURES],
                    dtype=float,
                )
            for idx, feat_name in enumerate(SEMANTIC_FEATURES):
                row[f"{feat_name}_{name}"] = float(rad_vec[idx])

        row["label"] = img_type
        seed_rows.append(row)

    per_seed_dfs.append(pd.DataFrame(seed_rows))
    print(f"  seed {seed}: {per_seed_dfs[-1].shape}  "
          f"NaN cols: {per_seed_dfs[-1].isna().any().sum()}")

print(f"\nPer-seed extraction done. {len(per_seed_dfs)} seeds × {len(per_seed_dfs[0])} subjects.")

# # Entropy aggregation helpers

_EPS = 1e-12

def _binary_entropy(p):
    """Numerically safe binary entropy (just added epsilon): H(p) = -p log p - (1-p) log(1-p)."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def _aggregate_probs(probs):
    """Aggregate an array of K probabilities into uncertainty statistics.

    predictive_entropy = H(mean_p)            — total uncertainty
    expected_entropy   = mean(H(p_i))         — aleatoric uncertainty
    mutual_info        = H(mean_p)-mean(H(p_i)) — epistemic (seed disagreement)
    """
    mean_p = float(np.mean(probs))
    return {
        "mean_p":             mean_p,
        "var_p":              float(np.var(probs)),
        "std_p":              float(np.std(probs)),
        "predictive_entropy": float(_binary_entropy(mean_p)),
        "expected_entropy":   float(np.mean([_binary_entropy(p) for p in probs])),
        "mutual_info":        float(_binary_entropy(mean_p) -
                                    np.mean([_binary_entropy(p) for p in probs])),
    }

y_global = np.array(df_full["label"])
N = len(y_global)

seeds = [10, 101, 202]

def get_folds(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((N, 1)), y_global))

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

    # just removed unc_* block

    X_df = df_full[selected_cols]
    X_np = X_df.to_numpy(dtype=np.float32)

    y_np = y_global.copy()
    N_samples = len(y_np)

    X_all = torch.from_numpy(X_np).float()
    y_all = torch.from_numpy((y_np == "Diseased").astype(np.float32)).to(device)

    seed_means = []
    per_seed_fold_metrics = []

    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        # pass a dummy X of shape (N_samples, 1) because StratifiedKFold only uses y
        folds = list(skf.split(np.zeros((N_samples, 1)), y_np))

        fold_accuracy_list = []
        fold_metrics_list = []

        for (train_idx, val_idx) in folds:
            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_val   = X_all[val_idx]
            y_val   = y_all[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled   = scaler.transform(X_val)  # kept for shape/reference

            X_train = torch.from_numpy(X_train_scaled).float().to(device)

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

            # Validation is now a 5-seed ensemble instead of a single forward pass
            #
            # so, for each validation subject:
            #   for each segmentation seed, extract features -> same model -> probability
            # Aggregate probabilities; final prediction = mean_p > 0.5
            model.eval()
            with torch.no_grad():
                val_seed_X = np.stack([
                    per_seed_dfs[s_i][selected_cols].to_numpy(dtype=np.float32)[val_idx]
                    for s_i in range(len(SEG_SEEDS))
                ], axis=0)  # (n_seeds, n_val, D)

                all_seed_probs = np.zeros((len(SEG_SEEDS), len(val_idx)))
                for s_i in range(len(SEG_SEEDS)):
                    x_scaled = scaler.transform(val_seed_X[s_i])  # same scaler as training
                    x_t      = torch.from_numpy(x_scaled).float().to(device)
                    all_seed_probs[s_i] = torch.sigmoid(
                        model(x_t).squeeze(1)
                    ).cpu().numpy()

                # aggregate across seeds
                val_probs = all_seed_probs.mean(axis=0)   # (n_val,)
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
            # AUC is added. It uses raw probability scores, so reflects the
            # model's confidence in its predictions, not just the binary outcome
            # try/except guards against the rare fold where only one class appears
            try:
                auc = float(roc_auc_score(y_true, val_probs))
            except ValueError:
                auc = float("nan")

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

# Run Optuna — separate study name from Experiments 1 & 2

study = optuna.create_study(
    study_name="prediction_ensemble",   
    direction="maximize"
)
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
        if not np.isnan(m['auc']):
            all_aucs.append(m['auc'])

acc_mean,  acc_std  = np.mean(all_accuracies),    np.std(all_accuracies)
prec_mean, prec_std = np.mean(all_precisions),   np.std(all_precisions)
rec_mean,  rec_std  = np.mean(all_recalls),      np.std(all_recalls)
f1_mean,   f1_std   = np.mean(all_f1s),          np.std(all_f1s)
sens_mean, sens_std = np.mean(all_sensitivities), np.std(all_sensitivities)
spec_mean, spec_std = np.mean(all_specificities), np.std(all_specificities)

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
if all_aucs:
    print(f"AUC         : {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")

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
        print(f"  AUC         = {m['auc']:.4f}")
        print(f"  Confusion Matrix:\n{m['confusion_matrix']}\n")

# Results CSV — separate file from Experiments 1 & 2
# Save per-fold results to a labelled CSV - useful for later analysis
_result_rows = []
for _si, _s in enumerate(best_fold_info['seeds']):
    for _fi, _m in enumerate(best_fold_info['per_seed_fold_metrics'][_si], start=1):
        _result_rows.append({
            "experiment":  "prediction_ensemble",
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
pd.DataFrame(_result_rows).to_csv("results_prediction_ensemble.csv", index=False)
print(f"\nFold-level results saved -> results_prediction_ensemble.csv")

# Also we save per-subject probability CSV in Experiment 3
#
# Re-run the best model once per (cv_seed, fold) to collect:
#   p_seed_<S>   — probability from each individual segmentation seed
#   mean_p       — ensemble mean probability (= final decision boundary)
#   var_p        — variance across seeds
#   std_p        — std across seeds
#   predictive_entropy  = H(mean_p)          — total uncertainty
#   expected_entropy    = mean(H(p_i))       — aleatoric uncertainty
#   mutual_info         = H(mean_p)-mean(H) — epistemic (from seed disagreement)
#   final_prediction    — mean_p > 0.5

print("\n--- Re-evaluating best model to save per-subject outputs ---")

hp          = best_hyperparams
sel_best    = []
for L, name in class_mapping.items():
    for i in range(hp["def_pc_amt"]):
        sel_best.append(f"def_pc{i+1}_{name}")
    for feat_name in SEMANTIC_FEATURES:
        sel_best.append(f"{feat_name}_{name}")

X_best = df_full[sel_best].to_numpy(dtype=np.float32)
D_best = X_best.shape[1]

subject_rows_out = []

for cv_seed in seeds:
    random.seed(cv_seed)
    np.random.seed(cv_seed)
    torch.manual_seed(cv_seed)

    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    folds = list(skf.split(np.zeros((N, 1)), y_global))

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train_raw = X_best[train_idx]
        y_train_bin = (y_global[train_idx] == "Diseased").astype(np.float32)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_train_t      = torch.from_numpy(X_train_scaled).float().to(device)
        y_train_t      = torch.from_numpy(y_train_bin).to(device)

        layers    = [hp["hidden_units"]] * hp["num_layers"] + [1]
        model     = MLP(D_best, layers, dropout=hp["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(1, hp["num_epochs"] + 1):
            model.train()
            logits = model(X_train_t).squeeze(1)
            loss   = criterion(logits, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for val_abs_idx in val_idx:
                seed_probs = []
                for s_i in range(len(SEG_SEEDS)):
                    x_seed  = per_seed_dfs[s_i][sel_best].to_numpy(dtype=np.float32)[[val_abs_idx]]
                    x_sc    = scaler.transform(x_seed)
                    x_t     = torch.from_numpy(x_sc).float().to(device)
                    p       = float(torch.sigmoid(model(x_t).squeeze(1)).cpu().item())
                    seed_probs.append(p)

                stats = _aggregate_probs(np.array(seed_probs))

                row_out = {
                    "subject_id":          val_abs_idx,
                    "fname":               subjects[val_abs_idx]["fname"],
                    "cv_seed":             cv_seed,
                    "fold":                fold_idx,
                    "ground_truth":        y_global[val_abs_idx],
                    "ground_truth_binary": int(y_global[val_abs_idx] == "Diseased"),
                }
                for s_i, seg_seed in enumerate(SEG_SEEDS):
                    row_out[f"p_seed_{seg_seed}"] = seed_probs[s_i]
                row_out.update(stats)
                row_out["final_prediction"] = int(stats["mean_p"] >= 0.5)
                subject_rows_out.append(row_out)

df_subjects = pd.DataFrame(subject_rows_out)
df_subjects.to_csv(SUBJECTS_CSV, index=False)
print(f"Per-subject outputs saved -> {SUBJECTS_CSV}")

# Summary of prediction uncertainty
print("\n=== Prediction Uncertainty Summary ===")
print(f"  mean_p              : {df_subjects['mean_p'].mean():.4f} ± {df_subjects['mean_p'].std():.4f}")
print(f"  var_p               : {df_subjects['var_p'].mean():.6f} ± {df_subjects['var_p'].std():.6f}")
print(f"  predictive_entropy  : {df_subjects['predictive_entropy'].mean():.4f} ± {df_subjects['predictive_entropy'].std():.4f}")
print(f"  expected_entropy    : {df_subjects['expected_entropy'].mean():.4f} ± {df_subjects['expected_entropy'].std():.4f}")
print(f"  mutual_info (epist) : {df_subjects['mutual_info'].mean():.4f} ± {df_subjects['mutual_info'].std():.4f}")
for cls in ["Normal", "Diseased"]:
    sub = df_subjects[df_subjects["ground_truth"] == cls]
    print(f"\n  {cls} subjects:")
    print(f"    mean_p             = {sub['mean_p'].mean():.4f}")
    print(f"    predictive_entropy = {sub['predictive_entropy'].mean():.4f}")
    print(f"    mutual_info        = {sub['mutual_info'].mean():.4f}")

# # ResNet Baseline
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
