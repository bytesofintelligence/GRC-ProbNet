#!/usr/bin/env python
# coding: utf-8

# Geo-Radio Classification (original + Experiment 1)
#
# This notebook is used to train the classification model on the ASOCA Dataset.
# The notebook assumes the previous scripts have ran, which have saved the following models:
# - a segmentation model trained using anatomix, currently pointing to `saved_models/segmentation/anatomix_trained_MM-WHS.pth`
# - a registration model trained using Atlas-ISTN, currently pointing to `output/mm-whs/full-stn/train/model/stn.pt`
# - an atlas labelmap created using Atlas-ISTN, currently pointing to `output/mm-whs/full-stn/train/model/atlas_labelmap_final.nii.gz`
#
# If you have run the `anatomix-fine-tuning.py` and the `atlas-istn-anatomix.py` files, these will automatically be generated for you.
#
# This model will segment the ASOCA images as directed by the config CSV file `data/config/inference.csv`.
# Core idea is: CT scan -> anatomix segmentation -> Atlas-ISTN registration ->
#     extract features (geometric, radiomics, uncertainty optional) -> MLP classifier

# Entire uncertainty experiment is controlled by --use-uncertainty flag (only the feature vector changes)

# Imports and Global Config

import sys
import os
import json
import argparse
sys.path.insert(
    0, "/vol/biomedic2/bglocker_studproj/<INSERT WHERE ANATOMIX IS FOR YOU>/anatomix/"
)
sys.path.insert(0, "/vol/biomedic2/bglocker_studproj/<USERNAME>/grc-net")

import torch
from monai.data import ThreadDataLoader, CacheDataset
from monai.transforms import Lambdad
from nets.stn import FullSTN3D
from img.datasets import ImageSegmentationOneHotDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

_parser = argparse.ArgumentParser(description="Geo-Radio Classification")
_parser.add_argument(
    "--use-uncertainty", action="store_true", default=False,
    help="Append 1 global entropy uncertainty feature per image to the classifier input. "
         "Not necessary for the baseline (radiomic + geometric features only)."
)
_parser.add_argument(
    "--uncertainty-csv",
    default="output/asoca/uncertainty_analysis_entropy_only_one/metrics/per_case_uncertainty.csv",
    help="Path to per_case_uncertainty.csv produced by compute_uncertainty-entropy_only_one.py."
)
_parser.add_argument(
    "--run-resnet", action="store_true", default=False,
    help="Run the ResNet 3D image-only baseline after the MLP experiment. "
         "Disabled by default."
)
_parser.add_argument(
    "--anatomix-model", default=None,
    help="If specified, use this Anatomix checkpoint for ASOCA segmentation inference "
         "(if so, then overrides automatic best-checkpoint selection)."
)
_args = _parser.parse_args()

spacing = (2.0, 2.0, 2.0)
anatomix_roi_size = (96, 96, 96)
crop_size = (96, 96, 96)
num_classes = 8

# The --use-uncertainty CLI flag can control this switch to experiment 1
USE_UNCERTAINTY_FEATURES = _args.use_uncertainty
UNCERTAINTY_CSV = _args.uncertainty_csv
RUN_RESNET = _args.run_resnet

# Class Mapping
# *(should match "class_mapping" in `data/config/config.json`)*

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

stn_path = "output/mm-whs/full-stn/train/model/stn.pt"
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

stn = FullSTN3D(input_size=crop_size, input_channels=2*(num_classes-1), device=device).to(device)
stn.load_state_dict(torch.load(stn_path))
stn.eval()

# Load dataset

dataset_test_base = ImageSegmentationOneHotDataset("data/config/inference.csv",
                                            num_classes, anatomix_roi_size, spacing,
                                            normalizer=Lambdad(keys=["image"], func=lambda x: x),
                                            binarize=0, augmentation=False,
                                            fixed_crop_size=crop_size,
                                            model_checkpoint=_args.anatomix_model)
dataset_test = CacheDataset(data=dataset_test_base, transform=None, cache_rate=1.0, num_workers=4)
dataset_test.get_sample = dataset_test_base.get_sample
dataloader_test = ThreadDataLoader(dataset_test, batch_size=1, shuffle=False)

# Extract radiomic and deformation data
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

atlas_label_itk = sitk.ReadImage("output/mm-whs/full-stn/train/model/atlas_labelmap_final.nii.gz")

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

example_batch = next(iter(dataloader_test))
batch_size = example_batch["image"].size(0)
atlas_label = atlas_label.repeat(batch_size, 1, 1, 1, 1)

# Precompute identity grid once (for displacement = T – grid)
identity_grid = stn.grid.unsqueeze(0)
identity_grid = stn.move_grid_dims(identity_grid)
identity_grid = identity_grid.repeat(batch_size, 1, 1, 1, 1).to(device)

subjects = []

# changed to enumerate to get sample_idx for matching with uncertainty CSV
for sample_idx, batch in enumerate(tqdm(dataloader_test, desc="extracting features")):
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
        disp_vox = disp_np[maskL] # all displacement vectors inside that structure
        struct_disp[L] = disp_vox

    # per‐structure radiomics
    img_np = image_tensor[0, 0].detach().cpu().numpy()
    sitk_img = sitk.GetImageFromArray(img_np)
    sitk_img.SetSpacing(spacing)

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

# MLP classification
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

# Load global-entropy CSV (one row per subject: sample_id, mean_entropy, voxel_count)
# Always loaded so df_full is consistent; USE_UNCERTAINTY_FEATURES controls
# whether "unc_entropy" enters selected_cols in objective().
_unc_df = pd.read_csv(UNCERTAINTY_CSV)

print("Uncertainty CSV verification")
print(f"  File     : {UNCERTAINTY_CSV}")
print(f"  Columns  : {list(_unc_df.columns)}")
print(f"  Shape    : {_unc_df.shape}   (will be {len(subjects)} rows once ASOCA uncertainty is complete)")
print(_unc_df.head(5).to_string(index=False))

# Simple Series: sample_id → mean_entropy (no pivot needed — already one row per subject)
_unc_series = _unc_df.set_index("sample_id")["mean_entropy"]

missing_ids = [subj["sample_idx"] for subj in subjects if subj["sample_idx"] not in _unc_series.index]
if missing_ids:
    print(f"\n  [WARN] {len(missing_ids)} subjects have no uncertainty entry yet "
          f"(IDs {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}). "
          f"Their unc_entropy value will be NaN until the ASOCA uncertainty run is complete.")
else:
    print(f"\n  All {len(subjects)} subjects have uncertainty entries. ✓")

rows = []
for subj in subjects:
    row = {}
    # Constructing uncertainty features 
    # for each subject, add a row for unc_{name} (one per structure) to the dataframe
    # these are mean binary entropy inside each structure ... computed across multi-seed 
    # Aantomix registrations  
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

    # Single global entropy feature — one scalar per image, no per-structure breakdown
    sid = subj["sample_idx"]
    row["unc_entropy"] = (
        float(_unc_series.loc[sid])
        if sid in _unc_series.index
        else float("nan")
    )

    row["label"] = subj["label"]  # "Normal" or "Diseased"
    rows.append(row)

df = pd.DataFrame(rows)

df_full = pd.DataFrame.from_records(rows)
print(df_full.isna().any()[lambda x: x])
print(df_full.head(4))
print("Shape of df_full:", df_full.shape)

# debug: confirm uncertainty column is present
print(f"\n=== Uncertainty column in df_full")
print(f"  ['unc_entropy']")
print(f"  Example subject 0: unc_entropy = {df_full['unc_entropy'].iloc[0]:.6f}")

# Useful to have a feature dimensionality summary
_n_structs   = len(class_mapping)                          # 7
_n_rad_feats = len(SEMANTIC_FEATURES) * _n_structs         # e.g. 107 × 7 = 749
_n_geo_feats = MAX_DEF_PC * _n_structs                     # 3 × 7 = 21
_n_unc_feats = 1                                           # 1 global entropy scalar

# some useful prints to confirm feature counts
print(f"Feature dimensionality")
print(f"  Radiomic features : {len(SEMANTIC_FEATURES)} per structure × {_n_structs} = {_n_rad_feats}")
print(f"  Geometric features: {MAX_DEF_PC} PCs × {_n_structs} = {_n_geo_feats}  "
      f"(Optuna searches def_pc_amt in 1–{MAX_DEF_PC})")
print(f"  Uncertainty feats : {_n_unc_feats}  (one global foreground entropy per image)")
print(f"  ── df_full total cols (excl. label): {df_full.shape[1] - 1}")
print(f"  ── Baseline  MLP input (def_pc_amt=MAX_DEF_PC): "
      f"{_n_geo_feats} + {_n_rad_feats} = {_n_geo_feats + _n_rad_feats}")
print(f"  ── Experiment MLP input (def_pc_amt=MAX_DEF_PC): "
      f"{_n_geo_feats} + {_n_rad_feats} + {_n_unc_feats} = {_n_geo_feats + _n_rad_feats + _n_unc_feats}")
print(f"  ── USE_UNCERTAINTY_FEATURES = {USE_UNCERTAINTY_FEATURES}")

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

    # Append the single global entropy feature if the flag is on
    if USE_UNCERTAINTY_FEATURES:
        selected_cols.append("unc_entropy")

    # If we're on trial 0, just confirm uncertainty columns are in X_np
    if trial.number == 0:
        unc_in_sel = [c for c in selected_cols if c.startswith("unc_")]
        print(f"\n[Trial 0] USE_UNCERTAINTY_FEATURES={USE_UNCERTAINTY_FEATURES}")
        print(f"[Trial 0] selected_cols count : {len(selected_cols)}")
        print(f"[Trial 0] unc_* cols present  : {unc_in_sel}")

    X_df = df_full[selected_cols]                   # shape (N_samples, def_pc_amt*7 + F_sem*7 [+ 7])
    X_np = X_df.to_numpy(dtype=np.float32)           # numpy array (N_samples, D_trial)

    if trial.number == 0:
        print(f"[Trial 0] X_np.shape          : {X_np.shape} <- uncertainty features {'included' if USE_UNCERTAINTY_FEATURES else 'excluded'}")
    y_np = y_global.copy()                           # numpy array (N_samples,), strings "Normal"/"Diseased"
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
                # AUC is added. It uses raw probability scores, so reflects the
                # model's confidence in its predictions, not just the binary outcome
                # try/except guards against the rare fold where only one class appears
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
_exp_tag = "uncertainty_entropy_one" if USE_UNCERTAINTY_FEATURES else "baseline"
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
_results_csv = f"results_{_exp_tag}.csv"
pd.DataFrame(_result_rows).to_csv(_results_csv, index=False)
print(f"\nResults saved -> {_results_csv}")


# For the uncertainty exp, feature importance analysis is used to understand
# whether model is actually relying on the uncertainty features, and which ones
#  Retrains with best hyperparameters on one seed (5 folds) to measure
#   (a) First-layer weight magnitudes, which is a proxy for how much each input feature is used.
#   ((b) Permutation importance for unc_* features, this is an accuracy-drop measure) - not used in report.
if USE_UNCERTAINTY_FEATURES:
    def analyse_uncertainty_importance():
        hp          = best_trial.user_attrs["hyperparams"]
        unc_cols = ["unc_entropy"]
        # reconstruct selected_cols exactly as the best trial used them
        sel = []
        for L, name in class_mapping.items():
            for i in range(hp["def_pc_amt"]):
                sel.append(f"def_pc{i+1}_{name}")
            for feat_name in SEMANTIC_FEATURES:
                sel.append(f"{feat_name}_{name}")
        sel.append("unc_entropy")

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

            # (a) first-layer weight magnitude per input feature
            # for each uncertainty feature, compute the mean absolute weight of 
            # the first layer's weights corresponding to that feature
            # larger magnitude means greater influence 
            first_lin = next(m for m in mdl.modules() if isinstance(m, torch.nn.Linear))
            w_imp += first_lin.weight.detach().abs().mean(dim=0).cpu().numpy()

            # Baseline accuracy on the val fold
            with torch.no_grad():
                Xvl_t  = torch.from_numpy(Xvl_s).float().to(device)
                b_pred = (torch.sigmoid(mdl(Xvl_t).squeeze(1)).cpu().numpy() >= 0.5).astype(int)
            base_acc = accuracy_score(yvl.astype(int), b_pred)

            # (b) Permutation importance for the 7 uncertainty features
            # e.g. unc_aorta gets shuffled, by rng.permutation. If the accuacy drops, 
            # then that unc_aorta feature is informative. If accuracy stays the same, 
            # then the model doesn't rely on it much 
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

        # Useful reporting of the results for analysis: 
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

# Overall, 7 new additional entropy features appended to baseline feature vector
# in the uncertainty experiments. This file consumes the "per_structure_uncertainty" csv

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
