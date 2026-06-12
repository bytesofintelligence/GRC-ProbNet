#!/usr/bin/env python
"""
This handles the coronary artery segmentation fine-tuning on the ASOCA 
dataset for one CV fold. Uses the same training pipeline as 
anatomix-fine-tuning.py, though just uses ASOCA GT coronary artery labels
for one fold of the 5-fold CV split.

Usage:
    python coronary-seg-finetune.py --fold 1
    python coronary-seg-finetune.py --fold 3 --seed 42

Outputs (separate from MM-WHS checkpoints):
    saved_models/segmentation_asoca/fold_{k}/
        finetuned_asoca_fold_{k}_{val_loss:.4f}.pth    <- best validation checkpoint
        anatomix_trained_asoca_fold_{k}.pth            <- final epoch checkpoint
"""

from tqdm import tqdm

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import SimpleITK as sitk

import monai
from monai.config import KeysCollection
from monai.data import list_data_collate, MetaTensor
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss, GeneralizedDiceLoss
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    MapTransform,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandSpatialCropd,
)
from torch.utils.data import DataLoader

sys.path.insert(0, "/vol/biomedic2/bglocker_studproj/<INSERT WHERE ANATOMIX IS FOR YOU>/anatomix/")
sys.path.insert(0, "anatomix")
from anatomix.segmentation.segmentation_utils import load_model, worker_init_fn

# CLI arguments

# Added --fold (required, 1-5) and made --seed explicit with a default.
# --fold selects which pre-built CV split to train. Everything else is unchanged.
_parser = argparse.ArgumentParser(
    description="Coronary segmentation fine-tuning (one ASOCA fold)"
)
_parser.add_argument(
    "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5],
    help="CV fold index (1-5). Loads data/config/asoca/fold_{k}/train.csv etc.",
)
_parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility (default: 42). Use 42, 123, 456, 789, or 999 for ensemble training.",
)
_args = _parser.parse_args()

FOLD = _args.fold
SEED = _args.seed

# Reproducibility

# We can train multiple segmentation seeds per fold to estimate ensemble uncertainty
# from prediction variability across independently initialised Anatomix models
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Output directory

# Each seed gets an isolated subdirectory so independently initialised models
# never overwrite one another
ckpt_dir = f"saved_models/segmentation_asoca/fold_{FOLD}/seed_{SEED}"
os.makedirs(ckpt_dir, exist_ok=True)
print(f"[INFO] Fold {FOLD} | seed {SEED} | checkpoints -> {ckpt_dir}")

# Hyperparameters 

iters_per_epoch = 75    # gradient steps per epoch
batch_size      = 4     # samples per step
n_epochs        = 200   # total fine-tuning epochs
lr              = 2e-4  # AdamW learning rate
crop_size       = 96    # spatial crop side length (voxels)
spacing         = 2.0   # isotropic mm spacing - matches ASOCA config.json
val_interval    = 5     # validate every N epochs

# One foreground class: coronary artery (background is implicit)
n_classes = 1

# Data loading

# CSV paths now point to the fold-specific ASOCA split (32 train / 8 test)
# Held-out test fold serves as the validation set during training
# Those 8 patients are never seen by the model during forward passe
fold_dir = f"data/config/asoca/fold_{FOLD}"

df_tr_imgs  = pd.read_csv(os.path.join(fold_dir, "train.csv"))
df_tr_segs  = pd.read_csv(os.path.join(fold_dir, "train.seg.csv"))
df_val_imgs = pd.read_csv(os.path.join(fold_dir, "test.csv"))
df_val_segs = pd.read_csv(os.path.join(fold_dir, "test.seg.csv"))

print("=" * 32)
print("ASOCA Coronary Fine-Tuning")
print("=" * 26)
print()
print(f"Fold:           fold_{FOLD}")
print(f"Seed:           {SEED}")
print(f"Train CSV:      {os.path.join(fold_dir, 'train.csv')}")
print(f"Val/Test CSV:   {os.path.join(fold_dir, 'test.csv')}")
print(f"Checkpoint dir: {ckpt_dir}")
print("=" * 19)

trimages = df_tr_imgs["img"].tolist()   # 32 training image paths
trsegs   = df_tr_segs["img"].tolist()   # 32 training label paths
vaimages = df_val_imgs["img"].tolist()  # 8 validation image paths
vasegs   = df_val_segs["img"].tolist()  # 8 validation label paths

print(f"[INFO] Train: {len(trimages)} images | Val (held-out test fold): {len(vaimages)} images")

# removed the subset-shuffle-slice block from the original (used a random 
# 14-of-20 subset for the few-shot learning experiments). For ASOCA CV the fold 
# CSVs already define the exact 32-image training split 

# Repeat logic is preserved, so by listing each image multiple times in
# train_files -> makes CacheDataset cache each repetition with its own
# independently-sampled random augmentation which gives augmentation variety
# without needing to re-read from disk.
finetuning_amount = len(trimages)                              # 32
samples_per_epoch = iters_per_epoch * batch_size               # 75 * 4 = 300
repeats           = max(1, samples_per_epoch // finetuning_amount)  # = 9

trimages = trimages * repeats   # 32 * 9 = 288 cache entries
trsegs   = trsegs   * repeats

train_files = [{"image": img, "label": seg} for img, seg in zip(trimages, trsegs)]
val_files   = [{"image": img, "label": seg} for img, seg in zip(vaimages, vasegs)]

print(
    f"[INFO] CacheDataset: {len(train_files)} entries "
    f"({finetuning_amount} images x {repeats} repeats) | "
    f"~{len(train_files) // batch_size} batches/epoch"
)

# Transform helpers 

def resample_image(
    image: MetaTensor,
    out_spacing=(1.0, 1.0, 1.0),
    out_size=None,
    is_label=False,
    pad_value=0,
):
    orig_affine  = np.array(image.meta.get("affine"))
    orig_spacing = np.linalg.norm(orig_affine[:3, :3], axis=0)
    direction    = orig_affine[:3, :3] / orig_spacing
    origin       = orig_affine[:3, 3]
    orig_size    = np.array(image.shape[1:], dtype=int)

    if out_size is None:
        out_size = np.round(orig_size * orig_spacing / np.array(out_spacing)).astype(int)
    else:
        out_size = np.array(out_size, dtype=int)

    orig_center = (orig_size - 1) / 2.0 * orig_spacing
    out_center  = (out_size  - 1) / 2.0 * np.array(out_spacing)
    new_origin  = origin + (direction.dot(orig_center) - direction.dot(out_center))

    new_affine         = np.eye(4, dtype=float)
    new_affine[:3, :3] = direction * np.array(out_spacing)
    new_affine[:3, 3]  = new_origin

    img = image.clone().detach()
    if not is_label:
        img = img.float()
    img = img.unsqueeze(0)

    mode    = "nearest" if is_label else "trilinear"
    align   = False if mode == "trilinear" else None
    resized = F.interpolate(img, size=tuple(out_size.tolist()), mode=mode, align_corners=align)
    resized = resized.squeeze(0)
    if is_label:
        resized = resized.long()

    new_meta = dict(image.meta)
    new_meta["affine"]          = new_affine
    new_meta["original_affine"] = new_affine
    new_meta["spatial_shape"]   = tuple(out_size.tolist())
    return MetaTensor(resized, meta=new_meta)


class Resamplerd(MapTransform):
    def __init__(self, keys, out_spacing, out_size=None, is_label=False,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.out_spacing = out_spacing
        self.out_size    = out_size
        self.is_label    = is_label

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = resample_image(d[key], self.out_spacing, self.out_size, self.is_label)
        return d


def load_sitk_array(path):
    img = sitk.ReadImage(path)
    img = sitk.DICOMOrient(img, "RAS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    sx, sy, sz   = img.GetSpacing()
    affine        = np.eye(4, dtype=np.float64)
    affine[0, 0]  = sz   # D axis spacing
    affine[1, 1]  = sy   # H axis spacing
    affine[2, 2]  = sx   # W axis spacing
    meta = {
        "affine":    affine,
        "spacing":   img.GetSpacing(),
        "origin":    img.GetOrigin(),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    return MetaTensor(torch.from_numpy(arr), meta=meta)


# remap_labels replaced with identity_labels since ASOCA binary labels are {0, 1} 
def identity_labels(label):
    return label


class RandPerSegmentElasticd(MapTransform):
    """Per-structure elastic deformation

    For binary ASOCA labels {0, 1}: skips background (cls == 0), applies
    elastic deformation to the single coronary class (cls == 1)
    """
    def __init__(self, keys: KeysCollection, sigma_range, magnitude_range, prob=0.3):
        super().__init__(keys)
        self.prob    = prob
        self.elastic = Rand3DElasticd(
            keys=keys,
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=1.0,
            spatial_size=None,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        )

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        d         = dict(data)
        image     = d[self.keys[0]]
        label     = d[self.keys[1]]

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        image_out = np.copy(image)
        label_out = np.zeros_like(label)

        for cls in np.unique(label):
            if cls == 0:
                continue
            mask       = (label == cls).astype(np.float32)
            img_masked = image * mask
            sample     = {self.keys[0]: img_masked, self.keys[1]: mask}
            aug        = self.elastic(sample)
            image_out  = np.where(mask, aug[self.keys[0]], image_out)
            # line below ensures only the deformed foreground region is assigned 
            # the class label (the rest remains background (0)) 
            # '(aug[...] > 0.5) * cls' evaluates to the augmented binary mask 
            # since cls == 1, which is correct
            label_out  = np.where(mask, (aug[self.keys[1]] > 0.5) * cls, label_out)

        d[self.keys[0]] = image_out.astype(np.float32)
        d[self.keys[1]] = label_out.astype(np.float32)
        return d


class RandPerSegmentInvertHUd(MapTransform):
    """Per-structure HU inversion

    For binary ASOCA labels, label_vals = [1.0] (coronary only). the single
    foreground class may have its HU values inverted with probability `prob`.
    """
    def __init__(self, keys: KeysCollection, prob=0.2):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        d         = dict(data)
        image     = d[self.keys[0]]
        label     = d[self.keys[1]]

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        image_out  = np.copy(image)
        label_vals = [v for v in np.unique(label) if v != 0]
        if not label_vals:
            return d

        selected = np.random.choice(
            label_vals,
            size=np.random.randint(1, len(label_vals) + 1),
            replace=False,
        )
        for cls in selected:
            mask = (label == cls)
            if not np.any(mask):
                continue
            region_vals = image[mask]
            vmin, vmax  = region_vals.min(), region_vals.max()
            image_out[mask] = (vmax + vmin) - image[mask]

        d[self.keys[0]] = image_out.astype(np.float32)
        return d


# Transform pipelines 

# only the label function is swapped (identity_labels instead of remap_labels) 
# since ASOCA labels are already binary and don't need remapping to 
# {0, 1, 2, 3, 4, 5, 6} like MM-WHS

def get_train_transforms(crop_size):
    return Compose([
        Lambdad(keys=["image", "label"], func=load_sitk_array),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"]),
        # out_size is changed to None, so that we resample to the correct 
        # physical spacing but keep original size, which can be larger than crop size
        Resamplerd(
            keys=["image", "label"],
            out_spacing=(spacing, spacing, spacing),
            out_size=None,
            is_label=[False, True],
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(crop_size, crop_size, crop_size),
            random_size=False,
        ),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        Lambdad(keys="label", func=identity_labels),   
        RandPerSegmentElasticd(
            keys=["image", "label"],
            sigma_range=(1, 9),
            magnitude_range=(1, 2),
            prob=0.9,
        ),
        RandPerSegmentInvertHUd(keys=["image", "label"], prob=0.2),
        RandGaussianNoised(keys=["image"], prob=0.33),
        RandBiasFieldd(keys=["image"], prob=0.33, coeff_range=(0.0, 0.05)),
        RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
        RandAdjustContrastd(keys=["image"], prob=0.33),
        RandGaussianSmoothd(
            keys=["image"], prob=0.33,
            sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
        ),
        RandGaussianSharpend(keys=["image"], prob=0.33),
        RandAffined(
            keys=["image", "label"],
            prob=0.98,
            mode=("bilinear", "nearest"),
            rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
            scale_range=(0.4, 0.4, 0.4),
            shear_range=(0.4, 0.4, 0.4),
            spatial_size=(crop_size, crop_size, crop_size),
            padding_mode="zeros",
        ),
    ])


def get_val_transforms():
    return Compose([
        Lambdad(keys=["image", "label"], func=load_sitk_array),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"], device="cpu"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        Resamplerd(
            keys=["image", "label"],
            out_spacing=(spacing, spacing, spacing),
            out_size=None,
            is_label=[False, True],
        ),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        Lambdad(keys="label", func=identity_labels),  
    ])


# Datasets and dataLoaders 

train_transforms = get_train_transforms(crop_size=crop_size)
val_transforms   = get_val_transforms()

train_ds = monai.data.CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=6,
)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
    worker_init_fn=worker_init_fn,
)

val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=0,
    collate_fn=list_data_collate,
    worker_init_fn=worker_init_fn,
    shuffle=False,
)

# removed the interactive visualisation loops 

# Model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# n_classes=n_classes instead of hardcoded. Anatomix treats n_classes as 
# foreground classes always adds a background channel 
# Pre-trained encoder weights are reused from anatomix 
new_model = load_model(
    pretrained_ckpt="anatomix/model-weights/anatomix.pth",
    n_classes=n_classes,
    device=device,
)

# Losses 

gdice = GeneralizedDiceLoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    w_type="square",
)
dicece = DiceCELoss(
    softmax=True,
    to_onehot_y=True,
    include_background=False,
)

loss_function          = dicece
valloss_function       = DiceLoss(softmax=True, to_onehot_y=True, include_background=False)
other_valloss_name     = "Gen Dice"
other_valloss_function = gdice

# Optimiser and scheduler 

optimizer = torch.optim.AdamW(new_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# Training loop 

best_val_loss          = float("inf")
best_loss_epoch        = 0
anatomix_train_loss_values = []
anatomix_val_loss_values   = []

for epoch in tqdm(range(n_epochs), desc="training"):
    print("-" * 10)
    print("epoch {:04d}/{:04d}".format(epoch + 1, n_epochs))
    new_model.train()
    epoch_loss = 0
    step       = 0

    for batch_data in train_loader:
        step   += 1
        inputs  = batch_data["image"].to(device)
        labels  = batch_data["label"].long().to(device)

        optimizer.zero_grad()
        outputs = new_model(inputs)
        loss    = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_len   = len(train_ds) // train_loader.batch_size

        if step % 10 == 0:
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

    epoch_loss /= step
    anatomix_train_loss_values.append(epoch_loss)
    scheduler.step()

    # Validation and checkpointing 
    if (epoch + 1) % val_interval == 0:
        new_model.eval()
        with torch.no_grad():
            val_loss            = 0.0
            val_loss_other_dice = 0.0
            valstep             = 0

            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].long().to(device)
                # Validation set volumes can be of any spatial size
                # So we're going to do sliding window inference at the
                # same crop size that we trained at
                roi_size      = (crop_size, crop_size, crop_size)
                sw_batch_size = 2
                val_outputs   = sliding_window_inference(
                    val_images, roi_size, sw_batch_size, new_model, overlap=0.7,
                )

                val_loss            += valloss_function(val_outputs, val_labels)
                val_loss_other_dice += other_valloss_function(val_outputs, val_labels)
                valstep             += 1

            val_loss = val_loss / valstep
            anatomix_val_loss_values.append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_loss_epoch = epoch + 1
                # renamed checkpoint name to include "asoca" and fold index
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"finetuned_asoca_fold_{FOLD}_{val_loss.item():.4f}.pth",
                )
                torch.save(new_model.state_dict(), ckpt_path)
                print(f"  [CKPT] Saved best checkpoint -> {ckpt_path}")

            print(
                "current epoch: {} current mean dice loss: {:.4f} ({} {:.4f})"
                " best mean dice loss: {:.4f} at epoch {}".format(
                    epoch + 1,
                    val_loss.item(),
                    other_valloss_name,
                    val_loss_other_dice.item(),
                    best_val_loss.item() if hasattr(best_val_loss, "item") else best_val_loss,
                    best_loss_epoch,
                )
            )

# final checkpoint follows the same ASOCA naming convention 
final_ckpt = os.path.join(ckpt_dir, f"anatomix_trained_asoca_fold_{FOLD}.pth")
torch.save(new_model.state_dict(), final_ckpt)

print(f"\n[INFO] Training complete.")
print(f"[INFO] Final checkpoint  -> {final_ckpt}")
print(f"[INFO] Best val Dice loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")