"""
This file fine-tunes the pretrained anatomix model on the MM-WHS dataset
"""
from tqdm.notebook import tqdm

import importlib
import sys

import torch
print(torch.cuda.is_available())

# Remove any previous broken anatomix attempts
for key in list(sys.modules.keys()):
    if 'anatomix' in key:
        del sys.modules[key]

# Add correct path
sys.path.insert(0, "/vol/biomedic2/bglocker_studproj/<INSERT WHERE ANATOMIX IS FOR YOU>/anatomix/")

from anatomix.segmentation.segmentation_utils import (
    worker_init_fn,
    load_model,
)
print("Success!")

import os
# print(os.getcwd())

import numpy as np
import random

import torch
import monai
from torch.utils.data import DataLoader
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Lambdad,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandAffined,
    EnsureTyped,
    EnsureChannelFirstd,
    RandSpatialCropd,
)
import sys
sys.path.insert(0, "anatomix")  
from anatomix.segmentation.segmentation_utils import (
    worker_init_fn,
    load_model,
)

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--seed', type=int, default=None)
_args, _ = _parser.parse_known_args()

# Set Python, NumPy, and PyTorch seeds - all four have independent RNGs 
SEED = _args.seed if _args.seed is not None else 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Ensure deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# we can run multiple seeds without overwriting each other’s checkpoints
ckpt_dir = f"saved_models/segmentation/seed_{SEED}"
os.makedirs(ckpt_dir, exist_ok=True)

import pandas as pd

n_trimages = 14
n_valimages = 6

df_tr_imgs = pd.read_csv('data/config/train.csv')
df_tr_segs = pd.read_csv('data/config/train.seg.csv')
df_val_imgs = pd.read_csv('data/config/val.csv')
df_val_segs = pd.read_csv('data/config/val.seg.csv')

trimages = sorted(df_tr_imgs['img'].tolist())[:n_trimages]
trsegs   = sorted(df_tr_segs['img'].tolist())[:n_trimages]

vaimages = sorted(df_val_imgs['img'].tolist())[:n_valimages]
vasegs   = sorted(df_val_segs['img'].tolist())[:n_valimages]

print('train images:', trimages)
print('train segs  :', trsegs)
print('val images  :', vaimages)
print('val segs    :', vasegs)

finetuning_amount = n_trimages  # amount of training set volumes to finetune on
iters_per_epoch = 75  # how many iterations to do per "epoch"
batch_size = 4  # batch size at every iteration
n_epochs = 200  # number of "epochs" to finetune for (reduced from 500 to 200)
lr = 2e-4  # learning rate for Adam
crop_size = 96  # train on (96, 96, 96) crops
spacing = 2.0
val_interval = 5  # validation loop frequency in epochs
n_classes = 7  # MM-WHS has 7 classes (background is already accounted for)

# Randomly select subset of training data for few-shot learning
trimages = np.random.RandomState(seed=SEED).permutation(trimages).tolist()
trsegs = np.random.RandomState(seed=SEED).permutation(trsegs).tolist()
trimages = trimages[:finetuning_amount]
trsegs = trsegs[:finetuning_amount]

# Calculate repeats needed to achieve desired iterations per epoch
samples_per_epoch = iters_per_epoch * batch_size
repeats = max(1, samples_per_epoch // finetuning_amount)

# Repeat training data to match desired samples per epoch
trimages = trimages * repeats
trsegs = trsegs * repeats

train_files = [
    {"image": img, "label": seg} for img, seg in zip(trimages, trsegs)
]
val_files = [
    {"image": img, "label": seg} for img, seg in zip(vaimages, vasegs)
]

from monai.transforms import MapTransform

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor

def resample_image(
    image: MetaTensor,
    out_spacing=(1.0, 1.0, 1.0),
    out_size=None,
    is_label=False,
    pad_value=0,
):
    """
    Resample a MetaTensor to given spacing and (optionally) size, matching the SITK logic.
    """

    orig_affine = np.array(image.meta.get("affine"))
    orig_spacing = np.linalg.norm(orig_affine[:3, :3], axis=0)
    direction = orig_affine[:3, :3] / orig_spacing
    origin = orig_affine[:3, 3]

    orig_size = np.array(image.shape[1:], dtype=int)

    if out_size is None:
        out_size = np.round(orig_size * orig_spacing / np.array(out_spacing)).astype(int)
    else:
        out_size = np.array(out_size, dtype=int)

    orig_center = (orig_size - 1) / 2.0 * orig_spacing
    out_center  = (out_size  - 1) / 2.0 * np.array(out_spacing)

    orig_ctr_phys = direction.dot(orig_center)
    out_ctr_phys  = direction.dot(out_center)
    new_origin = origin + (orig_ctr_phys - out_ctr_phys)

    new_affine = np.eye(4, dtype=float)
    new_affine[:3, :3] = direction * np.array(out_spacing)
    new_affine[:3, 3]  = new_origin

    img = image.clone().detach()
    dtype = img.dtype
    if not is_label:
        img = img.float()
    img = img.unsqueeze(0)

    mode = "nearest" if is_label else "trilinear"
    align = False if mode == "trilinear" else None
    resized = F.interpolate(img, size=tuple(out_size.tolist()), mode=mode, align_corners=align)

    resized = resized.squeeze(0)
    if is_label:
        resized = resized.long()

    new_meta = dict(image.meta)
    new_meta["affine"] = new_affine
    new_meta["original_affine"] = new_affine
    new_meta["spatial_shape"] = tuple(out_size.tolist())

    return MetaTensor(resized, meta=new_meta)


class Resamplerd(MapTransform):
    """
    Resamples each array in `keys` to `out_spacing` and (optionally) `out_size`,
    using nearest‐neighbour if `is_label=True`, else linear interpolation.
    """
    def __init__(self, keys, out_spacing, out_size=None, is_label=False, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.out_spacing = out_spacing
        self.out_size     = out_size
        self.is_label     = is_label

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # your existing resample_image takes (array, spacing, size, is_label)
            d[key] = resample_image(
                d[key],
                self.out_spacing,
                self.out_size,
                self.is_label,
            )
        return d

import SimpleITK as sitk
def load_sitk_array(path):
    img = sitk.ReadImage(path)
    img = sitk.DICOMOrient(img, "RAS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    tensor = torch.from_numpy(arr)

    meta = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    return MetaTensor(tensor, meta=meta)

def remap_labels(label):
    remap_dict = {
        0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7,
    }
    for raw_val, class_idx in remap_dict.items():
        label[label == raw_val] = class_idx
    return label


from monai.config import KeysCollection
from monai.transforms import MapTransform, Rand3DElasticd

import numpy as np
import torch

"""
data augmentation trick 
flips the image intensities within randomly selected segments,
which can help the model learn to be invariant to intensity variations and improve generalization.
"""
def invertHU(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32, copy=True)
    if np.random.rand() < 0.2:
        # ensure we don’t modify original
        mn, mx = float(arr.min()), float(arr.max())
        # negative: white ↔ black
        return (mx + mn) - arr
    
    return arr

"""
instead of applying elastic deformation to the whole image, 
we apply it separately to each labelled structure separately... 
so each organ/segment gets its own distortion 
"""
class RandPerSegmentElasticd(MapTransform):
    def __init__(self, keys: KeysCollection, sigma_range, magnitude_range, prob=0.3):
        super().__init__(keys)
        self.prob = prob
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

        d = dict(data)
        image = d[self.keys[0]]
        label = d[self.keys[1]]

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        image_out = np.copy(image)
        label_out = np.zeros_like(label)

        for cls in np.unique(label):
            if cls == 0:
                continue
            mask = (label == cls).astype(np.float32)
            img_masked = image * mask

            sample = {
                self.keys[0]: img_masked,
                self.keys[1]: mask
            }
            aug = self.elastic(sample)

            image_out = np.where(mask, aug[self.keys[0]], image_out)
            label_out = np.where(mask, (aug[self.keys[1]] > 0.5) * cls, label_out)

        d[self.keys[0]] = image_out.astype(np.float32)
        d[self.keys[1]] = label_out.astype(np.float32)
        return d

import numpy as np
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection

class RandPerSegmentInvertHUd(MapTransform):
    def __init__(self, keys: KeysCollection, prob=0.2):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        d = dict(data)
        image = d[self.keys[0]]
        label = d[self.keys[1]]

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        image_out = np.copy(image)
        label_vals = [v for v in np.unique(label) if v != 0]
        if not label_vals:
            return d

        # Choose random 1 to N segments
        selected = np.random.choice(label_vals, size=np.random.randint(1, len(label_vals) + 1), replace=False)

        for cls in selected:
            mask = (label == cls)
            if not np.any(mask):
                continue
            region_vals = image[mask]
            vmin, vmax = region_vals.min(), region_vals.max()
            image_out[mask] = (vmax + vmin) - image[mask]

        d[self.keys[0]] = image_out.astype(np.float32)
        return d

def get_train_transforms(crop_size):
    """
    Returns a MONAI composed transform object containing the specified
    data augmentations for the training dataset.
    """
    train_transforms = Compose(
        [
            Lambdad(keys=["image", "label"], func=load_sitk_array),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            EnsureTyped(keys=["image", "label"]),

            # Resamplerd(keys=["image", "label"], out_spacing=(spacing, spacing, spacing), out_size=(crop_size, crop_size, crop_size), is_label=[False, True]),
            # out_size is changed to None, so that we resample to the correct physical spacing but keep original size, which can be larger than crop size
            Resamplerd(keys=["image", "label"], out_spacing=(spacing, spacing, spacing), out_size=None, is_label=[False, True]), 
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(crop_size, crop_size, crop_size),
                random_size=False,
            ),

            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            Lambdad(keys="label", func=remap_labels),
            RandPerSegmentElasticd(
                keys=["image", "label"],
                sigma_range=(1, 9),
                magnitude_range=(1, 2),
                prob=0.9
            ),
            RandPerSegmentInvertHUd(keys=["image", "label"], prob=0.2),
            RandGaussianNoised(keys=["image"], prob=0.33),
            RandBiasFieldd(keys=["image"], prob=0.33, coeff_range=(0.0, 0.05)),
            RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
            RandAdjustContrastd(keys=["image"], prob=0.33),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.33,
                sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
            ),
            RandGaussianSharpend(keys=["image"], prob=0.33),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.4, 0.4, 0.4),
                shear_range=(0.4, 0.4, 0.4),
                spatial_size=(crop_size, crop_size, crop_size),
                padding_mode='zeros',
            ),
        ]
    )
    return train_transforms

def get_val_transforms():
    val_transforms = Compose(
        [
            Lambdad(keys=["image", "label"], func=load_sitk_array),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            EnsureTyped(keys=["image", "label"], device='cpu'),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            # Resamplerd(keys=["image", "label"], out_spacing=(spacing, spacing, spacing), out_size=(crop_size, crop_size, crop_size), is_label=[False, True]),
            Resamplerd(keys=["image", "label"], out_spacing=(spacing, spacing, spacing), out_size=None, is_label=[False, True]),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            Lambdad(keys="label", func=remap_labels),
        ]
    )
    return val_transforms

# define transforms for image and segmentation
train_transforms = get_train_transforms(crop_size=crop_size)
val_transforms = get_val_transforms()

# create a training data loader
train_ds = monai.data.CacheDataset(
    data=train_files, 
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=6
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
    worker_init_fn=worker_init_fn,
)

# create a validation data loader
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=0,
    collate_fn=list_data_collate,
    worker_init_fn=worker_init_fn,
    shuffle=True,
) # TODO do pca 1

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Define a generic color map generator
def get_colormap(num_classes):
    cmap_base = plt.get_cmap('tab20', num_classes)
    colors = cmap_base(np.linspace(0, 1, num_classes))
    colors[0] = [0, 0, 0, 1]
    return ListedColormap(colors)

# Visualize batches
for _ in range(5):
    batch_data = next(iter(train_loader))
    sampleimg, samplelab = batch_data["image"], batch_data["label"]

    img = sampleimg.cpu().numpy()[0].squeeze()
    lab = samplelab.cpu().numpy()[0].squeeze()

    # Automatically determine number of classes from label
    num_classes = int(lab.max()) + 1

    plt.figure(figsize=(7, 3.5))
    plt.suptitle(f'Batch {_}')
    plt.axis('off')

    plt.subplot(1, 2, 1)
    plt.imshow(img[48, :, :], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img[48, :, :], cmap='gray')
    plt.imshow(lab[48, :, :], cmap=get_colormap(num_classes), alpha=0.6)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

for _ in range(1):
    batch_data = next(iter(val_loader))
    sampleimg, samplelab = batch_data["image"], batch_data["label"]

    img = sampleimg.cpu().numpy()[0].squeeze()
    lab = samplelab.cpu().numpy()[0].squeeze()

    int(lab.max()) + 1

    plt.figure(figsize=(7, 3.5))
    plt.suptitle(f'Batch {_}')
    plt.axis('off')

    plt.subplot(1, 2, 1)
    plt.imshow(img[48, :, :], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img[48, :, :], cmap='gray')
    plt.imshow(lab[48, :, :], cmap=get_colormap(num_classes), alpha=0.6)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")
new_model = load_model(
    pretrained_ckpt='anatomix/model-weights/anatomix.pth',
    n_classes=7, # 8 - 1 for MM-WHS
    device=device,
)

import torch
import monai
from monai.losses import DiceCELoss, DiceLoss, GeneralizedDiceLoss

# Volume-adaptive Dice (“Gen Dice”) excluding background
gdice = GeneralizedDiceLoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    w_type="square",
)

# Standard Dice + CE (equal-weight)
dicece = DiceCELoss(
    softmax=True,
    to_onehot_y=True,
    include_background=False,
)

loss_function = dicece

valloss_function       = DiceLoss(
    softmax=True,
    to_onehot_y=True,
    include_background=False,
)
other_valloss_name     = "Gen Dice"
other_valloss_function = gdice


optimizer = torch.optim.AdamW(
    new_model.parameters(),
    lr=lr,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
)

# start a typical PyTorch training
best_val_loss = 10000000000
anatomix_train_loss_values = []
anatomix_val_loss_values = []

# Training loop
for epoch in tqdm(range(n_epochs), desc="training"):
    print("-" * 10)
    print("epoch {:04d}/{:04d}".format(epoch + 1, n_epochs))
    new_model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer.zero_grad()

        outputs = new_model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size

        if step % 10 == 0:
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

    epoch_loss /= step
    anatomix_train_loss_values.append(epoch_loss)
    scheduler.step()


    # Validation and checkpointing loop:
    if ((epoch + 1) % val_interval == 0):
        new_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_other_dice = 0.0
            valstep = 0
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                # Validation set volumes can be of any spatial size
                # So we're going to do sliding window inference at the
                # same crop size that we trained at
                roi_size = (crop_size, crop_size, crop_size)
                sw_batch_size = 2
                val_outputs = sliding_window_inference(
                    val_images, roi_size, sw_batch_size,
                    new_model, overlap=0.7,
                )
                
                val_loss += valloss_function(val_outputs, val_labels)
                val_loss_other_dice += other_valloss_function(val_outputs, val_labels)
                valstep += 1

            val_loss = val_loss / valstep
            anatomix_val_loss_values.append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_loss_epoch = epoch + 1
                torch.save(new_model.state_dict(), f"{ckpt_dir}/finetuned_MM-WHS{val_loss.item():.4f}.pth")

            print(
                "current epoch: {} current mean dice loss: {:.4f} ({} {:.4f})"
                " best mean dice loss: {:.4f} at epoch {}".format(
                    epoch + 1, val_loss.item(), other_valloss_name, val_loss_other_dice.item(),
                    best_val_loss.item(), best_loss_epoch,
                )
            )
torch.save(new_model.state_dict(), f"{ckpt_dir}/anatomix_trained_MM-WHS.pth")


