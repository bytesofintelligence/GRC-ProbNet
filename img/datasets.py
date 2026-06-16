import os
import os.path as path
import gc

from totalsegmentator.python_api import totalsegmentator
from anatomix.segmentation.segmentation_utils import load_model

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from monai.data import Dataset
from img.processing import one_hot_labelmap
from monai.transforms import Compose, Rand3DElasticd, ToTensord, EnsureTyped, Lambdad
from monai.transforms import EnsureChannelFirstd
from img.transforms import Resamplerd
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_dilation, label as cc_label

# remap the class labels in MM-WHS.
def remap_labels(label):
    remap_dict = {
        0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7,
    }
    for raw_val, class_idx in remap_dict.items():
        label[label == raw_val] = class_idx
    return label

def load_sitk_array(path):
    img = sitk.ReadImage(path)
    img = sitk.DICOMOrient(img, "RAS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    tensor = torch.from_numpy(arr)

    sx, sy, sz = img.GetSpacing()  # spacing in x, y, z
    
    # build affine so spacing is correctly interpreted in (D, H, W) space
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = sz  # D
    affine[1, 1] = sy  # H
    affine[2, 2] = sx  # W

    meta = {
        "affine": affine,
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    return MetaTensor(tensor, meta=meta)

def get_image_label_transforms(spacing, normalizer):
    # out_size is omitted
    # resample to spacing but keep native grid size, crop happens later 
    val_transforms = Compose(
        [
            Lambdad(keys=["image", "gt"], func=load_sitk_array, allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "gt"], channel_dim='no_channel', allow_missing_keys=True),
            EnsureTyped(keys=["image", "gt"], device='cpu', allow_missing_keys=True),
            EnsureTyped(keys=["image", "gt"], dtype=torch.float32, allow_missing_keys=True),
            Resamplerd(keys=["image", "gt"], out_spacing=spacing, is_label=[False, True], allow_missing_keys=True),
            EnsureTyped(keys=["image", "gt"], dtype=torch.float32, allow_missing_keys=True),
            Lambdad(keys="gt", func=remap_labels, allow_missing_keys=True),
            normalizer,
        ]
    )
    return val_transforms


def clean_segmentation_mask(label_vol: torch.Tensor, dilation_iters: int = 3, skip_lcc: bool = False) -> np.ndarray:
    """Convert label to binary mask, then morphological dilation and optionally keep largest connected region
    - returns numpy array (same spatial shape as label_vol).
    - dilation to give a safety margin around the heart. ensures the crop window includes the full cardiac structure
    - LCC filtering removes spurious Anatomix predictions that could just inflate the bounding box

    skip_lcc is set to True for coronary arteries - they might have legitimate connected components
    """
    binary = (label_vol > 0).numpy().astype(np.uint8)
    dilated = binary_dilation(binary, iterations=dilation_iters).astype(np.uint8)
    if skip_lcc:
        return dilated
    labeled_arr, n_components = cc_label(dilated)
    if n_components == 0:
        return dilated
    sizes = np.bincount(labeled_arr.ravel())
    sizes[0] = 0  # ignore background
    return (labeled_arr == int(sizes.argmax())).astype(np.uint8)


def get_bbox_size(binary_mask: np.ndarray) -> tuple:
    """Compute bounding box size of foreground, returns (D,H,W) tuple"""
    coords = np.where(binary_mask > 0)
    if len(coords[0]) == 0:
        return (0, 0, 0)
    return (
        int(coords[0].max() - coords[0].min()) + 1,
        int(coords[1].max() - coords[1].min()) + 1,
        int(coords[2].max() - coords[2].min()) + 1,
    )


def centroid_crop_fixed(
    ct_vol: torch.Tensor,
    label_vol: torch.Tensor,
    cleaned_mask: np.ndarray,
    crop_size: tuple,
    pad_value: float = 0.0,
) -> tuple:
    """Crop both volumes to crop_size centred on the cleaned-mask centroid
    if the crop window extends beyond the volume boundary, padding applied
    """
    coords = np.where(cleaned_mask > 0)
    if len(coords[0]) == 0:
        D, H, W = ct_vol.shape
        cz, cy, cx = D // 2, H // 2, W // 2
    else:
        cz = int(round(coords[0].mean()))
        cy = int(round(coords[1].mean()))
        cx = int(round(coords[2].mean()))

    cd, ch, cw = crop_size
    D, H, W = ct_vol.shape

    z0, y0, x0 = cz - cd // 2, cy - ch // 2, cx - cw // 2
    z1, y1, x1 = z0 + cd, y0 + ch, x0 + cw

    # padding if crop window exceeds the volume on any side
    pz0, pz1 = max(0, -z0), max(0, z1 - D)
    py0, py1 = max(0, -y0), max(0, y1 - H)
    px0, px1 = max(0, -x0), max(0, x1 - W)

    z0c, z1c = max(0, z0), min(D, z1)
    y0c, y1c = max(0, y0), min(H, y1)
    x0c, x1c = max(0, x0), min(W, x1)

    # F.pad uses (W, H, D) order for 5D tensors, like (W_left,W_right, H_top,H_bot, D_front,D_back)
    padding = (px0, px1, py0, py1, pz0, pz1)

    def _crop_and_pad(vol, fill):
        patch = vol.float()[z0c:z1c, y0c:y1c, x0c:x1c]
        return (
            F.pad(patch.unsqueeze(0).unsqueeze(0), padding, value=fill)
            .squeeze(0)
            .squeeze(0)
        )

    return _crop_and_pad(ct_vol, pad_value), _crop_and_pad(label_vol, 0.0)

def _find_best_anatomix_checkpoint(model_dir='saved_models/segmentation'):
    candidates = [
        f for f in os.listdir(model_dir)
        if f.startswith('finetuned_MM-WHS') and f.endswith('.pth')
    ]
    if not candidates:
        fallback = os.path.join(model_dir, 'anatomix_trained_MM-WHS.pth')
        print(f'[Anatomix] No finetuned checkpoints found, using {fallback}')
        return fallback
    def _loss(name):
        try:
            return float(name[len('finetuned_MM-WHS'):-len('.pth')])
        except ValueError:
            return float('inf')
        
    best = min(candidates, key=_loss)
    return os.path.join(model_dir, best)


class ImageSegmentationOneHotDataset(Dataset):
    """Dataset for medical image segmentation."""

    def __init__(self,
                csv_file_img,
                num_classes, spatial_size, spacing,
                mode="Anatomix", csv_file_seg=None, normalizer=None, binarize=False, augmentation=False, label_smoothing=0.0,
                fixed_crop_size=None, model_checkpoint=None, skip_lcc=False):
        """
        Args:
        :param csv_file_img: Path to csv file with image filenames.
        :param csv_file_seg: Path to csv file with segmentation filenames (optional).
        :param spatial_size: Anatomix sliding-window roi_size. Does NOT set the final crop shape.
        :param fixed_crop_size: (D,H,W) tuple for centroid crop applied to every image
               If None (training) then phase 1 computes it from max bounding box across the dataset, rounded up to nearest multiple of 16
               pass the training dataset's self.crop_size to all val/test constructors so they use identical dimensions 
        """
        # Anatomix inference runs on CPU to avoid competing with the STN/ITN
        # that already occupy GPU memory during training setup.
        infer_device = "cpu"
        is_labelled = csv_file_seg is not None and path.exists(csv_file_seg)

        if not path.exists(csv_file_img):
            raise FileNotFoundError(f"Image CSV not found: {csv_file_img!r}")
        
        self.img_data = pd.read_csv(csv_file_img, comment="#")
        if len(self.img_data) == 0:
            raise ValueError(f"Image CSV is empty: {csv_file_img!r}")
        
        self.seg_data = pd.read_csv(csv_file_seg, comment="#") if is_labelled else []

        self.augmentation = augmentation

        self.augment = Compose([
            ToTensord(keys=["image", "labelmap"], allow_missing_keys=True),
            Rand3DElasticd(
                keys=["image", "labelmap"],
                mode=["bilinear", "nearest"],
                sigma_range=(5, 7),
                magnitude_range=(4, 4),
                spatial_size=None,
                translate_range=0,
                rotate_range=0,
                scale_range=0,
                padding_mode="border",
                allow_missing_keys=True
            ),
        ])

        if not is_labelled:
            if model_checkpoint is None:
                model_checkpoint = _find_best_anatomix_checkpoint()

            print(f'[Anatomix] Loading checkpoint: {model_checkpoint}')

            model = load_model(
                pretrained_ckpt='anatomix/model-weights/anatomix.pth',
                n_classes=num_classes-1,
                device=infer_device,
            )
            model.load_state_dict(torch.load(model_checkpoint, map_location=infer_device))
            model.eval()

        # Run the full preprocessing pipeline, used in both phases
        def _load_image_and_label(idx):
            img_path = self.img_data.iloc[idx, 0]
            tfms     = get_image_label_transforms(spacing, normalizer)
            io_dict  = {"image": img_path}

            if is_labelled:
                io_dict["gt"] = self.seg_data.iloc[idx, 0]
                s         = tfms(io_dict)
                label_vol = s["gt"].cpu().squeeze()
            else:
                if mode == "Anatomix":
                    s     = tfms(io_dict)
                    img_t = s["image"].unsqueeze(0).to(infer_device)
                    with torch.no_grad():
                        # Decreased overlap from 0.7 to 0.35
                        logits = sliding_window_inference(
                            img_t, roi_size=spatial_size, sw_batch_size=1,
                            predictor=model, overlap=0.35,
                        )
                    label_vol = torch.argmax(logits, dim=1)[0].cpu()
                elif mode == "TotalSegmentator": # not used in this project, though left for future experimentation 
                    nib_img   = nib.load(img_path)
                    label_nib = segment_image(nib_img)
                    out_dir   = "output/totalseg"
                    os.makedirs(out_dir, exist_ok=True)
                    temp_seg_path = os.path.join(out_dir, f"totalseg_{idx}.nii.gz")
                    nib.save(label_nib, temp_seg_path)
                    io_dict["gt"] = temp_seg_path
                    s         = tfms(io_dict)
                    label_vol = s["gt"].cpu().squeeze()
                else:
                    raise NotImplementedError(f"mode {mode} not supported")

            return s["image"].cpu().squeeze(), label_vol

        # Phase 1: training only. Scans every image in dataset to collect bounding box sizes, 
        # to determine a fixed crop size that fits all cases 
        if fixed_crop_size is None:
            bbox_sizes = []
            for idx in tqdm(range(len(self.img_data)), desc='Phase 1/2: computing bounding boxes'):
                fname = os.path.basename(self.img_data.iloc[idx, 0])
                _, label_vol = _load_image_and_label(idx)
                cleaned_mask = clean_segmentation_mask(label_vol, skip_lcc=skip_lcc)
                size = get_bbox_size(cleaned_mask)
                bbox_sizes.append(size)

                if size == (0, 0, 0):
                    print(f'[PREPROC WARNING] {fname}: empty segmentation')
                else:
                    print(f'[PREPROC] {fname}: bbox (D,H,W) = {size}')

                gc.collect()
                torch.cuda.empty_cache()

            valid_sizes = [s for s in bbox_sizes if s != (0, 0, 0)]
            if not valid_sizes:
                raise ValueError(
                    "All segmentations produced empty bounding boxes"
                )

            def _round16(x):
                return (x + 15) // 16 * 16

            crop_size = tuple(_round16(max(s[i] for s in valid_sizes)) for i in range(3))

            n_empty = len(bbox_sizes) - len(valid_sizes)
            if n_empty:
                print(f'[PREPROC WARNING] {n_empty}/{len(bbox_sizes)} empty segmentations ignored')

            # Print some Summary statistics of per-axis bbox distribution
            print(f'[PREPROC] Bounding box stats across {len(valid_sizes)} valid images:')
            for ax, ax_name in enumerate(['D (z/axial)', 'H (y/coronal)', 'W (x/sagittal)']):
                vals = [s[ax] for s in valid_sizes]
                print(f'[PREPROC]   {ax_name}: min={min(vals)}, max={max(vals)} -> crop dim = {crop_size[ax]} (rounded to mult of 16)')

            print(f'[PREPROC] Fixed crop size = {crop_size}')
        else:
            crop_size = tuple(int(x) for x in fixed_crop_size)
            print(f'[PREPROC] Using provided fixed_crop_size = {crop_size}')

        self.crop_size = crop_size  # always set here for future calls 

        # Phase 2 begins - we reload every image, apply centroid crop
        # for val and test datasets, this is the only pass since fixed_crop_size is provided
        self.samples = []
        phase2_desc = 'Phase 2/2: cropping and encoding' if fixed_crop_size is None else 'Loading data'

        for idx in tqdm(range(len(self.img_data)), desc=phase2_desc):
            img_fname = os.path.basename(self.img_data.iloc[idx, 0])

            ct_vol, label_vol = _load_image_and_label(idx)
            cleaned_mask      = clean_segmentation_mask(label_vol, skip_lcc=skip_lcc)

            # Log centroid and whether padding is needed
            coords = np.where(cleaned_mask > 0)
            if len(coords[0]) > 0:
                cz = int(round(coords[0].mean()))
                cy = int(round(coords[1].mean()))
                cx = int(round(coords[2].mean()))
                D, H, W = ct_vol.shape
                cd, ch, cw = crop_size

                needs_pad = (cz - cd//2 < 0 or cz + cd//2 > D or
                             cy - ch//2 < 0 or cy + ch//2 > H or
                             cx - cw//2 < 0 or cx + cw//2 > W)
                
                pad_note = '[PADDING APPLIED]' if needs_pad else ''
                print(f'[PREPROC] {img_fname}: vol=({D},{H},{W}), centroid=({cz},{cy},{cx}){pad_note}')
            else:
                print(f'[PREPROC WARNING] {img_fname}: empty mask, cropping from volume centre')

            ct_vol, label_vol = centroid_crop_fixed(ct_vol, label_vol, cleaned_mask, crop_size)

            # quick sanity check since output must always equal crop_size
            assert tuple(ct_vol.shape) == crop_size, \
                f'[PREPROC ERROR] {img_fname}: shape mismatch -crop shape {tuple(ct_vol.shape)} != expected {crop_size}'

            image_sitk = sitk.GetImageFromArray(ct_vol.numpy())
            label_sitk = sitk.GetImageFromArray(label_vol.numpy())

            if len(image_sitk.GetSize()) == 3:
                image_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
                label_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            else:
                image_sitk.SetDirection((1, 0, 0, 1))
                label_sitk.SetDirection((1, 0, 0, 1))

            if binarize:
                label_sitk = sitk.Cast(label_sitk > 0, sitk.sitkInt64)

            labelmap = sitk.Cast(
                one_hot_labelmap(label_sitk, smoothing_sigma=label_smoothing, num_classes=num_classes),
                sitk.sitkVectorFloat32,
            )

            self.samples.append({'image': image_sitk, 'labelmap': labelmap, 'fname': img_fname})

            gc.collect()
            torch.cuda.empty_cache()

        if not is_labelled:
            del model

        gc.collect()
        torch.cuda.empty_cache()
        print(f'[PREPROC] Done. {len(self.samples)} samples loaded, crop_size={self.crop_size}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]

        image = torch.from_numpy(sitk.GetArrayFromImage(sample['image'])).unsqueeze(0)

        if len(image.size()) == 4:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(3, 0, 1, 2)
        else:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(2, 0, 1)

        if self.augmentation:
            device = "cuda" if torch.cuda.is_available() else "cpu" 

            subject_dict = {
                "image": image.to(device),
                "labelmap": labelmap.to(device),
            }

            subject = self.augment(subject_dict)
            def strip_meta(x):
                return x.as_tensor() if isinstance(x, MetaTensor) else x

            # Dynamically handle missing keys
            keys = ['image', 'labelmap']

            subject = {k: strip_meta(subject[k].data).cpu() for k in keys}
            return {'image': subject['image'], 'labelmap': subject['labelmap'], 'fname': sample['fname']}
        else:
            return {'image': image, 'labelmap': labelmap, 'fname': sample['fname']}

    def get_sample(self, item):
        return self.samples[item]


def segment_image(nifti_img):
    segmentation_nifti = totalsegmentator(nifti_img, task='heartchambers_highres', quiet=True)
    return segmentation_nifti