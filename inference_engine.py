"""
inference_engine.py
Modular SuPreM UNet inference and 3D CT processing engine.
Runs real sliding-window inference on whatever device is available
(CUDA GPU, Apple Silicon MPS, or CPU) -- there is no simulated/instant mode.
"""

import os
import sys
import io
import time
import json
import shutil
import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import nibabel as nib
from PIL import Image

import torch

# Add SuPreM to python path for model imports
suprem_path = os.path.join(os.path.dirname(__file__), "SuPreM", "direct_inference")
if suprem_path not in sys.path:
    sys.path.append(suprem_path)

try:
    from model.Universal_model import Universal_model
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
        Invertd,
    )
    MONAI_AVAILABLE = True
except Exception as e:
    MONAI_AVAILABLE = False
    print(f"Notice: MONAI/SuPreM imports: {e}")

# Target 9 primary abdominal organs evaluated in the BodyMaps benchmark
TARGET_ORGANS = {
    1: {"name": "spleen", "label": "Spleen", "color": "#FF4D4D", "rgb": (255, 77, 77)},
    2: {"name": "kidney_right", "label": "Right Kidney", "color": "#3399FF", "rgb": (51, 153, 255)},
    3: {"name": "kidney_left", "label": "Left Kidney", "color": "#33CCFF", "rgb": (51, 204, 255)},
    4: {"name": "gall_bladder", "label": "Gallbladder", "color": "#33CC33", "rgb": (51, 204, 51)},
    6: {"name": "liver", "label": "Liver", "color": "#FF9933", "rgb": (255, 153, 51)},
    7: {"name": "stomach", "label": "Stomach", "color": "#CC33FF", "rgb": (204, 51, 255)},
    8: {"name": "aorta", "label": "Aorta", "color": "#FF0000", "rgb": (255, 0, 0)},
    9: {"name": "postcava", "label": "IVC (Postcava)", "color": "#0055FF", "rgb": (0, 85, 255)},
    11: {"name": "pancreas", "label": "Pancreas", "color": "#FFCC00", "rgb": (255, 204, 0)},
}

WINDOW_PRESETS = {
    "abdomen": {"center": 40, "width": 400, "name": "Abdomen (Standard)"},
    "soft_tissue": {"center": 50, "width": 350, "name": "Soft Tissue"},
    "bone": {"center": 400, "width": 1800, "name": "Bone"},
    "lung": {"center": -600, "width": 1500, "name": "Lung"},
}


class VolumeCache:
    """In-memory cache for fast interactive slice serving."""
    def __init__(self):
        self.cached_case_id: Optional[str] = None
        self.ct_data: Optional[np.ndarray] = None
        self.mask_data: Optional[np.ndarray] = None
        self.affine: Optional[np.ndarray] = None
        self.header: Optional[Any] = None
        self.metadata: Optional[Dict[str, Any]] = None

volume_cache = VolumeCache()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Human-readable, honest description of the compute device actually used."""
    if device.type == "cuda":
        return f"NVIDIA GPU (CUDA) — {torch.cuda.get_device_name(0)}"
    elif device.type == "mps":
        return "Apple Silicon GPU (Metal/MPS)"
    return "CPU (no GPU acceleration detected)"


def load_ct_volume(ct_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Loads a NIfTI volume and extracts geometry & intensity statistics."""
    img = nib.load(ct_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    # Extract voxel spacing (dx, dy, dz)
    zooms = header.get_zooms()[:3]
    dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    voxel_volume_ml = (dx * dy * dz) / 1000.0

    meta = {
        "shape": list(data.shape),
        "slices_axial": data.shape[2],
        "slices_coronal": data.shape[1],
        "slices_sagittal": data.shape[0],
        "spacing_mm": [round(dx, 2), round(dy, 2), round(dz, 2)],
        "voxel_volume_ml": round(voxel_volume_ml, 4),
        "hu_min": float(np.min(data)),
        "hu_max": float(np.max(data)),
        "hu_mean": round(float(np.mean(data)), 1),
    }
    return data, affine, meta


def apply_window(slice_data: np.ndarray, center: float = 40, width: float = 400) -> np.ndarray:
    """Applies CT Hounsfield Unit windowing and scales to 0-255 uint8."""
    lower = center - (width / 2.0)
    upper = center + (width / 2.0)
    windowed = np.clip(slice_data, lower, upper)
    scaled = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return scaled


def calculate_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculates the Dice similarity coefficient between two binary masks."""
    pred_b = pred > 0
    gt_b = gt > 0
    intersection = np.logical_and(pred_b, gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def calculate_organ_statistics(mask_data: np.ndarray, voxel_vol_ml: float, gt_mask_data: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Computes voxel counts, physical volumes (mL), and Dice scores per organ."""
    stats = []
    for organ_id, info in TARGET_ORGANS.items():
        organ_binary = (mask_data == organ_id)
        voxels = int(np.count_nonzero(organ_binary))
        volume_ml = round(voxels * voxel_vol_ml, 2)
        
        entry = {
            "id": organ_id,
            "name": info["name"],
            "label": info["label"],
            "color": info["color"],
            "voxels": voxels,
            "volume_ml": volume_ml,
            "present": voxels > 0,
        }

        if gt_mask_data is not None:
            gt_binary = (gt_mask_data == organ_id)
            dice = calculate_dice(organ_binary, gt_binary)
            entry["dice"] = round(dice, 4)
            entry["gt_voxels"] = int(np.count_nonzero(gt_binary))
            entry["gt_volume_ml"] = round(entry["gt_voxels"] * voxel_vol_ml, 2)

        stats.append(entry)
    return stats


def render_slice_png(
    ct_data: np.ndarray,
    mask_data: Optional[np.ndarray],
    slice_idx: int,
    orientation: str = "axial",
    window_preset: str = "abdomen",
    active_organ_ids: Optional[List[int]] = None,
    opacity: float = 0.5,
) -> bytes:
    """
    Renders a 2D CT slice with color-blended organ segmentation overlays.
    Orientation supports: 'axial' (default, z-slice), 'coronal' (y-slice), 'sagittal' (x-slice).
    """
    # Extract 2D slice
    if orientation == "axial":
        slice_idx = max(0, min(slice_idx, ct_data.shape[2] - 1))
        ct_slice = ct_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx] if mask_data is not None else None
    elif orientation == "coronal":
        slice_idx = max(0, min(slice_idx, ct_data.shape[1] - 1))
        ct_slice = ct_data[:, slice_idx, :]
        mask_slice = mask_data[:, slice_idx, :] if mask_data is not None else None
    elif orientation == "sagittal":
        slice_idx = max(0, min(slice_idx, ct_data.shape[0] - 1))
        ct_slice = ct_data[slice_idx, :, :]
        mask_slice = mask_data[slice_idx, :, :] if mask_data is not None else None
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    # Transpose and flip to standard radiological viewing orientation
    # (Patient right is viewer left; dorsal is bottom)
    ct_slice = np.rot90(ct_slice)
    if mask_slice is not None:
        mask_slice = np.rot90(mask_slice)

    # Window/Level processing
    preset = WINDOW_PRESETS.get(window_preset, WINDOW_PRESETS["abdomen"])
    gray = apply_window(ct_slice, preset["center"], preset["width"])

    # Convert to RGB image
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # Apply organ mask overlays
    if mask_slice is not None and opacity > 0:
        if active_organ_ids is None:
            active_organ_ids = list(TARGET_ORGANS.keys())
        active_set = set(active_organ_ids)

        for organ_id in active_set:
            if organ_id in TARGET_ORGANS:
                info = TARGET_ORGANS[organ_id]
                organ_mask = (mask_slice == organ_id)
                if np.any(organ_mask):
                    target_color = np.array(info["rgb"], dtype=np.float32)
                    for c in range(3):
                        rgb[organ_mask, c] = (
                            (1.0 - opacity) * rgb[organ_mask, c]
                            + opacity * target_color[c]
                        )

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def load_suprem_model(checkpoint_path: str, device: torch.device) -> Any:
    """Constructs SuPreM UNet backbone and loads trained weights."""
    if not MONAI_AVAILABLE:
        raise RuntimeError(
            "MONAI or the SuPreM model definition failed to import. "
            "Check that the 'SuPreM' directory is present and 'monai' is installed "
            "(see requirements.txt)."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SuPreM checkpoint not found at '{checkpoint_path}'. "
            "Download 'supervised_suprem_unet_2100.pth' from "
            "https://huggingface.co/MrGiovanni/SuPreM and place it at that path "
            "before running inference."
        )

    model = Universal_model(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=32,
        backbone="unet",
        encoding="word_embedding",
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    load_dict = ckpt["net"] if "net" in ckpt else ckpt
    store_dict = model.state_dict()
    store_keys = list(store_dict.keys())
    load_values = list(load_dict.values())

    for i in range(len(store_keys)):
        store_dict[store_keys[i]] = load_values[i]

    model.load_state_dict(store_dict)
    model.to(device)
    model.eval()
    return model


def run_live_inference(
    ct_path: str,
    checkpoint_path: str,
    output_dir: str,
    device: Optional[torch.device] = None,
    overlap: float = 0.5,
) -> Dict[str, Any]:
    """
    Executes live sliding-window inference on a 3D CT volume using SuPreM UNet.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT volume not found at '{ct_path}'.")

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Stage outputs in a temp directory and only move them into place once the
    # full run succeeds, so a failure partway through never leaves a case with
    # a half-written / corrupted segmentation result.
    staging_dir = os.path.join(output_dir, f".inference_staging_{int(time.time() * 1000)}")
    seg_dir = os.path.join(staging_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)

    try:
        print(f"Starting SuPreM inference on {ct_path} (Device: {device})...")
        model = load_suprem_model(checkpoint_path, device)

        # Setup MONAI transforms matching SuPreM direct_inference.
        # EnsureChannelFirstd is required with modern MONAI (>=1.3): without it,
        # LoadImaged does not add a channel dim and sliding_window_inference
        # fails with a shape error.
        val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ])

        batch = val_transforms({"image": ct_path})
        image_tensor = batch["image"].unsqueeze(0).to(device)  # Shape: (1, 1, H, W, D)

        with torch.no_grad():
            preds = sliding_window_inference(
                inputs=image_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            preds = torch.sigmoid(preds)
            hard_preds = (preds > 0.5).cpu().numpy()[0]  # Shape: (32, H, W, D)

        orig_img = nib.load(ct_path)
        orig_affine = orig_img.affine
        orig_shape = orig_img.shape

        # Construct combined multi-label volume
        combined_labels = np.zeros(orig_shape, dtype=np.uint8)

        # Invert spatial transforms back to original image space
        invert_transform = Invertd(
            keys=["pred"],
            transform=val_transforms,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=False,
        )

        for organ_id in TARGET_ORGANS.keys():
            organ_channel = organ_id - 1
            binary_mask_resampled = hard_preds[organ_channel:organ_channel+1]  # (1, H, W, D)

            batch_inv = {"image": batch["image"], "pred": binary_mask_resampled}
            inverted = invert_transform(batch_inv)["pred"][0]  # (orig_H, orig_W, orig_D)

            # Ensure dimensions match original
            if inverted.shape != orig_shape:
                # Resample if needed
                from scipy.ndimage import zoom
                zoom_factors = [orig_shape[i] / inverted.shape[i] for i in range(3)]
                inverted = zoom(inverted.astype(np.float32), zoom_factors, order=0) > 0.5

            organ_name = TARGET_ORGANS[organ_id]["name"]
            organ_path = os.path.join(seg_dir, f"{organ_name}.nii.gz")
            nib.save(nib.Nifti1Image(inverted.astype(np.uint8), orig_affine), organ_path)

            combined_labels[inverted > 0] = organ_id

        staged_combined_path = os.path.join(staging_dir, "combined_labels.nii.gz")
        nib.save(nib.Nifti1Image(combined_labels, orig_affine), staged_combined_path)

        elapsed = round(time.time() - start_time, 2)
        run_meta = {
            "device": str(device),
            "device_label": device_label(device),
            "elapsed_seconds": elapsed,
            "overlap": overlap,
            "computed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": os.path.basename(checkpoint_path),
        }
        with open(os.path.join(staging_dir, "inference_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

        # Only now that every step has succeeded, atomically replace the case's
        # previous results with the freshly computed ones.
        final_seg_dir = os.path.join(output_dir, "segmentations")
        final_combined_path = os.path.join(output_dir, "combined_labels.nii.gz")
        final_meta_path = os.path.join(output_dir, "inference_meta.json")
        if os.path.exists(final_seg_dir):
            shutil.rmtree(final_seg_dir)
        shutil.move(seg_dir, final_seg_dir)
        shutil.move(staged_combined_path, final_combined_path)
        shutil.move(os.path.join(staging_dir, "inference_meta.json"), final_meta_path)
    finally:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Inference completed in {elapsed}s. Saved predictions to {output_dir}")

    return {
        "status": "success",
        "device": str(device),
        "device_label": device_label(device),
        "elapsed_seconds": elapsed,
        "combined_labels_path": final_combined_path,
        "segmentations_dir": final_seg_dir,
    }


def load_inference_meta(case_dir: str) -> Optional[Dict[str, Any]]:
    """Reads the sidecar file recording how/when the current prediction was computed."""
    meta_path = os.path.join(case_dir, "inference_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
