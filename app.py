"""
app.py
FastAPI backend for BodyMaps AI CT Segmentation Web App (SuPreM).
"""

import os
import io
import shutil
import zipfile
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import nibabel as nib
import torch

import inference_engine as ie

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "SuPreM", "direct_inference", "pretrained_checkpoints", "supervised_suprem_unet_2100.pth"
)

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="BodyMaps CT Segmentation API",
    description="Automated 3D Abdominal Organ Segmentation powered by SuPreM (CCVL / Johns Hopkins University)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_case_paths(case_id: str):
    """Resolves CT, segmentation, and ground-truth paths for a given case."""
    # Check default demo case in data/ or root
    candidates = [
        os.path.join(DATA_DIR, case_id),
        os.path.join(BASE_DIR, case_id),
        os.path.join(UPLOADS_DIR, case_id),
    ]
    for case_dir in candidates:
        ct_path = os.path.join(case_dir, "ct.nii.gz")
        if os.path.exists(ct_path):
            combined_labels = os.path.join(case_dir, "combined_labels.nii.gz")
            seg_dir = os.path.join(case_dir, "segmentations")
            # Ground truth (when shipped with a case) lives in its own dedicated
            # file, kept strictly separate from combined_labels.nii.gz (which is
            # the model's own prediction once inference has been run). This file
            # is NEVER inferred from combined_labels -- doing that previously
            # made Dice trivially 1.0 by comparing predictions to themselves.
            gt_combined = os.path.join(case_dir, "labels_groundtruth.nii.gz")

            return {
                "case_dir": case_dir,
                "ct_path": ct_path,
                "combined_labels_path": combined_labels if os.path.exists(combined_labels) else None,
                "seg_dir": seg_dir if os.path.exists(seg_dir) else None,
                "gt_path": gt_combined if os.path.exists(gt_combined) else None,
            }
    raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")


def load_case_into_cache(case_id: str):
    """Ensures volume data is cached in memory for rapid slice requests."""
    paths = get_case_paths(case_id)
    if ie.volume_cache.cached_case_id != case_id or ie.volume_cache.ct_data is None:
        ct_data, affine, meta = ie.load_ct_volume(paths["ct_path"])
        
        mask_data = None
        if paths["combined_labels_path"]:
            mask_img = nib.load(paths["combined_labels_path"])
            mask_data = mask_img.get_fdata().astype(np.uint8)

        ie.volume_cache.cached_case_id = case_id
        ie.volume_cache.ct_data = ct_data
        ie.volume_cache.mask_data = mask_data
        ie.volume_cache.affine = affine
        ie.volume_cache.metadata = meta

    return paths


@app.get("/api/health")
def health_check():
    """System status and hardware inspection. Reports the REAL device that
    /api/case/{id}/infer will actually run on -- CUDA, Apple Silicon MPS, or CPU.
    Nothing here is simulated; this reflects torch's own hardware detection."""
    device = ie.get_device()
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    accelerated = has_cuda or has_mps

    return {
        "status": "online",
        "device": str(device),
        "device_label": ie.device_label(device),
        "accelerated": accelerated,
        "has_gpu": has_cuda,  # kept for backwards compatibility; CUDA specifically
        "has_mps": has_mps,
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else None,
        "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
        "checkpoint_size_mb": round(os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024), 1) if os.path.exists(CHECKPOINT_PATH) else 0,
        "monai_available": ie.MONAI_AVAILABLE,
        "torch_version": torch.__version__,
    }


@app.get("/api/cases")
def list_cases():
    """Returns available CT scan cases."""
    cases = []
    
    # 1. Bundled demo case
    demo_paths = get_case_paths("BDMAP_00000338")
    cases.append({
        "case_id": "BDMAP_00000338",
        "title": "BDMAP_00000338 (AbdomenAtlas Benchmark Sample)",
        "is_demo": True,
        "has_segmentation": demo_paths["combined_labels_path"] is not None,
        "has_groundtruth": demo_paths["gt_path"] is not None,
    })

    # 2. Uploaded cases
    if os.path.exists(UPLOADS_DIR):
        for item in sorted(os.listdir(UPLOADS_DIR)):
            p = os.path.join(UPLOADS_DIR, item)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "ct.nii.gz")):
                has_seg = os.path.exists(os.path.join(p, "combined_labels.nii.gz"))
                cases.append({
                    "case_id": item,
                    "title": f"Upload: {item}",
                    "is_demo": False,
                    "has_segmentation": has_seg,
                    "has_groundtruth": False,
                })
    return {"cases": cases}


@app.get("/api/case/{case_id}/meta")
def get_case_meta(case_id: str):
    """Retrieves geometry, intensity statistics, and segmented organ metrics."""
    paths = load_case_into_cache(case_id)
    meta = dict(ie.volume_cache.metadata)

    # Compute organ statistics if mask is available
    if ie.volume_cache.mask_data is not None:
        # Ground truth is ONLY used if this case ships a dedicated, independent
        # ground-truth file distinct from the prediction file. There is no
        # fallback to combined_labels itself -- comparing predictions against
        # themselves would trivially always score Dice = 1.0.
        gt_mask = None
        if paths["gt_path"] and paths["gt_path"] != paths["combined_labels_path"]:
            gt_mask = nib.load(paths["gt_path"]).get_fdata().astype(np.uint8)

        organs = ie.calculate_organ_statistics(
            ie.volume_cache.mask_data,
            meta["voxel_volume_ml"],
            gt_mask_data=gt_mask
        )
        meta["organs"] = organs
        meta["has_segmentation"] = True
        meta["has_groundtruth_comparison"] = gt_mask is not None

        # Mean Dice
        dice_scores = [o["dice"] for o in organs if "dice" in o and o["present"]]
        meta["mean_dice"] = round(float(np.mean(dice_scores)), 4) if dice_scores else None
    else:
        meta["organs"] = [
            {"id": k, "name": v["name"], "label": v["label"], "color": v["color"], "present": False}
            for k, v in ie.TARGET_ORGANS.items()
        ]
        meta["has_segmentation"] = False
        meta["has_groundtruth_comparison"] = False
        meta["mean_dice"] = None

    # Honest provenance: how and when the CURRENT prediction was actually produced.
    # None if no inference has been run yet for this case.
    meta["inference_info"] = ie.load_inference_meta(paths["case_dir"])
    meta["presets"] = ie.WINDOW_PRESETS
    return meta


@app.get("/api/case/{case_id}/slice/{slice_idx}")
def get_slice_image(
    case_id: str,
    slice_idx: int,
    orientation: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    window: str = "abdomen",
    opacity: float = Query(0.5, ge=0.0, le=1.0),
    organs: Optional[str] = None,
):
    """Renders a 2D CT slice with color organ overlays."""
    load_case_into_cache(case_id)
    
    # Distinguish "no filter given at all" (organs is None -> show every organ,
    # the default) from "filter given but empty" (organs == "" -> show none).
    # Treating both the same way was a real bug: toggling every organ off in the
    # UI sent organs="", which a truthiness check silently ignored, so it kept
    # rendering all organs instead of none.
    active_organs = None
    if organs is not None:
        try:
            active_organs = [int(x.strip()) for x in organs.split(",") if x.strip()]
        except ValueError:
            active_organs = []

    png_bytes = ie.render_slice_png(
        ct_data=ie.volume_cache.ct_data,
        mask_data=ie.volume_cache.mask_data,
        slice_idx=slice_idx,
        orientation=orientation,
        window_preset=window,
        active_organ_ids=active_organs,
        opacity=opacity,
    )
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/upload")
async def upload_ct_scan(file: UploadFile = File(...)):
    """Uploads a new NIfTI CT volume."""
    if not file.filename.endswith((".nii.gz", ".nii")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .nii.gz or .nii file.")

    case_id = f"case_{int(time.time())}"
    case_dir = os.path.join(UPLOADS_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)
    target_ct = os.path.join(case_dir, "ct.nii.gz")

    with open(target_ct, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Validate file integrity and basic CT-volume shape with nibabel.
    try:
        data, affine, meta = ie.load_ct_volume(target_ct)
    except Exception as e:
        shutil.rmtree(case_dir)
        raise HTTPException(status_code=400, detail=f"Failed to parse NIfTI volume: {str(e)}")

    if data.ndim != 3:
        shutil.rmtree(case_dir)
        raise HTTPException(
            status_code=400,
            detail=f"Expected a 3D CT volume, but this file has {data.ndim} dimensions "
                   f"(shape {list(data.shape)}). This viewer only supports single-channel 3D volumes.",
        )

    MIN_DIM = 8
    if any(s < MIN_DIM for s in data.shape):
        shutil.rmtree(case_dir)
        raise HTTPException(
            status_code=400,
            detail=f"Volume shape {list(data.shape)} is too small to be a plausible CT scan "
                   f"(each dimension must be at least {MIN_DIM} voxels).",
        )

    return {
        "status": "success",
        "case_id": case_id,
        "filename": file.filename,
        "metadata": meta,
    }


@app.post("/api/case/{case_id}/infer")
def trigger_inference(case_id: str, confirm_slow: bool = False):
    """
    Runs REAL SuPreM sliding-window inference on the selected case, on whatever
    device this machine actually has (CUDA GPU > Apple Silicon MPS > CPU).
    There is no simulated/instant mode: this always executes the model.
    """
    paths = get_case_paths(case_id)
    device = ie.get_device()
    accelerated = device.type in ("cuda", "mps")

    if not accelerated and not confirm_slow:
        return {
            "status": "confirm_required",
            "device": str(device),
            "device_label": ie.device_label(device),
            "message": (
                "No GPU (CUDA) or Apple Silicon (MPS) acceleration was detected on this "
                "machine. Full 3D sliding-window inference on CPU can take a long time "
                "depending on volume size. You can run it anyway, or use the bundled "
                "Colab notebook for GPU-speed inference instead."
            ),
        }

    if not ie.MONAI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="MONAI / SuPreM model code failed to import in this environment. "
                   "Check that the 'SuPreM' directory and 'monai' package are installed.",
        )
    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Model checkpoint not found at '{CHECKPOINT_PATH}'. Download "
                   f"'supervised_suprem_unet_2100.pth' from Hugging Face before running inference.",
        )

    try:
        res = ie.run_live_inference(
            ct_path=paths["ct_path"],
            checkpoint_path=CHECKPOINT_PATH,
            output_dir=paths["case_dir"],
            device=device,
            overlap=0.5,
        )
        ie.volume_cache.cached_case_id = None
        load_case_into_cache(case_id)
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed partway through: {str(e)}")


@app.get("/api/case/{case_id}/download/{file_format}")
def download_results(case_id: str, file_format: str):
    """Downloads segmentation results as combined NIfTI or multi-organ ZIP."""
    paths = get_case_paths(case_id)
    if not paths["combined_labels_path"]:
        raise HTTPException(status_code=400, detail="Segmentation results not yet generated for this case.")

    if file_format == "nii":
        return FileResponse(
            paths["combined_labels_path"],
            media_type="application/gzip",
            filename=f"{case_id}_combined_labels.nii.gz",
        )
    elif file_format == "zip":
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(paths["combined_labels_path"], arcname="combined_labels.nii.gz")
            if paths["seg_dir"] and os.path.exists(paths["seg_dir"]):
                for mask_file in os.listdir(paths["seg_dir"]):
                    if mask_file.endswith(".nii.gz"):
                        zip_file.write(
                            os.path.join(paths["seg_dir"], mask_file),
                            arcname=f"segmentations/{mask_file}",
                        )
        zip_buffer.seek(0)
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={case_id}_segmentations.zip"},
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Choose 'nii' or 'zip'.")


# Serve frontend static assets
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting BodyMaps SuPreM Segmentation Web Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
