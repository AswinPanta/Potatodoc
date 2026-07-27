"""
Model downloader for Google Drive.

Downloads model files from a shared Google Drive folder on first startup,
then caches them locally. Subsequent startups use the cached files.

Google Drive Folder: https://drive.google.com/drive/folders/1EPPULrhp3P6Hcoma5rYqoJZxytSBmni4

Progress tracking is exposed via get_download_status() for the /setup/status endpoint.
"""

import os
import sys
import logging
import zipfile
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------- Google Drive File IDs ----------
GDRIVE_FILES = {
    "cnn_baseline": {
        "file_id": "1aChFCAlnch6WJ6iQP3G8yUnKzXgFlsiP",
        "filename": "cnn_baseline_model.zip",
        "extract_to": "saved_models/1",
        "is_zip": True,
        "display_name": "CNN Baseline",
        "size_estimate_mb": 2.5,
    },
    "transfer_learning": {
        "file_id": "13TOJFDotO_7aUwyBauqs20Fhn-mvurAh",
        "filename": "transfer_learning_model.zip",
        "extract_to": "saved_models/2",
        "is_zip": True,
        "display_name": "Transfer Learning",
        "size_estimate_mb": 227,
    },
    "mobilenetv2": {
        "file_id": "1XRfqEWrkUF9hf3ATTJXmlQ35BzLJjuV0",
        "filename": "mobilenetv2.h5",
        "extract_to": "saved_models/3/model.h5",
        "is_zip": False,
        "display_name": "MobileNetV2",
        "size_estimate_mb": 25,
    },
}

_MARKER_FILE = ".models_downloaded"

# ---------- Download Status Tracking ----------
_download_lock = threading.Lock()
_download_status = {
    "state": "idle",          # idle | downloading | extracting | complete | error
    "current_model": None,     # key from GDRIVE_FILES
    "current_model_name": "",  # display name
    "progress": 0,             # 0-100 per-model progress
    "models": {},              # {model_name: "pending"|"downloading"|"extracting"|"done"|"error"}
    "overall_progress": 0,     # 0-100 overall
    "message": "",
    "error": None,
}

# Initialize per-model status
for _k in GDRIVE_FILES:
    _download_status["models"][_k] = "pending"


def get_download_status() -> dict:
    """Return a deep copy of the current download status (thread-safe)."""
    with _download_lock:
        return {
            **_download_status,
            "models": dict(_download_status["models"]),
        }


def _update_status(**kwargs):
    """Update download status (thread-safe)."""
    with _download_lock:
        _download_status.update(kwargs)


def _get_base_dir():
    """Get the project base directory."""
    return Path(__file__).resolve().parent.parent


def _is_downloaded(base_dir: Path) -> bool:
    """Check if models have already been downloaded."""
    marker = base_dir / _MARKER_FILE
    return marker.exists()


def _mark_downloaded(base_dir: Path):
    """Create marker file after successful download."""
    marker = base_dir / _MARKER_FILE
    marker.write_text("Models downloaded from Google Drive")


def _download_file(file_id: str, destination: Path, model_name: str) -> bool:
    """Download a file from Google Drive using gdown with progress tracking."""
    try:
        import gdown

        def _progress_hook(current, total):
            if total and total > 0:
                pct = min(int(current * 100 / total), 100)
            else:
                pct = 0
            _update_status(progress=pct)

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=True,
                       callback=_progress_hook)
        return destination.exists()
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        return False
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False


def _extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a zip file to a directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted {zip_path.name} to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_models(base_dir: Path = None) -> dict:
    """
    Download all model files from Google Drive.
    Updates _download_status as it progresses.
    """
    if base_dir is None:
        base_dir = _get_base_dir()

    models_dir = base_dir / "saved_models"
    cache_dir = base_dir / ".model_cache"

    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check marker file — skip if already downloaded
    if _is_downloaded(base_dir):
        logger.info("Models already downloaded — skipping")
        _update_status(state="complete", overall_progress=100,
                       message="Models ready!", progress=100)
        for k in _download_status["models"]:
            _download_status["models"][k] = "done"
        return {name: True for name in GDRIVE_FILES}

    # Start download process
    total_models = len(GDRIVE_FILES)
    _update_status(state="downloading", message="Starting downloads...",
                   overall_progress=0)

    results = {}
    completed = 0

    for model_name, config in GDRIVE_FILES.items():
        dest_path = base_dir / config["extract_to"]

        # Check if already exists locally
        if config["is_zip"]:
            if dest_path.exists() and any(dest_path.iterdir()):
                logger.info(f"Model '{model_name}' already exists")
                results[model_name] = True
                _download_status["models"][model_name] = "done"
                completed += 1
                _update_status(
                    overall_progress=int(completed * 100 / total_models),
                    message=f"{config['display_name']} — already cached",
                )
                continue
        else:
            if dest_path.exists() and dest_path.stat().st_size > 1000:
                logger.info(f"Model '{model_name}' already exists")
                results[model_name] = True
                _download_status["models"][model_name] = "done"
                completed += 1
                _update_status(
                    overall_progress=int(completed * 100 / total_models),
                    message=f"{config['display_name']} — already cached",
                )
                continue

        # Download this model
        _update_status(
            state="downloading",
            current_model=model_name,
            current_model_name=config["display_name"],
            progress=0,
            message=f"Downloading {config['display_name']} ({config['size_estimate_mb']:.0f} MB)...",
        )
        _download_status["models"][model_name] = "downloading"

        zip_path = cache_dir / config["filename"]

        if not _download_file(config["file_id"], zip_path, model_name):
            logger.error(f"Failed to download {config['filename']}")
            results[model_name] = False
            _download_status["models"][model_name] = "error"
            _update_status(state="error", error=f"Failed to download {config['display_name']}")
            continue

        # Extract / copy
        _update_status(
            state="extracting",
            progress=100,
            message=f"Extracting {config['display_name']}...",
        )
        _download_status["models"][model_name] = "extracting"

        if config["is_zip"]:
            dest_path.mkdir(parents=True, exist_ok=True)
            if _extract_zip(zip_path, dest_path):
                results[model_name] = True
                _download_status["models"][model_name] = "done"
            else:
                results[model_name] = False
                _download_status["models"][model_name] = "error"
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, dest_path)
            results[model_name] = True
            _download_status["models"][model_name] = "done"

        completed += 1
        _update_status(
            overall_progress=int(completed * 100 / total_models),
        )

    # Mark as downloaded if all succeeded
    if results and all(results.values()):
        _mark_downloaded(base_dir)
        _update_status(
            state="complete",
            overall_progress=100,
            progress=100,
            message="All models ready!",
        )
        logger.info("All models downloaded — created marker file")
    else:
        failed = [k for k, v in results.items() if not v]
        _update_status(
            state="error",
            error=f"Failed to download: {', '.join(failed)}",
        )

    return results


def verify_models(base_dir: Path = None) -> dict:
    """Verify that all required model files exist locally."""
    if base_dir is None:
        base_dir = _get_base_dir()

    models_dir = base_dir / "saved_models"

    checks = {
        "cnn-baseline": models_dir / "1" / "saved_model.pb",
        "transfer-learning": models_dir / "2" / "model.h5",
        "mobilenetv2": models_dir / "3" / "model.h5",
    }

    results = {}
    for name, path in checks.items():
        exists = path.exists() and path.stat().st_size > 1000
        results[name] = exists
        if exists:
            logger.info(f"Model '{name}' found at {path}")
        else:
            logger.warning(f"Model '{name}' NOT found at {path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    print("=== Downloading Models from Google Drive ===\n")
    results = download_models()
    print("\n=== Verification ===\n")
    verification = verify_models()
    print("\n=== Summary ===\n")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")
    for name, exists in verification.items():
        status = "FOUND" if exists else "MISSING"
        print(f"  [{status}] {name}")
