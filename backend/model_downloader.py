"""
Model downloader for Google Drive.

Downloads model files from a shared Google Drive folder on first startup,
then caches them locally. Subsequent startups use the cached files.

Google Drive Folder: https://drive.google.com/drive/folders/1EPPULrhp3P6Hcoma5rYqoJZxytSBmni4
"""

import os
import sys
import logging
import zipfile
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------- Google Drive File IDs ----------
# These IDs correspond to files in the shared Google Drive folder.
# If you upload new models, update these IDs accordingly.

GDRIVE_FILES = {
    # CNN Baseline model (SavedModel format)
    "cnn_baseline": {
        "file_id": "1aChFCAlnch6WJ6iQP3G8yUnKzXgFlsiP",
        "filename": "cnn_baseline_model.zip",
        "extract_to": "saved_models/1",
        "is_zip": True,
    },
    # Transfer Learning model (SavedModel + H5 weights)
    "transfer_learning": {
        "file_id": "13TOJFDotO_7aUwyBauqs20Fhn-mvurAh",
        "filename": "transfer_learning_model.zip",
        "extract_to": "saved_models/2",
        "is_zip": True,
    },
    # MobileNetV2 model (new trained H5)
    "mobilenetv2": {
        "file_id": "1XRfqEWrkUF9hf3ATTJXmlQ35BzLJjuV0",
        "filename": "mobilenetv2.h5",
        "extract_to": "saved_models/3/model.h5",
        "is_zip": False,
    },
}

# Expected directory structure after download:
# saved_models/
#   1/              <- CNN Baseline (extracted from zip)
#   2/              <- Transfer Learning (extracted from zip)
#     model.h5
#     saved_model.pb
#     ...
#   3/
#     model.h5      <- MobileNetV2 (direct .h5 file)


def _get_base_dir():
    """Get the project base directory."""
    return Path(__file__).resolve().parent.parent


def _download_file(file_id: str, destination: Path) -> bool:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False, fuzzy=True)
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
    
    Returns a dict of {model_name: success_bool}.
    """
    if base_dir is None:
        base_dir = _get_base_dir()
    
    models_dir = base_dir / "saved_models"
    cache_dir = base_dir / ".model_cache"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_name, config in GDRIVE_FILES.items():
        dest_path = base_dir / config["extract_to"]
        
        # Check if model already exists locally
        if config["is_zip"]:
            # For zips, check if the extract directory has files
            if dest_path.exists() and any(dest_path.iterdir()):
                logger.info(f"Model '{model_name}' already exists at {dest_path}")
                results[model_name] = True
                continue
        else:
            # For direct files (like .h5), check if file exists
            if dest_path.exists() and dest_path.stat().st_size > 1000:
                logger.info(f"Model '{model_name}' already exists at {dest_path}")
                results[model_name] = True
                continue
        
        logger.info(f"Downloading model '{model_name}' from Google Drive...")
        zip_path = cache_dir / config["filename"]
        
        # Download
        if not _download_file(config["file_id"], zip_path):
            logger.error(f"Failed to download {config['filename']}")
            results[model_name] = False
            continue
        
        # Extract if zip
        if config["is_zip"]:
            if _extract_zip(zip_path, dest_path.parent):
                logger.info(f"Model '{model_name}' extracted to {dest_path.parent}")
                results[model_name] = True
            else:
                results[model_name] = False
        else:
            # Direct file (e.g., .h5) - copy to destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, dest_path)
            logger.info(f"Model '{model_name}' copied to {dest_path}")
            results[model_name] = True
    
    return results


def verify_models(base_dir: Path = None) -> dict:
    """
    Verify that all required model files exist locally.
    
    Returns a dict of {model_name: exists_bool}.
    """
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
            logger.info(f"✓ Model '{name}' found at {path}")
        else:
            logger.warning(f"✗ Model '{name}' NOT found at {path}")
    
    return results


if __name__ == "__main__":
    # Run as standalone script to download models
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    print("=== Downloading Models from Google Drive ===\n")
    results = download_models()
    
    print("\n=== Verification ===\n")
    verification = verify_models()
    
    print("\n=== Summary ===\n")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {'Downloaded' if success else 'FAILED'}")
    
    for name, exists in verification.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {'Found' if exists else 'MISSING'}")
