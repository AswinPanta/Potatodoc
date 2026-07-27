#!/bin/bash
# Script to prepare model files for Google Drive upload
# Run this script from the project root directory

set -e

echo "=== Preparing Model Files for Google Drive Upload ==="
echo ""

# Create a temporary directory for zipped models
TEMP_DIR="/tmp/potato_models_upload"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

zip_dir() {
    local src="$1"
    local dest="$2"
    local label="$3"
    if [ -d "$src" ]; then
        echo "Zipping $label..."
        (cd "$src" && zip -r "$TEMP_DIR/$dest" .)
        echo "  Created $dest"
    else
        echo "  WARNING: $src not found — skipping $label"
    fi
}

zip_dir "saved_models/1" "cnn_baseline_model.zip" "CNN Baseline"
zip_dir "saved_models/2" "transfer_learning_model.zip" "Transfer Learning"
zip_dir "saved_models/3" "mobilenetv2_model.zip" "MobileNetV2"
zip_dir "tf-lite-models" "tflite_models.zip" "TFLite Models"

echo ""
echo "=== Upload Instructions ==="
echo ""
echo "1. Go to Google Drive folder:"
echo "   https://drive.google.com/drive/folders/1EPPULrhp3P6Hcoma5rYqoJZxytSBmni4?usp=sharing"
echo ""
echo "2. Upload the following files from $TEMP_DIR:"
ls -lh "$TEMP_DIR"/*.zip 2>/dev/null || echo "  No files generated"
echo ""
echo "3. After upload, right-click each file → Share → Copy link"
echo "   (Make sure 'Anyone with the link' is selected)"
echo ""
echo "4. Update file IDs in backend/model_downloader.py"
