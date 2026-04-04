#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== AdStream Plugin Dependency Installer ==="
echo ""

PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using: $PYTHON_VERSION"
echo ""

install_pkg() {
    local pkg="$1"
    local required="${2:-true}"
    echo "  Installing $pkg..."
    if $PYTHON_CMD -m pip install "$pkg" 2>&1; then
        echo "  ✓ $pkg installed"
    else
        if [ "$required" = "true" ]; then
            echo "  ✗ FAILED to install $pkg (required)"
            exit 1
        else
            echo "  ⚠ $pkg skipped (optional — feature will be disabled)"
        fi
    fi
}

echo "Installing core dependencies..."
install_pkg "ultralytics>=8.0.0" true
install_pkg "opencv-python>=4.8.0" true
install_pkg "numpy>=1.24.0" true

echo ""
echo "Installing optional dependencies..."
install_pkg "mss>=9.0.0" false

echo ""
echo "Downloading YOLOv8n model (if not cached)..."
$PYTHON_CMD -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "⚠ Model download failed — will retry on first run"

echo ""
echo "=== Installation complete ==="
echo ""
echo "To use in OBS:"
echo "  1. Open OBS Studio"
echo "  2. Go to Tools -> Scripts"
echo "  3. Click '+' and select: $SCRIPT_DIR/ad_stream_plugin.py"
echo "  4. Configure settings in the script properties panel"
echo "  5. Place ad images in: $SCRIPT_DIR/assets/"
echo "  6. Click 'Start Processing'"
echo ""
echo "Replace Mode streams to: http://localhost:8765/stream"
echo "  Add a Media Source in OBS pointing to that URL."
