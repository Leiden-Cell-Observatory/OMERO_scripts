#!/bin/bash
# Launch script for ND2 to TIFF Converter GUI on macOS/Linux
# This script uses uv to run the GUI without needing a full installation

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run the GUI using uv
cd "$SCRIPT_DIR"
uvx --from . nd2-converter-gui
