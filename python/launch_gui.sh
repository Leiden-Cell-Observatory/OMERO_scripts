#!/bin/bash
# Launch script for ND2 to TIFF Converter GUI on macOS/Linux
# Run with: ./launch_gui.sh

cd "$(dirname "$0")/.."
pixi run nd2-gui
