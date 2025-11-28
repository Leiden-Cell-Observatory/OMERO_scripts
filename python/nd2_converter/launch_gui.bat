@echo off
REM Launch script for ND2 to TIFF Converter GUI on Windows
REM This script uses uv to run the GUI without needing a full installation

cd /d "%~dp0"
uvx --from . nd2-converter-gui
pause
