@echo off
REM Launch script for ND2 to TIFF Converter GUI on Windows
REM Run by double-clicking this file or running: launch_gui.bat

cd /d "%~dp0\.."
pixi run nd2-gui
pause
