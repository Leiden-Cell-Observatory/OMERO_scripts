# ND2 to TIFF Converter

A modern, cross-platform GUI and CLI tool for converting multi-position ND2 files to TIFF files.

![Platform Support](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

## Features

- **Modern GUI** built with CustomTkinter
  - Dark/Light mode support
  - Customizable UI scaling (80%-300%)
  - Real-time progress logging with color-coded output
  - Responsive, scrollable interface
- **Powerful CLI** for automation and scripting
- **Cross-platform** - Works on Windows, macOS, and Linux
- **Rich conversion options**:
  - OME-TIFF metadata support
  - Separate channels, Z-slices, or time points
  - Maximum intensity projections
  - Custom output folders and file prefixes
  - Auto-detection of missing position names

## Installation

### Quick Start (Recommended)

The easiest way to use the converter is with `uv` - no installation required!

#### Install uv (one-time setup)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Run the GUI

Just double-click the launch script:
- **Windows**: Double-click `launch_gui.bat`
- **macOS/Linux**: Double-click `launch_gui.sh` (or run `./launch_gui.sh`)

Or run directly from command line:
```bash
uvx --from . nd2-converter-gui
```

### Full Installation (Optional)

If you want to install the tool system-wide:

```bash
# Install with uv
uv pip install .

# Or install in development mode
uv pip install -e .
```

After installation, you can run:
```bash
nd2-converter-gui        # Launch GUI
nd2-converter --help     # CLI help
```

### Installation from GitHub

```bash
# Run GUI directly from GitHub (no clone needed!)
uvx --from git+https://github.com/Leiden-Cell-Observatory/OMERO_scripts#subdirectory=python/nd2_converter nd2-converter-gui

# Or install from GitHub
uv pip install git+https://github.com/Leiden-Cell-Observatory/OMERO_scripts#subdirectory=python/nd2_converter
```

## Using the GUI

### Interface Overview

1. **Appearance Settings** (top of window):
   - **Appearance**: System (auto), Dark, or Light mode
   - **UI Scale**: 80%-300% for different screen sizes and DPI settings
     - Default: 100% on Linux, 200% on Windows/macOS
     - For 4K displays: Use 200%-300%
     - For standard displays: Use 100%-150%

2. **Input File**: Select your ND2 file

3. **Output Settings** (optional):
   - **Export Folder**: Custom output location (default: `export` subfolder)
   - **File Prefix**: Add prefix to output filenames (e.g., `experiment_001_`)

4. **Conversion Options**:
   - **Skip OME metadata**: Create standard TIFF instead of OME-TIFF
   - **Separate channels**: Save each channel as separate file
   - **Separate Z-slices**: Save each z-slice separately
   - **Separate time points**: Save each time point as separate OME-TIFF
   - **Guess missing position names**: Auto-detect position names
   - **Max projection only**: Create only maximum intensity projections

5. **Progress Log**: Monitor conversion progress in real-time

### Example Workflow

1. Click **Browse** next to "ND2 File" and select your file
2. Optionally customize the export folder and file prefix
3. Select desired conversion options
4. Click **Convert**
5. Watch the progress log - you'll get a notification when complete!

## Using the CLI

### Basic Usage

```bash
# Convert with default settings
nd2-converter input.nd2

# Interactive mode (prompts for file)
nd2-converter

# With options
nd2-converter input.nd2 --separate-channels --max-projection

# Custom output location
nd2-converter input.nd2 --export-folder /path/to/output

# Add file prefix
nd2-converter input.nd2 --file-prefix "experiment_001_"
```

### CLI Options

```
positional arguments:
  nd2_file              Path to the ND2 file

optional arguments:
  -h, --help            Show this help message and exit
  --skip-ome            Skip generation of OME metadata
  --separate-channels   Save each channel as a separate file
  --separate-z          Save each z-slice as a separate file
  --separate-t          Save each time point as a separate file
  --guess-names         Attempt to guess missing position names
  --max-projection      Create only maximum intensity projections
  --export-folder PATH  Custom export folder path
  --file-prefix PREFIX  Prefix to add to output filenames
```

### CLI Examples

```bash
# Create max projections with separate channels
nd2-converter data.nd2 --max-projection --separate-channels

# Time-series data: separate time points with custom prefix
nd2-converter timeseries.nd2 --separate-t --file-prefix "ts_day7_"

# Full z-stacks, separate channels, custom output
nd2-converter screen.nd2 --separate-channels --separate-z --export-folder ./results
```

## Conversion Options Explained

### Skip OME metadata
Creates standard multi-page TIFF files instead of OME-TIFF. Useful for compatibility with older software that doesn't support OME-TIFF.

### Separate channels
Each fluorescence channel is saved as a separate TIFF file. Useful when processing channels independently.

### Separate Z-slices
Each z-position is saved as a separate file. Useful for 2D analysis of individual slices.

### Separate time points
For time-series data, each time point is saved as a separate OME-TIFF file instead of one file containing all time points.

### Guess missing position names
Some ND2 files have incomplete position metadata. This option attempts to automatically determine position names (e.g., well positions like "A1", "B2").

### Max projection only
Creates maximum intensity projections along the Z-axis. When enabled, only the projections are saved, not the original Z-stacks. Only works when multiple Z-slices are present.

## Dependencies

All dependencies are automatically handled by `uv`:

- **nd2** (>=0.10.4) - Reading ND2 files
- **tifffile** (>=2024.0.0) - Writing TIFF files
- **customtkinter** (>=5.2.0) - Modern GUI framework
- **numpy** (>=1.24.0) - Array operations

No OMERO dependencies required! This is a standalone tool.

## Troubleshooting

### "uv: command not found"
Install uv following the installation instructions above.

### GUI doesn't start
Make sure you have Python 3.10 or later:
```bash
python --version
```

### "Module not found" error
If you installed the package, try reinstalling:
```bash
uv pip install --force-reinstall .
```

### GUI display issues on Linux
- The GUI requires a display server (X11 or Wayland)
- On headless servers, use the CLI version instead
- Try adjusting the UI scaling if elements appear too large/small

### Launch script permission denied (macOS/Linux)
```bash
chmod +x launch_gui.sh
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/your-username/OMERO_scripts.git
cd OMERO_scripts/python/nd2_converter

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

### Code formatting

```bash
# Format code
black nd2_converter/

# Lint code
ruff check nd2_converter/
```

## Technical Details

### Output Structure

By default, files are saved to an `export` folder next to your input file:

```
input_file.nd2
export/
  ├── position1_stack.ome.tiff
  ├── position2_stack.ome.tiff
  └── ...
```

With `--separate-channels`:
```
export/
  ├── position1_C0.ome.tiff
  ├── position1_C1.ome.tiff
  ├── position2_C0.ome.tiff
  └── ...
```

### Metadata Preservation

OME-TIFF files preserve:
- Channel names and wavelengths
- Z-slice positions
- Time point information
- Physical pixel sizes
- All original ND2 metadata in OME-XML format

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions:
1. Check the progress log in the GUI for detailed error messages
2. Try the CLI version to get more detailed output
3. Open an issue on GitHub with the error message and your ND2 file metadata

## Acknowledgments

Built using the excellent [nd2](https://github.com/tlambert03/nd2/) library by Talley Lambert.
