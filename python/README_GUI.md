# ND2 to TIFF Converter GUI

A cross-platform graphical user interface for converting multi-position ND2 files to TIFF files.

## Features

- **Modern, sleek interface** built with CustomTkinter
- **Dark/Light mode support** - Choose from System, Dark, or Light appearance
- **Customizable UI scaling** - Scale from 80% to 150% for different screen sizes and DPI settings
- **Cross-platform support** - Works on Windows, macOS, and Linux
- **Real-time progress logging** with color-coded output
- **All conversion options** from the command-line version
- **Automatic output folder selection**
- **Responsive design** with scrollable interface

## Prerequisites

You need to have [pixi](https://pixi.sh) installed on your system.

## Installation

1. Navigate to the project directory:
   ```bash
   cd /path/to/OMERO_scripts
   ```

2. Install dependencies using pixi:
   ```bash
   pixi install
   ```

## Running the GUI

### Option 1: Using pixi (Recommended - works on all platforms)

From the project root directory:
```bash
pixi run nd2-gui
```

### Option 2: Using platform-specific launch scripts

#### macOS/Linux:
```bash
cd python
./launch_gui.sh
```

Or double-click `launch_gui.sh` in your file manager (if configured to run shell scripts).

#### Windows:
Double-click `launch_gui.bat` in the `python` folder, or run from command prompt:
```cmd
cd python
launch_gui.bat
```

## Using the GUI

### Appearance Settings

At the top of the window, you can customize the interface:
- **Appearance**: Choose between System (follows OS theme), Dark, or Light mode
- **UI Scale**: Adjust the interface size from 80% to 150% for optimal viewing on different displays
  - **Default**: 120% for better readability on most displays
  - The GUI starts with a larger window (1100x900) for comfortable viewing

### Converting Files

1. **Select Input File**: Click "Browse..." next to "ND2 File" to select your input ND2 file
2. **Configure Output** (optional):
   - **Export Folder**: Choose a custom output location (default is `export` subfolder next to input file)
   - **File Prefix**: Add a prefix to all output filenames (e.g., `250314_experiment_`)

3. **Choose Conversion Options**:
   - **Skip OME metadata**: Don't include OME-TIFF metadata
   - **Separate channels**: Save each channel as a separate file
   - **Separate Z-slices**: Save each z-slice as a separate file
   - **Separate time points**: Save each time point as a separate file
   - **Guess missing position names**: Auto-detect position names for files with incomplete metadata
   - **Max projection only**: Create only maximum intensity projections

4. **Convert**: Click the "Convert" button to start processing
5. **Monitor Progress**: Watch the progress log for status updates
6. **Done**: A popup will notify you when conversion is complete

## Conversion Options Explained

### Skip OME metadata
When enabled, output files will be standard TIFF files without OME-TIFF metadata. This can be useful for compatibility with some older software.

### Separate channels
Instead of saving all channels in one file, each channel will be saved as a separate TIFF file. Useful when you need to process channels independently.

### Separate Z-slices
Each z-slice will be saved as a separate file. Useful for 2D analysis of individual slices.

### Separate time points
For time-series data, each time point will be saved as a separate OME-TIFF file instead of one file containing all time points.

### Guess missing position names
Some ND2 files don't have complete position metadata. This option attempts to automatically determine position names based on the available data.

### Max projection only
Creates maximum intensity projections along the Z-axis. When enabled, the regular Z-stacks are not saved, only the max projections. Only works when multiple Z-slices are present.

## Troubleshooting

### "Module not found" error
Make sure you've installed all dependencies:
```bash
pixi install
```

### GUI doesn't start on macOS
Make sure the launch script is executable:
```bash
chmod +x python/launch_gui.sh
```

### Display issues on Linux
Tkinter requires a display. If you're on a headless Linux server, you'll need to use the command-line version instead:
```bash
pixi run python python/convert_nd2_screen_to_stacks.py --help
```

## Command-Line Version

If you prefer the command-line interface or need to automate conversions, the original script is still available:

```bash
pixi run python python/convert_nd2_screen_to_stacks.py <input.nd2> [options]
```

Use `--help` to see all available options:
```bash
pixi run python python/convert_nd2_screen_to_stacks.py --help
```

## Dependencies

- Python 3.10
- nd2 (for reading ND2 files)
- tifffile (for writing TIFF files)
- customtkinter (for the modern GUI interface)
- numpy (dependency of nd2)

All dependencies are managed through pixi and will be installed automatically.

## UI Customization

The GUI supports various customization options:

- **Appearance Modes**:
  - **System**: Automatically follows your operating system's dark/light mode preference
  - **Dark**: Dark theme with reduced eye strain for low-light environments
  - **Light**: Traditional light theme with high contrast

- **UI Scaling**: Perfect for:
  - High-DPI displays (use 120-150%)
  - Low-resolution screens (use 80-90%)
  - Accessibility needs (larger text and controls)
  - Multi-monitor setups with different resolutions

## Support

For issues, please check:
1. The progress log in the GUI for detailed error messages
2. The command-line version to see if it's a GUI-specific issue
3. File permissions on the input file and output directory
