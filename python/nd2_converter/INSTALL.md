# Installation Guide

## Quick Start (No Installation Required!)

The easiest way to use the ND2 Converter is with the provided launch scripts - no installation needed!

### Prerequisites

Install `uv` (one-time setup):

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Running the GUI

1. **Navigate to the nd2_converter folder**
   ```bash
   cd python/nd2_converter
   ```

2. **Run the launch script:**
   - **Windows**: Double-click `launch_gui.bat`
   - **macOS/Linux**:
     ```bash
     ./launch_gui.sh
     ```
     Or double-click `launch_gui.sh` in your file manager

That's it! The first time you run it, `uv` will automatically download and install all dependencies in an isolated environment. Subsequent launches will be much faster.

## Alternative Installation Methods

### Method 1: Install Globally with uv

```bash
cd python/nd2_converter
uv pip install .
```

After installation, you can run from anywhere:
```bash
nd2-converter-gui        # Launch GUI
nd2-converter --help     # CLI help
```

### Method 2: Development Installation

If you want to modify the code:

```bash
cd python/nd2_converter
uv pip install -e ".[dev]"
```

This creates an editable installation - changes to the code are immediately reflected.

### Method 3: Run Directly with uvx (No Clone Needed!)

You can run the tool directly from GitHub without cloning:

```bash
uvx --from git+https://github.com/Leiden-Cell-Observatory/OMERO_scripts#subdirectory=python/nd2_converter nd2-converter-gui
```

## Updating

### If using launch scripts:
Just run the script again - `uv` will automatically update dependencies if needed.

### If installed globally:
```bash
cd python/nd2_converter
uv pip install --upgrade .
```

### If installed from GitHub:
```bash
uv pip install --upgrade git+https://github.com/Leiden-Cell-Observatory/OMERO_scripts#subdirectory=python/nd2_converter
```

## Uninstalling

```bash
uv pip uninstall nd2-converter
```

Note: This doesn't remove `uv` itself - only the nd2-converter package.

## Troubleshooting

### "uv: command not found"
- Make sure you installed `uv` following the prerequisites above
- You may need to restart your terminal after installation
- On macOS/Linux, check that `~/.cargo/bin` is in your PATH

### Permission denied on launch script (macOS/Linux)
```bash
chmod +x launch_gui.sh
```

### Dependencies not installing
```bash
# Clear cache and try again
uv cache clean
```

### Python version issues
The tool requires Python 3.10 or later. Check your version:
```bash
python --version
```

If you have multiple Python versions, uv will automatically use a compatible one.

## Why uv?

- **Fast**: 10-100x faster than pip
- **Reliable**: Deterministic dependency resolution
- **Simple**: Single binary, no conda needed
- **Isolated**: Each project gets its own environment
- **Cross-platform**: Works the same on Windows, macOS, and Linux

## Comparison with pixi

The main OMERO scripts still use `pixi` because they require conda packages (OMERO dependencies). The ND2 converter has been separated out because it:
- Doesn't need OMERO
- All dependencies are available on PyPI
- Can be distributed more easily with uv

This makes it much simpler for end users who just want to convert ND2 files!
