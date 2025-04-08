Here's a concise documentation section for your README.md file that explains the script and provides usage examples:

```markdown
# ND2 to TIFF Converter

A Python script for extracting multi-position Nikon ND2 files from plate-based microscopy into individual TIFF files. The script supports extraction of well positions, z-stacks, and channels with multiple output options.

## Features

- Extracts well positions from ND2 files into separate TIFF files
- Preserves metadata (physical pixel size, channel info, z-spacing)
- Supports saving channels separately
- Supports saving z-slices separately
- Creates maximum intensity projections
- Can handle missing position names by guessing based on available metadata

## Usage

Basic usage:
```bash
python convert_nd2_screen_to_stacks.py path/to/file.nd2
```

### Command Line Options

- `--skip-ome`: Skip generation of OME metadata (makes the script much faster)
- `--separate-channels`: Save each channel as a separate file
- `--separate-z`: Save each z-slice as a separate file
- `--guess-names`: Attempt to guess missing position names
- `--max-projection`: Create maximum intensity projections along the Z axis

### Examples

Extract all positions with default settings:
```bash
python convert_nd2_screen_to_stacks.py "path/to/file.nd2"
```

Save channels separately:
```bash
python convert_nd2_screen_to_stacks.py "path/to/file.nd2" --separate-channels
```

Create max projections and guess missing position names:
```bash
python convert_nd2_screen_to_stacks.py "path/to/file.nd2" --max-projection --guess-names
```

Save each z-slice separately with channels separated:
```bash
python convert_nd2_screen_to_stacks.py "path/to/file.nd2" --separate-z --separate-channels
```

Interactive mode (will prompt for file path):
```bash
python convert_nd2_screen_to_stacks.py --max-projection
```

## Dependencies

- nd2 (https://github.com/tlambert03/nd2/)
- tifffile
- numpy

## Output Structure

Files are saved in an 'export' subdirectory in the same folder as the input file. 
Naming follows the pattern:
- Standard: `filename_position.ome.tif`
- Separate channels: `filename_position_ch1.ome.tif`, `filename_position_ch2.ome.tif`...
- Separate z-slices: `filename_position_z001.ome.tif`, `filename_position_z002.ome.tif`...
- Max projections: `filename_position_max.ome.tif` or `filename_position_ch1_max.tif`...
```

This documentation provides a clear overview of the script's capabilities along with practical examples that cover the main use cases. The examples demonstrate the various command-line options and their combinations.