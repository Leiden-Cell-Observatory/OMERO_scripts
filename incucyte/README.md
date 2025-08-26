# Incucyte OME-TIFF Converter

Python tools for converting Incucyte microscopy data to OME-TIFF format for OMERO import and analysis.
Note, in development!

## Overview

The Incucyte OME Generator converts Incucyte microscopy data to standardized OME-TIFF format for analysis and OMERO import. It handles the complex Incucyte directory structure and creates files compatible with bio-imaging platforms.


## Features

- **Single OME-TIFF Output**: Creates one comprehensive OME-TIFF file with complete plate structure
- **Well-Field OME-TIFFs**: Creates separate OME-TIFF files per well and field combination
- **Multi-file Companion**: Generates individual TIFF files with companion OME-XML file
- **Plate Structure**: Preserves well positions, fields, channels, and time series
- **Flexible Output**: Multiple output formats for different workflows

## Environment Setup

This project uses [Pixi](https://pixi.sh/) for dependency management. Pixi automatically manages Python environments and dependencies.

### Prerequisites

1. Install Pixi following the [official installation guide](https://pixi.sh/latest/#installation)

### Activate Environment

```bash
# Navigate to the incucyte directory
cd /path/to/OMERO_scripts/incucyte

# Activate the pixi environment (installs dependencies automatically)
pixi shell

# Or run commands directly with pixi
pixi run python IncucyteOMEGenerator.py --help
```

### Dependencies

The environment includes:
- `tifffile` - TIFF file reading/writing
- `numpy` - Array processing
- `ome-types` - OME metadata handling
- `pathlib` - Path operations

Dependencies are defined in `pixi.toml` and managed automatically by Pixi.

## Scripts

### IncucyteOMEGenerator.py

Main conversion script with multiple output formats.

#### Input Structure

Expects Incucyte export structure:
```
trial_dataset_incucyte/
├── EssenFiles/ScanData/
    └── YYMM/           # Year-Month (e.g., 2505)
        └── DD/         # Day (e.g., 16)
            └── HHMM/   # Time (e.g., 1124)
                └── XXXX/   # Fixed identifier (e.g., 1248)
                    ├── A1-1-C1.tif    # Well-Field-Channel
                    ├── A1-1-C2.tif
                    └── A1-1-Ph.tif    # Phase contrast
```

## Output Formats

The script offers three different output formats to suit various workflows:

### 1. Well-Field OME-TIFFs (🌟 Recommended for Analysis)

Creates individual OME-TIFF files for each well and field combination.

**Structure**: One file per well/field containing all channels and timepoints
**Format**: `WellName_FieldN.ome.tif` (e.g., `A1_Field1.ome.tif`)
**Dimensions**: Time × Channels × Y × X per file

**Best for:**
- Individual well analysis
- Parallel processing of different wells
- Manageable file sizes
- Easy data sharing and archival

```bash
pixi run python IncucyteOMEGenerator.py /path/to/data --well-field-ome-tiffs
```

### 2. Single Plate OME-TIFF

Creates one comprehensive file containing the entire experiment.

**Structure**: All data in one file with multiple image series
**Format**: `incucyte_plate.ome.tif`
**Dimensions**: Each series contains Time × Channels × Y × X

**Best for:**
- Complete experiment overview
- OMERO plate import
- Comprehensive analysis workflows

```bash
pixi run python IncucyteOMEGenerator.py /path/to/data --single-ome-tiff
```

### 3. Multi-File with Companion

Creates individual TIFF files with companion OME-XML metadata file.

**Structure**: One file per well/field/channel/timepoint + companion XML
**Format**: Individual `.tif` files + `companion.companion.ome`

**Best for:**
- Legacy OME workflows
- Maximum compatibility with OME standards

```bash
pixi run python IncucyteOMEGenerator.py /path/to/data
```

## Command Line Reference

```bash
python IncucyteOMEGenerator.py INPUT_DIR [options]
```

### Required Arguments
- `INPUT_DIR` - Path to Incucyte export directory

### Options
- `-o, --output-dir DIR` - Output directory (default: INPUT_DIR/converted)
- `-c, --companion FILE` - Companion file name (default: companion.companion.ome)
- `--well-field-ome-tiffs` - Create separate OME-TIFF files per well and field ⭐
- `--single-ome-tiff` - Create single OME-TIFF with plate structure
- `--simple-format` - Use simple 5D array (with --single-ome-tiff)
- `--skip-conversion` - Skip TIFF conversion, only create companion

### Quick Start Examples

```bash
# Most common: Well-field OME-TIFFs for analysis
pixi run python IncucyteOMEGenerator.py /path/to/data --well-field-ome-tiffs

# Single plate file for OMERO import
pixi run python IncucyteOMEGenerator.py /path/to/data --single-ome-tiff

# Multi-file approach
pixi run python IncucyteOMEGenerator.py /path/to/data
```

## Examples

### Basic Conversion (Recommended)
```bash
pixi shell
python IncucyteOMEGenerator.py /data/incucyte_experiment --well-field-ome-tiffs
```
**Output**: Individual OME-TIFF files per well/field in `well_field_ome_tiffs/` directory

### OMERO Import Preparation
```bash
pixi run python IncucyteOMEGenerator.py /data/experiment --single-ome-tiff
```
**Output**: Single `incucyte_plate.ome.tif` file ready for OMERO import

### Custom Output Directory
```bash
pixi run python IncucyteOMEGenerator.py /data/experiment -o /results/processed --well-field-ome-tiffs
```
**Output**: Files saved to `/results/processed/well_field_ome_tiffs/`

### Analysis-Ready Format
```bash
pixi run python IncucyteOMEGenerator.py /data/experiment --single-ome-tiff --simple-format
```
**Output**: Simple 5D array format for computational analysis

## OMERO Import

### Well-Field OME-TIFF Import (Individual Wells)
```bash
# Import individual well files
omero import well_field_ome_tiffs/A1_Field1.ome.tif

# Import all wells from a directory  
omero import well_field_ome_tiffs/

# Import specific wells matching a pattern
omero import well_field_ome_tiffs/A*_Field1.ome.tif
```
Each file imports as a separate multi-dimensional image with complete time series and channel information.

### Single Plate OME-TIFF Import
```bash
omero import incucyte_plate.ome.tif
```
Imports as a complete plate experiment with proper well structure and organization.

### Multi-File Companion Import
```bash
omero import companion.companion.ome
```
OMERO imports all referenced TIFF files as a unified plate experiment.

## Troubleshooting

### Common Issues

1. **"No timepoints found"**
   - Check input directory structure matches expected format
   - Verify TIFF files exist in deepest subdirectories

2. **Memory issues with large datasets**
   - Use `--simple-format` for more memory-efficient processing
   - Process subsets of timepoints separately

3. **Import issues in OMERO**
   - Ensure all TIFF files are in same directory as companion file
   - Check file paths use forward slashes in companion file

### Debug Mode
Add debugging to see detailed processing:
```bash
pixi run python -u IncucyteOMEGenerator.py /path/to/data --single-ome-tiff
```

## File Structure & Output

### Input Structure (Incucyte Export)
```
trial_dataset_incucyte/
├── EssenFiles/ScanData/
    └── YYMM/           # Year-Month (e.g., 2505)
        └── DD/         # Day (e.g., 16)
            └── HHMM/   # Time (e.g., 1124)
                └── XXXX/   # Fixed identifier (e.g., 1248)
                    ├── A1-1-C1.tif    # Well-Field-Channel
                    ├── A1-1-C2.tif
                    └── A1-1-Ph.tif    # Phase contrast
```

### Output Options

#### Well-Field OME-TIFFs
```
converted/
└── well_field_ome_tiffs/
    ├── A1_Field1.ome.tif     # All channels & timepoints for A1 Field1
    ├── A1_Field2.ome.tif     # All channels & timepoints for A1 Field2
    ├── B1_Field1.ome.tif
    └── ...
```

#### Single Plate OME-TIFF
```
converted/
└── incucyte_plate.ome.tif    # Complete experiment in one file
```

#### Multi-File Companion
```
converted/
├── 2505_16_1124/
│   ├── A1_F01_C1_T2505_16_1124.tif
│   ├── A1_F01_C2_T2505_16_1124.tif
│   └── ...
└── companion.companion.ome   # Metadata file
```

## File Naming & Channel Mapping

### Input Format
- **Pattern**: `WELL-FIELD-CHANNEL.tif`
- **Examples**: `A1-1-C1.tif`, `B2-1-Ph.tif`, `C3-2-C2.tif`

### Channel Mapping
| Incucyte Code | OME Name | Description |
|---------------|----------|-------------|
| `C1` | Green | Green fluorescence |
| `C2` | Red | Red fluorescence |
| `Ph` or `P` | Phase_Contrast | Phase contrast |

### Output Naming
- **Well-Field**: `A1_Field1.ome.tif`, `B2_Field2.ome.tif`
- **Multi-file**: `A1_F01_C1_T2505_16_1124.tif`

## Technical Details

- **Image Processing**: Extracts full-resolution images from Incucyte pyramid TIFFs
- **Metadata**: Preserves spatial and temporal information with proper OME-XML
- **Compression**: LZW compression with predictor for optimal file sizes
- **Tiling**: 256×256 pixel tiles for better I/O performance
- **UUID Support**: Proper file tracking for multi-file datasets
- **Pixel Types**: Maintains original data types (typically uint16)
- **Dimensions**: Time × Channels × Y × X organization

## Performance Notes

- **Well-Field OME-TIFFs**: Best balance of performance and usability
- **Single Plate**: Can create very large files for big experiments  
- **Multi-file**: Many small files, good for selective processing
- **Memory Usage**: Processes one timepoint at a time to minimize RAM requirements

## Support

For issues or questions:
1. Check this README for common solutions
2. Verify pixi environment is properly activated
3. Review error messages for specific guidance
4. Check input data structure matches expected format

## License

This project follows the license terms of the parent OMERO_scripts repository.
