#!/usr/bin/env python
"""
Incucyte HCS OME-NGFF Converter using ngff-zarr library (Fixed Version)

This script converts Incucyte pyramid TIFFs to HCS (High Content Screening) OME-NGFF format
using the ngff-zarr library with proper plate/well organization and fixes for WASM errors.

Creates a complete HCS dataset with:
- Plate level organization
- Well level with multiple fields
- Proper multiscale pyramids (with fallback methods)
- OME-Zarr 0.5 with sharding

Incucyte structure:
trial_dataset_incucyte/
├── EssenFiles/ScanData/
    └── YYMM/           # Year-Month (e.g., 2505)
        └── DD/         # Day (e.g., 16)
            └── HHMM/   # Time (e.g., 1124)
                └── XXXX/   # Fixed identifier (e.g., 1248)
                    ├── A1-1-C1.tif    # Well, Field, Channel
                    ├── A1-1-C2.tif
                    └── A1-1-Ph.tif    # Phase contrast

Requires:
    pip install "ngff-zarr[tensorstore]" tifffile numpy dask

Optional for better multiscale methods:
    pip install "ngff-zarr[dask-image]"  # For DASK_IMAGE_* methods (recommended)
    pip install "ngff-zarr[itk]"         # For ITK_* methods (alternative)

Note: The default ITKWASM_GAUSSIAN method may cause WASM errors on some systems.
      This script uses fallback methods to avoid those issues.
"""

import os
import re
import sys
import json
import numpy as np
import dask.array as da
from dask import delayed
import zarr

try:
    # Zarr v3
    from zarr.storage import DirectoryStore
except ImportError:
    try:
        # Zarr v2
        DirectoryStore = zarr.DirectoryStore
    except AttributeError:
        # Fallback
        DirectoryStore = None
from pathlib import Path
from collections import defaultdict
import tifffile

import ngff_zarr as nz

__version__ = "1.0.1"


class IncucyteHcsConverter:
    """
    Convert Incucyte pyramid TIFFs to HCS OME-NGFF format

    Creates a High Content Screening dataset with proper plate structure:
    - Plate level with well organization
    - Well level with field organization
    - Image level with time series and channels
    - Multiscale pyramids with sharding (with WASM error fixes)
    """

    def __init__(self, base_path, output_path=None):
        self.base_path = Path(base_path)
        self.scan_data_path = self.base_path / "EssenFiles" / "ScanData"
        self.output_path = (
            Path(output_path)
            if output_path
            else self.base_path / "incucyte_plate.ome.zarr"
        )

    def scan_structure(self):
        """Scan the Incucyte export directory structure"""
        timepoints = []
        wells = set()
        fields = set()
        channels = set()

        print(f"Scanning directory: {self.scan_data_path}")

        # Navigate through year/month directories
        for year_month in self.scan_data_path.iterdir():
            if not year_month.is_dir():
                continue
            # Navigate through day directories
            for day in year_month.iterdir():
                if not day.is_dir():
                    continue
                # Navigate through time directories
                for time_dir in day.iterdir():
                    if not time_dir.is_dir():
                        continue
                    # Navigate through fixed ID directories
                    for fixed_id in time_dir.iterdir():
                        if not fixed_id.is_dir():
                            continue

                        timepoint_path = fixed_id
                        timestamp = f"{year_month.name}_{day.name}_{time_dir.name}"
                        timepoints.append(
                            {"path": timepoint_path, "timestamp": timestamp}
                        )

                        # Scan TIFF files in this timepoint
                        tiff_files = list(timepoint_path.glob("*.tif"))
                        if tiff_files:
                            print(f"Found {len(tiff_files)} TIFF files in {timestamp}")

                        for file in tiff_files:
                            well, field, channel = self.parse_filename(file.name)
                            if well and field is not None and channel:
                                wells.add(well)
                                fields.add(field)
                                channels.add(channel)

        print(
            f"Found: {len(timepoints)} timepoints, {len(wells)} wells, {len(fields)} fields, {len(channels)} channels"
        )
        print(f"Wells: {sorted(wells)}")
        print(f"Channels: {sorted(channels)}")

        return {
            "timepoints": sorted(timepoints, key=lambda x: x["timestamp"]),
            "wells": sorted(wells),
            "fields": sorted(fields),
            "channels": sorted(channels),
        }

    def parse_filename(self, filename):
        """
        Parse Incucyte filename format: WELL-FIELD-CHANNEL.tif
        Examples: A1-1-C1.tif, B2-1-Ph.tif
        Returns: (well, field, channel)
        """
        pattern = r"([A-Z]\d+)-(\d+)-(.+)\.tif"
        match = re.match(pattern, filename)
        if match:
            well = match.group(1)
            field = int(match.group(2))
            channel = match.group(3)
            return well, field, channel
        return None, None, None

    def parse_well_position(self, well_name):
        """Convert well name (A1, B2) to row,column names and indices"""
        row_letter = well_name[0]
        col_number = int(well_name[1:])
        row_index = ord(row_letter) - ord("A")
        col_index = col_number - 1
        return row_letter, str(col_number), row_index, col_index

    def get_plate_dimensions(self, wells):
        """Determine plate dimensions from wells"""
        if not wells:
            return 1, 1

        max_row = 0
        max_col = 0

        for well in wells:
            _, _, row_idx, col_idx = self.parse_well_position(well)
            max_row = max(max_row, row_idx)
            max_col = max(max_col, col_idx)

        return max_row + 1, max_col + 1

    def create_plate_metadata(self, structure):
        """Create HCS plate metadata following OME-NGFF 0.5 spec"""
        rows, cols = self.get_plate_dimensions(structure["wells"])

        # Create row and column metadata
        row_list = []
        for i in range(rows):
            row_list.append({"name": chr(ord("A") + i)})

        col_list = []
        for i in range(cols):
            col_list.append({"name": str(i + 1)})

        # Create well list with acquisitions (fields)
        wells = []
        for well_name in structure["wells"]:
            row_name, col_name, row_idx, col_idx = self.parse_well_position(well_name)

            # Create acquisitions (fields) for this well
            acquisitions = []
            for field in structure["fields"]:
                acquisitions.append(
                    {
                        "id": field - 1,  # 0-based
                        "name": f"Field{field}",
                        "path": str(field - 1),  # 0-based path
                    }
                )

            wells.append(
                {
                    "path": f"{row_name}/{col_name}",  # Use actual row/column names
                    "rowIndex": row_idx,
                    "columnIndex": col_idx,
                    "acquisitions": acquisitions,
                }
            )

        plate_metadata = {
            "plate": {
                "acquisitions": [
                    {"id": field - 1, "name": f"Field{field}"}
                    for field in structure["fields"]
                ],
                "columns": col_list,
                "rows": row_list,
                "wells": wells,
                "version": "0.5",
            }
        }

        return plate_metadata

    def read_pyramid_tiff_dask(self, file_path):
        """Read TIFF file and return as dask array"""
        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                if len(tif.pages) > 0:
                    page = tif.pages[0]  # Full resolution is first page
                    shape = page.shape
                    dtype = page.dtype

                    def read_tiff():
                        return page.asarray()

                    # Create dask array from delayed function
                    dask_array = da.from_delayed(
                        delayed(read_tiff)(), shape=shape, dtype=dtype
                    )

                    return dask_array
                else:
                    print(f"  Warning: No pages found in {file_path.name}")
                    return None
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")
            return None

    def validate_chunks(self, chunks, array_shape):
        """Validate and fix chunk sizes to avoid zero or invalid chunks"""
        # Map dimension names to array indices for 5D TCZYX array
        dim_mapping = {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}

        validated_chunks = {}
        for dim_name, chunk_size in chunks.items():
            if dim_name in dim_mapping:
                idx = dim_mapping[dim_name]
                if idx < len(array_shape):
                    # Ensure chunk size is at least 1 and not larger than dimension
                    array_dim_size = array_shape[idx]
                    validated_chunk = max(1, min(chunk_size, array_dim_size))
                    validated_chunks[dim_name] = validated_chunk
                else:
                    validated_chunks[dim_name] = max(1, chunk_size)
            else:
                validated_chunks[dim_name] = max(1, chunk_size)

        return validated_chunks

    def organize_data_by_well_field(self, structure):
        """Organize file data by well and field for efficient processing"""
        well_field_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for timepoint in structure["timepoints"]:
            t_idx = structure["timepoints"].index(timepoint)

            for file_path in timepoint["path"].glob("*.tif"):
                well, field, channel = self.parse_filename(file_path.name)
                if well and field is not None and channel:
                    well_field_data[well][field][channel].append(
                        {
                            "file": file_path,
                            "timepoint_idx": t_idx,
                            "timepoint": timepoint,
                        }
                    )

        return well_field_data
        """Organize file data by well and field for efficient processing"""
        well_field_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for timepoint in structure["timepoints"]:
            t_idx = structure["timepoints"].index(timepoint)

            for file_path in timepoint["path"].glob("*.tif"):
                well, field, channel = self.parse_filename(file_path.name)
                if well and field is not None and channel:
                    well_field_data[well][field][channel].append(
                        {
                            "file": file_path,
                            "timepoint_idx": t_idx,
                            "timepoint": timepoint,
                        }
                    )

        return well_field_data

    def create_dask_array_for_well_field(self, file_list, structure):
        """Create a 5D dask array (T,C,Z,Y,X) from file list"""
        n_timepoints = len(structure["timepoints"])
        n_channels = len(structure["channels"])
        channels_sorted = sorted(structure["channels"])

        # Read first image to get dimensions
        first_file = next(iter(file_list.values()))[0]
        sample_dask = self.read_pyramid_tiff_dask(first_file["file"])
        if sample_dask is None:
            return None

        height, width = sample_dask.shape[:2]
        dtype = sample_dask.dtype

        # Create list of dask arrays for each timepoint and channel
        dask_arrays = []

        for t_idx in range(n_timepoints):
            channel_arrays = []
            for c_idx, channel in enumerate(channels_sorted):
                if channel in file_list:
                    matching_files = [
                        f for f in file_list[channel] if f["timepoint_idx"] == t_idx
                    ]
                    if matching_files:
                        file_info = matching_files[0]
                        dask_img = self.read_pyramid_tiff_dask(file_info["file"])
                        if dask_img is not None:
                            dask_img = dask_img[None, :, :]  # Add Z dimension
                            channel_arrays.append(dask_img)
                        else:
                            zero_array = da.zeros(
                                (1, height, width),
                                dtype=dtype,
                                chunks=(1, height, width),
                            )
                            channel_arrays.append(zero_array)
                    else:
                        zero_array = da.zeros(
                            (1, height, width), dtype=dtype, chunks=(1, height, width)
                        )
                        channel_arrays.append(zero_array)
                else:
                    zero_array = da.zeros(
                        (1, height, width), dtype=dtype, chunks=(1, height, width)
                    )
                    channel_arrays.append(zero_array)

            if channel_arrays:
                timepoint_array = da.stack(channel_arrays, axis=0)  # (C, Z, Y, X)
                dask_arrays.append(timepoint_array)

        if dask_arrays:
            full_array = da.stack(dask_arrays, axis=0)  # (T, C, Z, Y, X)
            return full_array
        else:
            return None

    def create_multiscale_with_fallback(self, ngff_image, scale_factors, chunk_size):
        """Create multiscale with fallback methods to avoid WASM errors"""

    def create_multiscale_with_fallback(self, ngff_image, scale_factors, chunk_size):
        """Create multiscale with fallback methods to avoid WASM errors"""

        # Check which methods are available and try them in order of preference
        methods_to_try = []

        # Check for DASK_IMAGE_GAUSSIAN (requires: pip install "ngff-zarr[dask-image]")
        if hasattr(nz.Methods, "DASK_IMAGE_GAUSSIAN"):
            methods_to_try.append(
                ("DASK_IMAGE_GAUSSIAN", nz.Methods.DASK_IMAGE_GAUSSIAN)
            )

        # Check for ITK methods (require: pip install "ngff-zarr[itk]")
        if hasattr(nz.Methods, "ITK_GAUSSIAN"):
            methods_to_try.append(("ITK_GAUSSIAN", nz.Methods.ITK_GAUSSIAN))

        if hasattr(nz.Methods, "ITK_BIN_SHRINK"):
            methods_to_try.append(("ITK_BIN_SHRINK", nz.Methods.ITK_BIN_SHRINK))

        # Check for other ITKWASM methods (safer than the problematic ITKWASM_GAUSSIAN)
        if hasattr(nz.Methods, "ITKWASM_BIN_SHRINK"):
            methods_to_try.append(("ITKWASM_BIN_SHRINK", nz.Methods.ITKWASM_BIN_SHRINK))

        if hasattr(nz.Methods, "ITKWASM_LABEL_IMAGE"):
            methods_to_try.append(
                ("ITKWASM_LABEL_IMAGE", nz.Methods.ITKWASM_LABEL_IMAGE)
            )

        # Check for DASK_IMAGE alternatives
        if hasattr(nz.Methods, "DASK_IMAGE_NEAREST"):
            methods_to_try.append(("DASK_IMAGE_NEAREST", nz.Methods.DASK_IMAGE_NEAREST))

        if hasattr(nz.Methods, "DASK_IMAGE_MODE"):
            methods_to_try.append(("DASK_IMAGE_MODE", nz.Methods.DASK_IMAGE_MODE))

        # Skip the problematic ITKWASM_GAUSSIAN method
        # if hasattr(nz.Methods, 'ITKWASM_GAUSSIAN'):
        #     methods_to_try.append(('ITKWASM_GAUSSIAN', nz.Methods.ITKWASM_GAUSSIAN))

        if not methods_to_try:
            print("    No multiscale methods available! Install optional dependencies:")
            print(
                '    pip install "ngff-zarr[dask-image]" or pip install "ngff-zarr[itk]"'
            )
            return self.create_single_scale_multiscales(ngff_image, chunk_size)

        chunks = {"t": 1, "c": 1, "z": 1, "y": chunk_size, "x": chunk_size}

        # Validate chunks against array shape
        validated_chunks = self.validate_chunks(chunks, ngff_image.data.shape)
        print(f"    Using chunks: {validated_chunks}")

        for method_name, method in methods_to_try:
            try:
                print(f"    Trying multiscale method: {method_name}")
                multiscales = nz.to_multiscales(
                    ngff_image,
                    scale_factors=scale_factors,
                    chunks=validated_chunks,
                    method=method,
                )
                print(f"    Success with method: {method_name}")
                return multiscales
            except Exception as e:
                print(f"    Failed with {method_name}: {str(e)}")
                continue

        # If all methods fail, create single-scale
        print("    All multiscale methods failed, creating single-scale")
        return self.create_single_scale_multiscales(ngff_image, validated_chunks)

    def create_single_scale_multiscales(self, ngff_image, chunks_dict):
        """Create single-scale multiscales as fallback"""

        # Create metadata for single scale
        metadata = nz.Metadata(
            axes=[
                nz.Axis(name="t", type="time", unit="second"),
                nz.Axis(name="c", type="channel"),
                nz.Axis(name="z", type="space", unit="micrometer"),
                nz.Axis(name="y", type="space", unit="micrometer"),
                nz.Axis(name="x", type="space", unit="micrometer"),
            ],
            datasets=[
                nz.Dataset(
                    path="0",
                    coordinateTransformations=[
                        nz.Scale(
                            scale=[
                                ngff_image.scale["t"],
                                ngff_image.scale["c"],
                                ngff_image.scale["z"],
                                ngff_image.scale["y"],
                                ngff_image.scale["x"],
                            ],
                            type="scale",
                        ),
                        nz.Translation(
                            translation=[
                                ngff_image.translation["t"],
                                ngff_image.translation["c"],
                                ngff_image.translation["z"],
                                ngff_image.translation["y"],
                                ngff_image.translation["x"],
                            ],
                            type="translation",
                        ),
                    ],
                )
            ],
            name=ngff_image.name,
            version="0.5",
        )

        # Re-chunk the image data - compatible with different zarr versions
        if isinstance(chunks_dict, dict):
            chunks = chunks_dict
        else:
            # Fallback if chunks_dict is actually chunk_size (old interface)
            chunk_size = chunks_dict if isinstance(chunks_dict, int) else 512
            chunks = {"t": 1, "c": 1, "z": 1, "y": chunk_size, "x": chunk_size}
            chunks = self.validate_chunks(chunks, ngff_image.data.shape)

        print(f"    Using single-scale chunks: {chunks}")

        try:
            rechunked_data = ngff_image.data.rechunk(chunks)
        except Exception as e:
            print(f"    Warning: Could not rechunk data: {e}")
            rechunked_data = ngff_image.data

        rechunked_image = nz.NgffImage(
            data=rechunked_data,
            dims=ngff_image.dims,
            scale=ngff_image.scale,
            translation=ngff_image.translation,
            name=ngff_image.name,
            axes_units=ngff_image.axes_units,
            computed_callbacks=ngff_image.computed_callbacks,
        )

        multiscales = nz.Multiscales(
            images=[rechunked_image],
            metadata=metadata,
            scale_factors=[],
            method=None,
            chunks=chunks,
        )

        return multiscales

    def convert_to_hcs_ngff(
        self, use_sharding=True, scale_factors=None, chunk_size=None
    ):
        """Convert Incucyte data to HCS OME-NGFF format"""
        structure = self.scan_structure()

        if not structure["wells"]:
            raise ValueError("No wells found in the dataset")

        # Set defaults
        if scale_factors is None:
            scale_factors = [2, 4]
        if chunk_size is None:
            chunk_size = 512

        # Organize data
        well_field_data = self.organize_data_by_well_field(structure)

        print(f"Creating HCS OME-NGFF dataset at: {self.output_path}")

        # Create zarr store (compatible with both Zarr v2 and v3)
        if DirectoryStore is not None:
            store = DirectoryStore(str(self.output_path))
            root_group = zarr.group(store=store, overwrite=True)
        else:
            # Fallback for different zarr versions
            root_group = zarr.open_group(str(self.output_path), mode="w")

        # Add plate metadata
        plate_metadata = self.create_plate_metadata(structure)
        root_group.attrs.update(plate_metadata)

        print("Created plate metadata")

        # Process each well
        for well_name in structure["wells"]:
            row_name, col_name, row_idx, col_idx = self.parse_well_position(well_name)
            print(f"Processing well {well_name} (row={row_name}, col={col_name})")

            # Create well groups using actual row/column names
            row_group = root_group.require_group(row_name)
            well_group = row_group.require_group(col_name)

            # Add well metadata
            well_metadata = {
                "well": {
                    "images": [
                        {"path": str(field - 1), "acquisition": field - 1}
                        for field in structure["fields"]
                        if field in well_field_data[well_name]
                    ],
                    "version": "0.5",
                }
            }
            well_group.attrs.update(well_metadata)

            # Process each field in this well
            for field_num in structure["fields"]:
                if field_num in well_field_data[well_name]:
                    print(f"  Processing field {field_num}")

                    # Create dask array for this well/field
                    field_data = well_field_data[well_name][field_num]
                    dask_array = self.create_dask_array_for_well_field(
                        field_data, structure
                    )

                    if dask_array is not None:
                        print(
                            f"    Creating NGFF image from array shape: {dask_array.shape}"
                        )

                        # Physical pixel size (adjust as needed)
                        pixel_size = 1.0

                        # Create NGFF image
                        ngff_image = nz.to_ngff_image(
                            dask_array,
                            dims=["t", "c", "z", "y", "x"],
                            scale={
                                "t": 1.0,
                                "c": 1.0,
                                "z": pixel_size,
                                "y": pixel_size,
                                "x": pixel_size,
                            },
                            translation={
                                "t": 0.0,
                                "c": 0.0,
                                "z": 0.0,
                                "y": 0.0,
                                "x": 0.0,
                            },
                            name=f"{well_name}_Field{field_num}",
                            axes_units={
                                "t": "second",
                                "c": None,
                                "z": "micrometer",
                                "y": "micrometer",
                                "x": "micrometer",
                            },
                        )

                        # Create multiscale pyramid with fallback methods
                        multiscales = self.create_multiscale_with_fallback(
                            ngff_image, scale_factors, chunk_size
                        )

                        # Create field group path using actual row/column names
                        field_path = str(
                            self.output_path / row_name / col_name / str(field_num - 1)
                        )

                        print(f"    Writing to {field_path}")

                        # Write field data with fallback for tensorstore issues
                        try:
                            if use_sharding:
                                # Create compatible chunks_per_shard
                                chunks_per_shard = {
                                    "t": 1,
                                    "c": 1,
                                    "z": 1,
                                    "y": 2,
                                    "x": 2,
                                }

                                # Validate chunks_per_shard against actual dimensions
                                validated_shard_chunks = self.validate_chunks(
                                    chunks_per_shard, dask_array.shape
                                )

                                print(
                                    f"    Using chunks_per_shard: {validated_shard_chunks}"
                                )

                                nz.to_ngff_zarr(
                                    field_path,
                                    multiscales,
                                    chunks_per_shard=validated_shard_chunks,
                                    use_tensorstore=True,
                                    version="0.5",
                                )
                            else:
                                nz.to_ngff_zarr(
                                    field_path,
                                    multiscales,
                                    use_tensorstore=True,
                                    version="0.4",
                                )
                        except ValueError as e:
                            if "chunk_shape" in str(e) and "received: 0" in str(e):
                                print(
                                    f"    Tensorstore chunk error, trying without tensorstore: {e}"
                                )
                                # Fallback: disable tensorstore
                                if use_sharding:
                                    print(
                                        "    Warning: Disabling sharding due to tensorstore issues"
                                    )
                                nz.to_ngff_zarr(
                                    field_path,
                                    multiscales,
                                    use_tensorstore=False,
                                    version="0.4",  # Use 0.4 for compatibility without tensorstore
                                )
                            else:
                                raise

                        print(f"    Written field {field_num}: {dask_array.shape}")

        print(f"HCS OME-NGFF conversion completed: {self.output_path}")
        return self.output_path


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Incucyte TIFFs to HCS OME-NGFF (Fixed Version)"
    )
    parser.add_argument("input_dir", help="Path to Incucyte export directory")
    parser.add_argument("-o", "--output", help="Output zarr path", default=None)
    parser.add_argument(
        "--no-sharding", action="store_true", help="Disable sharding (use OME-Zarr 0.4)"
    )
    parser.add_argument(
        "--scale-factors",
        nargs="+",
        type=int,
        default=[2, 4],
        help="Scale factors for pyramid levels (default: 2 4)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for spatial dimensions (default: 512)",
    )
    parser.add_argument(
        "--single-scale",
        action="store_true",
        help="Force single-scale output (no pyramid)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    try:
        converter = IncucyteHcsConverter(args.input_dir, args.output)

        # Force single scale if requested
        scale_factors = [] if args.single_scale else args.scale_factors

        output_path = converter.convert_to_hcs_ngff(
            use_sharding=not args.no_sharding,
            scale_factors=scale_factors,
            chunk_size=args.chunk_size,
        )

        print("\nHCS conversion completed successfully!")
        print(f"Output: {output_path}")
        print(f"\nDataset structure:")
        print(f"  {output_path}/")
        print(f"  ├── .zattrs          # Plate metadata")
        print(f"  ├── 0/               # Row 0 (A)")
        print(f"  │   ├── 0/           # Column 0 (Well A1)")
        print(f"  │   │   ├── .zattrs  # Well metadata")
        print(f"  │   │   ├── 0/       # Field 0")
        print(f"  │   │   └── 1/       # Field 1")
        print(f"  │   └── 1/           # Column 1 (Well A2)")
        print(f"  └── 1/               # Row 1 (B)")
        print(f"\nTo view the data:")
        print(f"  napari {output_path}")

        if args.no_sharding:
            print(f"\nDataset format: OME-Zarr 0.4 (no sharding)")
        else:
            print(f"\nDataset format: OME-Zarr 0.5 with sharding")

        if args.single_scale:
            print(f"Scale: Single scale (no pyramid)")
        else:
            print(f"Scale factors: {scale_factors}")
        print(f"Chunk size: {args.chunk_size}x{args.chunk_size}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
