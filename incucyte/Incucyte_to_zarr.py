#!/usr/bin/env python
"""
Incucyte to OME-NGFF (Zarr) Converter

This script converts Incucyte pyramid TIFFs to OME-NGFF format using ome-zarr-py.
Creates a high-content screening (HCS) dataset structure with proper plate/well organization.

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

Requires: pip install ome-zarr tifffile numpy
"""

import os
import re
import sys
import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
import tifffile

from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

__version__ = "1.0.0"


class IncucyteZarrConverter:
    """
    Convert Incucyte pyramid TIFFs to OME-NGFF format

    Creates an HCS (High Content Screening) dataset with proper plate structure:
    - Plate level with well organization
    - Well level with field organization
    - Image level with time series and channels
    """

    def __init__(self, base_path, output_path=None):
        self.base_path = Path(base_path)
        self.scan_data_path = self.base_path / "EssenFiles" / "ScanData"
        self.output_path = (
            Path(output_path) if output_path else self.base_path / "incucyte_data.zarr"
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
        """Convert well name (A1, B2) to row,column indices and names"""
        row_letter = well_name[0]
        col_number = int(well_name[1:])
        return row_letter, str(col_number)

    def determine_plate_dimensions(self, wells):
        """Determine plate dimensions and create row/column name lists"""
        if not wells:
            return ["A"], ["1"]  # Default minimal plate

        row_letters = set()
        col_numbers = set()

        for well in wells:
            row_letter = well[0]
            col_number = int(well[1:])
            row_letters.add(row_letter)
            col_numbers.add(col_number)

        # Create complete ranges
        min_row = min(ord(r) for r in row_letters)
        max_row = max(ord(r) for r in row_letters)
        row_names = [chr(i) for i in range(min_row, max_row + 1)]

        min_col = min(col_numbers)
        max_col = max(col_numbers)
        col_names = [str(i) for i in range(min_col, max_col + 1)]

        return row_names, col_names

    def get_channel_name(self, channel_code):
        """Convert Incucyte channel codes to readable names"""
        mapping = {
            "C1": "Green",
            "C2": "Red",
            "Ph": "Phase_Contrast",
            "P": "Phase_Contrast",
        }
        return mapping.get(channel_code, channel_code)

    def get_channel_color(self, channel_code):
        """Get display colors for channels"""
        mapping = {
            "C1": "00FF00",  # Green
            "C2": "FF0000",  # Red
            "Ph": "FFFFFF",  # White for phase contrast
            "P": "FFFFFF",  # White for phase contrast
        }
        return mapping.get(channel_code, "FFFFFF")

    def read_pyramid_tiff(self, file_path):
        """Read the full resolution image from a pyramid TIFF"""
        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                if len(tif.pages) > 0:
                    page = tif.pages[0]  # Full resolution is first page
                    image_data = page.asarray()
                    print(
                        f"  Read {file_path.name}: shape={image_data.shape}, dtype={image_data.dtype}, range={image_data.min()}-{image_data.max()}"
                    )
                    return image_data
                else:
                    print(f"  Warning: No pages found in {file_path.name}")
                    return None
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")
            return None

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

    def create_image_array(self, file_list, structure):
        """Create a 5D array (T,C,Z,Y,X) from file list"""
        n_timepoints = len(structure["timepoints"])
        n_channels = len(structure["channels"])
        channels_sorted = sorted(structure["channels"])

        # Read first image to get dimensions
        first_file = next(iter(file_list.values()))[0]
        sample_data = self.read_pyramid_tiff(first_file["file"])
        if sample_data is None:
            return None

        height, width = sample_data.shape[:2]

        # Initialize array: (T, C, Z, Y, X) - Z=1 for 2D images
        image_array = np.zeros(
            (n_timepoints, n_channels, 1, height, width), dtype=sample_data.dtype
        )

        # Fill array with data
        for channel, files in file_list.items():
            c_idx = channels_sorted.index(channel)

            for file_info in files:
                t_idx = file_info["timepoint_idx"]
                data = self.read_pyramid_tiff(file_info["file"])

                if data is not None:
                    image_array[t_idx, c_idx, 0, :, :] = data

        return image_array

    def create_omero_metadata(self, structure):
        """Create OMERO metadata for channel rendering"""
        channels = []

        for channel in sorted(structure["channels"]):
            channel_name = self.get_channel_name(channel)
            channel_color = self.get_channel_color(channel)

            # Set appropriate contrast limits based on channel type
            if channel in ["Ph", "P"]:  # Phase contrast
                window_settings = {"start": 0, "end": 255, "min": 0, "max": 255}
            else:  # Fluorescence channels
                window_settings = {"start": 0, "end": 1000, "min": 0, "max": 4095}

            channels.append(
                {
                    "color": channel_color,
                    "window": window_settings,
                    "label": channel_name,
                    "active": True,
                }
            )

        return {"channels": channels}

    def convert_to_zarr(self):
        """Convert Incucyte data to OME-NGFF format"""
        structure = self.scan_structure()

        if not structure["wells"]:
            raise ValueError("No wells found in the dataset")

        # Determine plate structure
        row_names, col_names = self.determine_plate_dimensions(structure["wells"])
        print(f"Creating plate with rows {row_names} and columns {col_names}")

        # Create well paths for HCS format
        well_paths = []
        for well in structure["wells"]:
            row, col = self.parse_well_position(well)
            well_paths.append(f"{row}/{col}")

        # Create field paths (convert to 0-based)
        field_paths = [
            str(f - 1) for f in sorted(structure["fields"])
        ]  # Convert to 0-based

        # Organize data
        well_field_data = self.organize_data_by_well_field(structure)

        # Create zarr store
        print(f"Creating OME-NGFF dataset at: {self.output_path}")
        store = parse_url(str(self.output_path), mode="w").store
        root = zarr.group(store=store)

        # Write plate metadata
        write_plate_metadata(root, row_names, col_names, well_paths)

        # Process each well
        for well_name in structure["wells"]:
            row, col = self.parse_well_position(well_name)
            print(f"Processing well {well_name} (row={row}, col={col})")

            # Create well group
            row_group = root.require_group(row)
            well_group = row_group.require_group(col)
            write_well_metadata(well_group, field_paths)

            # Process each field in this well
            for field_num in structure["fields"]:
                field_idx = field_num - 1  # Convert to 0-based

                if field_num in well_field_data[well_name]:
                    print(f"  Processing field {field_num}")

                    # Create image array for this well/field
                    field_data = well_field_data[well_name][field_num]
                    image_array = self.create_image_array(field_data, structure)

                    if image_array is not None:
                        # Create field group and write image
                        field_group = well_group.require_group(str(field_idx))

                        # Write image with time series (TCZYX axes)
                        write_image(
                            image=image_array,
                            group=field_group,
                            axes="tczyx",
                            storage_options=dict(chunks=(1, 1, 1, 1040, 1408)),
                        )

                        # Add OMERO metadata for rendering
                        omero_metadata = self.create_omero_metadata(structure)
                        field_group.attrs["omero"] = omero_metadata

                        print(
                            f"    Written image: {image_array.shape} {image_array.dtype}"
                        )

        print(f"OME-NGFF conversion completed: {self.output_path}")
        print(f"View with: napari {self.output_path}")

        return self.output_path


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Incucyte TIFFs to OME-NGFF format"
    )
    parser.add_argument("input_dir", help="Path to Incucyte export directory")
    parser.add_argument("-o", "--output", help="Output zarr path", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    try:
        converter = IncucyteZarrConverter(args.input_dir, args.output)
        output_path = converter.convert_to_zarr()

        print("\nConversion completed successfully!")
        print(f"Output: {output_path}")
        print(f"\nTo view the data:")
        print(f"  napari {output_path}")
        print(f"  # or in Python:")
        print(f"  import napari")
        print(f"  viewer = napari.Viewer()")
        print(f"  viewer.open('{output_path}', plugin='napari-ome-zarr')")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
