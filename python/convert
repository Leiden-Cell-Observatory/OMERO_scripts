import os
import sys
import nd2  # https://github.com/tlambert03/nd2/
import tifffile
import numpy as np
from pathlib import Path
from nd2._util import AXIS
import copy


# Script for converting multi-position ND2 files to individual OME-TIFF files
# Based on the nd2 library: https://github.com/tlambert03/nd2/


def convert_nd2_to_tiff_by_well_stack(nd2_path):
    """
    Convert an ND2 file containing well plate images to separate TIFF files per well stack.

    Args:
        nd2_path (str): Path to the input ND2 file
    """
    # Create output directory
    input_path = Path(nd2_path)
    output_dir = input_path.parent / "export"
    output_dir.mkdir(exist_ok=True)

    print(f"Processing file: {input_path.name}")
    print(f"Output directory: {output_dir}")

    # Open ND2 file
    with nd2.ND2File(nd2_path) as f:
        # Get dimensions and metadata
        print(f"File shape: {f.shape}")
        print(f"File dimensions: {f.ndim}")

        # Get the number of positions (first dimension)
        num_positions = f.sizes[AXIS.POSITION]
        num_z = f.sizes.get(AXIS.Z, 1)
        num_channels = f.sizes.get(AXIS.CHANNEL, 1)

        print(f"Number of positions: {num_positions}")
        print(f"Number of z-slices per position: {num_z}")
        print(f"Number of channels: {num_channels}")

        # Get physical pixel size for resolution information
        voxel_size = f.voxel_size()
        physical_pixel_size_x = voxel_size.x
        physical_pixel_size_y = voxel_size.y

        # Get original OME metadata
        full_ome_metadata = f.ome_metadata()

        # Process each position (well stack)
        for pos_idx in range(num_positions):
            # Get position name from the metadata
            # We'll read the first frame of this position's z-stack to get metadata
            frame_indices = list(f.loop_indices)
            pos_frame_indices = [
                (frame_num, idx)
                for frame_num, idx in enumerate(frame_indices)
                if idx.get(AXIS.POSITION, 0) == pos_idx and idx.get(AXIS.Z, 0) == 0
            ]

            if not pos_frame_indices:
                print(f"Warning: No frames found for position {pos_idx}")
                continue

            frame_num, _ = pos_frame_indices[0]
            metadata = f.frame_metadata(frame_num)

            # Extract position name
            try:
                position_name = metadata.channels[0].position.name

                # Verify that this appears to be a well position (should contain format like A01_0000)
                if not (
                    len(position_name) >= 4
                    and position_name[0].isalpha()
                    and any(c.isdigit() for c in position_name)
                ):
                    raise ValueError(
                        f"Position name '{position_name}' does not appear to be a valid well identifier"
                    )

                print(f"Processing position: {position_name}")
            except (AttributeError, IndexError):
                print(
                    f"Error: Position metadata missing or invalid for position {pos_idx}"
                )
                print(
                    "This script requires an ND2 file from a multi-position plate acquisition with proper well identifiers."
                )
                sys.exit(1)

            # Generate output filename
            base_name = input_path.stem
            output_filename = f"{base_name}_{position_name}.ome.tif"
            output_path = output_dir / output_filename

            # Create an iterator function to yield frames
            def position_frames():
                frames = []
                for z in range(num_z):
                    # Find the frame number for this position and z-slice
                    pos_z_frame_indices = [
                        (frame_num, idx)
                        for frame_num, idx in enumerate(frame_indices)
                        if idx.get(AXIS.POSITION, 0) == pos_idx
                        and idx.get(AXIS.Z, 0) == z
                    ]

                    if not pos_z_frame_indices:
                        print(
                            f"Warning: No frame found for position {pos_idx}, z-slice {z}"
                        )
                        # Use zeros for missing frames
                        frames.append(
                            np.zeros((num_channels, 512, 512), dtype=np.uint16)
                        )
                    else:
                        frame_num, _ = pos_z_frame_indices[0]
                        frames.append(f.read_frame(frame_num))
                return np.stack(frames, axis=0)

            # Get all frames for this position
            position_data = position_frames()

            # Modify OME metadata to include only this position
            position_ome_metadata = copy.deepcopy(full_ome_metadata)

            # Keep only the image for this position in the metadata
            if (
                hasattr(position_ome_metadata, "images")
                and len(position_ome_metadata.images) > pos_idx
            ):
                # Keep only the current position's image information
                position_ome_metadata.images = [position_ome_metadata.images[pos_idx]]

                # Update the image metadata to reflect it's a single position
                if hasattr(position_ome_metadata.images[0], "name"):
                    position_ome_metadata.images[
                        0
                    ].name = f"{base_name}_{position_name}"

            try:
                # Convert OME metadata to XML string
                ome_xml = position_ome_metadata.to_xml(exclude_unset=True).encode(
                    "utf-8"
                )
            except Exception as e:
                print(f"Warning: Could not generate OME metadata: {e}")
                ome_xml = None

            # Add time dimension for 5D TZCYX format expected by OME-TIFF
            position_data_with_t = np.expand_dims(position_data, axis=0)

            # Determine photometric interpretation
            photometric = "rgb" if f.is_rgb else "minisblack"

            print(f"Saving to: {output_path}")

            # Define axes for correct dimension ordering in OME-TIFF
            metadata = {"axes": "TZCYX"}

            # Save with tifffile
            tifffile.imwrite(
                output_path,
                position_data_with_t,
                resolution=(1 / physical_pixel_size_x, 1 / physical_pixel_size_y),
                resolutionunit="MICROMETER",
                photometric=photometric,
                metadata=metadata,
                description=ome_xml,
            )

    print(f"Completed processing {num_positions} positions")


if __name__ == "__main__":
    # Get file path from command line arguments or user input
    if len(sys.argv) > 1:
        nd2_file_path = sys.argv[1]
    else:
        nd2_file_path = input("Enter path to ND2 file: ")

    # Validate file exists
    if not os.path.exists(nd2_file_path):
        print(f"Error: File {nd2_file_path} does not exist")
        sys.exit(1)
    elif not nd2_file_path.lower().endswith(".nd2"):
        print(f"Error: File {nd2_file_path} is not an ND2 file")
        sys.exit(1)
    else:
        convert_nd2_to_tiff_by_well_stack(nd2_file_path)
