import os
import sys
import nd2  # https://github.com/tlambert03/nd2/
import tifffile
import numpy as np
from pathlib import Path
from nd2._util import AXIS
import copy
import argparse


# Script for converting multi-position ND2 files to individual TIFF files
# Based on the nd2 library: https://github.com/tlambert03/nd2/
#
# Version 2023-07-01: Initial version
# Version 2024-04-17: Added export-folder argument for custom output location
# Version 2024-04-17: Modified --max-projection to only save maximum intensity projections
# Version 2024-04-17: Added file-prefix option for custom file naming (e.g., 250314_PDLO_FUCCI_day7fixed_)
# Version 2025-11-14: Improved guessing of missing position names with numbering
# Version 2025-11-14: Added support for time series data
#

def guess_missing_position_names(f):
    """
    Analyze the ND2 file and guess missing position names.
    
    For files where not all positions have names, this function:
    1. Collects all available position names
    2. Creates a mapping of position indices to names
    3. Fills gaps by assuming consecutive unnamed positions belong to the last named position
    
    Args:
        f: An open ND2File object
        
    Returns:
        dict: Mapping of position indices to guessed position names with numbering (e.g., 'A1_0001')
    """
    print("Attempting to guess missing position names...")
    frame_indices = list(f.loop_indices)
    
    # First pass: collect all available position names
    position_names = {}
    current_position = None
    position_counters = {}  # To track sequencing within each position
    
    # Loop through frames to find all available position names
    for frame_num, idx in enumerate(frame_indices):
        pos_idx = idx.get(AXIS.POSITION, 0)
        
        # Skip if we already have a name for this position
        if pos_idx in position_names:
            continue
        
        # Try to get position name from metadata
        try:
            metadata = f.frame_metadata(frame_num)
            if hasattr(metadata, 'channels') and metadata.channels:
                if hasattr(metadata.channels[0], 'position') and hasattr(metadata.channels[0].position, 'name'):
                    name = metadata.channels[0].position.name
                    if name and len(name) > 0:
                        position_names[pos_idx] = name
                        current_position = name
                        # Initialize counter for this position if not already done
                        if name not in position_counters:
                            position_counters[name] = 0
        except (AttributeError, IndexError):
            pass
    
    print(f"Found {len(position_names)} named positions out of {f.sizes[AXIS.POSITION]} total positions")
    
    # If no position names found, use default naming
    if not position_names:
        print("No position names found. Using default position naming.")
        return {i: f"Pos{i+1}" for i in range(f.sizes[AXIS.POSITION])}
    
    # Second pass: fill in missing names based on sequence
    final_position_map = {}
    for pos_idx in range(f.sizes[AXIS.POSITION]):
        if pos_idx in position_names:
            # Use the actual position name
            position_name = position_names[pos_idx]
            current_position = position_name
        else:
            # Use the last known position name
            position_name = current_position if current_position else f"Pos{pos_idx+1}"
            print(f"Guessing position {pos_idx} belongs to {position_name}")
        
        # Increment counter for this position
        position_counters[position_name] = position_counters.get(position_name, 0) + 1
        
        # Create the final name with numbering
        final_name = f"{position_name}_{position_counters[position_name]:04d}"
        final_position_map[pos_idx] = final_name
    
    return final_position_map


def convert_nd2_to_tiff_by_well_stack(
    nd2_path, 
    skip_ome=False, 
    separate_channels=False,
    separate_z=False,
    separate_t=False,
    guess_names=False,
    max_projection=False,
    export_folder=None,
    file_prefix=None
):
    """
    Convert an ND2 file containing well plate images to separate TIFF files.

    Args:
        nd2_path (str): Path to the input ND2 file
        skip_ome (bool): If True, skip generation of OME metadata
        separate_channels (bool): If True, save each channel as a separate file
        separate_z (bool): If True, save each z-slice as a separate file
        separate_t (bool): If True, save each time point as a separate file
        guess_names (bool): If True, attempt to guess missing position names
        max_projection (bool): If True, create maximum intensity projections only
        export_folder (str): Custom export folder path (default is "export" subfolder)
        file_prefix (str): Optional prefix to add to all output filenames (e.g., '250314_PDLO_FUCCI_day7fixed_')
    """
    # Create output directory
    input_path = Path(nd2_path)
    
    # Use custom export folder if provided, otherwise use default
    if export_folder:
        export_path = Path(export_folder)
        # If the export path is relative, make it relative to the input file's directory
        if not export_path.is_absolute():
            output_dir = input_path.parent / export_path
        else:
            output_dir = export_path
    else:
        output_dir = input_path.parent / "export"
        
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Processing file: {input_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Options: skip_ome={skip_ome}, separate_channels={separate_channels}, "
          f"separate_z={separate_z}, separate_t={separate_t}, guess_names={guess_names}, max_projection={max_projection}")
    
    if file_prefix:
        print(f"Using file prefix: {file_prefix}")
    
    if max_projection:
        print("Max projection mode: Only saving maximum intensity projections")
    
    # Open ND2 file
    with nd2.ND2File(nd2_path) as f:
        # Get dimensions and metadata
        print(f"File shape: {f.shape}")
        print(f"File dimensions: {f.ndim}")

        # Get the number of positions (first dimension)
        num_positions = f.sizes[AXIS.POSITION]
        num_z = f.sizes.get(AXIS.Z, 1)
        num_channels = f.sizes.get(AXIS.CHANNEL, 1)
        num_time = f.sizes.get(AXIS.TIME, 1)
        has_time = num_time > 1

        print(f"Number of positions: {num_positions}")
        print(f"Number of time points: {num_time}")
        print(f"Number of z-slices per position: {num_z}")
        print(f"Number of channels: {num_channels}")
        
        if has_time:
            if separate_t:
                print(f"Time series detected - will create separate OME-TIFF files for each of {num_time} time points per position")
            else:
                print(f"Time series detected - will create one OME-TIFF per position with all {num_time} time points")

        # Get physical pixel size for resolution information
        voxel_size = f.voxel_size()
        physical_pixel_size_x = voxel_size.x
        physical_pixel_size_y = voxel_size.y

        # Get position names mapping if guessing names
        position_name_map = None
        if guess_names:
            position_name_map = guess_missing_position_names(f)

        # Get original OME metadata if needed
        full_ome_metadata = None
        if not skip_ome:
            full_ome_metadata = f.ome_metadata()

        # Get all frame indices once
        frame_indices = list(f.loop_indices)

        # Track position name counts to handle duplicates
        position_name_counts = {}

        # Process each position (well stack)
        for pos_idx in range(num_positions):
            # Get position name based on the method
            if guess_names and position_name_map:
                position_name = position_name_map[pos_idx]
                print(f"Using guessed position name: {position_name}")
            else:
                # Use the original method to get position name
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
                        len(position_name) >= 1
                        and position_name[0].isalpha()
                        and any(c.isdigit() for c in position_name)
                    ):
                        raise ValueError(
                            f"Position name '{position_name}' does not appear to be a valid well identifier"
                        )

                    print(f"Processing position: {position_name}")
                except (AttributeError, IndexError, ValueError) as e:
                    if not guess_names:
                        print(f"Error: Position metadata issue for position {pos_idx}: {e}")
                        print("Consider using the --guess-names option for this file.")
                        sys.exit(1)
                    else:
                        # This shouldn't happen if guess_names is True and position_name_map exists
                        position_name = f"Pos{pos_idx+1}"
                        print(f"Falling back to default position name: {position_name}")

            # Handle duplicate position names by adding sequential numbering
            # Skip this if guess_names is enabled since it already includes numbering
            if guess_names and position_name_map:
                numbered_position_name = position_name
            else:
                if position_name in position_name_counts:
                    position_name_counts[position_name] += 1
                    numbered_position_name = f"{position_name}_{position_name_counts[position_name]:04d}"
                    print(f"Duplicate position name detected. Using: {numbered_position_name}")
                else:
                    position_name_counts[position_name] = 1
                    numbered_position_name = f"{position_name}_{position_name_counts[position_name]:04d}"

            # Generate base output filename
            base_name = input_path.stem
            base_output_filename = f"{base_name}_{numbered_position_name}"
            # Create a function to fetch frames for this position
            # Use local frame_indices to avoid scope issues
            def get_position_frames(pos_idx, frame_indices_local):
                if has_time:
                    # For time series data: (T, Z, C, Y, X)
                    frames = []
                    for t in range(num_time):
                        z_slices = []
                        for z in range(num_z):
                            # Find the frame number for this position, time, and z-slice
                            pos_t_z_frame_indices = [
                                (frame_num, idx)
                                for frame_num, idx in enumerate(frame_indices_local)
                                if idx.get(AXIS.POSITION, 0) == pos_idx
                                and idx.get(AXIS.TIME, 0) == t
                                and idx.get(AXIS.Z, 0) == z
                            ]

                            if not pos_t_z_frame_indices:
                                print(
                                    f"Warning: No frame found for position {pos_idx}, time {t}, z-slice {z}"
                                )
                                # Use zeros for missing frames
                                z_slices.append(
                                    np.zeros((num_channels, 512, 512), dtype=np.uint16)
                                )
                            else:
                                frame_num, _ = pos_t_z_frame_indices[0]
                                z_slices.append(f.read_frame(frame_num))
                        frames.append(np.stack(z_slices, axis=0))
                    return np.stack(frames, axis=0)  # Shape: (T, Z, C, Y, X)
                else:
                    # Original code for non-time series data: (Z, C, Y, X)
                    frames = []
                    for z in range(num_z):
                        # Find the frame number for this position and z-slice
                        pos_z_frame_indices = [
                            (frame_num, idx)
                            for frame_num, idx in enumerate(frame_indices_local)
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
                    return np.stack(frames, axis=0)  # Shape: (Z, C, Y, X)

            # Get all frames for this position, passing frame_indices to avoid scope issues
            position_data = get_position_frames(pos_idx, frame_indices)

            # Process OME metadata if not skipped
            ome_xml = None
            if not skip_ome:
                position_ome_metadata = copy.deepcopy(full_ome_metadata)

                # Keep only the image for this position in the metadata
                if (
                    hasattr(position_ome_metadata, 'images')
                    and len(position_ome_metadata.images) > pos_idx
                ):
                    # Keep only the current position's image information
                    position_ome_metadata.images = [position_ome_metadata.images[pos_idx]]

                    # Update the image metadata to reflect it's a single position
                    if hasattr(position_ome_metadata.images[0], 'name'):
                        position_ome_metadata.images[0].name = base_output_filename

                try:
                    # Convert OME metadata to XML string
                    ome_xml = position_ome_metadata.to_xml(exclude_unset=True).encode(
                        "utf-8"
                    )
                except Exception as e:
                    print(f"Warning: Could not generate OME metadata: {e}")
                    ome_xml = None

            # Determine photometric interpretation
            photometric = "rgb" if f.is_rgb else "minisblack"

            # Set common parameters for tifffile.imwrite
            common_imwrite_params = {
                "resolution": (1 / physical_pixel_size_x, 1 / physical_pixel_size_y),
                "resolutionunit": "MICROMETER",
                "photometric": photometric,
            }

            # Save the data based on the requested options
            # Skip saving regular files if max_projection is True
            if not max_projection:
                if has_time and separate_t:
                    # Save each time point separately
                    for t in range(num_time):
                        if file_prefix:
                            output_filename = f"{file_prefix}{position_name}_t{t+1:03d}.ome.tif"
                        else:
                            output_filename = f"{base_output_filename}_t{t+1:03d}.ome.tif"
                        
                        output_path = output_dir / output_filename
                        print(f"Saving time point {t+1} to: {output_path}")
                        
                        # Extract single time point: (Z, C, Y, X)
                        time_data = position_data[t]
                        
                        # Add time dimension for 5D TZCYX format (T is 1)
                        time_data_with_t = np.expand_dims(time_data, axis=0)
                        
                        # Save with modified metadata
                        tifffile.imwrite(
                            output_path,
                            time_data_with_t,
                            metadata={"axes": "TZCYX"} if not skip_ome else None,
                            description=ome_xml,
                            **common_imwrite_params
                        )
                elif has_time:
                    # For time series data, only save the full stack (no separate channels/z-slices)
                    if file_prefix:
                        output_filename = f"{file_prefix}{position_name}.ome.tif"
                    else:
                        output_filename = f"{base_output_filename}.ome.tif"
                    
                    output_path = output_dir / output_filename
                    print(f"Saving time series to: {output_path}")
                    
                    # Data is already in TZCYX format
                    # Save with full OME metadata
                    tifffile.imwrite(
                        output_path,
                        position_data,
                        metadata={"axes": "TZCYX"} if not skip_ome else None,
                        description=ome_xml,
                        **common_imwrite_params
                    )
                elif separate_channels and separate_z:
                    # Save each channel and z-slice separately
                    for c in range(num_channels):
                        for z in range(num_z):
                            if file_prefix:
                                output_filename = f"{file_prefix}{position_name}_ch{c+1}_z{z+1:03d}.tif"
                            else:
                                output_filename = f"{base_output_filename}_ch{c+1}_z{z+1:03d}.tif"
                            
                            output_path = output_dir / output_filename
                            print(f"Saving to: {output_path}")
                            
                            # Extract single channel and z-slice
                            channel_z_data = position_data[z, c]
                            
                            # Save without OME metadata (not applicable for single images)
                            tifffile.imwrite(
                                output_path,
                                channel_z_data,
                                **common_imwrite_params
                            )
                            
                elif separate_channels:
                    # Save each channel separately (with all z-slices)
                    for c in range(num_channels):
                        if file_prefix:
                            output_filename = f"{file_prefix}{position_name}_ch{c+1}.ome.tif"
                        else:
                            output_filename = f"{base_output_filename}_ch{c+1}.ome.tif"
                        
                        output_path = output_dir / output_filename
                        print(f"Saving to: {output_path}")
                        
                        # Extract single channel across all z-slices
                        channel_data = position_data[:, c]
                        
                        # Add time dimension for 5D TZCYX format expected by OME-TIFF
                        channel_data_with_t = np.expand_dims(np.expand_dims(channel_data, axis=1), axis=0)
                        
                        # Save with modified metadata
                        tifffile.imwrite(
                            output_path,
                            channel_data_with_t,
                            metadata={"axes": "TZCYX"} if not skip_ome else None,
                            description=ome_xml,
                            **common_imwrite_params
                        )
                        
                elif separate_z:
                    # Save each z-slice separately (with all channels)
                    for z in range(num_z):
                        if file_prefix:
                            output_filename = f"{file_prefix}{position_name}_z{z+1:03d}.ome.tif"
                        else:
                            output_filename = f"{base_output_filename}_z{z+1:03d}.ome.tif"
                        
                        output_path = output_dir / output_filename
                        print(f"Saving to: {output_path}")
                        
                        # Extract single z-slice with all channels
                        z_data = position_data[z]
                        
                        # Add time and z dimensions for 5D TZCYX format
                        z_data_with_tz = np.expand_dims(np.expand_dims(z_data, axis=0), axis=0)
                        
                        # Save with modified metadata
                        tifffile.imwrite(
                            output_path,
                            z_data_with_tz,
                            metadata={"axes": "TZCYX"} if not skip_ome else None,
                            description=ome_xml,
                            **common_imwrite_params
                        )
                        
                else:
                    # Standard case: Save the whole position with all channels and z-slices
                    if file_prefix:
                        output_filename = f"{file_prefix}{position_name}.ome.tif"
                    else:
                        output_filename = f"{base_output_filename}.ome.tif"
                    
                    output_path = output_dir / output_filename
                    print(f"Saving to: {output_path}")
                    
                    # Add time dimension for 5D TZCYX format expected by OME-TIFF
                    position_data_with_t = np.expand_dims(position_data, axis=0)
                    
                    # Save with full OME metadata
                    tifffile.imwrite(
                        output_path,
                        position_data_with_t,
                        metadata={"axes": "TZCYX"} if not skip_ome else None,
                        description=ome_xml,
                        **common_imwrite_params
                    )
            
            # Create and save maximum intensity projections if requested and we have multiple Z slices
            if max_projection and num_z > 1:
                if has_time and separate_t:
                    # For time series with separate_t: save each time point's max projection separately
                    # Input shape: (T, Z, C, Y, X) -> Output shape per file: (1, 1, C, Y, X)
                    max_proj_data = np.max(position_data, axis=1)  # Shape: (T, C, Y, X)
                    
                    for t in range(num_time):
                        if file_prefix:
                            proj_filename = f"{file_prefix}{position_name}_t{t+1:03d}.ome.tif"
                        else:
                            proj_filename = f"{base_output_filename}_t{t+1:03d}_max.ome.tif"
                        
                        proj_path = output_dir / proj_filename
                        print(f"Saving max projection for time point {t+1} to: {proj_path}")
                        
                        # Extract single time point and add T and Z dimensions for TZCYX format
                        time_proj = max_proj_data[t]  # Shape: (C, Y, X)
                        time_proj_with_tz = np.expand_dims(np.expand_dims(time_proj, axis=0), axis=0)
                        
                        # Save with modified metadata
                        tifffile.imwrite(
                            proj_path,
                            time_proj_with_tz,
                            metadata={"axes": "TZCYX"} if not skip_ome else None,
                            description=ome_xml,
                            **common_imwrite_params
                        )
                elif has_time:
                    # For time series: create max projection along Z for each time point
                    # Input shape: (T, Z, C, Y, X) -> Output shape: (T, C, Y, X)
                    max_proj_data = np.max(position_data, axis=1)
                    
                    # Save with time dimension
                    if file_prefix:
                        proj_filename = f"{file_prefix}{position_name}.ome.tif"
                    else:
                        proj_filename = f"{base_output_filename}_max.ome.tif"
                    
                    proj_path = output_dir / proj_filename
                    print(f"Saving time series max projection to: {proj_path}")
                    
                    # Add Z dimension (size 1) for TZCYX format
                    max_proj_with_z = np.expand_dims(max_proj_data, axis=1)
                    
                    # Save with modified metadata
                    tifffile.imwrite(
                        proj_path,
                        max_proj_with_z,
                        metadata={"axes": "TZCYX"} if not skip_ome else None,
                        description=ome_xml,
                        **common_imwrite_params
                    )
                else:
                    # Original code for non-time series data
                    # Create maximum intensity projection along the Z axis (axis 0)
                    max_proj_data = np.max(position_data, axis=0)
                    
                    if separate_channels:
                        # Save separate max projection for each channel
                        for c in range(num_channels):
                            # Apply custom prefix if provided, otherwise use the base filename
                            if file_prefix:
                                proj_filename = f"{file_prefix}{position_name}_ch{c+1}.tif"
                            else:
                                proj_filename = f"{base_output_filename}_ch{c+1}_max.tif"
                            
                            proj_path = output_dir / proj_filename
                            print(f"Saving max projection to: {proj_path}")
                            
                            # Extract the channel from max projection
                            channel_proj = max_proj_data[c]
                            
                            # Save without OME metadata
                            tifffile.imwrite(
                                proj_path,
                                channel_proj,
                                **common_imwrite_params
                            )
                    else:
                        # Save all channels in one max projection file
                        if file_prefix:
                            proj_filename = f"{file_prefix}{position_name}.ome.tif"
                        else:
                            proj_filename = f"{base_output_filename}_max.ome.tif"
                        
                        proj_path = output_dir / proj_filename
                        print(f"Saving max projection to: {proj_path}")
                        
                        # Add time and Z dimensions for 5D TZCYX format (Z is 1)
                        max_proj_with_tz = np.expand_dims(np.expand_dims(max_proj_data, axis=0), axis=0)
                        
                        # Save with modified metadata
                        tifffile.imwrite(
                            proj_path,
                            max_proj_with_tz,
                            metadata={"axes": "TZCYX"} if not skip_ome else None,
                            description=ome_xml,
                            **common_imwrite_params
                        )

    print(f"Completed processing {num_positions} positions")


def main():
    """Main entry point for the CLI application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert ND2 files to TIFF files with various options.")
    parser.add_argument("nd2_file", nargs="?", help="Path to the ND2 file (optional, can be entered interactively)")
    parser.add_argument("--skip-ome", action="store_true", help="Skip generation of OME metadata")
    parser.add_argument("--separate-channels", action="store_true", help="Save each channel as a separate file")
    parser.add_argument("--separate-z", action="store_true", help="Save each z-slice as a separate file")
    parser.add_argument("--separate-t", action="store_true", help="Save each time point as a separate file")
    parser.add_argument("--guess-names", action="store_true",
                       help="Attempt to guess missing position names by assuming sequential frames belong to the same well")
    parser.add_argument("--max-projection", action="store_true",
                       help="Create and save only maximum intensity projections along the Z axis")
    parser.add_argument("--export-folder", help="Custom export folder path (default is 'export' subfolder)")
    parser.add_argument("--file-prefix",
                        help="Prefix to add to all output filenames (e.g., '250314_PDLO_FUCCI_day7fixed_')")

    args = parser.parse_args()

    # Get file path from command line arguments or user input
    nd2_file_path = args.nd2_file
    if not nd2_file_path:
        nd2_file_path = input("Enter path to ND2 file: ")

    # Validate file exists
    if not os.path.exists(nd2_file_path):
        print(f"Error: File {nd2_file_path} does not exist")
        sys.exit(1)
    elif not nd2_file_path.lower().endswith(".nd2"):
        print(f"Error: File {nd2_file_path} is not an ND2 file")
        sys.exit(1)
    else:
        convert_nd2_to_tiff_by_well_stack(
            nd2_file_path,
            skip_ome=args.skip_ome,
            separate_channels=args.separate_channels,
            separate_z=args.separate_z,
            separate_t=args.separate_t,
            guess_names=args.guess_names,
            max_projection=args.max_projection,
            export_folder=args.export_folder,
            file_prefix=args.file_prefix
        )


if __name__ == "__main__":
    main()
