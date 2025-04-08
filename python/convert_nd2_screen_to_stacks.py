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
# TO DO add option to rename files based on metadata

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
    guess_names=False,
    max_projection=False
):
    """
    Convert an ND2 file containing well plate images to separate TIFF files.

    Args:
        nd2_path (str): Path to the input ND2 file
        skip_ome (bool): If True, skip generation of OME metadata
        separate_channels (bool): If True, save each channel as a separate file
        separate_z (bool): If True, save each z-slice as a separate file
        guess_names (bool): If True, attempt to guess missing position names
    """
    # Create output directory
    input_path = Path(nd2_path)
    output_dir = input_path.parent / "export"
    output_dir.mkdir(exist_ok=True)

    print(f"Processing file: {input_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Options: skip_ome={skip_ome}, separate_channels={separate_channels}, "
          f"separate_z={separate_z}, guess_names={guess_names}, max_projection={max_projection}")

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

            # Generate base output filename
            base_name = input_path.stem
            base_output_filename = f"{base_name}_{position_name}"

            # Create a function to fetch frames for this position
            # Use local frame_indices to avoid scope issues
            def get_position_frames(pos_idx, frame_indices_local):
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
                return np.stack(frames, axis=0)

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
                        position_ome_metadata.images[
                            0
                        ].name = base_output_filename

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
            if separate_channels and separate_z:
                # Save each channel and z-slice separately
                for c in range(num_channels):
                    for z in range(num_z):
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
                # Create maximum intensity projection along the Z axis (axis 0)
                max_proj_data = np.max(position_data, axis=0)
                
                if separate_channels:
                    # Save separate max projection for each channel
                    for c in range(num_channels):
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


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert ND2 files to TIFF files with various options.")
    parser.add_argument("nd2_file", nargs="?", help="Path to the ND2 file (optional, can be entered interactively)")
    parser.add_argument("--skip-ome", action="store_true", help="Skip generation of OME metadata")
    parser.add_argument("--separate-channels", action="store_true", help="Save each channel as a separate file")
    parser.add_argument("--separate-z", action="store_true", help="Save each z-slice as a separate file")
    parser.add_argument("--guess-names", action="store_true", 
                       help="Attempt to guess missing position names by assuming sequential frames belong to the same well")
    parser.add_argument("--max-projection", action="store_true",
                       help="Create maximum intensity projections along the Z axis")
    
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
            guess_names=args.guess_names,
            max_projection=args.max_projection
        )