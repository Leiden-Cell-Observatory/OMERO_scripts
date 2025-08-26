#!/usr/bin/env python
"""
Incucyte OME Companion File Generator with TIFF Conversion

This script:
1. Converts Incucyte pyramid TIFFs to single-plane TIFFs using tifffile
2. Generates OME companion files for the converted data

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

Requires: pip install tifffile
"""

import os
import re
import sys
import uuid as uuid_module
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import tifffile
import argparse

__version__ = "1.1.0"

# OME companion file constants - master UUID will be set dynamically
OME_ATTRIBUTES_TEMPLATE = {
    "Creator": "incucyte_ome_generator %s" % __version__,
    "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
    "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 \
http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
}


class Channel(object):
    """OME Channel object"""

    ID = 0

    def __init__(self, image, name=None, color=None, samplesPerPixel=1):
        self.data = {
            "ID": "Channel:%s" % self.ID,
            "SamplesPerPixel": str(samplesPerPixel),
        }
        if name:
            self.data["Name"] = name
        if color:
            self.data["Color"] = str(color)
        Channel.ID += 1


class Plane(object):
    """OME Plane object"""

    ALLOWED_KEYS = (
        "DeltaT",
        "DeltaTUnit",
        "ExposureTime",
        "ExposureTimeUnit",
        "PositionX",
        "PositionXUnit",
        "PositionY",
        "PositionYUnit",
        "PositionZ",
        "PositionZUnit",
    )

    def __init__(self, TheC=0, TheZ=0, TheT=0, options={}):
        self.data = {
            "TheC": str(TheC),
            "TheZ": str(TheZ),
            "TheT": str(TheT),
        }
        if options:
            for key, value in options.items():
                if key in self.ALLOWED_KEYS:
                    self.data[key] = value


class UUID(object):
    """OME UUID object"""

    def __init__(self, filename=None, uuid_value=None):
        self.data = {"FileName": filename}
        # Use provided UUID or generate new one
        if uuid_value:
            self.value = uuid_value
        else:
            self.value = "urn:uuid:%s" % str(uuid_module.uuid4())


class TiffData(object):
    """OME TiffData object"""

    def __init__(
        self, firstC=0, firstT=0, firstZ=0, ifd=None, planeCount=None, uuid=None
    ):
        self.data = {
            "FirstC": str(firstC),
            "FirstT": str(firstT),
            "FirstZ": str(firstZ),
        }
        self.uuid = uuid
        # Always include IFD and PlaneCount for standard TIFFs
        self.data["IFD"] = str(ifd) if ifd is not None else "0"
        self.data["PlaneCount"] = str(planeCount) if planeCount is not None else "1"


class Image(object):
    """OME Image object"""

    ID = 0

    def __init__(
        self,
        name,
        sizeX,
        sizeY,
        sizeZ,
        sizeC,
        sizeT,
        tiffs=[],
        order="XYZTC",
        type="uint16",
        physSizeX="1.0",
        physSizeY="1.0",
        physSizeZ="1.0",
        unitX="µm",
        unitY="µm",
        unitZ="µm",
    ):
        self.data = {
            "Image": {"ID": "Image:%s" % self.ID, "Name": name},
            "Pixels": {
                "ID": "Pixels:%s:%s" % (self.ID, self.ID),
                "DimensionOrder": order,
                "Type": type,
                "SizeX": str(sizeX),
                "SizeY": str(sizeY),
                "SizeZ": str(sizeZ),
                "SizeT": str(sizeT),
                "SizeC": str(sizeC),
                "PhysicalSizeX": str(physSizeX),
                "PhysicalSizeXUnit": unitX,
                "PhysicalSizeY": str(physSizeY),
                "PhysicalSizeYUnit": unitY,
                "PhysicalSizeZ": str(physSizeZ),
                "PhysicalSizeZUnit": unitZ,
            },
            "Channels": [],
            "TIFFs": [],
            "Planes": [],
        }
        Image.ID += 1
        for tiff in tiffs:
            self.add_tiff(tiff)

    def add_channel(self, name=None, color=None, samplesPerPixel=1):
        """Add a channel to the image"""
        self.data["Channels"].append(
            Channel(self, name=name, color=color, samplesPerPixel=samplesPerPixel)
        )

    def add_tiff(self, filename, c=0, t=0, z=0, ifd=None, planeCount=None, uuid_value=None):
        """Add a TIFF file reference to the image with optional UUID"""
        self.data["TIFFs"].append(
            TiffData(
                firstC=c,
                firstT=t,
                firstZ=z,
                ifd=ifd,
                planeCount=planeCount,
                uuid=UUID(filename, uuid_value=uuid_value),
            )
        )

    def add_plane(self, c=0, t=0, z=0, options={}):
        """Add plane metadata"""
        assert c >= 0 and c < int(self.data["Pixels"]["SizeC"])
        assert z >= 0 and z < int(self.data["Pixels"]["SizeZ"])
        assert t >= 0 and t < int(self.data["Pixels"]["SizeT"])
        self.data["Planes"].append(Plane(TheC=c, TheT=t, TheZ=z, options=options))

    def validate(self):
        """Validate image metadata"""
        sizeC = int(self.data["Pixels"]["SizeC"])
        assert len(self.data["Channels"]) <= sizeC, str(self.data)
        channel_samples = sum(
            [int(x.data["SamplesPerPixel"]) for x in self.data["Channels"]]
        )
        assert channel_samples <= sizeC, str(self.data)
        return self.data


class Plate(object):
    """OME Plate object"""

    ID = 0

    def __init__(self, name, rows=None, columns=None):
        self.data = {
            "Plate": {"ID": "Plate:%s" % self.ID, "Name": name},
            "Wells": [],
        }
        Plate.ID += 1

    def add_well(self, row, column):
        """Add a well to the plate"""
        well = Well(self, row, column)
        self.data["Wells"].append(well)
        return well


class Well(object):
    """OME Well object"""

    ID = 0

    def __init__(self, plate, row, column):
        self.data = {
            "Well": {
                "ID": "Well:%s" % self.ID,
                "Row": "%s" % row,
                "Column": "%s" % column,
            },
            "WellSamples": [],
        }
        Well.ID += 1

    def add_wellsample(self, index, image):
        """Add a well sample (field) to the well"""
        wellsample = WellSample(self, index, image)
        self.data["WellSamples"].append(wellsample)
        return wellsample


class WellSample(object):
    """OME WellSample object"""

    ID = 0

    def __init__(self, well, index, image):
        self.data = {
            "WellSample": {
                "ID": "WellSample:%s" % self.ID,
                "Index": "%s" % index,
            },
            "Image": image,
        }
        WellSample.ID += 1


def create_companion(plates=[], images=[], out=None, master_uuid=None):
    """Create a companion OME-XML for a given experiment with proper UUID"""
    # Generate master UUID if not provided
    if not master_uuid:
        master_uuid = "urn:uuid:%s" % str(uuid_module.uuid4())
    
    # Create OME attributes with the master UUID
    ome_attributes = OME_ATTRIBUTES_TEMPLATE.copy()
    ome_attributes["UUID"] = master_uuid
    
    root = ET.Element("OME", attrib=ome_attributes)

    for plate in plates:
        p = ET.SubElement(root, "Plate", attrib=plate.data["Plate"])
        for well in plate.data["Wells"]:
            w = ET.SubElement(p, "Well", attrib=well.data["Well"])
            for wellsample in well.data["WellSamples"]:
                ws = ET.SubElement(
                    w, "WellSample", attrib=wellsample.data["WellSample"]
                )
                image = wellsample.data["Image"]
                images.append(image)
                ET.SubElement(ws, "ImageRef", attrib={"ID": image.data["Image"]["ID"]})

    for img in images:
        i = img.validate()
        image = ET.SubElement(root, "Image", attrib=i["Image"])
        pixels = ET.SubElement(image, "Pixels", attrib=i["Pixels"])

        for channel in i["Channels"]:
            c = channel.data
            ET.SubElement(pixels, "Channel", attrib=c)

        for tiff in i["TIFFs"]:
            tiffdata = ET.SubElement(pixels, "TiffData", attrib=tiff.data)
            if tiff.uuid:
                ET.SubElement(tiffdata, "UUID", tiff.uuid.data).text = tiff.uuid.value

        for plane in i["Planes"]:
            ET.SubElement(pixels, "Plane", attrib=plane.data)

    # Write XML with proper formatting
    if not out:
        out = sys.stdout

    # For file objects, write as string; for stdout, handle encoding properly
    if hasattr(out, "write"):
        xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
        out.write(xml_str)
    else:
        ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    
    return master_uuid


class IncucyteConverter:
    """
    Convert Incucyte pyramid TIFFs to single-plane TIFFs and generate OME companion

    Handles the specific directory structure:
    EssenFiles/ScanData/YYMM/DD/HHMM/XXXX/*.tif

    Filenames follow pattern: WELL-FIELD-CHANNEL.tif
    e.g., A1-1-C1.tif, B2-1-Ph.tif
    """

    def __init__(self, base_path, output_dir=None):
        self.base_path = Path(base_path)
        self.scan_data_path = self.base_path / "EssenFiles" / "ScanData"
        self.output_dir = (
            Path(output_dir) if output_dir else self.base_path / "converted"
        )
        self.output_dir.mkdir(exist_ok=True)
        # Store UUID mappings
        self.file_uuids = {}
        self.master_uuid = None

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
        """Convert well name (A1, B2) to row,column indices (0,0), (1,1)"""
        row = ord(well_name[0]) - ord("A")
        col = int(well_name[1:]) - 1
        return row, col

    def determine_plate_dimensions(self, wells):
        """Determine plate dimensions from actual well names"""
        if not wells:
            return 8, 12  # Default 96-well plate

        max_row = 0
        max_col = 0

        for well in wells:
            row_letter = well[0]
            col_number = int(well[1:])

            row_idx = ord(row_letter) - ord("A")
            max_row = max(max_row, row_idx + 1)
            max_col = max(max_col, col_number)

        return max_row, max_col

    def get_channel_name(self, channel_code):
        """Convert Incucyte channel codes to readable names"""
        mapping = {
            "C1": "Green",
            "C2": "Red",
            "Ph": "Phase_Contrast",
            "P": "Phase_Contrast",  # Alternative phase contrast naming
        }
        return mapping.get(channel_code, channel_code)

    def convert_tiffs_with_tifffile(self):
        """Convert pyramid TIFFs to single-plane TIFFs using tifffile with UUID support"""
        structure = self.scan_structure()
        converted_files = []

        print("Converting pyramid TIFFs to single-plane TIFFs using tifffile...")
        
        # Generate master UUID for companion file
        self.master_uuid = "urn:uuid:%s" % str(uuid_module.uuid4())

        # Create organized output structure
        for timepoint in structure["timepoints"]:
            timepoint_dir = self.output_dir / timepoint["timestamp"]
            timepoint_dir.mkdir(exist_ok=True)

            for tiff_file in timepoint["path"].glob("*.tif"):
                well, field, channel = self.parse_filename(tiff_file.name)
                if well and field is not None and channel:
                    # Output filename with .ome.tif extension
                    output_name = f"{well}_F{field:02d}_{channel}_T{timepoint['timestamp']}.ome.tif"
                    output_path = timepoint_dir / output_name
                    
                    # Generate UUID for this file
                    file_uuid = "urn:uuid:%s" % str(uuid_module.uuid4())
                    # Use forward slashes for OME compatibility
                    relative_path = str(output_path.relative_to(self.output_dir.parent)).replace("\\", "/")
                    self.file_uuids[relative_path] = file_uuid

                    # Convert using tifffile with UUID support
                    if self.convert_single_tiff_with_uuid(tiff_file, output_path, file_uuid):
                        converted_files.append(
                            {
                                "original": tiff_file,
                                "converted": output_path,
                                "well": well,
                                "field": field,
                                "channel": channel,
                                "timepoint": timepoint,
                                "uuid": file_uuid,
                            }
                        )

        return converted_files

    def convert_single_tiff(self, input_path, output_path):
        """Convert a single TIFF file using tifffile to extract full resolution image (legacy)"""
        # This method is kept for backward compatibility but not used with UUID support
        return self.convert_single_tiff_with_uuid(input_path, output_path, None)
    
    def convert_single_tiff_with_uuid(self, input_path, output_path, file_uuid):
        """Convert a single TIFF file with proper UUID embedding"""
        try:
            

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Change extension to .ome.tif for proper OME-TIFF format
            output_path = output_path.with_suffix(".ome.tif")

            print(f"Converting: {input_path.name} -> {output_path.name}")

            # Read the TIFF file with tifffile
            with tifffile.TiffFile(str(input_path)) as tif:
                # Debug: print TIFF structure
                print(f"  Original TIFF has {len(tif.pages)} pages")

                # For pyramid TIFFs, the first page is usually the full resolution
                # Get the first page (full resolution)
                if len(tif.pages) > 0:
                    page = tif.pages[0]
                    image_data = page.asarray()

                    # Debug: print image statistics
                    print(
                        f"  Image shape: {image_data.shape}, dtype: {image_data.dtype}"
                    )
                    print(f"  Image range: {image_data.min()} - {image_data.max()}")

                    # Create minimal OME-XML with BinaryOnly element for multi-file dataset
                    if file_uuid:
                        minimal_ome = self.create_minimal_ome_xml_with_uuid(file_uuid)
                    else:
                        # Fallback to regular OME-XML if no UUID provided
                        minimal_ome = self.create_minimal_ome_xml(
                            image_data.shape, image_data.dtype
                        )

                    # Save as OME-TIFF with embedded UUID metadata
                    tifffile.imwrite(
                        str(output_path),
                        image_data,
                        imagej=False,  # Don't use ImageJ format for proper OME-TIFF
                        photometric="minisblack"
                        if len(image_data.shape) == 2
                        else "rgb",
                        description=minimal_ome,  # Embed the minimal OME-XML
                    )

                    if file_uuid:
                        print(f"✓ Converted successfully with UUID: {file_uuid[:50]}...")
                    else:
                        print(f"✓ Converted successfully")
                    return True
                else:
                    print(f"✗ No pages found in TIFF file")
                    return False

        except ImportError:
            print(
                "Error: tifffile package not found. Please install with: pip install tifffile"
            )
            return False
        except Exception as e:
            print(f"Exception during conversion of {input_path.name}: {e}")
            return False
    
    def create_minimal_ome_xml_with_uuid(self, file_uuid):
        """Create minimal OME-XML with BinaryOnly element referencing companion file"""
        minimal_ome = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
     http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
     UUID="{file_uuid}">
   <BinaryOnly MetadataFile="companion.companion.ome" UUID="{self.master_uuid}"/>
</OME>'''
        return minimal_ome

    def find_converted_files(self):
        """Find already converted files in output directory"""
        converted_files = []
        
        # Generate master UUID if not already set
        if not self.master_uuid:
            self.master_uuid = "urn:uuid:%s" % str(uuid_module.uuid4())

        for timepoint_dir in self.output_dir.iterdir():
            if not timepoint_dir.is_dir():
                continue

            timestamp = timepoint_dir.name
            for tiff_file in timepoint_dir.glob("*.ome.tif"):
                # Parse converted filename: WELL_FXX_CHANNEL_TTIMESTAMP.ome.tif
                match = re.match(
                    r"([A-Z]\d+)_F(\d+)_(.+)_T(.+)\.ome\.tif", tiff_file.name
                )
                if match:
                    well = match.group(1)
                    field = int(match.group(2))
                    channel = match.group(3)
                    
                    # Use forward slashes for OME compatibility
                    relative_path = str(tiff_file.relative_to(self.output_dir.parent)).replace("\\", "/")
                    # Get or create UUID for existing file
                    if relative_path not in self.file_uuids:
                        self.file_uuids[relative_path] = "urn:uuid:%s" % str(uuid_module.uuid4())

                    converted_files.append(
                        {
                            "converted": tiff_file,
                            "well": well,
                            "field": field,
                            "channel": channel,
                            "timepoint": {"timestamp": timestamp},
                            "uuid": self.file_uuids[relative_path],
                        }
                    )

        return converted_files

    def create_companion_for_converted(self, converted_files, output_file=None):
        """Create OME companion file for converted TIFFs with proper UUID references"""
        if not converted_files:
            raise ValueError("No converted files found")

        # Analyze converted files structure
        wells = set()
        channels = set()
        fields = set()
        timepoints = set()

        for file_info in converted_files:
            wells.add(file_info["well"])
            channels.add(file_info["channel"])
            fields.add(file_info["field"])
            timepoints.add(file_info["timepoint"]["timestamp"])

        # Sort for consistent ordering
        wells = sorted(wells)
        channels = sorted(channels)
        fields = sorted(fields)
        timepoints = sorted(timepoints)

        print(
            f"Creating companion for: {len(wells)} wells, {len(channels)} channels, {len(timepoints)} timepoints"
        )

        # Create plate
        rows, cols = self.determine_plate_dimensions(wells)
        plate = Plate("Incucyte_Converted")

        # Create images
        images = []
        for well_name in wells:
            row, col = self.parse_well_position(well_name)
            well = plate.add_well(row, col)

            for field_num in fields:
                field_files = [
                    f
                    for f in converted_files
                    if f["well"] == well_name and f["field"] == field_num
                ]

                if field_files:
                    image = self.create_image_from_converted_files(
                        field_files, well_name, field_num, channels, timepoints
                    )
                    images.append(image)
                    well.add_wellsample(field_num - 1, image)

        # Generate companion file with master UUID
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                master_uuid = create_companion(plates=[plate], images=images, out=f, master_uuid=self.master_uuid)
            print(f"OME companion file written to: {output_file}")
            print(f"Master UUID: {master_uuid}")
        else:
            create_companion(plates=[plate], images=images, master_uuid=self.master_uuid)

        return [plate], images

    def create_image_from_converted_files(
        self, files, well_name, field_num, all_channels, all_timepoints
    ):
        """Create Image object from converted files"""
        # Get actual image dimensions from first file
        first_file = files[0]["converted"]
        sizeX, sizeY = self.get_tiff_dimensions(first_file)

        sizeT = len(all_timepoints)
        sizeC = len(all_channels)
        sizeZ = 1

        print(
            f"Creating image for {well_name}_Field{field_num}: {sizeX}x{sizeY}, {sizeC}C x {sizeT}T x {sizeZ}Z"
        )
        print(f"  Files for this image: {len(files)}")

        image_name = f"{well_name}_Field{field_num}"
        image = Image(
            image_name,
            sizeX=sizeX,
            sizeY=sizeY,
            sizeZ=sizeZ,
            sizeC=sizeC,
            sizeT=sizeT,
            order="XYZTC",
            type="uint16",
        )

        # Add channels
        for channel in all_channels:
            channel_name = self.get_channel_name(channel)
            image.add_channel(name=channel_name)
            print(f"  Added channel: {channel} -> {channel_name}")

        # Add TIFF references with UUIDs
        for file_info in files:
            c_idx = all_channels.index(file_info["channel"])
            t_idx = all_timepoints.index(file_info["timepoint"]["timestamp"])

            # Relative path from output directory's parent - use forward slashes for OME compatibility
            relative_path = str(
            file_info["converted"].relative_to(self.output_dir.parent)
            )
            # Convert Windows backslashes to forward slashes for OME compatibility
            relative_path = relative_path.replace("\\", "/")
            
            # Get UUID for this file
            uuid_value = file_info.get("uuid") or self.file_uuids.get(relative_path)
        
            print(f"  Adding TIFF: {relative_path} -> C={c_idx}, T={t_idx}, Z=0, UUID={uuid_value[:50] if uuid_value else 'None'}...")
            image.add_tiff(relative_path, c=c_idx, t=t_idx, z=0, ifd=0, planeCount=1, uuid_value=uuid_value)

        return image

    def create_minimal_ome_xml(self, shape, dtype):
        """Create minimal OME-XML metadata for a single image plane"""
        # Map numpy dtypes to OME types
        dtype_map = {
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "float32": "float",
            "float64": "double",
        }

        ome_type = dtype_map.get(str(dtype), "uint16")
        height, width = shape[:2]

        # Generate unique UUID for this image
        image_uuid = str(uuid.uuid4())

        ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
     Creator="incucyte_ome_generator {__version__}"
     UUID="urn:uuid:{image_uuid}">
    <Image ID="Image:0" Name="Converted_Image">
        <Pixels ID="Pixels:0" DimensionOrder="XYZTC" Type="{ome_type}"
                SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="1" SizeT="1">
            <Channel ID="Channel:0" SamplesPerPixel="1"/>
            <TiffData IFD="0" PlaneCount="1"/>
        </Pixels>
    </Image>
</OME>'''
        return ome_xml

    def get_tiff_dimensions(self, tiff_path):
        """Get dimensions from a TIFF file"""
        try:

            with tifffile.TiffFile(str(tiff_path)) as tif:
                page = tif.pages[0]
                return page.shape[1], page.shape[0]  # width, height
        except ImportError:
            print("tifffile not available, using default dimensions")
            return 2048, 2048
        except Exception as e:
            print(f"Error reading TIFF dimensions: {e}, using defaults")
            return 2048, 2048


def main():
    """Main function"""

    parser = argparse.ArgumentParser(
        description="Convert Incucyte TIFFs and generate OME companion file"
    )
    parser.add_argument("input_dir", help="Path to Incucyte export directory")
    parser.add_argument(
        "-o", "--output-dir", help="Output directory for converted files", default=None
    )
    parser.add_argument(
        "-c",
        "--companion",
        help="Companion file name",
        default="companion.companion.ome",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip TIFF conversion, only create companion",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    try:
        converter = IncucyteConverter(args.input_dir, args.output_dir)

        if not args.skip_conversion:
            print("Starting TIFF conversion...")
            converted_files = converter.convert_tiffs_with_tifffile()
            print(f"Converted {len(converted_files)} files")
        else:
            print("Skipping conversion, looking for existing converted files...")
            converted_files = converter.find_converted_files()
            print(f"Found {len(converted_files)} converted files")

        if converted_files:
            # Save companion file at the same level as the converted folder, not inside it
            companion_path = converter.base_path / args.companion
            converter.create_companion_for_converted(converted_files, companion_path)
            print("Conversion and companion file generation completed!")
        else:
            print("No files to process")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
