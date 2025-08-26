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
import numpy as np

__version__ = "1.0.0"

# Try to import ome_types for proper OME-XML handling
try:
    from ome_types.model import OME, BinaryOnly
    HAS_OME_TYPES = True
except ImportError:
    HAS_OME_TYPES = False
    print("Warning: ome_types not available. Install with: pip install ome-types")
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

            # Change extension to .tif (not .ome.tif) to avoid auto-generation
            output_path = output_path.with_suffix(".tif")
            
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
                        print(f"  Generated OME-XML with UUID: {file_uuid}")
                        print(f"  XML preview: {minimal_ome[:150]}...")
                    else:
                        # Fallback to regular OME-XML if no UUID provided
                        minimal_ome = self.create_minimal_ome_xml(
                            image_data.shape, image_data.dtype
                        )
                        print(f"  XML preview: {minimal_ome[:150]}...")

                    # Save as regular TIFF with embedded OME-XML (no auto-generation)
                    # Using .tif extension prevents tifffile from auto-generating OME structure
                    tifffile.imwrite(
                        str(output_path),
                        image_data,
                        compression=None,
                        description=minimal_ome,  # Our clean OME-XML with UUID
                        photometric='minisblack' if len(image_data.shape) == 2 else 'rgb',
                        software=f"incucyte_ome_generator {__version__}",
                    )
                    
                    if file_uuid:
                        print("  Saved as .tif with embedded OME-XML and UUID")
                    else:
                        print("  Saved as .tif with embedded OME-XML")

                    # VERIFY the XML was embedded correctly
                    if file_uuid:
                        try:
                            with tifffile.TiffFile(str(output_path)) as verify_tif:
                                embedded_desc = verify_tif.pages[0].description
                                if embedded_desc and file_uuid in embedded_desc:
                                    print(f"✓ UUID successfully embedded and verified")
                                elif embedded_desc:
                                    print(f"⚠ XML embedded but UUID not found in: {embedded_desc[:100]}...")
                                else:
                                    print(f"⚠ No description/XML found in written TIFF")
                        except Exception as verify_e:
                            print(f"⚠ Could not verify embedding: {verify_e}")

                    if file_uuid:
                        print(f"✓ Converted successfully with UUID")
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
        # Ensure master UUID exists
        if not self.master_uuid:
            self.master_uuid = "urn:uuid:%s" % str(uuid_module.uuid4())
        
        # Use ome_types if available for proper OME-XML structure
        if HAS_OME_TYPES:
            try:
                print(f"  Using ome-types to generate OME-XML")
                # Create OME object with UUID
                ome = OME(uuid=file_uuid)
                
                # Add BinaryOnly reference to companion file
                binary_only = BinaryOnly(
                    metadata_file="companion.companion.ome",
                    uuid=self.master_uuid
                )
                ome.binary_only = binary_only
                
                # Convert to XML string
                xml_output = ome.to_xml()
                print(f"  ome-types generated {len(xml_output)} characters of XML")
                return xml_output
                
            except Exception as e:
                print(f"Warning: ome_types failed ({e}), falling back to manual XML")
        else:
            print(f"  ome-types not available, using manual XML generation")
        
        # Fallback to manual XML generation
        print(f"  Using manual XML generation")
        minimal_ome = f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" UUID="{file_uuid}"><BinaryOnly MetadataFile="companion.companion.ome" UUID="{self.master_uuid}"/></OME>'
        return minimal_ome

    def test_uuid_embedding(self):
        """Test UUID embedding in OME-TIFF files"""
        print("\n=== Testing UUID embedding ===")
        
        # Find test files in your converted output (now .tif files)
        test_files = list(self.output_dir.glob("*/*.tif"))
        if not test_files:
            print("No TIFF files found to test")
            return False
        
        # Test first 3 files
        for i, test_file in enumerate(test_files[:3]):
            print(f"\n--- Testing file {i+1}: {test_file.name} ---")
            
            try:
                with tifffile.TiffFile(str(test_file)) as tif:
                    # Check if description exists
                    desc = tif.pages[0].description
                    if desc:
                        print(f"Description found: {len(desc)} characters")
                        
                        # Check if it starts with XML declaration (good sign)
                        if desc.startswith('<?xml'):
                            print("✓ Starts with XML declaration")
                        else:
                            print("⚠ Does not start with XML declaration")
                        
                        # Check for nested OME elements (bad sign)
                        ome_count = desc.count('<OME')
                        if ome_count == 1:
                            print("✓ Single OME root element found")
                        elif ome_count > 1:
                            print(f"⚠ Multiple OME elements found ({ome_count}) - indicates nesting issue")
                        else:
                            print("✗ No OME elements found")
                        
                        # Print the structure for analysis
                        print(f"Full XML content:")
                        print(desc)
                        print("--- End XML ---")
                        
                        # Check for UUID patterns
                        import re
                        uuid_pattern = r'UUID="(urn:uuid:[a-f0-9-]+)"'
                        matches = re.findall(uuid_pattern, desc)
                        if matches:
                            print(f"✓ Found {len(matches)} UUID(s):")
                            for j, uuid in enumerate(matches):
                                print(f"  UUID {j+1}: {uuid}")
                        else:
                            print("✗ No UUIDs found in XML")
                        
                        # Check for BinaryOnly element
                        if "BinaryOnly" in desc:
                            print("✓ BinaryOnly element found")
                        else:
                            print("✗ BinaryOnly element not found")
                            
                        # Check for companion reference
                        if "companion.companion.ome" in desc:
                            print("✓ Companion file reference found")
                        else:
                            print("✗ Companion file reference not found")
                            
                    else:
                        print("✗ No description/XML found in TIFF")
                        
            except Exception as e:
                print(f"Error reading file: {e}")
                
        return True

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
            for tiff_file in timepoint_dir.glob("*.tif"):
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
        image_uuid = str(uuid_module.uuid4())

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

    def convert_to_single_ome_tiff(self):
        """Convert all Incucyte TIFFs to a single OME-TIFF file with plate structure"""
        structure = self.scan_structure()
        
        if not structure["timepoints"]:
            print("No timepoints found to convert")
            return False
        
        print("Converting to single OME-TIFF with plate structure...")
        
        # Collect all image data first
        wells = sorted(structure["wells"])
        channels = sorted(structure["channels"])
        timepoints = sorted(structure["timepoints"], key=lambda x: x["timestamp"])
        fields = sorted(structure["fields"])
        
        print(f"Processing: {len(wells)} wells, {len(channels)} channels, {len(timepoints)} timepoints, {len(fields)} fields")
        
        # Organize data by well -> field -> channel -> time
        organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for timepoint in timepoints:
            print(f"Processing timepoint: {timepoint['timestamp']}")
            
            for tiff_file in timepoint["path"].glob("*.tif"):
                well, field, channel = self.parse_filename(tiff_file.name)
                if well and field is not None and channel:
                    try:
                        with tifffile.TiffFile(str(tiff_file)) as tif:
                            if len(tif.pages) > 0:
                                image_data = tif.pages[0].asarray()
                                organized_data[well][field][channel].append({
                                    'data': image_data,
                                    'timepoint': timepoint['timestamp']
                                })
                    except Exception as e:
                        print(f"Error reading {tiff_file}: {e}")
        
        if not organized_data:
            print("No valid image data found")
            return False
        
        # Get image dimensions from first image
        first_well = next(iter(organized_data.keys()))
        first_field = next(iter(organized_data[first_well].keys()))
        first_channel = next(iter(organized_data[first_well][first_field].keys()))
        first_image = organized_data[first_well][first_field][first_channel][0]['data']
        height, width = first_image.shape[:2]
        dtype = first_image.dtype
        
        print(f"Image dimensions: {width}x{height}, dtype: {dtype}")
        
        # Calculate total dimensions
        num_wells = len(wells)
        num_fields = len(fields)
        num_channels = len(channels) 
        num_timepoints = len(timepoints)
        
        # Create single OME-TIFF file
        output_file = self.output_dir / "incucyte_plate.ome.tif"
        
        print(f"Writing single OME-TIFF: {output_file}")
        
        # Create OME metadata for plate structure
        pixel_size_um = 1.0  # Adjust as needed
        
        with tifffile.TiffWriter(str(output_file), bigtiff=True) as tif:
            # Build comprehensive OME metadata
            metadata = {
                'axes': 'TCYX',  # Time, Channel, Y, X (per image series)
                'SignificantBits': 16,
                'PhysicalSizeX': pixel_size_um,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': pixel_size_um, 
                'PhysicalSizeYUnit': 'µm',
                'TimeIncrement': 1.0,  # Adjust based on your time intervals
                'TimeIncrementUnit': 'h',
                'Channel': {
                    'Name': [self.get_channel_name(ch) for ch in channels],
                    'SamplesPerPixel': [1] * len(channels)
                },
                'Plate': {
                    'ID': 'Plate:0',
                    'Name': 'Incucyte_Experiment',
                    'RowNamingConvention': 'letter',
                    'ColumnNamingConvention': 'number',
                    'Rows': max(ord(w[0]) - ord('A') + 1 for w in wells),
                    'Columns': max(int(w[1:]) for w in wells)
                },
                'Description': f'Incucyte experiment with {len(wells)} wells, {len(channels)} channels, {len(timepoints)} timepoints'
            }
            
            # Write image data organized by plate position
            image_count = 0
            for well_idx, well_name in enumerate(wells):
                if well_name not in organized_data:
                    continue
                    
                for field_idx, field_num in enumerate(fields):
                    if field_num not in organized_data[well_name]:
                        continue
                    
                    # Create 4D array: TCYX (time, channel, y, x)
                    well_field_data = np.zeros((num_timepoints, num_channels, height, width), dtype=dtype)
                    
                    # Fill the array with actual data
                    for channel_idx, channel in enumerate(channels):
                        if channel in organized_data[well_name][field_num]:
                            channel_data = organized_data[well_name][field_num][channel]
                            
                            # Sort by timepoint
                            channel_data.sort(key=lambda x: x['timepoint'])
                            
                            for time_idx, time_data in enumerate(channel_data):
                                if time_idx < num_timepoints:
                                    well_field_data[time_idx, channel_idx] = time_data['data']
                    
                    # Add well/field specific metadata
                    image_metadata = metadata.copy()
                    image_metadata.update({
                        'Name': f'{well_name}_Field{field_num}',
                        'Well': {
                            'ID': f'Well:{well_idx}',
                            'Row': ord(well_name[0]) - ord('A'),
                            'Column': int(well_name[1:]) - 1
                        },
                        'WellSample': {
                            'ID': f'WellSample:{image_count}',
                            'Index': field_idx,
                            'PositionX': 0.0,  # Adjust if you have stage positions
                            'PositionY': 0.0,
                            'PositionXUnit': 'µm',
                            'PositionYUnit': 'µm'
                        }
                    })
                    
                    print(f"Writing {well_name}_Field{field_num}: {well_field_data.shape}")
                    
                    # Write this well/field as a series in the OME-TIFF
                    tif.write(
                        well_field_data,
                        photometric='minisblack',
                        compression='lzw',  # Use compression to save space
                        predictor=True,     # Improve compression
                        tile=(256, 256),    # Use tiling for better performance
                        metadata=image_metadata,
                        resolution=(10000.0/pixel_size_um, 10000.0/pixel_size_um),  # Convert to pixels per cm
                        resolutionunit='CENTIMETER'
                    )
                    
                    image_count += 1
        
        print(f"✓ Successfully created single OME-TIFF: {output_file}")
        print(f"  Contains {image_count} image series (well/field combinations)")
        print(f"  Each series: {num_timepoints}T x {num_channels}C x {height}x{width}px")
        
        return str(output_file)

    def convert_to_single_ome_tiff_simple(self):
        """Simpler version - create one big 5D array with all data"""
        structure = self.scan_structure()
        
        if not structure["timepoints"]:
            print("No timepoints found to convert")
            return False
        
        print("Converting to single 5D OME-TIFF...")
        
        wells = sorted(structure["wells"])
        channels = sorted(structure["channels"]) 
        timepoints = sorted(structure["timepoints"], key=lambda x: x["timestamp"])
        fields = sorted(structure["fields"])
        
        # For simplicity, assume 1 field per well
        if len(fields) > 1:
            print("Warning: Multiple fields detected, using field 1 only")
            fields = [1]
        
        print(f"Dimensions: {len(wells)} wells, {len(channels)} channels, {len(timepoints)} timepoints")
        
        # Get image dimensions
        first_file = None
        for timepoint in timepoints:
            for tiff_file in timepoint["path"].glob("*.tif"):
                first_file = tiff_file
                break
            if first_file:
                break
        
        if not first_file:
            print("No TIFF files found")
            return False
            
        with tifffile.TiffFile(str(first_file)) as tif:
            sample_image = tif.pages[0].asarray()
            height, width = sample_image.shape[:2]
            dtype = sample_image.dtype
        
        print(f"Image dimensions: {width}x{height}, dtype: {dtype}")
        
        # Create 5D array: PTCYX (Position/Well, Time, Channel, Y, X)
        full_data = np.zeros((len(wells), len(timepoints), len(channels), height, width), dtype=dtype)
        
        # Fill the array
        for t_idx, timepoint in enumerate(timepoints):
            print(f"Processing timepoint {t_idx+1}/{len(timepoints)}: {timepoint['timestamp']}")
            
            for tiff_file in timepoint["path"].glob("*.tif"):
                well, field, channel = self.parse_filename(tiff_file.name)
                if well and field == 1 and channel:  # Only field 1
                    try:
                        well_idx = wells.index(well)
                        channel_idx = channels.index(channel)
                        
                        with tifffile.TiffFile(str(tiff_file)) as tif:
                            if len(tif.pages) > 0:
                                image_data = tif.pages[0].asarray()
                                full_data[well_idx, t_idx, channel_idx] = image_data
                                
                    except (ValueError, IndexError) as e:
                        print(f"Skipping {tiff_file.name}: {e}")
                    except Exception as e:
                        print(f"Error reading {tiff_file.name}: {e}")
        
        # Write single OME-TIFF
        output_file = self.output_dir / "incucyte_simple.ome.tif"
        
        # Calculate plate dimensions
        max_row = max(ord(w[0]) - ord('A') + 1 for w in wells)
        max_col = max(int(w[1:]) for w in wells)
        
        metadata = {
            'axes': 'PTCYX',
            'SignificantBits': 16,
            'PhysicalSizeX': 1.0,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': 1.0,
            'PhysicalSizeYUnit': 'µm',
            'TimeIncrement': 1.0,
            'TimeIncrementUnit': 'h',
            'Channel': {
                'Name': [self.get_channel_name(ch) for ch in channels]
            },
            'Plate': {
                'Name': 'Incucyte_Experiment',
                'Rows': max_row,
                'Columns': max_col
            },
            'Well': [{'Name': well, 'Row': ord(well[0]) - ord('A'), 'Column': int(well[1:]) - 1} for well in wells]
        }
        
        print(f"Writing OME-TIFF: {output_file}")
        
        tifffile.imwrite(
            str(output_file),
            full_data,
            bigtiff=True,
            photometric='minisblack',
            compression='lzw',
            predictor=True,
            tile=(256, 256),
            metadata=metadata
        )
        
        print(f"✓ Created single OME-TIFF: {output_file}")
        print(f"  Shape: {full_data.shape} (Wells, Time, Channels, Y, X)")
        
        return str(output_file)

    def convert_to_well_field_ome_tiffs(self):
        """Convert Incucyte TIFFs to separate OME-TIFF files per well and field combination"""
        structure = self.scan_structure()
        
        if not structure["timepoints"]:
            print("No timepoints found to convert")
            return []
        
        print("Converting to separate OME-TIFF files per well and field...")
        
        wells = sorted(structure["wells"])
        channels = sorted(structure["channels"])
        timepoints = sorted(structure["timepoints"], key=lambda x: x["timestamp"])
        fields = sorted(structure["fields"])
        
        print(f"Processing: {len(wells)} wells, {len(channels)} channels, {len(timepoints)} timepoints, {len(fields)} fields")
        
        # Organize data by well -> field -> channel -> time
        organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for timepoint in timepoints:
            print(f"Processing timepoint: {timepoint['timestamp']}")
            
            for tiff_file in timepoint["path"].glob("*.tif"):
                well, field, channel = self.parse_filename(tiff_file.name)
                if well and field is not None and channel:
                    try:
                        with tifffile.TiffFile(str(tiff_file)) as tif:
                            if len(tif.pages) > 0:
                                image_data = tif.pages[0].asarray()
                                organized_data[well][field][channel].append({
                                    'data': image_data,
                                    'timepoint': timepoint['timestamp']
                                })
                    except Exception as e:
                        print(f"Error reading {tiff_file}: {e}")
        
        if not organized_data:
            print("No valid image data found")
            return []
        
        # Get image dimensions from first image
        first_well = next(iter(organized_data.keys()))
        first_field = next(iter(organized_data[first_well].keys()))
        first_channel = next(iter(organized_data[first_well][first_field].keys()))
        first_image = organized_data[first_well][first_field][first_channel][0]['data']
        height, width = first_image.shape[:2]
        dtype = first_image.dtype
        
        print(f"Image dimensions: {width}x{height}, dtype: {dtype}")
        
        # Create output directory for well-field files
        well_field_dir = self.output_dir / "well_field_ome_tiffs"
        well_field_dir.mkdir(exist_ok=True)
        
        created_files = []
        pixel_size_um = 1.0  # Adjust as needed
        
        # Process each well-field combination
        for well_name in wells:
            if well_name not in organized_data:
                continue
                
            for field_num in fields:
                if field_num not in organized_data[well_name]:
                    continue
                
                print(f"Creating OME-TIFF for {well_name}_Field{field_num}")
                
                # Create 4D array: TCYX (time, channel, y, x)
                well_field_data = np.zeros((len(timepoints), len(channels), height, width), dtype=dtype)
                
                # Fill the array with actual data
                for channel_idx, channel in enumerate(channels):
                    if channel in organized_data[well_name][field_num]:
                        channel_data = organized_data[well_name][field_num][channel]
                        
                        # Sort by timepoint
                        channel_data.sort(key=lambda x: x['timepoint'])
                        
                        for time_idx, time_data in enumerate(channel_data):
                            if time_idx < len(timepoints):
                                well_field_data[time_idx, channel_idx] = time_data['data']
                
                # Create filename: WellName_Field{X}.ome.tif
                output_filename = f"{well_name}_Field{field_num}.ome.tif"
                output_path = well_field_dir / output_filename
                
                # Create metadata for this well/field
                metadata = {
                    'axes': 'TCYX',  # Time, Channel, Y, X
                    'SignificantBits': 16,
                    'PhysicalSizeX': pixel_size_um,
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': pixel_size_um,
                    'PhysicalSizeYUnit': 'µm',
                    'TimeIncrement': 1.0,  # Adjust based on your time intervals
                    'TimeIncrementUnit': 'h',
                    'Channel': {
                        'Name': [self.get_channel_name(ch) for ch in channels],
                        'SamplesPerPixel': [1] * len(channels)
                    },
                    'Description': f'Incucyte {well_name} Field {field_num} - {len(timepoints)} timepoints, {len(channels)} channels',
                    'Name': f'{well_name}_Field{field_num}',
                    'Well': {
                        'Name': well_name,
                        'Row': ord(well_name[0]) - ord('A'),
                        'Column': int(well_name[1:]) - 1
                    },
                    'Field': field_num
                }
                
                print(f"  Writing {output_filename}: {well_field_data.shape} (T={len(timepoints)}, C={len(channels)}, Y={height}, X={width})")
                
                # Write the OME-TIFF file
                tifffile.imwrite(
                    str(output_path),
                    well_field_data,
                    bigtiff=True,
                    photometric='minisblack',
                    compression='lzw',  # Use compression to save space
                    predictor=True,     # Improve compression
                    tile=(256, 256),    # Use tiling for better performance
                    metadata=metadata,
                    resolution=(10000.0/pixel_size_um, 10000.0/pixel_size_um),  # Convert to pixels per cm
                    resolutionunit='CENTIMETER'
                )
                
                created_files.append({
                    'file': output_path,
                    'well': well_name,
                    'field': field_num,
                    'shape': well_field_data.shape,
                    'channels': len(channels),
                    'timepoints': len(timepoints)
                })
                
                print(f"  ✓ Created: {output_filename}")
        
        print(f"\n✓ Successfully created {len(created_files)} OME-TIFF files in: {well_field_dir}")
        print(f"  Files contain: {len(timepoints)} timepoints × {len(channels)} channels × {height}×{width} pixels")
        
        # Print summary of created files
        print("\nCreated files:")
        for file_info in created_files:
            print(f"  {file_info['file'].name} - {file_info['well']} Field {file_info['field']} - {file_info['timepoints']}T × {file_info['channels']}C")
        
        return created_files


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
    parser.add_argument(
        "--single-ome-tiff",
        action="store_true",
        help="Create single OME-TIFF with plate structure instead of companion file",
    )
    parser.add_argument(
        "--simple-format",
        action="store_true",
        help="Use simpler 5D array format (use with --single-ome-tiff)",
    )
    parser.add_argument(
        "--well-field-ome-tiffs",
        action="store_true",
        help="Create separate OME-TIFF files per well and field combination",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    try:
        converter = IncucyteConverter(args.input_dir, args.output_dir)

        if args.well_field_ome_tiffs:
            print("Creating separate OME-TIFF files per well and field...")
            result = converter.convert_to_well_field_ome_tiffs()
            
            if result:
                print(f"✓ Created {len(result)} OME-TIFF files successfully")
            else:
                print("✗ Failed to create well-field OME-TIFF files")
                
        elif args.single_ome_tiff:
            print("Creating single OME-TIFF with plate structure...")
            if args.simple_format:
                result = converter.convert_to_single_ome_tiff_simple()
            else:
                result = converter.convert_to_single_ome_tiff()
            
            if result:
                print(f"✓ Single OME-TIFF created successfully: {result}")
            else:
                print("✗ Failed to create single OME-TIFF")
                
        else:
            # Original multi-file approach
            if not args.skip_conversion:
                print("Starting TIFF conversion...")
                converted_files = converter.convert_tiffs_with_tifffile()
                print(f"Converted {len(converted_files)} files")
            else:
                print("Skipping conversion, looking for existing converted files...")
                converted_files = converter.find_converted_files()
                print(f"Found {len(converted_files)} converted files")

            if converted_files:
                # Test UUID embedding
                converter.test_uuid_embedding()
                
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
