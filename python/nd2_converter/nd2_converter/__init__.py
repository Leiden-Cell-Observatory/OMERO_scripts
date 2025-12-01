"""
ND2 to TIFF Converter

A modern GUI and CLI tool for converting multi-position ND2 files to TIFF files.
"""

__version__ = "1.0.0"

from .converter import convert_nd2_to_tiff_by_well_stack

__all__ = ["convert_nd2_to_tiff_by_well_stack"]
