# Convert an ImageXpress metadata.csv to a CellProfiler LoadData CSV.
# Input has one row per channel; output has one row per field of view (image set)
# with an Image_FileName_<channel> and Image_PathName_<channel> column per channel.

library(tidyverse)

# --- Settings: edit these for your experiment -------------------------------

input_csv  <- "image_metadata_1.csv"   # ImageXpress metadata export
output_csv <- "loaddata.csv"   # CellProfiler LoadData file to create

# Folder that the image subfolders (timepoint0, timepoint1, ...) sit in.
# Leave "" to keep PathNames relative to CellProfiler's Default Input Folder,
# or set an absolute path to write absolute PathNames.
base_path <- ""

# Channel names are taken from the ExcitationEmissionFilter column. CellProfiler
# names must be alphanumeric/underscore, so any disallowed character (space,
# hyphen, etc.) is replaced by an underscore, e.g. "CFP-YFP FRET" -> "CFP_YFP_FRET".
sanitize_name <- function(x) {
  x <- gsub("[^A-Za-z0-9]+", "_", x)   # runs of disallowed characters -> underscore
  x <- gsub("^_+|_+$", "", x)          # trim leading/trailing underscores
  x
}

# --- Conversion -------------------------------------------------------------

meta <- read_csv(input_csv, show_col_types = FALSE)

# Assign a clean channel name from the filter recorded in the metadata
meta <- meta %>%
  mutate(Channel = sanitize_name(ExcitationEmissionFilter))

# Use the per-FOV subfolder as the path; optionally make it absolute
meta <- meta %>%
  mutate(PathName = ImageSubFolderPath)
if (base_path != "") {
  meta <- meta %>% mutate(PathName = file.path(base_path, ImageSubFolderPath))
}

# Reshape from one row per channel to one row per field of view, creating
# Image_FileName_<channel> and Image_PathName_<channel> columns
loaddata <- meta %>%
  select(Well, Row, Column, Field, Timepoint, ZIndex, Channel,
         FileName = ImageFileName, PathName) %>%
  pivot_wider(
    names_from  = Channel,
    values_from = c(FileName, PathName),
    names_glue  = "Image_{.value}_{Channel}"
  ) %>%
  rename(
    Metadata_Well      = Well,
    Metadata_Row       = Row,
    Metadata_Column    = Column,
    Metadata_Field     = Field,
    Metadata_Timepoint = Timepoint,
    Metadata_ZIndex    = ZIndex
  )

# Write the LoadData file for CellProfiler
write_csv(loaddata, output_csv)

# Sanity check: this should equal wells x fields x timepoints x z-planes
cat(sprintf("Wrote %d image sets to %s\n", nrow(loaddata), output_csv))
