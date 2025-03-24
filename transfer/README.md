# Setup OMERO CLI
OMERO CLI only works only Linux (possibly Mac)

### install conda
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

```
### create dedicated conda environment

```

conda create -n omero -c ome python=3.6 zeroc-ice36-python omero-py
conda activate omero

```
### Connect to OMERO

```
omero login
omero transfer pack Plate:1 Plate1.tar

#extract only metadata
omero transfer pack --binaries none Screen:1 Screen2_metadata.tar


```

### issues with timeouts

In that case you could try use the omero_download_plate.sh script.

```
./omero_download.sh [OMERO_OBJECT_ID]
```
