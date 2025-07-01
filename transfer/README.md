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
#provide the connection details and credentials

omero transfer pack Plate:1 Plate1.tar

#This can be Image, Project, Dataset, Plate or Screen

#extract only metadata
omero transfer pack --binaries none Screen:1 Screen2_metadata.tar

#create a simplified folder structure
omero transfer pack --simple Plate:1 Plate1.tar

```


### issues with timeouts

In that case you could try use the omero_download_plate.sh script.
```
./omero_download.sh [OMERO_OBJECT_ID]
```
