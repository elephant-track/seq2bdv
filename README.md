# seq2bdv

Convert time-lapse images to BigDataviewer HDF5/XML data.

## Prerequisites

Please prepare a directory containing time-lapse image data.

```
images/
├── t000.tif
├── t001.tif
├── t002.tif
├── t003.tif
├── t004.tif
├── t005.tif
├── t006.tif
├── t007.tif
├── t008.tif
├── t009.tif
...
```

## Install

Clone this repository.

```bash
git clone git@github.com:elephant-track/seq2bdv.git
cd seq2bdv
```

- Using Poetry

```bash
poetry install
```

- Using Pip

```bash
python -m pip install .
```

## Usage

The following example shows converting 3D image stacks with the voxelsize `(0.09, 0.09, 1.0)` um^3.

```bash
poetry run python src/seq2bdv/main.py /path/to/input/images /path/to/output.xml --unit um --voxelsize 0.09 0.09 1.0
```

For 2D data, you need to explicitly specify the voxelsize with the length `2`.

```bash
poetry run python src/seq2bdv/main.py /path/to/input2d/images /path/to/output2d.xml --unit um --voxelsize 0.09 0.09
```

All options can be found below.

```
usage: seq2bdv [-h] [--unit UNIT] [--voxelsize VOXELSIZE [VOXELSIZE ...]] [--first FIRST]
               [--extention EXTENTION]
               input output

Convert image sequence to BDV .xml/.h5

positional arguments:
  input                 input directory path
  output                output path for .xml/.h5 without extension (extensions are ignored)

options:
  -h, --help            show this help message and exit
  --unit UNIT           unit for pixel (e.g. um, default: px)
  --voxelsize VOXELSIZE [VOXELSIZE ...]
                        voxel size in the order of x y z (e.g. 0.09 0.09 1.0, default: 1.0 1.0 1.0)
  --first FIRST         first timepoint
  --extention EXTENTION
                        file extention (e.g. jpg, png, tif, default: tif)
```