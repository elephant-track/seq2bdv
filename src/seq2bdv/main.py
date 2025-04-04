#!/usr/bin/env python
import argparse
import math
from pathlib import Path
from typing import (
    List,
    Tuple,
    Union,
)
from xml.dom import minidom
import xml.etree.ElementTree as ET

import h5py
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

MAX_NUM_ELEMENTS = 4096
KEY_RESOLUTIONS = "resolutions"
KEY_SUBDIVISIONS = "subdivisions"
KEY_CELLS = "cells"


def validate_image(
    image: np.ndarray,
    expected_shape: Union[Tuple[int, int], Tuple[int, int, int]],
) -> None:
    if image.shape != expected_shape:
        raise ValueError(
            f"image.shape is expected to be {expected_shape}, but got {image.shape}"
        )
    if image.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(
            f"Only 8, 16, 32-bit grayscale images are supported, but got {image.dtype}"
        )


def normalize_voxelsize(
    voxelsize: Union[Tuple[float, float], Tuple[float, float, float]]
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    min_voxel_dim = float("inf")
    for d in range(len(voxelsize)):
        min_voxel_dim = min(min_voxel_dim, voxelsize[d])
    return [size / min_voxel_dim for size in voxelsize]


def suggest_pot_block_size(
    voxelscale: Union[Tuple[float, float], Tuple[float, float, float]],
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    max_num_elements: int,
) -> Union[Tuple[int, int], Tuple[int, int, int]]:
    ndim = len(voxelscale)
    bias = [0.01 * (ndim - d) for d in range(ndim)]
    shape = [1.0 / voxelscale[d] for d in range(ndim)]
    shape_vol = math.prod(shape)
    m = math.pow(max_num_elements / shape_vol, 1.0 / ndim)
    sum_num_bits = math.log(max_num_elements) / math.log(2)
    num_bits = [math.log(m * shape[d]) / math.log(2) for d in range(ndim)]
    int_num_bits = [max(0, int(num_bits[d])) for d in range(ndim)]
    if size is not None:
        full_size_bits = [
            max(0, int(math.log(size[d] - 1) / math.log(2)) + 1) for d in range(ndim)
        ]
        int_num_bits = [min(int_num_bits[d], full_size_bits[d]) for d in range(ndim)]
    sum_int_num_bits = sum(int_num_bits)
    while sum_int_num_bits + 1 <= sum_num_bits:
        max_diff = -float("inf")
        max_diff_dim = 0
        for d in range(ndim):
            if size is not None and int_num_bits[d] >= full_size_bits[d]:
                continue
            diff = num_bits[d] - int_num_bits[d] + bias[d]
            if diff > max_diff:
                max_diff = diff
                max_diff_dim = d
        int_num_bits[max_diff_dim] += 1
        sum_int_num_bits += 1
    block_size = [min(size[d], 1 << int_num_bits[d]) for d in range(ndim)]
    return block_size


def propose_mipmaps(
    shape: Union[Tuple[int, int], Tuple[int, int, int]],
    voxelsize: Union[Tuple[float, float], Tuple[float, float, float]],
):
    ndim = len(shape)
    assert len(voxelsize) == ndim
    voxelscale = normalize_voxelsize(voxelsize)
    res = [1] * ndim
    resolutions = []
    subdivisions = []
    while True:
        resolution = list(res)
        if ndim == 2:
            resolution.append(1.0)
        resolutions.append(tuple(resolution))
        size = list(shape)
        max_size = 0
        for d in range(ndim):
            size[d] = max(1, size[d] // res[d])
            max_size = max(max_size, size[d])
        subdivision = suggest_pot_block_size(voxelscale, size, MAX_NUM_ELEMENTS)
        if ndim == 2:
            subdivision.append(1)
        subdivisions.append(tuple(subdivision))

        if max_size <= 256:
            break

        any_dimension_changed = False
        for d in range(ndim):
            if voxelscale[d] <= 2.0 and size[d] > 1:
                res[d] *= 2
                voxelscale[d] *= 2
                any_dimension_changed = True

        if not any_dimension_changed:
            for d in range(ndim):
                res[d] *= 2

        voxelscale = normalize_voxelsize(voxelscale)
    return resolutions, subdivisions


def main(
    input: Union[str, Path],
    output: Union[str, Path],
    unit: str,
    voxelsize: Union[Tuple[float, float], Tuple[float, float, float]],
    first_timepoint: int = 0,
    extention: str = "tif",
):
    input = Path(input)
    output = Path(output)
    if not input.is_dir():
        raise ValueError("input needs to be a directory")

    files = list(sorted(input.glob(f"*.{extention}")))
    last_timepoint = first_timepoint + len(files) - 1
    shape = iio.imread(files[0]).shape[::-1]
    resolutions, subdivisions = propose_mipmaps(shape, voxelsize)
    ndim = len(voxelsize)
    if ndim == 2:
        voxelsize = tuple(list(voxelsize) + [1])

    output.parent.mkdir(parents=True, exist_ok=True)
    output_h5 = str(output.parent / output.stem) + ".h5"
    output_xml = str(output.parent / output.stem) + ".xml"
    elem_spimdata = ET.Element("SpimData", {"version": "0.2"})
    elem_basepath = ET.SubElement(elem_spimdata, "BasePath", {"type": "relative"})
    elem_basepath.text = "."
    elem_seqdesc = ET.SubElement(elem_spimdata, "SequenceDescription")
    elem_imgloader = ET.SubElement(elem_seqdesc, "ImageLoader", {"format": "bdv.hdf5"})
    elem_hdf5 = ET.SubElement(elem_imgloader, "hdf5", {"type": "relative"})
    elem_hdf5.text = Path(output_h5).name
    elem_viewsetups = ET.SubElement(elem_seqdesc, "ViewSetups")
    elem_viewsetup = ET.SubElement(elem_viewsetups, "ViewSetup")
    elem_elem_viewsetup_id = ET.SubElement(elem_viewsetup, "id")
    elem_elem_viewsetup_id.text = "0"
    elem_elem_viewsetup_name = ET.SubElement(elem_viewsetup, "name")
    elem_elem_viewsetup_name.text = "channel 1"
    elem_elem_viewsetup_size = ET.SubElement(elem_viewsetup, "size")
    elem_elem_viewsetup_size.text = " ".join((str(el) for el in shape))
    elem_voxelsize = ET.SubElement(elem_viewsetup, "voxelSize")
    elem_voxelsize_unit = ET.SubElement(elem_voxelsize, "unit")
    elem_voxelsize_unit.text = unit
    elem_voxelsize_size = ET.SubElement(elem_voxelsize, "size")
    elem_voxelsize_size.text = " ".join(str(el) for el in voxelsize)
    elem_viewattributes = ET.SubElement(elem_viewsetup, "attributes")
    elem_channel = ET.SubElement(elem_viewattributes, "channel")
    elem_channel.text = "1"
    elem_attributes = ET.SubElement(elem_viewsetups, "Attributes", {"name": "channel"})
    elem_channel = ET.SubElement(elem_attributes, "Channel")
    elem_channel_id = ET.SubElement(elem_channel, "id")
    elem_channel_id.text = "1"
    elem_channel_name = ET.SubElement(elem_channel, "name")
    elem_channel_name.text = "1"
    elem_timepoints = ET.SubElement(elem_seqdesc, "Timepoints", {"type": "range"})
    elem_timepoints_first = ET.SubElement(elem_timepoints, "first")
    elem_timepoints_first.text = str(first_timepoint)
    elem_timepoints_last = ET.SubElement(elem_timepoints, "last")
    elem_timepoints_last.text = str(last_timepoint)
    elem_viewregs = ET.SubElement(elem_spimdata, "ViewRegistrations")
    affine = ["0.0"] * 12
    affine[0] = str(voxelsize[0])
    affine[5] = str(voxelsize[1])
    affine[10] = str(voxelsize[2])
    for timepoint in range(first_timepoint, last_timepoint + 1):
        elem_viewreg = ET.SubElement(
            elem_viewregs,
            "ViewRegistration",
            {"timepoint": str(timepoint), "setup": "0"},
        )
        elem_viewtrans = ET.SubElement(
            elem_viewreg,
            "ViewTransform",
            {"type": "affine"},
        )
        elem_affine = ET.SubElement(elem_viewtrans, "affine")
        elem_affine.text = " ".join(affine)
    xmlstr = minidom.parseString(ET.tostring(elem_spimdata)).toprettyxml(
        indent="   ", encoding="utf-8"
    )
    with open(output_xml, "wb") as f:
        f.write(xmlstr)

    # Write HDF5
    with h5py.File(output_h5, "w") as h5:
        setup = 0
        mipmap_group = h5.create_group(f"s{setup:02d}")
        mipmap_group.create_dataset(
            name=KEY_RESOLUTIONS,
            dtype=np.float64,
            data=resolutions,
        )
        mipmap_group.create_dataset(
            name=KEY_SUBDIVISIONS,
            dtype=np.int32,
            data=subdivisions,
        )
        for timepoint in tqdm(range(first_timepoint, last_timepoint + 1)):
            img = iio.imread(files[timepoint - first_timepoint])
            if img.ndim != ndim:
                raise RuntimeError(
                    f"{ndim}d image dim is expected, but got {img.ndim}d data."
                )
            image_group = h5.create_group(f"t{timepoint:05d}").create_group(
                f"s{setup:02d}"
            )
            for i, (resolution, subdivision) in enumerate(
                zip(resolutions, subdivisions)
            ):
                resolution = resolution[::-1]
                subdivision = subdivision[::-1]
                if ndim == 2:
                    data = img[:: resolution[1], :: resolution[2]][np.newaxis]
                elif ndim == 3:
                    data = img[:: resolution[0], :: resolution[1], :: resolution[2]]
                image_group.create_group(str(i)).create_dataset(
                    name=KEY_CELLS,
                    dtype=np.int16,
                    data=data,
                    chunks=subdivision,
                    compression="gzip",
                    compression_opts=6,
                    scaleoffset=0,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="seq2bdv",
        description="Convert image sequence to BDV .xml/.h5",
    )
    parser.add_argument(
        "input",
        type=str,
        help="input directory path",
    )
    parser.add_argument(
        "output",
        type=str,
        help="output path for .xml/.h5 without extension (extensions are ignored)",
    )
    parser.add_argument(
        "--unit",
        type=str,
        required=False,
        default="px",
        help="unit for pixel (e.g. um, default: px)",
    )
    parser.add_argument(
        "--voxelsize",
        type=float,
        nargs="+",
        required=False,
        default=(1.0, 1.0, 1.0),
        help="voxel size in the order of x y z (e.g. 0.09 0.09 1.0, default: 1.0 1.0 1.0)",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=0,
        required=False,
        help="first timepoint",
    )
    parser.add_argument(
        "--extention",
        type=str,
        required=False,
        default="tif",
        help="file extention (e.g. jpg, png, tif, default: tif)",
    )

    args = parser.parse_args()

    main(args.input, args.output, args.unit, args.voxelsize, args.first, args.extention)
