#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import imageio.v3 as iio
from tqdm import tqdm


def main():
    p_input = Path("/lustre1/users/sugawara/dpf-singularity-biology/yuki.sato/2023.06.28-30_Quail time-lapse/Day1/00_RawData/Quail_Day1_Ch1/") 
    p_output = p_input.parent / "Quail_Day1_Ch1_3d"
    p_output.mkdir(parents=True, exist_ok=True)
    for t in tqdm(range(187)):
        img = np.array([iio.imread(str(p_input / f"Quail_Day1_Ch1t{t:04d}z{z:02d}.bmp")) for z in range(25)])
        print(img.shape)
        iio.imwrite(str(p_output / f"Quail_Day1_Ch1t{t:04d}.bsdf"), img)
        break

if __name__ == "__main__":
    main()

