import argparse
from pathlib import Path
from typing import Union

import imageio.v3 as iio
from tqdm import tqdm


def main(
    input: Union[str, Path],
    output: Union[str, Path],
):
    p_input = Path(input)
    p_output = Path(output)
    p_output.mkdir(parents=True, exist_ok=True)
    img = iio.imread(str(p_input))
    num_digits = min(3, len(str(img.shape[0] - 1)))
    for t in tqdm(range(img.shape[0])):
        iio.imwrite(str(p_output / f"t{t:0{num_digits}d}.tif"), img[t])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="seq2bdv",
        description="Convert 4D image sequence to 3D+t image",
    )
    parser.add_argument(
        "input",
        type=str,
        help="input file path",
    )
    parser.add_argument(
        "output",
        type=str,
        help="output dir path for 3d+t tif files",
    )

    args = parser.parse_args()

    main(args.input, args.output)
