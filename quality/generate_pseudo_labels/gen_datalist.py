import argparse
import os
from pathlib import Path

from torchvision.datasets import ImageFolder
from tqdm import tqdm


class GenDatalistNamespace(argparse.Namespace):
    outfile: str = "./DATA.label"
    data_root: str


def parse_args() -> GenDatalistNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--outfile", help="File to write results")
    parser.add_argument("data_root", help="Path to dataset root")

    return parser.parse_args(namespace=GenDatalistNamespace())


def gentxt(data_root: str, outfile: str) -> int:
    """
    Use ImageFolder method to travel the target dataset
    Save to two files including '.label' and '.labelpath'
    """

    # output file1
    outfile_1 = open(outfile, "w")
    # output file2
    outfile_2 = open(outfile + "path", "w")
    data = ImageFolder(data_root)
    image_counter = 0

    # travel the target dataset
    for (image_path, person_index) in tqdm(data.imgs, total=len(data)):
        image_counter += 1

        path = Path(image_path)
        img = os.path.join(*path.parts[-2:])

        print(img, file=outfile_1)
        print(str(path) + "\t" + str(person_index), file=outfile_2)

    outfile_1.close()
    outfile_2.close()

    return image_counter


def main(args: GenDatalistNamespace):
    gentxt_res = gentxt(args.data_root, args.outfile)
    print(f"Total images: {gentxt_res}")


if __name__ == "__main__":
    """
    This method is to obtain data list from dataset
    and save to txt files
    """

    args = parse_args()
    main(args)
