import argparse
from pathlib import Path


def fix_mmap_length_is_greater_than_file_size_eof(preproc_path):
    """See https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/common_problems_and_solutions.md#nnu-net-training-error-mmap-length-is-greater-than-file-size-and-eoferror"""

    npys = list(preproc_path.rglob("*.npy"))

    for npy in npys:
        print(npy)

    proceed = input("Do you want to delete all files shown above? (y/n) ")

    if proceed.lower() == "y" or proceed.lower() == "yes":
        for npy in npys:
            npy.unlink()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmap",
        type=Path,
        help="Fix 'mmap length is greater than file size and EOFError'. Path to 'nnUNet_preprocessed' must be provided. See https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/common_problems_and_solutions.md#nnu-net-training-error-mmap-length-is-greater-than-file-size-and-eoferror",
    )
    args = parser.parse_args()

    if args.mmap is not None:
        # /mnt/DataRAID/nabil/projects/Glioblastoma/brats21/dataset/nnUNet_preprocessed/Task500_Brats21
        fix_mmap_length_is_greater_than_file_size_eof(args.mmap)
