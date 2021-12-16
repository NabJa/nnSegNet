import argparse
from pathlib import Path
import json
from collections import OrderedDict
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm


def prepare_brats_json(target_base, pateint_names):

    target_base = Path(target_base)

    json_dict = OrderedDict()
    json_dict["name"] = "BraTS2021"
    json_dict["description"] = "nothing"
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = "see BraTS2021"
    json_dict["licence"] = "see BraTS2021 license"
    json_dict["release"] = "0.0"
    json_dict["modality"] = {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"}
    json_dict["labels"] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict["numTraining"] = len(pateint_names)
    json_dict["numTest"] = 0
    json_dict["training"] = [
        {"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
        for i in pateint_names
    ]
    json_dict["test"] = []

    with open(target_base / "dataset.json", mode="w+") as file:
        json.dump(json_dict, file)


def copy_image_files(source_path, target_path, target_label_path=None):
    """Copy files from source_path to target_path with nnUNet naming convention."""

    patient_names = []
    target_path = Path(target_path)
    target_path.mkdir(exist_ok=True, parents=True)

    for patient in tqdm(source_path.iterdir()):
        name = patient.stem.split("_")[1]
        t1 = patient / f"{patient.stem}_t1.nii.gz"
        t1c = patient / f"{patient.stem}_t1ce.nii.gz"
        t2 = patient / f"{patient.stem}_t2.nii.gz"
        flair = patient / f"{patient.stem}_flair.nii.gz"

        if target_label_path is not None:
            seg = patient / f"{patient.stem}_seg.nii.gz"
            assert seg.is_file(), "Segmentation is not a file"

        assert all(
            [t1.is_file(), t1c.is_file(), t2.is_file(), flair.is_file()]
        ), f"Missing modality in {patient}"

        shutil.copy(t1, target_path / f"{name}_0000.nii.gz")
        shutil.copy(t1c, target_path / f"{name}_0001.nii.gz")
        shutil.copy(t2, target_path / f"{name}_0002.nii.gz")
        shutil.copy(flair, target_path / f"{name}_0003.nii.gz")

        if target_label_path is not None:
            copy_BraTS_segmentation_and_convert_labels(
                seg, target_label_path / f"{name}.nii.gz"
            )

        patient_names.append(name)

    return patient_names


def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    """nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3"""
    img = nib.load(in_file)
    img_npy = np.array(img.dataobj)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError("unexpected label")

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2

    nifti_new = nib.Nifti1Image(seg_new, affine=img.affine, header=img.header)
    nib.save(nifti_new, out_file)


def prepare_brats():
    source_path = Path("dataset/RSNA_ASNR_MICCAI_BraTS2021_TrainingData")
    task_path = Path("dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Brats21/")

    target_train_path = task_path / "imagesTr"
    target_label_path = task_path / "labelsTr"

    target_train_path.mkdir(parents=True, exist_ok=True)
    target_label_path.mkdir(parents=True, exist_ok=True)

    patient_names = copy_image_files(source_path, target_train_path, target_label_path)

    prepare_brats_json(task_path, patient_names)


def postprocess_prediction_labels(folder):
    """Change labels to BraTS21 convention."""
    folder = Path(folder)
    images = list(folder.glob("*.nii.gz"))

    for image_path in tqdm(images):
        img_nib = nib.load(image_path)
        img_npy = np.array(img_nib.dataobj)
        seg_new = np.zeros_like(img_npy)

        seg_new[img_npy == 3] = 4
        seg_new[img_npy == 1] = 2
        seg_new[img_npy == 2] = 1

        img = nib.Nifti1Image(
            seg_new.astype(np.uint16),
            img_nib.affine,
            img_nib.header,
            img_nib.extra,
            img_nib.file_map,
        )
        nib.save(img, image_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_brats_train", default=False, action="store_true")
    parser.add_argument("--prepare_brats_validation", default=False, nargs=2)
    parser.add_argument(
        "--postprocess_predictions",
        default=False,
        help="Path to folder to be processed. Changes labels of targets to BraTS21 convention.",
    )
    args = parser.parse_args()

    print(args)

    if args.prepare_brats_train:
        prepare_brats()

    if args.prepare_brats_validation:
        copy_image_files(
            Path(args.prepare_brats_validation[0]),
            Path(args.prepare_brats_validation[1]),
        )

    if args.postprocess_predictions:
        postprocess_prediction_labels(Path(args.postprocess_predictions))


if __name__ == "__main__":
    main()
