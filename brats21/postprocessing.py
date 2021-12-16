import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import shutil

from tqdm import tqdm
import nibabel as nib
import numpy as np

from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder
from brats21 import utils as bu
from brats21.prepare_nnunet import postprocess_prediction_labels


# ################################# #
# Main postprocessing functions     #
# ################################# #


def surpress_labels(
    do_supress: List[bool],
    prediction_dir: Union[Path, str],
    out_dir: Union[Path, str],
    label: int,
    regions: Tuple[int] = (1, 2, 3),
    thresholds: Tuple[float] = (0.5, 0.5, 0.5),
    sort_func: Optional[callable] = None,
) -> None:
    """Supress labels based on a given list of booleans.

    Args:
        do_supress: List of booleans whether to supress label with the same order as glob(*.nii.gz). (See sort_func).
        prediction_dir: Directory containing all predictions to be processed saved as *.nii.gz.
        out_dir: Output direcoty to save output. If does not exist, it will be generated.
        label: Value of the label that should be replaced.
        regions: All region classes in dataset.
        thresholds: For every region class the corresponding threshold.
        sort_func: Optional sort function used on img_dir to make sure do_supress is in the same order as glob on prediction_dir.
    """

    prediction_paths = sorted(Path(prediction_dir).glob("*.nii.gz"), key=sort_func)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove "label" from regions and thresholds
    regions_removed = [i for i in regions if i != label]
    thresholds_removed = [t for i, t in zip(regions, thresholds) if i != label]

    for supress, pred_path in tqdm(zip(do_supress, prediction_paths)):

        pred = nib.load(pred_path)
        image_id = pred_path.name.split(".")[0]
        logits, props = read_image_meta_and_softmax(prediction_dir, image_id)

        if supress:
            classes = logits_to_classes(logits, regions_removed, thresholds_removed)
        else:
            classes = logits_to_classes(logits, regions, thresholds)

        classes = revert_bbox_classes(
            classes, props["crop_bbox"], props["original_size_of_raw_data"]
        )
        classes = classes_transformations(classes)

        img = nib.Nifti1Image(
            classes.astype(np.uint8),
            pred.affine,
            pred.header,
            pred.extra,
            pred.file_map,
        )
        out_img_path = out_dir / pred_path.name
        nib.save(img, out_img_path)


def read_image_meta_and_softmax(
    folder_path: Union[str, Path], image_id: str
) -> Tuple[np.ndarray, dict]:
    """Load softmax from .npz file and meta info from .pkl file.

    Args:
        folder_path: Folder containing images (.nii.gz), softmax (.npz) and meta information (.pkl).
        image_id: ID of image. Must be the same for image, softmax and meta information.

    Returns:
        Tuple of softmax and meta data
    """

    folder_path = Path(folder_path)

    softmax_path = folder_path / f"{image_id}.npz"
    plans_path = folder_path / f"{image_id}.pkl"

    softmax = np.load(softmax_path)["softmax"]
    properties_dict = bu.load_pickle(plans_path)

    return softmax, properties_dict


def segnet_unet_ensamble(
    segnet_pred_path: Union[Path, str],
    unet_pred_path: Union[Path, str],
    output_path: Union[Path, str],
    segnet_labels: Tuple[int] = (3),
    unet_labels: Tuple[int] = (1, 2),
    save_softmax: bool = True,
    save_mask: bool = True,
    save_meta: bool = True,
) -> None:
    """Combine SegNet softmax with UNet softmax.

    Args:
        sgnet_pred_path: Directory containing softmax files (.npz) for SegNet.
        unet_pred_path: Directory containing softmax files (.npz) for UNet.
        output_path: Directory to save output. If does not exist, will be created. Files will be overwritten.
        segnet_labels: Region classes used from segnet. Defaults to (3).
        unet_labels: Region classes used from unet. Defaults to (1, 2).
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    segnet_softmax_paths = list(segnet_pred_path.glob("*.npz"))
    unet_softmax_path = list(unet_pred_path.glob("*.npz"))

    assert np.all(
        [i.name for i in segnet_softmax_paths] == [i.name for i in unet_softmax_path]
    ), "Not all softmax files have a corresponding match."

    # We have to go from region classes to index
    segnet_labels = np.array(segnet_labels) - 1
    unet_labels = np.array(unet_labels) - 1

    for segnet_softmax_path in tqdm(segnet_softmax_paths):
        unet_softmax_path = unet_pred_path / segnet_softmax_path.name

        # Load softmax
        segnet_softmax = np.load(segnet_softmax_path)["softmax"]
        unet_softmax = np.load(unet_softmax_path)["softmax"]

        # Generate ensamble softmax
        ensable_softmax = np.zeros_like(segnet_softmax)

        ensable_softmax[segnet_labels, ...] = segnet_softmax[segnet_labels, ...]
        ensable_softmax[unet_labels, ...] = unet_softmax[unet_labels, ...]

        # Read meta information
        segnet_meta_path = segnet_softmax_path.with_suffix(".pkl")
        segnet_meta = bu.load_pickle(segnet_meta_path)
        unet_meta = bu.load_pickle(unet_softmax_path.with_suffix(".pkl"))
        assert segnet_meta["crop_bbox"] == unet_meta["crop_bbox"]
        assert np.all(
            segnet_meta["original_size_of_raw_data"]
            == unet_meta["original_size_of_raw_data"]
        )

        if save_softmax:
            np.savez(output_path / unet_softmax_path.name, softmax=ensable_softmax)

        if save_meta:
            shutil.copy(segnet_meta_path, output_path / segnet_meta_path.name)

        if save_mask:
            segnet_nib = nib.load(segnet_softmax_path.with_suffix(".nii.gz"))
            unet_nib = nib.load(unet_softmax_path.with_suffix(".nii.gz"))
            assert np.all(unet_nib.affine == segnet_nib.affine)

            classes = logits_to_classes(ensable_softmax, (1, 2, 3), (0.5, 0.5, 0.5))
            classes = revert_bbox_classes(
                classes,
                segnet_meta["crop_bbox"],
                segnet_meta["original_size_of_raw_data"],
            )
            classes = classes_transformations(classes)

            mask = nib.Nifti1Image(
                classes.astype(np.uint8),
                segnet_nib.affine,
                segnet_nib.header,
                segnet_nib.extra,
                segnet_nib.file_map,
            )
            out_img_path = output_path / unet_softmax_path.with_suffix(".nii.gz").name
            nib.save(mask, out_img_path)


# ################################# #
# Softmax transformation functions  #
# ################################# #


def revert_bbox_classes(
    classes: np.ndarray,
    bbox: Iterable[Iterable],
    shape_original_before_cropping: np.ndarray,
) -> np.ndarray:
    """Recreate original image shape after bbox cropping for given classes.
    Basically just put classes on bbox position.

    Args:
        classes: Volume with classes as indexes.
        bbox: List of tuple (x, y) for every dimension in target volume.
        shape_original_before_cropping: Original shape to recreate.

    Returns:
        Resized segmentation class volume.
    """

    seg_original_size = np.zeros(shape_original_before_cropping)
    for c in range(3):
        bbox[c][1] = np.min(
            (bbox[c][0] + classes.shape[c], shape_original_before_cropping[c])
        )
        seg_original_size[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ] = classes
    return seg_original_size


def revert_bbox_softmax(
    softmax: np.ndarray,
    bbox: Iterable[Iterable],
    shape_original_before_cropping: np.ndarray,
) -> np.ndarray:
    """Recreate original image shape after bbox cropping for given softmax volume.
    Basically just put softmax on bbox position.

    Args:
        softmax: Softmax volume [C, D, H, W].
        bbox: List of tuple (x, y) for every dimension in target volume.
        shape_original_before_cropping: Original shape to recreate.

    Returns:
        Resized segmentation class volume.
    """
    seg_original_size = np.zeros(shape_original_before_cropping)
    for c in range(3):
        bbox[c][1] = np.min(
            (bbox[c][0] + softmax.shape[c + 1], shape_original_before_cropping[c + 1])
        )
        seg_original_size[
            c, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ] = softmax[c]
    return seg_original_size


def logits_to_classes(
    logits: np.ndarray,
    region_class_order: tuple = (1, 2, 3),
    thresholds: tuple = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Transform logit volume to class volume while having nested regions.

    Args:
        logits: Volume with logits. Must be one hot encoded with classes in dimension 0. Shape = C, D, H, W.
        region_class_order: Region classes. Order of this tuple resambles the region order of the final mask!
        thresholds: Softmax thresholds for every class.

    Returns:
        Class volume with shape D, H, W
    """
    assert len(region_class_order) == len(
        thresholds
    ), f"For every region class (given: {region_class_order}) one threshold (given: {thresholds}) is needed."

    assert len(logits.shape) == 4, "Logits must be C, D, H, W format."

    classes = np.zeros(logits.shape[1:])
    for i, c in enumerate(region_class_order):
        classes[logits[i] > thresholds[i]] = c

    return classes


def classes_transformations(classes: np.ndarray) -> np.ndarray:
    """Transformations to match input image.

    Args:
        classes: Class volume of shape D, H, W

    Returns:
        Transformed class volume
    """
    # Depth last in prediction output
    classes = np.moveaxis(classes, 0, -1)

    # Rotate as in predictions
    classes = np.rot90(classes, k=3)
    classes = np.fliplr(classes)
    return classes


def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("-output_folder", "--o", type=Path)
    parser.add_argument("-input_folder", "--i", type=Path)

    # Choose postprocessing action. At least one have to be choosen.
    parser.add_argument(
        "-supress_enhancing", "--se", default=False, action="store_true"
    )
    parser.add_argument("-ensemble", "--en", default=False, action="store_true")
    parser.add_argument(
        "-connected_component", "--cc", default=False, action="store_true"
    )
    parser.add_argument("-transform_labels", "--tl", default=False, action="store_true")

    # Depending on postprocessing action we need these.
    parser.add_argument("-segnet", "--seg", type=Path)
    parser.add_argument("-unet", "--unet", type=Path)
    parser.add_argument("-decision_tree", "--dtr", type=Path)
    parser.add_argument("-decision_tree_predictions", "--dtp", type=Path)
    parser.add_argument("-supress_label", "--sup", default=3, type=int)
    parser.add_argument("-labels", "--lab", default=(1, 2, 3), nargs="+")
    parser.add_argument(
        "-thresholds", "--thr", default=(0.475, 0.449, 0.681), type=float, nargs=3
    )

    # Parse and check
    args = parser.parse_args()
    if sum([args.se, args.en, args.cc, args.tl]) < 1:
        print("No postprocessing action defined. One of --en, --dt or --c must be set.")
        quit()

    # Run tasks
    if args.en:
        segnet_unet_ensamble(args.seg, args.unet, args.o)

    if args.se:
        if args.dtp is not None:
            preds = bu.load_pickle(args.dtp)
            surpress_labels(
                preds,
                args.i,
                args.o,
                label=args.sup,
                regions=args.lab,
                thresholds=args.thr,
            )
        elif args.dtr is not None:
            print("Still not implemented")
        else:
            print(
                "Provide predictions or decision tree for enhancing label supression."
            )

    if args.cc:
        apply_postprocessing_to_folder(args.i, args.o, (1, 2, 3))

    if args.tl:
        postprocess_prediction_labels(args.i)


if __name__ == "__main__":
    main()
