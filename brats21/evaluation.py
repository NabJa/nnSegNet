import os
import argparse
import shutil
import json
from multiprocessing import Pool
from collections import defaultdict
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

import numpy as np
import pandas as pd
import nibabel as nib
from medpy import metric

import plotly.express as px

from brats21 import utils as bu
from brats21.slurm import batch


@dataclass
class Result:
    values: dict
    classes: dict = field(
        default_factory=lambda: {"0": "Edema", "1": "Necrotic", "2": "Enhancing"}
    )

    def get_property(self, key) -> dict:
        return {self.classes[k]: self.values[k][key] for k in self.classes.keys()}

    def __repr__(self) -> str:
        return f"Dice: {self.dice}\nJaccard: {self.jaccard}\nPrecision: {self.precision}\nRecall: {self.recall}"

    @property
    def dice(self):
        return self.get_property("Dice")

    @property
    def fdr(self):
        return self.get_property("False Discovery Rate")

    @property
    def fnr(self):
        return self.get_property("False Negative Rate")

    @property
    def fpr(self):
        return self.get_property("False Positive Rate")

    @property
    def tnr(self):
        return self.get_property("True Negative Rate")

    @property
    def false_omission(self):
        return self.get_property("False Omission Rate")

    @property
    def jaccard(self):
        return self.get_property("Jaccard")

    @property
    def npv(self):
        return self.get_property("Negative Predictive Value")

    @property
    def precision(self):
        return self.get_property("Precision")

    @property
    def recall(self):
        return self.get_property("Recall")

    @property
    def total_positives_reference(self):
        return self.get_property("Total Positives Reference")

    @property
    def total_positive_test(self):
        return self.get_property("Total Positives Test")

    @property
    def hausdorff(self):
        return self.get_property("hausdorff")

    @property
    def hausdorff95(self):
        return self.get_property("hausdorff95")


class Summary:
    def __init__(self, path, name=None):
        self.name = name
        self.path = Path(path)
        self.json = json.load(self.path.open(mode="r"))

    def __repr__(self) -> str:
        return f"Name:\t{self.name}\#Evaluations:\t{len(self.image)}\nMean Resuluts:\n{self.mean}"

    def __getitem__(self, idx):
        return self.image[idx]

    @property
    def mean(self):
        return Result(self.json["results"]["mean"])

    @property
    def image(self):
        return [Result(x) for x in self.json["results"]["all"]]


class ImageSummary:
    """Summarizing functions for a single image instance."""

    def __init__(self, img_id: str, img_path: str, label_path: str, pred_path: str):
        self.img_id = img_id
        self.img_path = Path(img_path)
        self.label_path = Path(label_path)
        self.pred_path = Path(pred_path)

    def parse_modality(self, modality: str) -> int:
        if modality == "t1":
            return 0
        elif modality == "t1c" or modality == "t1ce":
            return 1
        elif modality == "t2":
            return 2
        elif modality == "flair":
            return 3
        else:
            print("WARNING: Unknown modality", modality)
            return -1

    def plot_volume(self, modality: str, **kwargs):
        vol = self.get_volume(modality)
        fig = px.imshow(vol, animation_frame=2, binary_string=True, **kwargs)
        return fig

    def get_volume(self, modality: str, rot90k: int = 1) -> np.ndarray:
        midx = self.parse_modality(modality)
        path = self.img_path / f"{self.img_id}_{midx:04}.nii.gz"
        img = np.array(nib.load(path).dataobj)
        img = np.rot90(img, k=rot90k)
        return img

    def plot_label(self, **kwargs):
        vol = self.get_label()
        fig = px.imshow(vol, animation_frame=2, **kwargs)
        return fig

    def get_label(self, rot90k: int = 1) -> np.ndarray:
        path = self.label_path / f"{self.img_id}.nii.gz"
        img = np.array(nib.load(path).dataobj)
        img = np.rot90(img, k=rot90k)
        return img

    def plot_pred(self, **kwargs):
        vol = self.get_pred()
        fig = px.imshow(vol, animation_frame=2, **kwargs)
        return fig

    def get_pred(self, rot90k: int = 1) -> np.ndarray:
        path = self.pred_path / f"{self.img_id}.nii.gz"
        img = np.array(nib.load(path).dataobj)
        img = np.rot90(img, k=rot90k)
        return img


class FoldDataGenerator:
    """Generates directories to evaluate model performance."""

    def __init__(
        self, target_path, raw_data_path=None, processed_data_path=None, threads=8
    ):
        self.target_path = self.parse_target_dir(target_path)

        self.raw_data_path = self.parse_env_variable(
            raw_data_path, "nnUNet_raw_data_base"
        )
        self.processed_data_path = self.parse_env_variable(
            processed_data_path, "nnUNet_preprocessed"
        )

        self.folds = bu.load_pickle(
            next(self.processed_data_path.rglob("splits_final.pkl"))
        )

        self.threads = threads

    def parse_target_dir(self, target):
        """Transform target to path and make directories if they dont exist."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def parse_env_variable(self, path, fallback):
        """Parse input path or look in environment variables for fallback."""
        if path is not None:
            return Path(path)
        if fallback in os.environ:
            return Path(os.environ[fallback])
        raise ValueError(f"Required path not found: {fallback}")

    def get_training_image_paths(self):
        """Generates folder for every fold and returns to lists of source image paths and their destination."""
        train_data_path = list(self.raw_data_path.rglob("imagesTr"))[0]

        srcs, dsts = [], []
        for i, fold in enumerate(self.folds):
            val_fold = fold["val"]

            fold_path = self.target_path / "input" / f"fold_{i}"
            fold_path.mkdir(parents=True, exist_ok=True)

            # For every image in cross validation split
            for img_id in val_fold:

                # For every modality
                for img_path in train_data_path.glob(str(img_id) + "*"):
                    destination = fold_path / img_path.name

                    if not destination.is_file():
                        srcs.append(img_path)
                        dsts.append(destination)
        return srcs, dsts

    def prepare_data(self):
        """Main function to prepare data"""

        # Copy all images to fold folder in target_path directory
        srcs, dsts = self.get_training_image_paths()
        with Pool(self.threads) as p:
            p.starmap(shutil.copy, zip(srcs, dsts))


class PredictionGenerator:
    """Predictor for all folds and all models given."""

    def __init__(self, model_dir, input_dir, output_dir):

        self.command = (
            "nnUNet_predict -i {} -o {} -t Task500_Brats21 -tr {} -m 3d_fullres -f {}"
        )

        self.model_dir = Path(model_dir)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.all_folds = {str(fold.name[-1]) for fold in self.input_dir.glob("fold_*")}

    def prepare_prediction_commands(self):
        prediction_commands = []
        for model_fold in self.model_dir.rglob("fold_*"):
            for val_fold in self.input_dir.glob("fold_*"):
                if model_fold.name == val_fold.name:
                    trainer_name = model_fold.parent.name.split("_")[0]
                    out = self.generate_out_path(trainer_name, val_fold)
                    cmd = self.generate_pred_command(val_fold, out, trainer_name)
                    prediction_commands.append(cmd)
        return prediction_commands

    def generate_out_path(self, trainer_name, inp):
        p = self.output_dir / trainer_name / inp.name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def generate_pred_command(self, inp, out, trainer):
        return f"nnUNet_predict -i {inp} -o {out} -t Task500_Brats21 -tr {trainer} -m 3d_fullres"

    def run_predictions(self):
        prediction_commands = self.prepare_prediction_commands()
        log = open(self.output_dir / "log.txt", "w")
        for cmd in tqdm(prediction_commands):
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.close()


class ValidationLoader:
    def __init__(self, prediction_dir, ground_truth_dir):
        self.prediction_dir = Path(prediction_dir)
        self.ground_truth_dir = Path(ground_truth_dir)

        self.predictions = list(self.prediction_dir.glob("*.nii.gz"))
        self.ground_truths = list(self.ground_truth_dir.glob("*.nii.gz"))

        assert len(self.predictions) == len(
            self.ground_truths
        ), f"Not same number of predictions {len(self.predictions)} as GTs {len(self.ground_truths)}. \
            \nPrediction dir: {prediction_dir}\nGT dir: {ground_truth_dir}"

    def __getitem__(self, idx):

        sample_id = self.predictions[idx].name.split(".")[0]

        pred = nib.load(self.predictions[idx])
        gt = nib.load(self.ground_truths[idx])

        pred = np.array(pred.dataobj)
        gt = np.array(gt.dataobj)

        return pred, gt, sample_id

    def __len__(self):
        return len(self.predictions)

    @property
    def name(self):
        return self.prediction_dir.parent.name


def class_wise_metric(pred, ref, metric_func, classes=(1, 2, 3), background=0):
    res = {}
    for c in classes:
        pred_copy = pred.copy()
        ref_copy = ref.copy()

        if metric_func.__name__ in ["hd", "hd95"] and (
            np.all(ref_copy == c)
            or np.all(pred_copy == c)
            or not np.any(ref_copy == c)
            or not np.any(pred_copy == c)
        ):
            res[str(c)] = np.NaN
        else:
            pred_copy[pred_copy != c] = background
            ref_copy[ref_copy != c] = background
            res[str(c)] = metric_func(pred_copy, ref_copy)
    return res


def make_validation(loader):
    results = defaultdict(list)

    class_dict = {"1": "Edema", "2": "Necrotic", "3": "Enhancing"}

    for pred, gt, sample_id in tqdm(loader, desc=loader.name):
        results["id"].append(sample_id)
        for metric_func in [
            metric.hd,
            metric.hd95,
            metric.dc,
            metric.sensitivity,
            metric.specificity,
        ]:
            class_wise_performance = class_wise_metric(pred, gt, metric_func)

            for k, v in class_wise_performance.items():
                results[f"{metric_func.__name__}_{class_dict[k]}"].append(v)

    return pd.DataFrame(results)


def run_evaluation(inp, gt):
    loader = ValidationLoader(inp, gt)
    results = make_validation(loader)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-prediction",
        "--p",
        type=Path,
        required=True,
        help="Path to prediction directory containing .nii.gz files.",
    ),
    parser.add_argument(
        "-label",
        "--l",
        type=Path,
        required=True,
        help="Path to label directory containing .nii.gz files.",
    )
    parser.add_argument(
        "-output_path",
        "--o",
        type=Path,
        help="Absolute path to csv file to save prediction. Parent directories will be created. \
            DEFAULT: Save into prediction path.",
    )
    args = parser.parse_args()

    # Check input args
    assert args.p.is_dir()
    assert args.l.is_dir()
    if args.o is not None:
        args.o.parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    performance = run_evaluation(args.p, args.l)

    # Save performance
    if args.o is not None:
        performance.to_csv(args.o)
    else:
        performance.to_csv(args.p / "performance.csv")


if __name__ == "__main__":
    main()
