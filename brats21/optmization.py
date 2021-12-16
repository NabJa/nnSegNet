import argparse
from pathlib import Path
from functools import partial

import numpy as np
import nibabel as nib
from medpy import metric
import optuna

from brats21 import utils as bu
from brats21.postprocessing import (
    logits_to_classes,
    revert_bbox_classes,
    classes_transformations,
)


class TargetLoader:
    """Loade labels saved as '.nii.gz' files."""

    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        self.targets = list(self.input_dir.glob("*.nii.gz"))

        self.id_to_path = {i.name.split(".")[0]: i for i in self.targets}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = self.targets[idx]
        img_id = img_path.name.split(".")[0]
        img = nib.load(img_path).get_fdata()
        return img, img_id

    def get_from_id(self, target_id):
        img = nib.load(self.id_to_path[target_id])
        return np.array(img.dataobj)


class LogitsLoader:
    """Loade and process logits from a directory as returned from nnUNet."""

    def __init__(self, input_dir, thresholds=(0.5, 0.5, 0.5)):
        self.input_dir = Path(input_dir)
        self.thresholds = thresholds

        self.logits = list(self.input_dir.glob("*.npz"))
        assert len(self.logits) > 0, f"No softmax found in {self.input_dir}"

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        path = self.logits[idx]

        logit = np.load(path)["softmax"]
        logit_id = path.name.split(".")[0]

        # Load meta information
        meta = bu.load_pickle(path.with_suffix(".pkl"))
        bbox = meta["crop_bbox"]
        shape = meta["original_size_of_raw_data"]

        # Logits to classes
        classes = logits_to_classes(logit, thresholds=self.thresholds)
        classes = revert_bbox_classes(classes, bbox, shape)
        classes = classes_transformations(classes)

        return classes, logit_id

    def get_id(self, idx):
        return self.logits[idx].stem


def hd95(pred, targ):
    pred_empty = not np.any(pred)
    pred_full = np.all(pred)
    targ_empty = not np.any(targ)
    targ_full = np.all(targ)

    if np.any([pred_empty, pred_full, targ_empty, targ_full]):
        return 372.0  # Weird magic Synapse number! Should be NaN...
    else:
        return metric.hd95(pred, targ)


def class_wise_performance(pred, targ, metric, classes=(1, 2, 3)):
    class_dcs = []
    for c in classes:
        pred_copy = pred.copy()
        targ_copy = targ.copy()

        pred_copy[pred_copy != c] = 0
        targ_copy[targ_copy != c] = 0

        class_dcs.append(metric(pred_copy, targ_copy))
    return class_dcs


def objective(trial, pred_path, targ_path):

    # Suggest parameters
    t_edema = trial.suggest_float("edema", 0.001, 0.999)
    t_enhan = trial.suggest_float("enhancing", 0.001, 0.999)
    t_necro = trial.suggest_float("necrotic", 0.001, 0.999)

    # Use thresholds to generate output masks
    logits_loader = LogitsLoader(pred_path, thresholds=(t_edema, t_enhan, t_necro))
    target_loader = TargetLoader(targ_path)

    # Evaluate masks
    performance = []
    for pred, pred_id in logits_loader:
        targ = target_loader.get_from_id(pred_id)

        # Calculate and save performace
        class_performance = class_wise_performance(pred, targ, metric.dc)
        performance.append(class_performance)

    # Return metric
    return np.mean(performance)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-study_name", "--n", type=str, required=True)
    parser.add_argument("-model_path", "--m", type=Path, required=True)
    parser.add_argument("-target_path", "--t", type=Path, required=True)
    parser.add_argument(
        "-database_dir",
        "--db",
        type=Path,
        required=False,
        default="/sc-projects/sc-proj-gbm-radiomics/brats21/brats21/optuna_dbs",
    )
    args = parser.parse_args()

    assert args.m.is_dir()
    assert args.t.is_dir()
    assert len(list(args.m.glob("*.npz"))) == len(
        list(args.t.glob("*.nii.gz"))
    ), f"Not as many softmax as labels!"

    args.db.mkdir(exist_ok=True, parents=True)

    return args


def main():
    args = parse_args()

    # Instantiate partial objective function with given paths
    this_objective = partial(objective, pred_path=args.m, targ_path=args.t)

    # We want to maximize our score (dice). If you change to hausdorff set minimize here!
    study = optuna.create_study(
        study_name=args.n,
        storage=f"sqlite:///{args.db}/{args.n}.db",
        direction="maximize",
    )

    # Start optimization.
    study.optimize(this_objective, n_trials=1000)


if __name__ == "__main__":
    main()
