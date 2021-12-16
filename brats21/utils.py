from pathlib import Path
from collections import defaultdict
from typing import List, Union
import nibabel as nib
import pickle

import torch
import numpy as np


NAME_MAP = {
    "nnUNetTrainerV2BraTSSegnet__nnUNetPlansv2.1": "SegNet",
    "nnUNetTrainerV2__nnUNetPlansv2.1": "nnUNet",
    "nnUNetTrainerV2BraTSRegions__nnUNetPlansv2.1": "BraTS20",
    "nnUNetTrainerV2BraTSSmallSegnet__nnUNetPlansv2.1": "miniSegNet",
    "nnUNetTrainerMHSA__nnUNetPlansv2.1": "MHSA",
    "nnUNetTrainerMHCA__nnUNetPlansv2.1": "MHCA",
    "nnUNetTrainerTransformer__nnUNetPlansv2.1": "Transformer",
    "nnUNetTrainerResNetDec__nnUNetPlansv2.1": "ResNetDec",
    "nnUNetTrainerResNetEnc__nnUNetPlansv2.1": "ResNetEnc",
    "nnUNetTrainerSegNetTransformer__nnUNetPlansv2.1": "SegNetTransformer",
    "nnUNetTrainerSegNetMHSA__nnUNetPlansv2.1": "SegNetMHSA",
    "nnUNetTrainerSegResNetEnc__nnUNetPlansv2.1": "SegResNetEnc",
    "nnUNetTrainerSegResNetDec__nnUNetPlansv2.1": "SegResNetDec",
    "nnUNetTrainerSegNetMHCA__nnUNetPlansv2.1": "SegNetMHCA",
    "nnUNetTrainerSegNetPool5Conv3__nnUNetPlansv2.1": "SegNetPool5Conv3",
    "nnUNetTrainerSegNetPool5Conv4__nnUNetPlansv2.1": "SegNetPool5Conv4",
    "nnUNetTrainerSegNetPool6Conv2__nnUNetPlansv2.1": "SegNetPool6Conv2",
    "nnUNetTrainerSegNetPool6Conv3__nnUNetPlansv2.1": "SegNetPool6Conv3",
    "nnUNetTrainerSegNetPool6Conv4__nnUNetPlansv2.1": "SegNetPool6Conv4",
    "nnUNetTrainerSegNetPool432__nnUNetPlansv2.1": "SegNetUnpool2",
    "nnUNetTrainerSegNetPool43__nnUNetPlansv2.1": "SegNetUnpool3",
    "nnUNetTrainerSegNetPool4__nnUNetPlansv2.1": "SegNetUnpool4",
    "nnUNetTrainerV2SegNetTversky__nnUNetPlansv2.1": "Tversky",
    "nnUNetTrainerV2_focalLoss__nnUNetPlansv2.1": "Focal",
}


class NNUnetDataGenerator:
    """This class can read data as defined in nnUNet_raw_data."""

    def __init__(self, path, nmods=4):
        self.path = Path(path)
        self.nmods = nmods
        self.ids = sorted({p.name.split("_")[0] for p in self.path.iterdir()})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx_id = self.ids[idx]
        imgs = []
        for mod in range(self.nmods):
            img = self.load_and_process_modality(idx_id, mod)
            imgs.append(img)
        imgs = torch.tensor(imgs)
        return torch.unsqueeze(imgs, 0).float()

    def load_and_process_modality(self, idx, mod):
        mod_path = self.path / f"{idx}_{mod:04}.nii.gz"
        img = nib.load(mod_path).dataobj
        img = (img - np.mean(img)) / np.std(img)
        return img[50:178, 50:178, 14:142]  # TODO Random crop to patch size!


def load_pickle(file: str, mode: str = "rb"):
    """Load pickle as used in nnunet."""
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def find_all_model_meta_paths(data_dir: Union[str, Path]) -> List[Path]:
    data_dir = Path(data_dir)
    folds = data_dir.rglob(r"fold_*/")

    metas = []
    for fold in folds:
        models = {model.stem: model for model in fold.rglob("*.model")}
        if "model_final_checkpoint" in models:
            metas.append(models["model_final_checkpoint"])
        elif "model_latest" in models:
            metas.append(models["model_latest"])
        elif "model_best" in models:
            metas.append(models["model_best"])
    return metas


def load_flair(patient: str) -> nib.nifti1.Nifti1Image:
    patient = Path(patient)
    return nib.load(patient / f"{patient.name}_flair.nii.gz")


def load_t1(patient: str) -> nib.nifti1.Nifti1Image:
    patient = Path(patient)
    return nib.load(patient / f"{patient.name}_t1.nii.gz")


def load_t2(patient: str) -> nib.nifti1.Nifti1Image:
    patient = Path(patient)
    return nib.load(patient / f"{patient.name}_t2.nii.gz")


def load_t1ce(patient: str) -> nib.nifti1.Nifti1Image:
    patient = Path(patient)
    return nib.load(patient / f"{patient.name}_t1ce.nii.gz")


def load_seg(patient: str) -> nib.nifti1.Nifti1Image:
    patient = Path(patient)
    return nib.load(patient / f"{patient.name}_seg.nii.gz")


def mean_smooth(x, window_length=5):
    kernel = window_length * [(1 / window_length)]
    smoothed = np.convolve(x, kernel, mode="same")
    # last = -(window_length - 1)
    # smoothed[last:] = x[last:]
    smoothed[-window_length:] = x[-window_length:]
    return smoothed


def get_model_histroy(data_path: str, nepochs: int = 1000, name_map: dict = NAME_MAP):
    latest_models = find_all_model_meta_paths(data_path)

    performance = defaultdict(list)

    for model in latest_models:
        name, fold, checkpoint = model.parts[-3:]

        plot_stuff = torch.load(model, map_location=torch.device("cpu"))["plot_stuff"]
        tr_losses, val_losses, val_losses_tr_mode, val_eval_metrics = plot_stuff

        if len(tr_losses) < 1:
            continue

        for i in range(nepochs):
            performance["name"].append(name_map.get(name, name))
            performance["fold"].append(fold[-1])
            performance["checkpoint"].append(checkpoint)

            if len(tr_losses) > i:
                tr_loss = tr_losses[i]
                val_loss = val_losses[i]
                val_eval_metric = val_eval_metrics[i]
            else:
                tr_loss = np.nan
                val_loss = np.nan
                val_eval_metric = np.nan

            performance["train_loss"].append(tr_loss)
            performance["valid_loss"].append(val_loss)
            performance["valid_metric"].append(val_eval_metric)
            performance["iteration"].append(i)

    return performance
