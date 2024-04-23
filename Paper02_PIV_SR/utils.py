"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of GitHub.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The code reference comes from Lornatang's DRRN-PyTorch code repository. Thanks again for Lornatang's excellent work and open source contribution.
The link to the reference code repository is as follows:
    https://github.com/Lornatang/RDN-PyTorch
"""
# ==============================================================================
import os
import shutil
from enum import Enum
from typing import Any, Union, Optional, Tuple

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer

__all__ = [
    "load_state_dict", "make_directory", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter"
]


def load_state_dict(
        model: nn.Module,
        model_weights_path: str,
        ema_model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> Union[
    Tuple[Module, Module, Any, Any, Optional[Optimizer], Any], Tuple[Module, Any, Any, Optional[
        Optimizer], Any], Module]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    if load_mode == "resume":
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None:
            # Load the scheduler model
            scheduler.load_state_dict(checkpoint["scheduler"])

        if ema_model is not None:
            # Load ema model state dict. Extract the fitted model weights
            ema_model_state_dict = ema_model.state_dict()
            ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
            # Overwrite the model weights to the current model (ema model)
            ema_model_state_dict.update(ema_state_dict)
            ema_model.load_state_dict(ema_model_state_dict)

        return model, ema_model, start_epoch, best_psnr, optimizer, scheduler
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

        return model


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        best_file_name: str,
        last_file_name: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    # torch.save(state_dict, checkpoint_path)

    if is_best:
        torch.save(state_dict, checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, best_file_name))
    if is_last:
        torch.save(state_dict, checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, last_file_name))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# Processing large amounts of data named according to instantaneous order
def sort_key(i):
    """

    Args:
        i: The order of the processed data files, the number of files

    Returns: Sequential numbering of files with uniform string length

    """
    if i < 10:
        Num_tag = '00000' + str(i)
    elif 10 <= i < 100:
        Num_tag = '0000' + str(i)
    elif 100 <= i < 1000:
        Num_tag = '000' + str(i)
    elif 1000 <= i < 10000:
        Num_tag = '00' + str(i)
    elif 10000 <= i < 100000:
        Num_tag = '0' + str(i)
    else:
        Num_tag = str(i)
    return Num_tag

