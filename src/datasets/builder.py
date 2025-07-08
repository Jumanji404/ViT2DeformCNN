"""Builder functions for constructing the Datasets."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def build_dataloader(cfg: DictConfig, is_train: bool) -> DataLoader[tuple[Tensor, Tensor]]:
    """
    Build and return a Dataloader instance using the given configuration.

    Args:
        cfg: Configuration object containing the dataset details.
        is_train: a boolean specifying whether to create the training dataset or not.

    Returns:
        TeacherStudentModel: An instance containing the dataloader specified in the config.
    """
    if cfg.name == "CIFAR10":
        dataset = CIFAR10(
            root=cfg.data_dir,
            train=is_train,
            transform=None,
            download=True,
        )
    else:
        raise ValueError(f"Unsuported dataset with name {cfg.name}!")
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=is_train,
        sampler=torch.utils.data.DistributedSampler(dataset) if torch.distributed.is_initialized() else None,
        num_workers=cfg.num_workers,
    )
