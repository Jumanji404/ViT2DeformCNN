"""Tests for the dataloader."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from datasets.builder import build_dataloader


@pytest.fixture
def dataloader() -> DataLoader[tuple[Tensor, Tensor]]:
    """
    Fixture that builds and returns a Dataloader instance based on a sample config.
    """
    cfg = DictConfig(
        {
            "data": {
                "name": "CIFAR10",
                "batch_size": 32,
                "num_workers": 8,
                "data_dir": "src/data",
            }
        }
    )
    model = build_dataloader(cfg.data, is_train=True)
    return model


def test_dataloader(dataloader: DataLoader[tuple[Tensor, Tensor]]) -> None:  # pylint: disable=W0621
    """
    Test the attributes of the Dataloader.
    """
    assert isinstance(dataloader.dataset, Dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 8


def test_invalid_dataloader() -> None:
    """
    Test the invalid Dataloader.
    """
    cfg = DictConfig(
        {
            "data": {
                "name": "INVALID",
                "batch_size": 32,
                "num_workers": 8,
                "data_dir": "src/data",
            }
        }
    )
    with pytest.raises(ValueError, match="Unsuported dataset with name INVALID!"):
        build_dataloader(cfg.data, is_train=True)
