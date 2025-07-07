"""Tests for the TeacherStudentModel, ensuring correct initialization and forward pass behavior."""

from __future__ import annotations

from typing import Generator

import pytest
import torch
from omegaconf import DictConfig

from models import TeacherStudentModel
from models.builder import build_teacher_student_model
from models.definitions import DistillationOutput
from models.layers.utils import to_tuple


@pytest.fixture
def teacher_student_model() -> Generator[TeacherStudentModel, None, None]:
    """
    Fixture that builds and returns a TeacherStudentModel instance based on a sample config.
    """
    cfg = DictConfig(
        {
            "model": {
                "teacher": {
                    "name": "vit_large_patch14_dinov2",
                    "args": {
                        "global_pool": "",
                    },
                },
                "student": {
                    "name": "dummy_model",
                    "args": {
                        "in_channels": 3,
                        "out_channels": 10,
                    },
                },
            }
        }
    )
    model = build_teacher_student_model(cfg.model)
    return model


def test_model_initialization(teacher_student_model: TeacherStudentModel) -> None:  # pylint: disable=W0621
    """
    Test that the TeacherStudentModel is correctly initialized and is an instance of the class.
    """
    assert isinstance(teacher_student_model, TeacherStudentModel)


def test_forward(teacher_student_model: TeacherStudentModel) -> None:  # pylint: disable=W0621
    """
    Test the forward pass of the model, verifying output types and shapes.
    """
    inputs = torch.randn(2, 3, 518, 518)
    output = teacher_student_model(inputs)
    assert isinstance(output, DistillationOutput)
    assert hasattr(output, "teacher_output")
    assert hasattr(output, "student_output")
    assert output.teacher_output.shape == (2, 1370, 1024)
    assert output.student_output.shape == (2, 10, 518, 518)


def test_utils_functions() -> None:  # pylint: disable=W0621
    """
    Test the utility functions.
    """
    assert to_tuple(2) == (2, 2)
    assert to_tuple((2, 2)) == (2, 2)
    with pytest.raises(ValueError, match="Parameter must be int or tuple of length 2, got abc"):
        to_tuple("abc")  # type: ignore[arg-type]
