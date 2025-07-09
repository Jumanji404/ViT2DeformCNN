"""Tests for the TeacherStudentModel, ensuring correct initialization and forward pass behavior."""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from pytest import FixtureRequest

from models import TeacherStudentModel
from models.builder import build_teacher_student_model
from models.definitions import DistillationOutput
from models.layers.utils import to_tuple
from models.resnet import ResNetBase
from models.vgg import VGGBase


@pytest.fixture
def teacher_student_model(request: FixtureRequest) -> TeacherStudentModel:
    """
    Fixture that builds and returns a TeacherStudentModel instance based on a sample config.
    """
    model_name, conv_layer_type = request.param
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
                    "name": model_name,
                    "args": {
                        "conv_layer_type": conv_layer_type,
                    },
                },
            }
        }
    )
    model = build_teacher_student_model(cfg.model)
    return model


@pytest.mark.parametrize(
    "teacher_student_model",
    [
        ("custom_vgg11", "nn.Conv2d"),
        ("custom_vgg13", "nn.Conv2d"),
        ("custom_vgg16", "nn.Conv2d"),
        ("custom_vgg19", "nn.Conv2d"),
        ("custom_vgg11", "DeformableConv2d"),
        ("custom_vgg13", "DeformableConv2d"),
        ("custom_vgg16", "DeformableConv2d"),
        ("custom_vgg19", "DeformableConv2d"),
        ("custom_resnet18", "nn.Conv2d"),
        ("custom_resnet50", "nn.Conv2d"),
        ("custom_resnet101", "nn.Conv2d"),
        ("custom_resnet152", "nn.Conv2d"),
        ("custom_resnet18", "DeformableConv2d"),
        ("custom_resnet50", "DeformableConv2d"),
        ("custom_resnet101", "DeformableConv2d"),
        ("custom_resnet152", "DeformableConv2d"),
    ],
    indirect=True,  # Tells pytest to pass this param to the fixture
)
def test_model_initialization(teacher_student_model: TeacherStudentModel) -> None:  # pylint: disable=W0621
    """
    Test that the TeacherStudentModel is correctly initialized and is an instance of the class.
    """
    assert isinstance(teacher_student_model, TeacherStudentModel)


@pytest.mark.parametrize(
    "teacher_student_model",
    [
        ("custom_vgg11", "nn.Conv2d"),
        ("custom_vgg11", "DeformableConv2d"),
        ("custom_resnet18", "nn.Conv2d"),
        ("custom_resnet18", "DeformableConv2d"),
    ],
    indirect=True,  # Tells pytest to pass this param to the fixture
)
def test_forward(teacher_student_model: TeacherStudentModel) -> None:  # pylint: disable=W0621
    """
    Test the forward pass of the model, verifying output types and shapes.
    """
    inputs = torch.randn(1, 3, 518, 518)
    output = teacher_student_model(inputs)
    assert isinstance(output, DistillationOutput)
    assert hasattr(output, "teacher_output")
    assert hasattr(output, "student_output")
    assert hasattr(teacher_student_model, "student")
    assert hasattr(teacher_student_model, "teacher")
    assert output.teacher_output.shape == (1, 1370, 1024)
    if isinstance(teacher_student_model.student, VGGBase):
        assert output.student_output.shape == (1, 25088)
    elif isinstance(teacher_student_model.student, ResNetBase):
        assert output.student_output.shape == (1, 1000)


def test_utils_functions() -> None:
    """
    Test the utility functions.
    """
    assert to_tuple(2) == (2, 2)
    assert to_tuple((2, 2)) == (2, 2)
    with pytest.raises(ValueError, match="Parameter must be int or tuple of length 2, got abc"):
        to_tuple("abc")  # type: ignore[arg-type]
