"""Model definition for the teacher-student distillation architecture."""

from __future__ import annotations

import torch
from torch import nn

from models.definitions import DistillationOutput


class TeacherStudentModel(nn.Module):
    """
    A PyTorch model that encapsulates a teacher-student framework.

    The teacher is a pre-trained model. The student learns to mimic the teacher's predictions.

    Attributes:
        _teacher: The teacher model, usually a pre-trained network.
        _student: The student model, which learns from the teacher's output.
    """

    _teacher: nn.Module
    _student: nn.Module

    def __init__(self, teacher: nn.Module, student: nn.Module) -> None:
        """
        Initializes the TeacherStudentModel with the provided teacher and student models.

        Args:
            teacher: A pre-trained model used as the teacher.
            student: The model that learns from the teacher's output.
        """
        super().__init__()
        self._teacher = teacher
        self._student = student

    def forward(self, inputs: torch.Tensor) -> DistillationOutput:
        """
        The forward pass for the TeacherStudentModel.

        Args:
            inputs: Input tensor passed through both the teacher and student models.

        Returns:
            DistillationOutput: Contains outputs from both teacher and student models.
        """
        teacher_output = self._teacher(inputs)
        student_output = self._student(inputs)
        return DistillationOutput(
            teacher_output=teacher_output,
            student_output=student_output,
        )
