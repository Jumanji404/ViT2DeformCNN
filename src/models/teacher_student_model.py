from __future__ import annotations

import torch
from torch import nn

from utils.definitions import DistillationOutput


class TeacherStudentModel(nn.Module):
    """
    A PyTorch model that encapsulates a teacher-student framework, where the teacher is a pre-trained model
    and the student is a model that learns to mimic or approximate the teacher's predictions.

    Attributes:
        _teacher (nn.Module): The teacher model, usually a pre-trained network.
        _student (nn.Module): The student model, which learns from the teacher's output.

    Methods:
        __init__(teacher: nn.Module, student: nn.Module):
    """

    _teacher: nn.Module
    _student: nn.Module

    def __init__(self, teacher: nn.Module, student: nn.Module) -> None:
        """
        Initializes the TeacherStudentModel with the provided teacher and student models.

        Args:
            teacher (nn.Module): A pre-trained model used as the teacher.
            student (nn.Module): The model that learns from the teacher's output.
            *args: Additional arguments passed to the parent nn.Module's constructor.
            **kwargs: Additional keyword arguments passed to the parent nn.Module's constructor.
        """
        super().__init__()
        self._teacher = teacher
        self._student = student

    def forward(self, inputs: torch.Tensor) -> DistillationOutput:
        """
        The forward pass for the TeacherStudentModel. This method takes input data and passes it through both
        the teacher and student models, returning their respective outputs.

        Args:
            inputs (torch.Tensor): The input tensor to be passed through both the teacher and student models.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - teacher_output (torch.Tensor): The output from the teacher model.
                - student_output (torch.Tensor): The output from the student model.
        """
        teacher_output = self._teacher(inputs)
        student_output = self._student(inputs)
        return DistillationOutput(teacher_output=teacher_output, student_output=student_output)
