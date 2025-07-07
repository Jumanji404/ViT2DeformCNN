"""Data structures used in the distillation framework."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DistillationOutput:
    """
    A data class that encapsulates the outputs of the teacher and student models.

    Attributes:
        teacher_output: The output from the teacher model.
        student_output: The output from the student model.
    """

    teacher_output: torch.Tensor
    student_output: torch.Tensor
