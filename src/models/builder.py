"""Builder functions for constructing teacher and student models."""

from __future__ import annotations

import timm
from omegaconf import DictConfig

from models import TeacherStudentModel


def build_teacher_student_model(cfg: DictConfig) -> TeacherStudentModel:
    """
    Build and return a TeacherStudentModel instance using the given configuration.

    Args:
        cfg: Configuration object containing teacher and student model details.

    Returns:
        TeacherStudentModel: An instance containing initialized teacher and student models.
    """
    teacher = timm.create_model(
        cfg.teacher.name,
        pretrained=True,
        **getattr(cfg.teacher, "args", {}),
    )
    student = timm.create_model(
        cfg.student.name,
        **getattr(cfg.student, "args", {}),
    )

    return TeacherStudentModel(
        teacher=teacher,
        student=student,
    )
