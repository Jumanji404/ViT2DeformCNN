"""Entry point for starting ditillation experiments."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from datasets.builder import build_dataloader
from models.builder import build_teacher_student_model


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    model = build_teacher_student_model(cfg.model)
    train_dataloader = build_dataloader(cfg.data, is_train=True)
    # TODO: Add training logic here or call training function using model # pylint: disable=W0511
    print(f"Model built: {model.__class__.__name__}")
    print(f"Dataset built: {train_dataloader.dataset.__class__.__name__}")


if __name__ == "__main__":
    main()  # pylint: disable=E1120
