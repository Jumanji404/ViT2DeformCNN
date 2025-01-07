import torch


def test_mock() -> None:
    """ceva"""
    a = torch.ones(1)
    b = torch.zeros(1)
    assert not (a == b).all()
