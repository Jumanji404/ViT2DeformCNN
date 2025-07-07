"""
Helper for converting convolution params to 2-element tuples.
"""

from __future__ import annotations


def to_tuple(param: int | tuple[int, int]) -> tuple[int, int]:
    """
    Ensure param is a 2-element tuple.

    Args:
        param: An int or a 2-tuple of ints.

    Returns:
        A 2-tuple of ints.

    Raises:
        ValueError: If input is not valid.
    """
    if isinstance(param, int):
        return (param, param)
    if isinstance(param, tuple) and len(param) == 2:
        return param
    raise ValueError(f"Parameter must be int or tuple of length 2, got {param}")
