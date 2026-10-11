"""Pure-Python helpers for summarizing evaluation returns."""

from __future__ import annotations

from collections.abc import Iterable


def mean_return(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


__all__ = ["mean_return"]
