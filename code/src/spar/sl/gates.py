from __future__ import annotations

import math


def meets_minimum(value: float, minimum: float, *, abs_tol: float = 1e-12) -> bool:
    """Return whether value meets a threshold, robust to floating-point roundoff."""
    return value > minimum or math.isclose(value, minimum, rel_tol=0.0, abs_tol=abs_tol)

