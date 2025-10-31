from __future__ import annotations

from .ic_loss import IntensityCategoricalLoss


class ICLoss(IntensityCategoricalLoss):
    """Backward compatible alias for the IC-Loss implementation."""


__all__ = ["ICLoss", "IntensityCategoricalLoss"]
