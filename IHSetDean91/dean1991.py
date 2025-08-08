# =============================================================
# Dean (1991) – Equlibrium Beach Profiles - Main function
# =============================================================
#
# • IHSetDean91/dean1991.py → Dean eq. functiobn (kernel model)
#   h(y) = A⋅y²/³
# • IHSetDean91/cal_dean1991.py → class calibration
# • IHSetDean91/run_dean1991.py → class run model
#
# -------------------------------------------------------------

from __future__ import annotations # type hints - avoids circular reference
import numpy as np

__all__ = ["dean1991", "dean1991_rev"]  # Control what is exported with command: import *

def dean1991(x: np.ndarray, A: float, HTL: float = 0.0) -> np.ndarray:  # noqa: N802
    """Dean's static equilibrium profile (1991)..

    Parameters
    ----------
    x : ndarray
        Cross-shore distance (m), measured from the waterline..
    A : float
        Dean's parameter (m\ :sup:`1/3`). Must be positive.
    sl : float, default 0.0
        Instantaneous sea level (vertical reference) – allows you to move the
        profile up/down without changing *A*.   

    Returns
    -------
    h : ndarray
        Profile elevation (m, positive upwards). Depth is
        *h* > 0.
    """
    x = np.asarray(x, dtype=float)
    h = A * np.power(x, 2.0 / 3.0) + HTL
    return h

def dean1991_rev(h: np.ndarray, A: float, HTL: float = 0.0) -> np.ndarray:
    """Inverso: x = ((h - HTL)/A)^(3/2)."""
    return np.power((h - HTL) / A, 3.0 / 2.0)