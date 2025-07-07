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

__all__ = ["dean1991"]  # Control what is exported with command: import *

def dean1991(y: np.ndarray, A: float, sl: float = 0.0) -> np.ndarray:  # noqa: N802
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
    y = np.asarray(y, dtype=float)
    h = A * np.power(y, 2.0 / 3.0) + sl
    return h
