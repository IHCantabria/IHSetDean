# =============================================================
# calibration/cal_dean1991.py
# =============================================================
"""
The goal is to:

1. **Read** a *simple* CSV (two columns: X, Y) with the measured profile.

2. **Calibrate** the *A* (and optionally *k*) parameter using
:pyfunc:`scipy.optimize.fsolve` so as to minimize the squared error.

3. **Provide** the *wrappers* `model_sim`, `init_par` and `run_model` to
maintain compatibility with the rest of the infrastructure (optimizer,
GUI, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from dean1991 import dean1991  # type: ignore

__all__ = ["cal_dean1991", "dean_A_from_d50_no_calib"]

# -----------------------------------------------------------------
# DEAN EQUILIBRIUM BEACH PROFILE - NO CALIBRATION
# -----------------------------------------------------------------
def dean_A_from_d50_no_calib(d50_mm: float) -> float:
    """
    Dean's empirical relationship between median-grain size and A (m ^1/3).
    Parameters
    ----------
    d50_mm : float
        Median grain size in millimetres.

    Returns
    -------
    float
    float
        Empirical A = 0.067 x (d50[m])^0.44
    """
    if d50_mm is None:
        raise ValueError("D50 must be provided.")
    return 0.067 * (d50_mm) ** 0.44

# -----------------------------------------------------------------
# DEAN EQUILIBRIUM BEACH PROFILE - CSV CALIBRATION
# -----------------------------------------------------------------
class cal_dean1991:  # noqa: N801
    """Calibration class for Dean (1991).

        Parameters
        ----------
        - csv_path : str | Path
            CSV path containing two columns, ``X`` and ``Y``.
        - sl : float, optional
            Reference sea level (m). Default = ``0.0``.
        - d50 : float | None, optional
            Median grain size (mm). Not required for this
            calibration method, but stored for future use.
        - Xm : tuple[float, float] | None, optional
            Range (min, max) of X that will be used to generate a *dense*
            vector for the modeled profile (same role as in the other
            models). If ``None``, use ``(min(X_obs), max(X_obs))``.
    """

    # -----------------------------------------------------------------
    # 1. CONSTRUCTION AND PRE-PROCESSING
    # -----------------------------------------------------------------

    def __init__(
        self,
        csv_path: str | Path,
        sl: float = 0.0,
        d50: float | None = None,
        Xm: Tuple[float, float] | None = None,
    ):  # noqa: D401
        self.csv_path = Path(csv_path)
        self.sl = float(sl)
        self.d50 = d50

        # read CSV → np.ndarray
        df = pd.read_csv(self.csv_path, header=None, names=["X", "Y"])
        self.x_obs: np.ndarray = df["X"].to_numpy(float)
        self.z_obs: np.ndarray = df["Y"].to_numpy(float)
        
        # By IHSet convention: positive depth *down*.
        # If "Y" is negative, invert here.
        if np.mean(self.z_obs) < 0: 
            self.z_obs = -self.z_obs

        # Ensures that the origin (SL) is removed from the observation
        self.z_obs_rel = self.z_obs - self.sl

        # Generate "dense" X vector for smooth curves
        if Xm is None:
            Xm = (float(self.x_obs.min()), float(self.x_obs.max()))
        self.Xm = Xm
        self.x_dense = np.linspace(Xm[0], Xm[1], 1_000)

        # Placeholders to be filled after calibration/execution
        self.A: float | None = None
        self.z_model: np.ndarray | None = None

    # -----------------------------------------------------------------
    # 2. INTERNAL SUPPORT FUNCTIONS
    # -----------------------------------------------------------------

    def _sse_gradient(self, A: float) -> float:  # noqa: N802
        """Gradient (∂SSE/∂A) of the total squared error.

        Used as *equation = 0* for :pyfunc:`scipy.optimize.fsolve`.
        """
        diff = dean1991(self.x_obs, A, 0.0) - self.z_obs_rel  # relative to SL
        grad = float(np.sum(diff * np.power(self.x_obs, 2.0 / 3.0)))
        return grad

    # -----------------------------------------------------------------
    # 3. CALIBRATION (FINDING THE BEST A VALUE)
    # -----------------------------------------------------------------

    def calibrate(self, guess: float | None = None) -> "cal_dean1991":
        """Fits *A* with *fsolve*; uses D50 as guess if it exists.

           The initial guess is the analytical solution by the least-squares method 
           with fixed slope (derived from the log transformation).
        """
        # Uses D50 as seed for fsolve guess ----------------------------
        if guess is None and self.d50 is None:
            assert self.d50 is not None, "D50 must be provided"
            #lnA_est = np.mean(np.log(self.z_obs_rel) - (2.0 / 3.0) * np.log(self.x_obs))
            #guess = float(np.exp(lnA_est))
        # --------------------------------------------------------------
        if guess is None and self.d50 is not None:
            d50_m = self.d50 / 1000  # mm → m
            guess = 0.067 * 0.3 ** 0.44  # empirical guess from Dean (1991)
        
        [A_opt] = fsolve(self._sse_gradient, x0=guess, xtol=1e-6, maxfev=10_000)
        self.A = float(A_opt)
        return self
    
    # -----------------------------------------------------------------
    # 4. EXECUTION OF THE COMPLETE MODEL
    # -----------------------------------------------------------------

    def run_model(self) -> "cal_dean1991":
        """Computes the profile along `self.x_dense`."""
        if self.A is None:
            raise RuntimeError("First execute `.calibrate()` to define A.")
        self.z_model = dean1991(self.x_dense, self.A, self.sl)
        return self

    # -----------------------------------------------------------------
    # 5. WRAPPERS COMPATIBLE WITH fast_optimization  (opcionais)
    # -----------------------------------------------------------------

    def model_sim(self, par: np.ndarray) -> np.ndarray:  # noqa: N802
        """Wrapper to provide the same signature required by GA/PSO."""
        A = float(par[0])
        return dean1991(self.x_obs, A, 0.0)

    def init_par(self, pop_size: int) -> np.ndarray:  # noqa: N802
        """Random starting population within ±50% of analytical guess."""
        if self.A is None:
            # calculate guess via analytical method if not yet calibrated
            lnA_est = np.mean(np.log(self.z_obs_rel) - (2.0 / 3.0) * np.log(self.x_obs))
            estA = float(np.exp(lnA_est))
        else:
            estA = self.A
        low, high = estA * 0.5, estA * 1.5
        return np.random.uniform(low, high, size=(pop_size, 1))

    # -----------------------------------------------------------------
    # 6. BASIC METRICS (RMSE, Bias)
    # -----------------------------------------------------------------

    def metrics(self) -> dict[str, float]:  # noqa: D401
        if self.z_model is None:
            raise RuntimeError("Execute `.run_model()` antes de pedir métricas.")
        # Interpola modelo nos pontos observados para comparar
        z_interp = np.interp(self.x_obs, self.x_dense, self.z_model)
        rmse = float(np.sqrt(np.mean((z_interp - self.z_obs) ** 2)))
        bias = float(np.mean(z_interp - self.z_obs))
        return {"RMSE": rmse, "Bias": bias}
  