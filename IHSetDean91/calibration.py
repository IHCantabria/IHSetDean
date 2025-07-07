# =============================================================
# calibration.py  –  EqP + Cal classes (rev. 2025-07-04)
# =============================================================
"""Dean Equilibrium Beach Profile (1991).

* **eqp_dean1991** - generates profiles *without* CSV:
    - ``dean_d50`` → calculates *A* via D50 (mm).
    - ``dean_A`` → *A* (m^1/3) value provided by user.

* **cal_dean1991** - fits *A* by fsolve function *with* CSV, using only the
portion of the beach profile between *SL* and *DoC*.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from dean1991 import dean1991

__all__ = [
    "eqp_dean1991",
    "cal_dean1991",
]

# =============================================================
# Class 1 – EQP (Empirical Model - NO CSV)
# =============================================================

class eqp_dean1991:
    """Directly generated Dean profile (without calibration).

        Parameters
        ----------
        sl : float
            Water level (m).
        doc : float
            depth of closure (m) *positive for down.
        """

    # ---------------------------------------------------------
    def __init__(self, sl: float, doc: float):
        self.sl = float(sl)
        self.doc = float(doc)
    
    # --- METHOD 1 - Empirical (D50) --------------------------
    @staticmethod
    def _A_from_d50(d50_mm: float) -> float:
        """Moore (1982) compiled 30+ natural and laboratory profiles 
           and fitted A vs D to a log-log plot. Dean (1991) took these 
           points and added cases from Hughes (1983) and Swart (1974) 
           to plot the best-fit line, whose equation is the Empirical 
           Model: **A = 0.067·(D₅₀ [m])^0.44** (D₅₀ mm)."""
        if d50_mm is None:
            raise ValueError("D50 must be provided.")
        return (0.067 * ((d50_mm) ** 0.44))

    # Calculate A from D50 (call: _A_from_d50)
    def dean_d50(self, d50_mm: float) -> tuple[np.ndarray, np.ndarray]:
        self.A = self._A_from_d50(d50_mm)
        return self.dean_A(self.A)

    # --- METHOD 2 - Empirical (A) ----------------------------
    def dean_A(self, A: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) untill *y* reaches *DoC* (±0.05 m)."""
        # Preliminar domain (2 km, 0.1 m resolution)
        x = np.linspace(0.0, 2_000.0, 20_001)
        y = dean1991(x, A, self.sl)
        # Select points up to DoC (tolerance 0.05 m)
        mask = y <= self.doc + 0.5
        if not np.any(mask):
            raise RuntimeError("DoC not reached within 2km - increase domain.")
        
        return x[mask], y[mask]

# =============================================================
# Class 2 – CAL (with CSV provided)
# =============================================================

class cal_dean1991:  # noqa: N801
    """Calibrates *A* using CSV, but **only** between SL and DoC."""

    # ---------------- STRUCTURE ------------------------------
    def __init__(
        self,
        csv_path: str | Path,
        sl: float,
        doc: float,
    ):
        self.sl = float(sl)
        self.doc = float(doc)
        df = pd.read_csv(Path(csv_path), header=None, names=["X", "Y"])
        x_raw = df["X"].to_numpy(float)
        y_raw = df["Y"].to_numpy(float)

        # positive depth down
        if np.mean(y_raw) < 0:
            y_raw = -y_raw

        # cut between SL and DoC
        m = (y_raw >= self.sl) & (y_raw <= self.doc)
        if not np.any(m):
            raise ValueError("CSV does not contain values between SL and DoC.")
        self.x_obs = x_raw[m]
        self.y_obs = y_raw[m]
        self.y_obs_rel = self.y_obs - self.sl  # relative to SL

        # Initialize main variables
        self.x_dense = np.linspace(float(self.x_obs.min()), float(self.x_obs.max()), 1_000)
        self.A: float | None = None
        self.y_model: np.ndarray | None = None

    # ------------ CALCULATE FSOLVE GRAD FUNCTION --------------------
    def _grad_sse(self, A: float) -> float:
        diff = dean1991(self.x_obs, A, 0.0) - self.y_obs_rel
        return float(np.sum(diff * np.power(self.x_obs, 2.0 / 3.0)))

    # ------------ FSOLVE CALIBRATION --------------------------------
    def calibrate(self) -> "cal_dean1991":
        guess = 0.067 * (0.3) ** 0.44  # fixed seed (~D50 = 0.30 mm)
        [A_opt] = fsolve(self._grad_sse, x0=guess, xtol=1e-6, maxfev=5_000)
        self.A = float(A_opt)
        return self

    # ------------ RUN CALIBRATION MODEL -----------------------------
    def run_model(self) -> "cal_dean1991":
        if self.A is None:
            raise RuntimeError("Execute .calibrate() before .run_model().")
        self.y_model = dean1991(self.x_dense, self.A, self.sl)
        return self

    # ------------ FINAL METRICS -------------------------------------
    def metrics(self) -> dict[str, float]:
        if self.y_model is None:
            raise RuntimeError("Call .run_model() before metrics.")
        y_i = np.interp(self.x_obs, self.x_dense, self.y_model)
        rmse = float(np.sqrt(np.mean((y_i - self.y_obs) ** 2)))
        bias = float(np.mean(y_i - self.y_obs))
        return {"RMSE": rmse, "Bias": bias}
