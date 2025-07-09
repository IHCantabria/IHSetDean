import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from dean1991 import dean1991, dean1991_rev

class cal_dean1991_2(object):  

    def __init__(self):
        
        self.sl = None
        self.doc = None
        self.A = None
        self.D50 = None
        self.x_data = None
        self.y_data = None

    @staticmethod
    def _A_from_d50(d50_mm: float) -> float:
        return (0.067 * ((d50_mm) ** 0.44))

    def dean_d50(self, d50_mm: float) -> tuple[np.ndarray, np.ndarray]:
        self.d50 = d50_mm
        if self.data:
            return self.dean_A(self._A_from_d50(d50_mm))
        else:
            return self.dean_A_rev(self._A_from_d50(d50_mm))

    def dean_A_rev(self, A: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) untill *y* reaches *DoC* (±0.05 m)."""
        self.A = A
        y = np.linspace(self.sl, self.doc + 0.5, 1e+3)
        x = dean1991_rev(y, A, self.sl)
        return x, y
    
    def dean_A(self, A: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) untill *y* reaches *DoC* (±0.05 m)."""
        self.A = A
        y = dean1991(self.x_dense, A, self.sl)
        return self.x_dense, y
    
    def add_data(self, path):
        
        df = pd.read_csv(path, header=None, names=["X", "Y"])
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
        self.x_dense = np.linspace(float(self.x_obs.min()), float(self.x_obs.max()), 1e+3)

        return self.calibrate()

    def _grad_sse(self, A: float) -> float:
        diff = dean1991(self.x_obs, A, 0.0) - self.y_obs_rel
        return float(np.sum(diff * np.power(self.x_obs, 2.0 / 3.0)))

    def calibrate(self):
        guess = 0.067 * (0.3) ** 0.44  # fixed seed (~D50 = 0.30 mm)
        [A_opt] = fsolve(self._grad_sse, x0=guess, xtol=1e-6, maxfev=5e+3)
        return self.dean_A(float(A_opt))

