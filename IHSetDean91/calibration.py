import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from dean1991 import dean1991, dean1991_rev
 
class cal_dean1991(object):  
 
    def __init__(self, sl, doc, d50):
       
        self.sl = sl
        self.doc = doc
        self.A = None
        self.D50 = d50
        self.x_obs = None
        self.y_obs = None
        self.data = False
        self.x_raw = None
        self.y_raw = None
        self.x_offset = 0
        self.x_drift = 0
 
    @staticmethod
    def _A_from_d50(d50_mm: float) -> float:
        return 0.067 * ((d50_mm) ** 0.44)
 
    def from_d50(self, d50_mm: float) -> tuple[np.ndarray, np.ndarray]:
        self.d50 = d50_mm
   
        if self.data:
            return self.dean_A(self._A_from_d50(d50_mm))
        else:
            return self.dean_A_rev(self._A_from_d50(d50_mm))
       
    def from_A(self, A: float) -> tuple[np.ndarray, np.ndarray]:
            self.A = A
            if self.data:
                return self.dean_A(A)
            else:
                return self.dean_A_rev(A)
           
    def change_sl(self, sl: float):
        self.sl = sl
        if self.data:
            self.recalc_x_dense()
            return self.dean_A(self.A)
        else:
            return self.dean_A_rev(self.A)
       
    def change_doc(self, doc: float):
        self.doc = doc
        if self.data:
            self.recalc_x_dense()
            return self.dean_A(self.A)
        else:
            return self.dean_A_rev(self.A)
           
    def add_data(self, path):
        df = pd.read_csv(path, header=None, names=["X", "Y"])
        self.x_raw = df["X"].to_numpy(float)
        self.y_raw = df["Y"].to_numpy(float)
 
        self.recalc_x_dense()
 
        self.data = True
        return self.calibrate()
 
    def dean_A_rev(self, A: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) untill *y* reaches *DoC* (±0.05 m)."""
        self.A = A
        y = np.linspace(self.sl, self.doc + 0.5, 1000)
        x = dean1991_rev(y, A, self.sl)
 
        return x, y
   
    def dean_A(self, A: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x, y) untill *y* reaches *DoC* (±0.05 m)."""
        self.A = A
        y = dean1991(self.x_dense, A, self.sl)
        return self.x_dense + self.x_drift, y
   
    def _grad_sse(self, A: float) -> float:
        diff = dean1991(self.x_obs, A, 0.0) - self.y_obs_rel
        return float(np.sum(diff * np.power(self.x_obs, 2.0 / 3.0)))
 
    def calibrate(self):
        guess = 0.067 * (0.3) ** 0.44  # fixed seed (~D50 = 0.30 mm)
        [A_opt] = fsolve(self._grad_sse, x0=guess, xtol=1e-6, maxfev=5000)
        return self.dean_A(float(A_opt))
       
    def recalc_x_dense(self):
        # positive depth down
        if np.mean(self.y_raw) < 0:
            self.y_raw = -self.y_raw
 
        # cut between SL and DoC
        mask = (self.y_raw >= self.sl) & (self.y_raw <= self.doc)
        if not np.any(mask):
            raise ValueError("CSV does not contain values between SL and DoC.")
        x_cut = self.x_raw[mask]
        y_cut = self.y_raw[mask]

        # interpolate x where y == SL
        x_at_sl = np.interp(self.sl, y_cut, x_cut)
        self.x_drift = x_at_sl

        # observed profile relative to SL and shifted such that x=0 at SL
        self.x_obs = x_cut - self.x_drift
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.sl  # relative to SL

        # Initialize main variables
        self.x_dense = np.linspace(float(self.x_obs.min()), float(self.x_obs.max()), 1000)
        