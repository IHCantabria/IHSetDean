
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from dean1991 import dean1991, dean1991_rev

class cal_dean1991(object):
    """
    HTL   : reference level (e.g., High Tide Level) [m, positive upward]
    doc   : depth of closure [m, positive downward; used to filter CSV between HTL and doc]
    d50   : median grain size [mm], optional
    K     : empirical constant (dimensionless). If None, 0.51 (Dean) will be used when needed.
    """

    def __init__(self, HTL: float, doc: float, d50: float | None = None, K: float | None = None):
        self.HTL = HTL
        self.doc = doc
        self.d50 = d50
        self.K = K
        self.A = None

        # field data
        self.data = False
        self.x_raw = None
        self.y_raw = None

        # processed / working arrays
        self.x_obs = None
        self.y_obs = None
        self.y_obs_rel = None
        self.x_dense = None
        self.x_drift = 0.0

        if self.d50 is not None:
            self.A = self._A_from_d50(self.d50, self.K if self.K is not None else 0.51)

    # ---------- utilities ----------
    @staticmethod
    def _w_from_d50(d50_mm: float) -> float:
        if d50_mm < 0.1:
            return 1.1e6 * (d50_mm ** 2)
        elif 0.1 <= d50_mm <= 1:
            return 273 * (d50_mm ** 1.1)
        else:
            return 4.36 * (d50_mm ** 0.5)

    def _A_from_d50(self, d50_mm: float, K: float = 0.51) -> float:
        w = self._w_from_d50(d50_mm)
        return K * (w ** 0.44)

    # ---------- profile generators ----------
    def dean_A_rev(self, A: float):
        self.A = A
        y = np.linspace(self.HTL, self.doc + 0.5, 1000)
        x = dean1991_rev(y, A, self.HTL)
        return x, y

    def dean_A(self, A: float):
        self.A = A
        y = dean1991(self.x_dense, A, self.HTL)
        return self.x_dense + self.x_drift, y

    # ---------- inputs ----------
    def from_d50(self, d50_mm: float, K: float | None = None):
        self.d50 = d50_mm
        A = self._A_from_d50(d50_mm, K if K is not None else (self.K if self.K is not None else 0.51))
        if self.data:
            return self.dean_A(A)
        else:
            return self.dean_A_rev(A)

    def from_K(self, K: float):
        if self.d50 is None:
            raise ValueError("Para usar K, forneça também d50 para computar A.")
        A = self._A_from_d50(self.d50, K)
        self.K = K
        if self.data:
            return self.dean_A(A)
        else:
            return self.dean_A_rev(A)

    # ---------- HTL/DoC tweaks ----------
    def change_HTL(self, HTL: float):
        self.HTL = HTL
        if self.data:
            self.recalc_x_dense()
            return self.dean_A(self.A if self.A is not None else self._A_from_d50(self.d50, self.K or 0.51))
        else:
            return self.dean_A_rev(self.A if self.A is not None else self._A_from_d50(self.d50, self.K or 0.51))

    def change_doc(self, doc: float):
        self.doc = doc
        if self.data:
            self.recalc_x_dense()
            return self.dean_A(self.A if self.A is not None else self._A_from_d50(self.d50, self.K or 0.51))
        else:
            return self.dean_A_rev(self.A if self.A is not None else self._A_from_d50(self.d50, self.K or 0.51))

    def shift_HTL(self, new_x0: float):
        """Translate the Dean profile horizontally by setting x_drift. No recalibration; HTL/DoC unchanged.
        Returns (x_dean, y_dean, x_drift).
        """
        self.x_drift = float(new_x0)
        if self.A is None:
            if self.d50 is None:
                raise ValueError("A não definido. Forneça d50 (+K opcional) ou calibre com CSV antes de transladar.")
            self.A = self._A_from_d50(self.d50, self.K if self.K is not None else 0.51)

        if self.data and self.x_dense is not None:
            x_dean, y_dean = self.dean_A(self.A)
            return x_dean, y_dean, self.x_drift
        else:
            x_dean, y_dean = self.dean_A_rev(self.A)
            return x_dean + self.x_drift, y_dean, self.x_drift

    # convenience alias
    def shift_x0(self, new_x0: float):
        return self.shift_HTL(new_x0)

    # ---------- data / calibration ----------
    def add_data(self, path: str):
        df = pd.read_csv(path, dtype={"X": float, "Y": float})
        self.x_raw = pd.to_numeric(df["X"], errors="coerce").to_numpy()
        self.y_raw = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
        self.recalc_x_dense()
        self.data = True
        return self.calibrate()

    def _grad_sse(self, A: float) -> float:
        diff = dean1991(self.x_obs, A, 0.0) - self.y_obs_rel
        return float(np.sum(diff * np.power(self.x_obs, 2.0 / 3.0)))

    def calibrate(self):
        guess = 0.51 * (273 * (0.3 ** 1.1)) ** 0.44  # seed (~d50=0.30 mm)
        [A_opt] = fsolve(self._grad_sse, x0=guess, xtol=1e-6, maxfev=5000)
        self.A = float(A_opt)
        return self.dean_A(self.A)

    def recalc_x_dense(self):
        y_tmp = self.y_raw.copy()
        if np.mean(y_tmp[np.isfinite(y_tmp)]) < 0:
            y_tmp = -y_tmp

        mask = (np.isfinite(self.x_raw) & np.isfinite(y_tmp) &
                (y_tmp >= self.HTL) & (y_tmp <= self.doc))
        if not np.any(mask):
            raise ValueError("CSV não contém dados entre HTL e DoC.")

        x_cut = self.x_raw[mask]
        y_cut = y_tmp[mask]

        # horizontal drift as the x where y==HTL (linear interpolation)
        x_at_HTL = np.interp(self.HTL, y_cut, x_cut)
        self.x_drift = x_at_HTL

        # observed profile relative to HTL
        self.x_obs = x_cut - self.x_drift
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL

        # dense x for Dean on observed window
        self.x_dense = np.linspace(float(np.nanmin(self.x_obs)), float(np.nanmax(self.x_obs)), 1000)

    # ---------- metrics ----------
    def metrics(self, x_dean: np.ndarray, y_dean: np.ndarray):
        """Compute RMSE and R^2 between observed (filtered to [HTL, doc]) and Dean profile.
        Interpolates y_dean(x) to the observed x grid where both exist.
        Returns (rmse, r2). If no CSV/overlap, returns (None, None).
        """
        if self.x_raw is None or self.y_raw is None:
            return None, None
        m = (np.isfinite(self.x_raw) & np.isfinite(self.y_raw) &
             (self.y_raw >= self.HTL) & (self.y_raw <= self.doc))
        if not np.any(m):
            return None, None
        x_o = self.x_raw[m]
        y_o = self.y_raw[m]

        xmin, xmax = float(np.nanmin(x_dean)), float(np.nanmax(x_dean))
        mm = (x_o >= xmin) & (x_o <= xmax)
        if not np.any(mm):
            return None, None
        x_use = x_o[mm]
        y_obs = y_o[mm]
        y_pred = np.interp(x_use, x_dean, y_dean)
        resid = y_obs - y_pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y_obs - np.mean(y_obs))**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else None
        return rmse, r2

    # ---------- main ----------
    def run_model(self, A: float | None = None, csv: str | None = None):
        if self.d50 is not None and not (0.06 <= self.d50 <= 4.0):
            raise ValueError("D50 deve estar entre 0.06 mm e 4.0 mm.")
        if not (-17.0 <= self.HTL <= 17.0):
            raise ValueError("HTL deve estar entre -17 m e +17 m.")
        if not (0.5 <= self.doc <= 20.0):
            raise ValueError("DoC deve estar entre 0.5 m e 20 m.")

        if csv:
            x_dean, y_dean = self.add_data(csv)
            x_obs = self.x_obs + self.x_drift
            y_obs = self.y_obs
            return x_obs, y_obs, x_dean, y_dean

        x_obs = np.array([])
        y_obs = np.array([])

        if A is not None:
            self.A = A
            x_dean, y_dean = self.dean_A_rev(A)
        elif self.d50 is not None:
            A_calc = self._A_from_d50(self.d50, self.K if self.K is not None else 0.51)
            self.A = A_calc
            x_dean, y_dean = self.dean_A_rev(A_calc)
        else:
            raise ValueError("Forneça A diretamente, ou d50 (com K opcional).")

        return x_obs, y_obs, x_dean, y_dean
