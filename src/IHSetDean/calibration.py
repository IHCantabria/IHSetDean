# calibration.py — robust Dean(1991) calibration (X- or Y-fit)
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from .dean1991 import dean1991, dean1991_rev

class cal_dean1991(object):
    """
    HTL   : reference level (e.g., High Tide Level) [m, positive upward]
    doc   : depth of closure [m, positive downward; used to filter CSV between HTL and doc]
    d50   : median grain size [mm], optional
    K     : empirical constant (dimensionless). If None, 0.51 (Dean) will be used when needed.
    fit_mode : "y" (legacy) or "x" (new). If "x", calibration minimizes residuals in X.
    """

    def __init__(self, HTL: float, doc: float, d50: float | None = None, K: float | None = None,
                 fit_mode: str = "x"):
        self.HTL = HTL
        self.doc = doc
        self.d50 = d50
        self.K = K
        self.A = None

        # field data
        self.data = False
        self.x_raw = None
        self.y_raw = None

        # processed
        self.x_obs = None            # observed x relative to x_drift
        self.y_obs = None            # observed y (absolute)
        self.y_obs_rel = None        # observed y relative to HTL
        self.x_dense = None          # kept only for back-compat (not used for plotting)
        self.x_drift = 0.0           # shoreline position (x where y==HTL)

        self.fit_mode = (fit_mode or "y").lower().strip()
        if self.fit_mode not in ("y", "x"):
            raise ValueError("fit_mode must be 'y' or 'x'")

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
    
    # ---------- helpers específicos para a calibração entre X0 e DoC ----------

    def _interp_y_at_x_csv(self, xq: float) -> float:
        """
        Interpola (com extrapolação linear nas bordas) o Y do CSV no X=xq.
        Requer que self.data==True e self.x_raw/self.y_raw populados (add_data).
        """
        if not self.data or self.x_raw is None or self.y_raw is None:
            raise RuntimeError("CSV não carregado. Chame add_data() antes.")

        xr = np.asarray(self.x_raw, dtype=float)
        yr = np.asarray(self.y_raw, dtype=float)

        # Caso trivial
        if xr.size < 2:
            return float(yr[0])

        # Interpola com extrapolação linear manual
        if xq <= xr.min():
            i0, i1 = 0, 1
        elif xq >= xr.max():
            i0, i1 = -2, -1
        else:
            # intervalo interno
            i1 = np.searchsorted(xr, xq)
            i0 = max(0, i1 - 1)
        x0, x1 = float(xr[i0]), float(xr[i1])
        y0, y1 = float(yr[i0]), float(yr[i1])
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * ( (xq - x0) / (x1 - x0) )

    def _x_doc_from_csv(self) -> float:
        """
        Retorna o X (absoluto) onde o CSV cru cruza Y=DoC (interpola).
        Se a série não atinge exatamente o DoC, interpola entre os dois pontos
        que o envolvem; se estiver todo acima/abaixo, faz extrapolação linear de borda.
        """
        if not self.data or self.x_raw is None or self.y_raw is None:
            raise RuntimeError("CSV não carregado. Chame add_data() antes.")

        xr = np.asarray(self.x_raw, dtype=float)
        yr = np.asarray(self.y_raw, dtype=float)

        # Procurar mudança de sinal de (yr - doc)
        dif = yr - self.doc
        s = np.sign(dif)
        cross = np.where(s[:-1] * s[1:] <= 0)[0]
        if cross.size:
            i = cross[0]
            x0, x1 = xr[i], xr[i+1]
            y0, y1 = yr[i], yr[i+1]
            if y1 == y0:
                return float(x0)
            # interpola x para y==doc
            return float(x0 + (self.doc - y0) * (x1 - x0) / (y1 - y0))

        # Sem cruzamento: extrapola pela borda mais próxima
        # (escolhe o ponto cujo |y-doc| é menor e usa o seu segmento vizinho)
        k = int(np.argmin(np.abs(dif)))
        if k == 0:
            x0, x1 = xr[0], xr[1]
            y0, y1 = yr[0], yr[1]
        else:
            x0, x1 = xr[-2], xr[-1]
            y0, y1 = yr[-2], yr[-1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (self.doc - y0) * (x1 - x0) / (y1 - y0))

    def calibrate_segment_x(self, x0_abs: float):
        """
        <<< MÉTODO-CHAVE >>>
        1) Obtém Y0 = Y_csv(X=x0_abs) e X_doc a partir do CSV bruto.
        2) Seleciona o segmento do CSV: X ∈ [X0, X_doc].
        3) Ajusta 'A' minimizando SSE em X usando o modelo:
        x_pred(y; A) = X0 + dean1991_rev(y, A, HTL) - dean1991_rev(Y0, A, HTL)
        (garante que a curva passe por (X0, Y0)).
        4) Retorna (x_dean_seg, y_dean_seg, A, X0, Y0, X_doc) do segmento calibrado.
        """
        if not self.data:
            raise RuntimeError("Sem CSV. Carregue o CSV com add_data() antes.")

        X0 = float(x0_abs)
        Y0 = float(self._interp_y_at_x_csv(X0))
        Xdoc = float(self._x_doc_from_csv())

        xr = np.asarray(self.x_raw, dtype=float)
        yr = np.asarray(self.y_raw, dtype=float)

        # Garante ordem crescente de X para o intervalo
        xmin, xmax = (X0, Xdoc) if X0 <= Xdoc else (Xdoc, X0)
        m = (xr >= xmin) & (xr <= xmax)
        if not np.any(m):
            raise RuntimeError("Não há pontos do CSV entre X0 e DoC para calibrar.")

        x_seg = xr[m]
        y_seg = yr[m]

        # Objetivo: minimizar erro em X
        def sse_in_x(A):
            A = float(A)
            if not np.isfinite(A) or A <= 0:
                return 1e30
            xpred = X0 + dean1991_rev(y_seg, A, self.HTL) - dean1991_rev(Y0, A, self.HTL)
            res = xpred - x_seg
            return float(np.sum(res * res))

        # Busca A — usa A prévio como palpite se houver
        A0 = self.A if (self.A is not None and np.isfinite(self.A) and self.A > 0) else 0.3
        opt = minimize_scalar(sse_in_x, bounds=(1e-4, 5.0), method="bounded")
        Ahat = float(opt.x if np.isfinite(opt.x) else A0)
        self.A = Ahat  # salva A calibrado

        # Gera o perfil de Dean APENAS no segmento [Y0 .. DoC]
        y_grid = np.linspace(Y0, self.doc, 400) if Y0 <= self.doc else np.linspace(self.doc, Y0, 400)
        x_grid = X0 + dean1991_rev(y_grid, Ahat, self.HTL) - dean1991_rev(Y0, Ahat, self.HTL)

        # Se X0 > Xdoc, inverter para manter X crescente (apenas estética do plot)
        if x_grid[0] > x_grid[-1]:
            x_grid = x_grid[::-1]
            y_grid = y_grid[::-1]

        return x_grid, y_grid, Ahat, X0, Y0, Xdoc


    # ---------- profile generators (always draw full HTL→DoC) ----------
    def dean_A_rev(self, A: float):
        """Draw Dean curve on a Y-grid across the full [HTL, doc] domain."""
        self.A = A
        y = np.linspace(self.HTL, self.doc, 1000)
        x_rel = dean1991_rev(y, A, self.HTL)  # x >= 0
        return x_rel + self.x_drift, y

    def dean_A(self, A: float):
        """Public plotting helper (same as 'rev' to guarantee full-width curve)."""
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

    # ---------- shifting ----------
    def shift_HTL(self, new_x0: float, recalibrate: bool = False):
        """Translate horizontally by setting x_drift. If `recalibrate=True`, recalibrates A."""
        self.x_drift = float(new_x0)
        if self.A is None:
            if self.d50 is None:
                raise ValueError("A not defined. Provide d50 (+K optional) or calibrate with CSV before shifting.")
            self.A = self._A_from_d50(self.d50, self.K if self.K is not None else 0.51)
        if self.data and recalibrate:
            self._rebuild_obs_keep_drift()
            self.calibrate()  # robust bounds + masking
        return self.dean_A(self.A)

    def shift_x0(self, new_x0: float, recalibrate: bool = False):
        return self.shift_HTL(new_x0, recalibrate=recalibrate)

    # ---------- data ----------
    def add_data(self, path: str):
        df = pd.read_csv(path, dtype={"X": float, "Y": float})
        self.x_raw = pd.to_numeric(df["X"], errors="coerce").to_numpy()
        self.y_raw = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
        self.recalc_x_dense()
        self.data = True
        return self.calibrate()

    # ---------- robust masks for calibration ----------
    # NEW: ensure we never feed x<=0 to x^(2/3) in Y-fit
    def _mask_yfit(self):
        if self.x_obs is None or self.y_obs_rel is None:
            return None
        m = (np.isfinite(self.x_obs) & np.isfinite(self.y_obs_rel) &
             (self.x_obs > 0.0) & (self.y_obs_rel >= 0.0))
        return m

    # NEW: ensure valid y for inverse Dean in X-fit
    def _mask_xfit(self):
        if self.y_obs is None or self.x_obs is None:
            return None
        m = (np.isfinite(self.y_obs) & np.isfinite(self.x_obs) &
            (self.x_obs >= 0.0) &
            (self.y_obs <= self.doc))
        return m

    # ---------- SSE objectives ----------
    def _sse_y(self, A: float) -> float:
        if A <= 0 or not np.isfinite(A):
            return 1e30
        m = self._mask_yfit()
        if m is None or not np.any(m):
            return 1e30
        x = self.x_obs[m]
        yrel = self.y_obs_rel[m]
        # NEW: clip a copy to keep >0 safely (don’t alter stored arrays)
        x_clip = np.maximum(x, 1e-6)
        y_hat_rel = dean1991(x_clip, A, 0.0)
        diff = y_hat_rel - yrel
        return float(np.sum(diff * diff))

    def _sse_x(self, A: float) -> float:
        if A <= 0 or not np.isfinite(A):
            return 1e30
        m = self._mask_xfit()
        if m is None or not np.any(m):
            return 1e30
        y = self.y_obs[m]
        x = self.x_obs[m]  # may have negatives, ok (we compare after predicting x_hat from y)
        x_hat_rel = dean1991_rev(y, A, self.HTL)
        diff = x_hat_rel - x
        return float(np.sum(diff * diff))

    # ---------- bounded minimization with data-driven bracket ----------
    # NEW: build [lo, hi] around a robust A guess from data (if available)
    def _A_bounds_guess(self, mode: str) -> tuple[float, float]:
        lo_default, hi_default = 1e-3, 1.0
        if mode == "y":
            m = self._mask_yfit()
            if m is None or not np.any(m):
                return lo_default, hi_default
            x = np.maximum(self.x_obs[m], 1e-6)
            yrel = self.y_obs_rel[m]
            xr23 = np.power(x, 2.0/3.0)
            ratio = yrel / xr23
        else:
            m = self._mask_xfit()
            if m is None or not np.any(m):
                return lo_default, hi_default
            # from x = ((yrel)/A)^(3/2)  =>  A = yrel / x^(2/3)
            yrel = self.y_obs[m] - self.HTL
            x = np.maximum(self.x_obs[m], 1e-6)
            xr23 = np.power(x, 2.0/3.0)
            ratio = yrel / xr23

        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 5:
            return lo_default, hi_default
        A0 = float(np.median(ratio))
        # bracket around A0 (avoid silly ranges)
        lo = max(1e-4, A0/6.0)
        hi = min(5.0,  A0*6.0)
        if lo >= hi:
            lo, hi = lo_default, hi_default
        return lo, hi

    def _minimize_A(self, mode: str) -> float:
        lo, hi = self._A_bounds_guess(mode)
        obj = self._sse_x if mode == "x" else self._sse_y
        res = minimize_scalar(obj, method="bounded", bounds=(lo, hi),
                              options={"xatol": 1e-8, "maxiter": 1500})
        if not res.success:
            # widen once
            res = minimize_scalar(obj, method="bounded", bounds=(1e-5, 5.0),
                                  options={"xatol": 1e-8, "maxiter": 2500})
        return float(res.x)

    # ---------- calibration ----------
    def calibrate(self):
        """Estimate A minimizing SSE in the chosen space ('x' or 'y')."""
        # no CSV: nothing to calibrate
        if self.x_obs is None or self.y_obs is None:
            if self.A is None and self.d50 is not None:
                self.A = self._A_from_d50(self.d50, self.K if self.K is not None else 0.51)
            return self.dean_A(self.A)

        # choose mode
        mode = self.fit_mode
        # if mask is empty (e.g. slider very landward), try the other mask as fallback
        if mode == "y" and not np.any(self._mask_yfit()):
            mode = "x"
        if mode == "x" and not np.any(self._mask_xfit()):
            mode = "y"

        A_opt = self._minimize_A(mode)
        self.A = A_opt
        # Perfil recalibrado (roxo): usado apenas internamente se quiser plotar depois
        x_calib, y_calib = dean1991_rev(self.y_obs[self._mask_xfit()], self.A, self.HTL) + self.x_drift, self.y_obs[self._mask_xfit()]
        # Perfil atualizado completo de Dean entre HTL e DoC (lilás)
        x_dean_full, y_dean_full = self.dean_A_rev(self.A)
        return x_dean_full, y_dean_full

    # ---------- observed arrays ----------
    def recalc_x_dense(self):
        """(Re)compute observed arrays and x_drift (x where y==HTL)."""
        y_tmp = self.y_raw.copy()
        if np.mean(y_tmp[np.isfinite(y_tmp)]) < 0:
            y_tmp = -y_tmp

        mask = (np.isfinite(self.x_raw) & np.isfinite(y_tmp) &
            (y_tmp <= self.doc))
        if not np.any(mask):
            raise ValueError("CSV não contém dados entre HTL e DoC.")

        x_cut = self.x_raw[mask]
        y_cut = y_tmp[mask]

        # robust shoreline position: sort-by-y before interpolation
        idx = np.argsort(y_cut)
        y_sorted = y_cut[idx]
        x_sorted = x_cut[idx]
        x_at_HTL = np.interp(self.HTL, y_sorted, x_sorted)
        self.x_drift = float(x_at_HTL)

        # observed arrays relative to current drift
        self.x_obs = x_cut - self.x_drift
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL

        # kept for back-compat (not used to plot Dean)
        self.x_dense = np.linspace(max(0.0, float(np.nanmin(self.x_obs))),
                                   float(np.nanmax(self.x_obs)), 1000)

    def _rebuild_obs_keep_drift(self):
        """Rebuilds x_obs/y_obs after changing x_drift externally (preserve drift)."""
        if self.x_raw is None or self.y_raw is None:
            raise ValueError("No CSV loaded.")
        y_tmp = self.y_raw.copy()
        if np.mean(y_tmp[np.isfinite(y_tmp)]) < 0:
            y_tmp = -y_tmp

        mask = (np.isfinite(self.x_raw) & np.isfinite(y_tmp) &
            (y_tmp <= self.doc))
        if not np.any(mask):
            raise ValueError("CSV não contém dados entre HTL e DoC.")

        x_cut = self.x_raw[mask]
        y_cut = y_tmp[mask]

        self.x_obs = x_cut - float(self.x_drift)
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL

        self.x_dense = np.linspace(max(0.0, float(np.nanmin(self.x_obs))),
                                   float(np.nanmax(self.x_obs)) if np.isfinite(self.x_obs).any() else 1.0,
                                   1000)

    # ---------- metrics ----------
    def metrics(self, x_dean: np.ndarray, y_dean: np.ndarray):
        """Compute RMSE and R^2 by interpolating y_dean(x) to observed X."""
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
        """Helper: returns (x_obs, y_obs, x_dean, y_dean)."""
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

        x_obs = np.array([]); y_obs = np.array([])
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
