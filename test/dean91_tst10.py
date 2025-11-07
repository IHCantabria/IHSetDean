# dean91_tst10.py — Dean(1991) with AUTO mode:
#   - calib_mode if CSV is provided (dominates any other setting)
#   - A_mode     if A_user is provided and CSV is not provided
#   - d50_mode   if D50 is provided and neither CSV nor A_user are provided

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import matplotlib

# More stable dynamic chart backend for sliders
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["MPLBACKEND"] = "TkAgg"
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ------------------------------------------------------------------ imports IHSet
ROOT = Path(__file__).resolve().parent.parent  # .../IHSetDean/test -> IHSetDean
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.IHSetDean.calibration import cal_dean1991
    from src.IHSetDean.dean1991 import dean1991_rev
except Exception:
    # fallback se instalado como pacote local
    from IHSetDean.calibration import cal_dean1991
    from IHSetDean.dean1991 import dean1991_rev

# ------------------------------ color palette
WATER      = "#a6cee3"
SAND_LIGHT = "#f4dcb8"
SAND_DARK  = "#d6b07a"


class DeanTest:
    def __init__(self, *, HTL: float, doc: float, d50: float | None,
                 K: float | None, A_user: float | None,
                 csv: str | None):
        # ------------------ INPUTS
        self.HTL = float(HTL)
        self.doc = float(doc)
        self.d50 = None if d50 is None else float(d50)
        self.K   = None if K   is None else float(K)
        self.A_user = None if A_user is None else float(A_user)
        self.csv = csv

        # ------------------ figure/handles instance
        self.fig, self.ax = plt.subplots(figsize=(12, 6.2), constrained_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.78, top=0.96, bottom=0.18)
        self.water_fill = None
        self.htl_line   = None
        self.obs_line   = None
        self.obs_fill   = None
        self.dean_line  = None
        self.dean_fill  = None
        self.doc_scatter= None

        # Surveyed data instance
        self.x_obs = np.array([])
        self.y_obs = np.array([])

        # slider instance
        self.slider = None
        self._x0_current = None

        # model/calibration instance
        self.model = cal_dean1991(HTL=self.HTL, doc=self.doc, d50=self.d50, K=self.K, fit_mode="x")

        # ------------------ AUTOMATIC MODE SELECTION
        mode = self._auto_choose_mode()
        # print(f"[DeanTest] AUTO mode: {mode}")  # for debug

        if mode == "calib_mode":
            self._run_calib_with_slider()
        elif mode == "A_mode":
            self._run_dean_A()
        elif mode == "d50_mode":
            self._run_dean_d50()
        else:
            raise ValueError("Modo inválido após detecção automática.")

        plt.show()

    # ------------------ auto chooser
    def _auto_choose_mode(self) -> str:
        """
        1) If CSV was provided (non-empty string) -> calib_mode
        2) Otherwise, if A_user was provided (not None) -> A_mode
        3) Otherwise, if D50 was provided (not None) -> d50_mode
        4) Otherwise, error.
        """
        if self.csv and str(self.csv).strip():
            return "calib_mode"
        if self.A_user is not None:
            return "A_mode"
        if self.d50 is not None:
            return "d50_mode"
        raise ValueError(
            "Unable to select Dean mode: "
            "Please provide CSV (calib_mode) OR A_user (A_mode) OR D50 (d50_mode)."
        )

    # ---------- Graphic utils
    def _draw_water_htl(self, x_min, x_max, y_bottom):
        if self.water_fill is not None:
            try: self.water_fill.remove()
            except Exception: pass
            self.water_fill = None
        self.water_fill = self.ax.fill_between([x_min, x_max], self.HTL, y_bottom,
                                               color=WATER, alpha=0.7, zorder=0)
        if self.htl_line is None:
            (self.htl_line,) = self.ax.plot([x_min, x_max], [self.HTL, self.HTL],
                                            color="blue", lw=1.8, label="High Tide level", zorder=1)
        else:
            self.htl_line.set_data([x_min, x_max], [self.HTL, self.HTL])

    def _style_axes(self, x_min, x_max, y_top, y_bottom):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_top, y_bottom)
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")
        self.ax.invert_yaxis()
        self.ax.grid(True, linestyle=":", lw=0.6)

    def _legend_metrics(self, A: float | None, r2: float | None,
                        rmse: float | None, d50_mm: float | None, show_calib_fill: bool):
        lines = []; labels = []
        lines.append(self.htl_line); labels.append("High Tide level")
        if self.obs_line is not None:
            lines.append(self.obs_line); labels.append("Observed (CSV)")
        if self.doc_scatter is not None:
            lines.append(self.doc_scatter); labels.append(f"DoC={self.doc:.1f} m")
        if d50_mm is not None:
            dummy, = self.ax.plot([], [], alpha=0)
            lines.append(dummy); labels.append(f"D50={d50_mm:.2f} mm")
        if show_calib_fill and self.dean_fill is not None:
            dummy2 = self.ax.fill_between([0,0],[0,0],[0,0], color=SAND_DARK, alpha=0.7)
            lines.append(dummy2); labels.append("Calibrated zone")
        if self.dean_line is not None:
            lbl = ["Dean Profile"]
            if A   is not None:  lbl.append(f"A={A:.4f}")
            if r2  is not None:  lbl.append(f"R²={r2:.3f}")
            if rmse is not None: lbl.append(f"RMSE={rmse:.3f} m")
            self.dean_line.set_label("\n".join(lbl))
            lines.append(self.dean_line); labels.append(self.dean_line.get_label())
        self.ax.legend(handles=lines, labels=labels, loc="upper left",
                       bbox_to_anchor=(1.02, 1), frameon=False)

    # ---------- Dean profile via D50 mm (uses run_model)
    # Note: dean_line receives x_dean, y_dean from run_model
    def _run_dean_d50(self):
        _, _, x_dean, y_dean = self.model.run_model(A=None, csv=None)
        A = float(self.model.A)

        x_min, x_max = float(np.nanmin(x_dean)), float(np.nanmax(x_dean))
        y_bottom = self.doc
        y_top    = self.HTL - 2.0

        self._draw_water_htl(x_min, x_max, y_bottom)

        if self.dean_fill is not None:
            try: self.dean_fill.remove()
            except Exception: pass
        self.dean_fill = self.ax.fill_between(x_dean, y_dean, y_bottom, color=SAND_LIGHT, zorder=2)

        if self.dean_line is None:
            (self.dean_line,) = self.ax.plot(x_dean, y_dean, "r--", lw=1.6, zorder=4)
        else:
            self.dean_line.set_data(x_dean, y_dean)

        if self.doc_scatter is None:
            self.doc_scatter = self.ax.scatter([x_dean[-1]], [self.doc], c="k", s=18, zorder=5)

        self._style_axes(x_min, x_max, y_top, y_bottom)
        self._legend_metrics(A=A, r2=None, rmse=None, d50_mm=self.d50, show_calib_fill=False)

    # ---------- Dean profile via A value defined by the User "A_user" (uses run_model)
    # Note: dean_line receives x_dean, y_dean from run_model
    def _run_dean_A(self):
        _, _, x_dean, y_dean = self.model.run_model(A=self.A_user, csv=None)
        A = float(self.A_user)

        x_min, x_max = float(np.nanmin(x_dean)), float(np.nanmax(x_dean))
        y_bottom = self.doc
        y_top    = self.HTL - 2.0

        self._draw_water_htl(x_min, x_max, y_bottom)

        if self.dean_fill is not None:
            try: self.dean_fill.remove()
            except Exception: pass
        self.dean_fill = self.ax.fill_between(x_dean, y_dean, y_bottom, color=SAND_LIGHT, zorder=2)

        if self.dean_line is None:
            (self.dean_line,) = self.ax.plot(x_dean, y_dean, "r--", lw=1.6, zorder=4)
        else:
            self.dean_line.set_data(x_dean, y_dean)

        if self.doc_scatter is None:
            self.doc_scatter = self.ax.scatter([x_dean[-1]], [self.doc], c="k", s=18, zorder=5)

        self._style_axes(x_min, x_max, y_top, y_bottom)
        self._legend_metrics(A=A, r2=None, rmse=None, d50_mm=None, show_calib_fill=False)

    # ---------- Dean calibrated + slider (X0 → A via calibrate_segment_x)
    def _run_calib_with_slider(self):
        # Loads CSV and calculates x_drift(x where y==HTL) internally.
        #Note: x_obs = distance of the beach profile observed in the CSV file
        #Note: y_obs = elevation of the beach profile observed in the CSV file
        self.model.add_data(self.csv)
        self.x_obs = np.asarray(self.model.x_raw, float)
        self.y_obs = np.asarray(self.model.y_raw, float)

        # Initial X0 (without slider interference) = position where CSV crosses HTL.
        self._x0_current = float(self.model.x_drift)
        self._recalibrate_and_redraw(self._x0_current)

    # ---------- Slider callback outside of the graphic area
        ax_sl = self.fig.add_axes([0.10, 0.065, 0.66, 0.035])
        self.slider = Slider(
            ax=ax_sl, label="X0 (for calib)",
            valmin=float(np.nanmin(self.x_obs)),
            valmax=float(np.nanmax(self.x_obs)),
            valinit=self._x0_current
        )
        self.slider.on_changed(self._on_slider)

    # ---------- Slider callback
    def _on_slider(self, val):
        x0 = float(val)
        if x0 == self._x0_current:
            return
        self._x0_current = x0
        self._recalibrate_and_redraw(x0)

    # ---------- Recalibrate and redraw plot based on new X0 from slider
    def _recalibrate_and_redraw(self, x0):
        #Note: y_full = Elevations of Dean Profile calibrated and extrapoleted from HTL to DoC
        #Note: x_full = Distances of Dean Profile calibrated and extrapoleted from HTL to DoC
        
        # 1) Calibrates Dean's profile on the [X0→DoC] segment
        # Note x_seg and y_seg are supporting segments for calibration between X0 and DoC to determine Ahat, 
        # which is the calibrated Dean parameter A. From Ahat, x_full and y_full are defined, replacing x_seg and y_seg.
        x_seg, y_seg, Ahat, X0, Y0, Xdoc = self.model.calibrate_segment_x(x0)

        # 2) Constructs a complete HTL→DoC profile anchored at (X0 = HTL and Yfinal = DoC):
        # Note: The reverse Dean model "dean1991_rev" needs y (y_full mesh), the estimated A (Ahat), and the HTL vertical offset.
        # Note: The horizontal shift is applied later to align X0 with the Y in the CSV observed profile.
        y_full = np.linspace(self.HTL, self.doc, 1000)
        x_rel_full = dean1991_rev(y_full, Ahat, self.HTL)
        shift = X0 - float(dean1991_rev(Y0, Ahat, self.HTL))
        x_full = x_rel_full + shift

        # 3) Error Metrics: Observed Profile vs. Dean Profile
        rmse, r2 = self.model.metrics(x_full, y_full)

        # 4) Chart limitations: use CSV extension
        x_min = float(np.nanmin(self.x_obs))
        x_max = float(np.nanmax(self.x_obs))
        y_bottom = float(np.nanmax(self.y_obs))   # greatest depth observed (inverted axis)
        y_top    = float(min(np.nanmin(self.y_obs), self.HTL))

        # 5) Draw Water and HTL line
        self._draw_water_htl(x_min, x_max, y_bottom)

        # 6) Draw the field survey beach profile line and filling of (X,Y observed) - black line
        if self.obs_fill is not None:
            try: self.obs_fill.remove()
            except Exception: pass
        self.obs_fill = self.ax.fill_between(self.x_obs, self.y_obs, y_bottom,
                                             color=SAND_LIGHT, zorder=1.5)

        if self.obs_line is None:
            (self.obs_line,) = self.ax.plot(self.x_obs, self.y_obs, color="k", lw=1.6, zorder=3)
        else:
            self.obs_line.set_data(self.x_obs, self.y_obs)

        # 7) Draw the calibrated Dean profile - red line (X_full and Y_full | projection up to HTL)
        if self.dean_line is None:
            (self.dean_line,) = self.ax.plot(x_full, y_full, "r--", lw=1.6, zorder=4)
        else:
            self.dean_line.set_data(x_full, y_full)

        # 8) Fill calibrated zone only between [X0 drifted and Xdoc]
        if self.dean_fill is not None:
            try: self.dean_fill.remove()
            except Exception: pass
        x_end = float(Xdoc) if Xdoc >= X0 else float(X0)
        mask = (x_full >= min(X0, x_end)) & (x_full <= max(X0, x_end))
        self.dean_fill = self.ax.fill_between(x_full[mask], y_full[mask], y_bottom,
                                              color=SAND_DARK, alpha=0.7, zorder=2.8)

        # 9) DoC marker (in CSV)
        if self.doc_scatter is None:
            i_doc = int(np.nanargmin(np.abs(self.y_obs - self.doc)))
            self.doc_scatter = self.ax.scatter([self.x_obs[i_doc]], [self.doc], c="k", s=18, zorder=5)
        else:
            i_doc = int(np.nanargmin(np.abs(self.y_obs - self.doc)))
            self.doc_scatter.set_offsets(np.c_[self.x_obs[i_doc], self.doc])

        self._style_axes(x_min, x_max, y_top, y_bottom)
        self._legend_metrics(A=Ahat, r2=r2, rmse=rmse, d50_mm=self.d50, show_calib_fill=True)
        self.fig.canvas.draw_idle()


# -------------------------- Define INPUT parameters and Run test 
if __name__ == "__main__":
    HTL = -2.0
    DoC =  8.0
    D50 =  0.30   # None or 0.30 mm
    K   =  None
    A_user = None # None or 0.25

    #CSV = ""
    CSV = str(Path(__file__).with_name("XY_PuertoChiquito_clean.csv"))  # define "" to disable calib_mode
    
    # Rules to run Dean model: CSV->calib_mode; else A_user->A_mode; else D50->d50_mode
    app = DeanTest(
        HTL=HTL, doc=DoC, d50=D50, K=K, A_user=A_user,
        csv=CSV
    )
