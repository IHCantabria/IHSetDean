# dean91_tst.py – Dean (1991) profile – legacy structure with robust calibration
# Keeps: root_path + class DeanTest
# NEW: bounded minimization for A (robust), Dean curve from Y-grid (HTL→DoC),
#      slider recalibrates (keeps HTL/DoC), constrained_layout to avoid warnings.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ------------------------------------------------------------------------------
# User path config (legacy style)
# ------------------------------------------------------------------------------
#root_path = "D:/mario_mascagni/IH_Cantabria/IHSet/04_models/dean1991/IHSetDean/IHSetDean91" # windows
root_path = "/home/mmascagni/work/IHCantabria/IHSet/models/dean1991/IHSetDean/IHSetDean91/" # linux

sys.path.insert(0, str(root_path))
# Try plain import first (legacy), then package-style if available
try:
    from calibration import cal_dean1991
except Exception:
    from IHSetDean91.calibration import cal_dean1991  # pragma: no cover


class DeanTest:
    def __init__(self):
        # ---------------- user params ----------------
        self.HTL = -2.0
        self.doc = 8.0
        self.d50 = 0.30
        self.K   = 0.51

        # NEW: choose calibration mode: "x" (new) or "y" (legacy)
        self.fit_mode = "x"

        # CSV path (in the same folder as root_path)
        #self.csv = str(Path(root_path) / "XY_PuertoChiquito_clean.csv")
        self.csv = "XY_PuertoChiquito_clean.csv"

        # DoC marker appearance
        self.doc_marker_color = "k"
        self.doc_marker_size  = 90.0

        # ---------------- model ----------------
        self.model = cal_dean1991(
            HTL=self.HTL, doc=self.doc, d50=self.d50, K=self.K, fit_mode=self.fit_mode
        )

        # raw CSV (absolute X,Y)
        self.x_raw = np.array([])
        self.y_raw = np.array([])

        # Load CSV and initial calibration
        if self.csv:
            try:
                df = pd.read_csv(self.csv, dtype={'X': float, 'Y': float})
                self.x_raw = pd.to_numeric(df['X'], errors='coerce').to_numpy()
                self.y_raw = pd.to_numeric(df['Y'], errors='coerce').to_numpy()
                if np.nanmean(self.y_raw) < 0:
                    self.y_raw = -self.y_raw
                self.model.add_data(self.csv)
            except Exception as e:
                print(f"[Warning] Could not load CSV '{self.csv}': {e}")
                self.csv = None

        # Produce initial Dean + observed slice (if CSV present)
        self.x_obs, self.y_obs, self.x_dean, self.y_dean = self.model.run_model(csv=self.csv)
        self._x0_init = float(getattr(self.model, "x_drift", 0.0))

    def _has_csv(self) -> bool:
        return self.x_raw.size > 0 and np.isfinite(self.x_raw).any()

    def _doc_point(self):
        """Return a (x_doc, y_doc) point near DoC for plotting."""
        if not self._has_csv():
            return None
        mfin = np.isfinite(self.x_raw) & np.isfinite(self.y_raw)
        if not np.any(mfin):
            return None
        x = self.x_raw[mfin]
        y = self.y_raw[mfin]
        if np.nanmin(y) <= self.doc <= np.nanmax(y):
            diff = y - self.doc
            cross = np.where(((diff[:-1] <= 0) & (diff[1:] >= 0)) |
                             ((diff[:-1] >= 0) & (diff[1:] <= 0)))[0]
            if cross.size > 0:
                i = cross[0]
                x0, x1 = x[i], x[i+1]
                y0, y1 = y[i], y[i+1]
                if np.isfinite(y1 - y0) and abs(y1 - y0) > 1e-12:
                    x_doc = x0 + (self.doc - y0) * (x1 - x0) / (y1 - y0)
                else:
                    x_doc = 0.5 * (x0 + x1)
                return float(x_doc), float(self.doc)
        k = int(np.nanargmin(np.abs(y - self.doc)))
        return float(x[k]), float(y[k])

    def plot(self):
        # NEW: use constrained_layout to avoid tight_layout warning with widgets
        fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)

        LIGHTSAND = "#f4dcb8"; DARKSAND  = "#d6b07a"; WATER = "#a6cee3"

        if self._has_csv():
            x_min, x_max = float(np.nanmin(self.x_raw)), float(np.nanmax(self.x_raw))
            y_bottom = float(np.nanmax(self.y_raw))
            y_top = float(min(np.nanmin(self.y_raw), self.HTL)) - 1.0
        else:
            x_min, x_max = float(np.nanmin(self.x_dean)), float(np.nanmax(self.x_dean))
            y_bottom = float(np.nanmax(self.y_dean))
            y_top = float(min(np.nanmin(self.y_dean), self.HTL)) - 1.0

        ax.fill_between([x_min, x_max], self.HTL, y_bottom, color=WATER, alpha=0.7, zorder=0)
        fill_dean = ax.fill_between(self.x_dean, self.y_dean, y_bottom, color=DARKSAND, alpha=0.7, zorder=3)
        line_dean, = ax.plot(self.x_dean, self.y_dean, "r--", lw=1.6, label="Dean Profile", zorder=5)
        line_HTL,  = ax.plot([x_min, x_max], [self.HTL, self.HTL], color="blue", lw=1.5, label="High Tide level", zorder=1)

        line_obs = None
        sc_doc = None
        if self._has_csv():
            ax.fill_between(self.x_raw, self.y_raw, y_bottom, color=LIGHTSAND, zorder=2)
            line_obs, = ax.plot(self.x_raw, self.y_raw, color="black", lw=1.6, label="Observed (CSV)", zorder=4)
            doc_pt = self._doc_point()
            if doc_pt is not None:
                sc_doc = ax.scatter([doc_pt[0]], [doc_pt[1]], s=self.doc_marker_size, c=self.doc_marker_color,
                                    zorder=6, label=f"DoC ({self.doc:.1f} m)")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_top, y_bottom)
        ax.set_xlabel("Cross-shore distance X [m]")
        ax.set_ylabel("Elevation / Depth [m]")
        ax.invert_yaxis()
        ax.grid(True, linestyle=":", lw=0.5)

        def rebuild_legend():
            handles = [line_HTL]
            labels  = ["High Tide level"]
            if line_obs is not None:
                handles.append(line_obs); labels.append("Observed (CSV)")
            if sc_doc is not None:
                handles.append(sc_doc);  labels.append(f"DoC={self.doc:.1f} m")
            handles.append(line_dean);  labels.append(line_dean.get_label())
            ax.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

        # Initial legend with metrics
        rmse, r2 = self.model.metrics(self.x_dean, self.y_dean) if self._has_csv() else (None, None)
        A_now = getattr(self.model, "A", None)
        lbl = ["Dean Profile"]
        if A_now is not None: lbl.append(f"A={A_now:.4f}")
        if r2    is not None: lbl.append(f"R²={r2:.3f}")
        if rmse  is not None: lbl.append(f"RMSE={rmse:.3f} m")
        line_dean.set_label("\n".join(lbl))
        rebuild_legend()

        # ---------------- slider (x0 with recalibration) ----------------
        if self._has_csv():
            x0_min = float(np.nanmin(self.x_raw))
            x0_max = float(np.nanmax(self.x_raw))
            init_x0 = float(getattr(self.model, "x_drift", self._x0_init))

            # reserve a little bottom space for the slider
            fig.subplots_adjust(bottom=0.12)

            ax_sl = plt.axes([0.09, 0.03, 0.72, 0.03])
            slider = Slider(
                ax=ax_sl,
                label="x0 (m) – drift (recalibration)",
                valmin=x0_min, valmax=x0_max,
                valinit=init_x0, valstep=0.5,
            )

            def update(val):
                x0 = slider.val
                try:
                    x_dean_new, y_dean_new = self.model.shift_HTL(x0, recalibrate=True)[:2]
                except Exception as e:
                    print(f"[Warning] Shift failed x0={x0:.2f}: {e}")
                    return

                line_dean.set_data(x_dean_new, y_dean_new)
                nonlocal fill_dean
                fill_dean.remove()
                fill_dean = ax.fill_between(x_dean_new, y_dean_new, y_bottom, color=DARKSAND, alpha=0.7, zorder=3)

                rmse, r2 = self.model.metrics(x_dean_new, y_dean_new)
                A_now = getattr(self.model, "A", None)
                lbl = ["Dean Profile"]
                if A_now is not None: lbl.append(f"A={A_now:.4f}")
                if r2    is not None: lbl.append(f"R²={r2:.3f}")
                if rmse  is not None: lbl.append(f"RMSE={rmse:.3f} m")
                line_dean.set_label("\n".join(lbl))
                rebuild_legend()
                fig.canvas.draw_idle()

            slider.on_changed(update)

        plt.show()


if __name__ == "__main__":
    DeanTest().plot()
