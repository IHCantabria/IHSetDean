# dean91_tst7.py  (com “Calibrated zone” e D50 na legenda)
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["MPLBACKEND"] = "TkAgg"
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon, Patch

# ----- imports do projeto -----
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
try:
    from calibration import cal_dean1991
except Exception:
    from src.IHSetDean.calibration import cal_dean1991  # pragma: no cover
try:
    from dean1991 import dean1991_rev
except Exception:
    from src.IHSetDean.dean1991 import dean1991_rev  # pragma: no cover

# Paleta
WATER = "#a6cee3"; LIGHTSAND = "#f4dcb8"; DARKSAND = "#d6b07a"
SHOW_CALIB_SPAN = False

class DeanTest:
    def __init__(self):
        # ------- parâmetros do usuário -------
        self.HTL = -2.0
        self.doc = 8.0
        self.d50 = 0.30   # mm
        self.K   = 0.51
        self.csv = str(root_path / "XY_PuertoChiquito_clean.csv")

        # ------- modelo / dados -------
        self.model = cal_dean1991(HTL=self.HTL, doc=self.doc, d50=self.d50, K=self.K)
        self.model.add_data(self.csv)
        self.xr = np.asarray(self.model.x_raw, dtype=float)
        self.yr = np.asarray(self.model.y_raw, dtype=float)

        self.xmin = float(np.nanmin(self.xr)); self.xmax = float(np.nanmax(self.xr))
        self.Xdoc = float(self.model._x_doc_from_csv())
        self.y_bottom = float(np.nanmax(self.yr))
        self.y_top    = float(min(np.nanmin(self.yr), self.HTL)) - 1.0

        self.Xhtl    = float(self._x_at_y_csv(self.HTL))
        self.x0_init = np.clip(self.Xhtl, self.xmin, self.Xdoc - 1e-6)

        # ------- figura / layout -------
        self.fig, self.ax = plt.subplots(figsize=(12, 6.8), constrained_layout=False)
        plt.subplots_adjust(right=0.80, bottom=0.16)

        # água (abaixo da HTL)
        self.ax.fill_between([self.xmin, self.xmax], self.HTL, self.y_bottom,
                             color=WATER, alpha=0.7, zorder=0)
        (self.htl_line,) = self.ax.plot([self.xmin, self.xmax], [self.HTL, self.HTL],
                                        color="blue", lw=1.8, label="High Tide level", zorder=1)

        # areia clara abaixo do observado
        order_csv = np.argsort(self.xr)
        self.ax.fill_between(self.xr[order_csv], self.yr[order_csv], self.y_bottom,
                             color=LIGHTSAND, alpha=0.95, zorder=2)
        (self.obs_line,) = self.ax.plot(self.xr, self.yr, color="black", lw=1.8,
                                        label="Observed (CSV)", zorder=4)

        # Dean curve + polygon (setados depois)
        (self.dean_line,) = self.ax.plot([], [], "r--", lw=1.8,
                                         label="Dean Profile", zorder=5)
        self.sand_dean_poly = None

        # DoC marker/label
        self.doc_pt   = self.ax.scatter([self.Xdoc], [self.doc], s=38, c="k", zorder=6)
        #self.doc_anno = self.ax.annotate(f"DoC={self.doc:.1f} m", xy=(self.Xdoc, self.doc),
        #                                 xytext=(6, -4), textcoords="offset points", color="k", zorder=6)

        self.calib_span = None
        if SHOW_CALIB_SPAN:
            self.calib_span = self.ax.axvspan(self.x0_init, self.Xdoc, color="#C09048",
                                              alpha=0.25, zorder=3)

        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.y_top, self.y_bottom)
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")
        self.ax.grid(True, linestyle=":", lw=0.5)

        # ------- legenda (com zona calibrada e D50) -------
        calib_proxy = Patch(facecolor=DARKSAND, edgecolor="none", alpha=0.95,
                            label="Calibrated zone")
        d50_proxy   = Patch(facecolor="none", edgecolor="none",
                            label=f"D50={self.d50:.2f} mm")

        self._legend_labels = ["High Tide level", "Observed (CSV)",
                               f"DoC={self.doc:.1f} m", f"D50={self.d50:.2f} mm", "Calibrated zone", 
                               "Dean Profile"
                               ]
        handles = [self.htl_line, self.obs_line, self.doc_pt, d50_proxy, calib_proxy, 
                   self.dean_line
                   ]

        self._legend = self.ax.legend(handles=handles, labels=self._legend_labels,
                                      loc="upper left", bbox_to_anchor=(1.02, 1.00),
                                      frameon=False, borderaxespad=0.0)

        dean_idx = self._legend_labels.index("Dean Profile")
        self.dean_label_text = self._legend.get_texts()[dean_idx]

        # ------- slider (com debounce) -------
        ax_sl = self.fig.add_axes([0.15, 0.04, 0.70, 0.035])
        self.slider = Slider(ax=ax_sl, label="X0 Drift",
                             valmin=self.Xhtl, valmax=self.Xdoc - 1e-6,
                             valinit=self.x0_init, valstep=0.5)

        self._pending_x0 = None
        self._debounce_timer = self.fig.canvas.new_timer(interval=40)
        try:
            self._debounce_timer.single_shot = True
        except Exception:
            pass
        self._debounce_timer.add_callback(self._fire_update)
        self.slider.on_changed(self._on_slider)

        self._really_update(self.x0_init)

    # ---------- utilidades ----------
    def _x_at_y_csv(self, yq: float) -> float:
        xr, yr = self.xr, self.yr
        d = yr - yq; s = np.sign(d)
        cross = np.where(s[:-1] * s[1:] <= 0)[0]
        if cross.size:
            i = cross[0]
            x0, x1 = xr[i], xr[i+1]; y0, y1 = yr[i], yr[i+1]
            if y1 == y0: return float(x0)
            return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))
        k = int(np.argmin(np.abs(d)))
        if k == 0:
            x0, x1, y0, y1 = xr[0], xr[1], yr[0], yr[1]
        else:
            x0, x1, y0, y1 = xr[-2], xr[-1], yr[-2], yr[-1]
        if y1 == y0: return float(x0)
        return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))

    def _rmse_on_segment(self, x_seg, y_seg) -> float:
        if len(x_seg) < 3: return np.nan
        m = (self.xr >= x_seg.min()) & (self.xr <= x_seg.max())
        if not np.any(m): return np.nan
        y_csv = np.interp(x_seg, self.xr[m], self.yr[m])
        return float(np.sqrt(np.mean((y_csv - y_seg) ** 2)))

    def _r2_on_segment(self, x_seg, y_seg) -> float:
        if len(x_seg) < 3: return np.nan
        m = (self.xr >= x_seg.min()) & (self.xr <= x_seg.max())
        if not np.any(m): return np.nan
        y_csv = np.interp(x_seg, self.xr[m], self.yr[m])
        ss_res = float(np.sum((y_csv - y_seg) ** 2))
        ss_tot = float(np.sum((y_csv - np.mean(y_csv)) ** 2))
        if ss_tot <= 0: return np.nan
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def _build_fill_verts(x, y, y_bottom):
        return np.vstack([np.column_stack([x, y]),
                          [x[-1], y_bottom],
                          [x[0],  y_bottom]])

    # ---------- debounce ----------
    def _on_slider(self, val):
        self._pending_x0 = float(val)
        try: self._debounce_timer.stop()
        except Exception: pass
        self._debounce_timer.start()

    def _fire_update(self):
        if self._pending_x0 is None:
            try: self._debounce_timer.stop()
            except Exception: pass
            return
        x0 = self._pending_x0
        self._pending_x0 = None
        self._really_update(x0)
        self.fig.canvas.draw_idle()
        try: self._debounce_timer.stop()
        except Exception: pass

    # ---------- atualização ----------
    def _really_update(self, x0_abs):
        try:
            x0 = float(x0_abs)
            x_seg, y_seg, Ahat, X0, Y0, Xdoc = self.model.calibrate_segment_x(x0)
        except Exception:
            self.dean_line.set_data([], [])
            if self.sand_dean_poly is not None:
                tiny = 1e-6
                self.sand_dean_poly.set_xy(np.array([[0,0],[tiny,0],[0,tiny]]))
            return

        # prolonga até HTL (gráfico apenas)
        if Y0 > self.HTL:
            n_up = max(20, int(200 * (Y0 - self.HTL) / max(1e-6, (self.doc - self.HTL))))
            y_up = np.linspace(self.HTL, Y0, n_up, endpoint=False)
            x_up = X0 + dean1991_rev(y_up, Ahat, self.HTL) - dean1991_rev(Y0, Ahat, self.HTL)
            x_plot = np.concatenate([x_up, x_seg])
            y_plot = np.concatenate([y_up, y_seg])
        else:
            x_plot, y_plot = x_seg, y_seg
        self.dean_line.set_data(x_plot, y_plot)

        # fill somente no trecho calibrado X0→Xdoc
        order = np.argsort(x_seg)
        xs, ys = x_seg[order], y_seg[order]
        verts = self._build_fill_verts(xs, ys, self.y_bottom)
        if self.sand_dean_poly is None:
            self.sand_dean_poly = Polygon(verts, closed=True,
                                          facecolor=DARKSAND, edgecolor="none",
                                          alpha=0.95, zorder=3)
            self.ax.add_patch(self.sand_dean_poly)
        else:
            self.sand_dean_poly.set_xy(verts)

        # DoC
        self.doc_pt.set_offsets(np.c_[[Xdoc],[self.doc]])
        #self.doc_anno.set_text(f"DoC={self.doc:.1f} m")
        #self.doc_anno.xy = (Xdoc, self.doc)

        # métricas no item “Dean Profile” da legenda
        rmse = self._rmse_on_segment(xs, ys)
        r2   = self._r2_on_segment(xs, ys)
        lbl = ["Dean Profile"]
        if np.isfinite(Ahat): lbl.append(f"A={Ahat:.4f}")
        if np.isfinite(r2):   lbl.append(f"R²={r2:.3f}")
        if np.isfinite(rmse): lbl.append(f"RMSE={rmse:.3f} m")
        self.dean_label_text.set_text("\n".join(lbl))

if __name__ == "__main__":
    DeanTest()
    plt.show()

