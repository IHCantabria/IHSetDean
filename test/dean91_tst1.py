# dean91_tst2.py  — Dean (1991) profile — SLIDER = SHIFT ONLY (no recalibration)
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Backend estável (evita Wayland/Qt travar)
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["MPLBACKEND"] = "TkAgg"
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon

# ------------------------------------------------------------------------------
# User path config (robusto)
# ------------------------------------------------------------------------------
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

# Try plain import first (local), then package-style if available
try:
    from calibration import cal_dean1991
except Exception:
    from src.IHSetDean.calibration import cal_dean1991  # pragma: no cover

WATER     = "#a6cee3"
LIGHTSAND = "#f4dcb8"
DARKSAND  = "#d6b07a"

class DeanTestShiftOnly:
    def __init__(self):
        # ---------------- user params ----------------
        self.HTL = -2.0
        self.doc = 8.0
        self.d50 = 0.30
        self.K   = 0.51

        # CSV path
        self.csv = str(root_path / "XY_PuertoChiquito_clean.csv")

        # ---------------- model ----------------
        self.model = cal_dean1991(HTL=self.HTL, doc=self.doc, d50=self.d50, K=self.K)
        self.x_raw = np.array([])
        self.y_raw = np.array([])

        # Load CSV e calibração inicial (uma única vez)
        try:
            df = pd.read_csv(self.csv, dtype={"X": float, "Y": float})
            self.x_raw = pd.to_numeric(df["X"], errors="coerce").to_numpy()
            self.y_raw = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
            if np.nanmean(self.y_raw) < 0:
                # manter convenção: positivo para baixo (profundidade), negativo para cima
                self.y_raw = -self.y_raw
            self.model.add_data(self.csv)
        except Exception as e:
            print(f"[Warning] Could not load CSV '{self.csv}': {e}")
            self.csv = None

        # Curvas inicial (observado + Dean calibrado entre HTL e DoC)
        self.x_obs, self.y_obs, self.x_dean0, self.y_dean0 = self.model.run_model(csv=self.csv)

        # Guardar A inicial para legend
        self.A0 = getattr(self.model, "A", np.nan)

        # Geometria geral do gráfico
        if self.x_raw.size:
            self.xmin = float(np.nanmin(self.x_raw))
            self.xmax = float(np.nanmax(self.x_raw))
            self.y_bottom = float(np.nanmax(self.y_raw))
            self.y_top = float(min(np.nanmin(self.y_raw), self.HTL)) - 1.0
        else:
            self.xmin = float(np.nanmin(self.x_dean0))
            self.xmax = float(np.nanmax(self.x_dean0))
            self.y_bottom = float(np.nanmax(self.y_dean0))
            self.y_top = float(min(np.nanmin(self.y_dean0), self.HTL)) - 1.0

        # X do cruzamento OBS com HTL — usaremos como âncora do shift
        self.Xhtl = self._x_at_y_csv(self.HTL) if self.x_raw.size else float(self.x_dean0[0])

        # Deslocamento inicial (0.0). O slider controla Δx; NÃO recalibra.
        self.dx_init = 0.0

        # X no DoC (para referência visual/legenda)
        self.Xdoc = float(self.model._x_doc_from_csv()) if self.x_raw.size else float(self.x_dean0[-1])

        # -------- figura/layout --------
        self.fig, self.ax = plt.subplots(figsize=(12, 6.8), constrained_layout=False)
        plt.subplots_adjust(right=0.80, bottom=0.16)

        # (1) água (abaixo da HTL)
        self.ax.fill_between([self.xmin, self.xmax], self.HTL, self.y_bottom,
                             color=WATER, alpha=0.7, zorder=0)

        # (2) HTL
        (self.htl_line,) = self.ax.plot([self.xmin, self.xmax], [self.HTL, self.HTL],
                                        color="blue", lw=1.8, label="High Tide level", zorder=1)

        # (3) areia clara (abaixo do OBS)
        if self.x_raw.size:
            sort_idx = np.argsort(self.x_raw)
            self.ax.fill_between(self.x_raw[sort_idx], self.y_raw[sort_idx], self.y_bottom,
                                 color=LIGHTSAND, alpha=0.95, zorder=2)

        # (4) observado (linha)
        (self.obs_line,) = self.ax.plot(self.x_raw, self.y_raw, color="black", lw=1.8,
                                        label="Observed (CSV)", zorder=4)

        # (5) Dean inicial + fill escuro via Polygon (reaproveitado)
        (self.dean_line,) = self.ax.plot([], [], "r--", lw=1.8, label="Dean Profile", zorder=5)
        self.sand_dean_poly = None  # Polygon do fill escuro

        # (6) DoC marker/label
        self.doc_pt   = self.ax.scatter([self.Xdoc], [self.doc], s=38, c="k", zorder=6)
        #self.doc_anno = self.ax.annotate(f"DoC={self.doc:.1f} m", xy=(self.Xdoc, self.doc),
        #                                 xytext=(6, -4), textcoords="offset points", color="k", zorder=6)

        # Eixos
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.y_top, self.y_bottom)
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")
        self.ax.grid(True, linestyle=":", lw=0.5)

        # -------- Legenda única (atualizo só o item Dean) --------
        handles = [self.htl_line, self.obs_line, self.doc_pt, self.dean_line]
        labels  = ["High Tide level", "Observed (CSV)", f"DoC={self.doc:.1f} m", "Dean Profile"]
        self._legend = self.ax.legend(handles=handles, labels=labels,
                                      loc="upper left", bbox_to_anchor=(1.02, 1.00),
                                      frameon=False, borderaxespad=0.0)
        self.dean_label_text = self._legend.get_texts()[3]

        # -------- Slider (shift Δx) com debounce --------
        ax_sl = self.fig.add_axes([0.15, 0.04, 0.70, 0.035])
        # Limites do shift: deixe algum espaço de segurança
        dx_min = (self.xmin - self.Xhtl) - 5.0
        dx_max = (self.xmax - self.Xhtl) + 5.0
        self.slider = Slider(ax=ax_sl, label="X0 Drift",
                             valmin=dx_min, valmax=dx_max,
                             valinit=self.dx_init, valstep=0.5)

        self._pending_dx = None
        self._debounce_timer = self.fig.canvas.new_timer(interval=40)  # ~25 fps
        try:
            self._debounce_timer.single_shot = True
        except Exception:
            pass
        self._debounce_timer.add_callback(self._fire_update)
        self.slider.on_changed(self._on_slider)

        # Desenho inicial (Δx = 0)
        self._really_update(self.dx_init)

    # ---------------- helpers ----------------
    def _x_at_y_csv(self, yq: float) -> float:
        xr, yr = self.x_raw, self.y_raw
        d = yr - yq
        s = np.sign(d)
        cross = np.where(s[:-1] * s[1:] <= 0)[0]
        if cross.size:
            i = cross[0]
            x0, x1 = xr[i], xr[i + 1]
            y0, y1 = yr[i], yr[i + 1]
            if y1 == y0:
                return float(x0)
            return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))
        k = int(np.argmin(np.abs(d)))
        if k == 0:
            x0, x1, y0, y1 = xr[0], xr[1], yr[0], yr[1]
        else:
            x0, x1, y0, y1 = xr[-2], xr[-1], yr[-2], yr[-1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))

    def _rmse_on_curve(self, x_curve, y_curve) -> float:
        if self.x_raw.size < 3 or x_curve.size < 3:
            return np.nan
        m = (self.x_raw >= x_curve.min()) & (self.x_raw <= x_curve.max())
        if not np.any(m):
            return np.nan
        y_csv = np.interp(x_curve, self.x_raw[m], self.y_raw[m])
        return float(np.sqrt(np.mean((y_csv - y_curve) ** 2)))

    def _r2_on_curve(self, x_curve, y_curve) -> float:
        if self.x_raw.size < 3 or x_curve.size < 3:
            return np.nan
        m = (self.x_raw >= x_curve.min()) & (self.x_raw <= x_curve.max())
        if not np.any(m):
            return np.nan
        y_csv = np.interp(x_curve, self.x_raw[m], self.y_raw[m])
        ss_res = float(np.sum((y_csv - y_curve) ** 2))
        ss_tot = float(np.sum((y_csv - np.mean(y_csv)) ** 2))
        if ss_tot <= 0:
            return np.nan
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def _build_fill_verts(x, y, y_bottom):
        return np.vstack([np.column_stack([x, y]),
                          [x[-1], y_bottom],
                          [x[0],  y_bottom]])

    # ---------------- debounce ----------------
    def _on_slider(self, val):
        self._pending_dx = float(val)
        try:
            self._debounce_timer.stop()
        except Exception:
            pass
        self._debounce_timer.start()

    def _fire_update(self):
        if self._pending_dx is None:
            try:
                self._debounce_timer.stop()
            except Exception:
                pass
            return
        dx = self._pending_dx
        self._pending_dx = None
        self._really_update(dx)
        self.fig.canvas.draw_idle()
        try:
            self._debounce_timer.stop()
        except Exception:
            pass

    # ---------------- update principal (shift-only) ----------------
    def _really_update(self, dx):
        # Translada a curva inicial do Dean (sem recalibração)
        x_shift = self.x_dean0 + dx
        y_shift = self.y_dean0.copy()

        # Atualiza linha do Dean
        self.dean_line.set_data(x_shift, y_shift)

        # Atualiza/Cria fill escuro via Polygon
        verts = self._build_fill_verts(x_shift, y_shift, self.y_bottom)
        if getattr(self, "sand_dean_poly", None) is None:
            self.sand_dean_poly = Polygon(verts, closed=True,
                                          facecolor=DARKSAND, edgecolor="none",
                                          alpha=0.95, zorder=3)
            self.ax.add_patch(self.sand_dean_poly)
        else:
            self.sand_dean_poly.set_xy(verts)

        # DoC (do observado) permanece igual — mas realinhamos o texto se quiser
        self.doc_pt.set_offsets(np.c_[[self.Xdoc], [self.doc]])
        #self.doc_anno.set_text(f"DoC={self.doc:.1f} m")
        #self.doc_anno.xy = (self.Xdoc, self.doc)

        # Métricas vs CSV para a curva deslocada (faz sentido mudar com o shift)
        rmse = self._rmse_on_curve(x_shift, y_shift)
        r2   = self._r2_on_curve(x_shift, y_shift)

        # Atualiza somente o texto do item "Dean Profile" na legenda
        lab = ["Dean Profile"]
        if np.isfinite(self.A0): lab.append(f"A={self.A0:.4f}")
        if np.isfinite(r2):      lab.append(f"R²={r2:.3f}")
        if np.isfinite(rmse):    lab.append(f"RMSE={rmse:.3f} m")
        self.dean_label_text.set_text("\n".join(lab))


if __name__ == "__main__":
    DeanTestShiftOnly()
    plt.show()
