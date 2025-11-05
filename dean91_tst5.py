# dean91_tst5.py
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib

# backend estável (evita Wayland/Qt travar)
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["MPLBACKEND"] = "TkAgg"
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- import robusto do calibrador ---
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
try:
    from calibration import cal_dean1991
except Exception:
    from IHSetDean91.calibration import cal_dean1991  # pragma: no cover
    
# --- dean inverse to extend curve up to HTL (robust import) ---
try:
    # se estiver no mesmo diretório
    from dean1991 import dean1991_rev
except Exception:
    # pacote-style
    from IHSetDean91.dean1991 import dean1991_rev  # pragma: no cover

# paleta
WATER     = "#a6cee3"
LIGHTSAND = "#f4dcb8"
DARKSAND  = "#d6b07a"

# mostrar (ou não) a faixa de calibração X0→Xdoc
SHOW_CALIB_SPAN = False


class DeanTest:
    def __init__(self):
        # ---------------- user params ----------------
        self.HTL = -2.0
        self.doc = 8.0
        self.d50 = 0.30
        self.K   = 0.51

        self.csv = str(root_path / "XY_PuertoChiquito_clean.csv")

        # ---------------- model ----------------
        self.model = cal_dean1991(HTL=self.HTL, doc=self.doc, d50=self.d50, K=self.K)
        self.model.add_data(self.csv)  # popula x_raw/y_raw

        # dados observados absolutos
        self.xr = np.asarray(self.model.x_raw, dtype=float)
        self.yr = np.asarray(self.model.y_raw, dtype=float)

        # limites horizontais/verticais
        self.xmin = float(np.nanmin(self.xr))
        self.xmax = float(np.nanmax(self.xr))
        self.Xdoc = float(self.model._x_doc_from_csv())

        # y_bottom = valor mais profundo que aparece; y_top = topo (acima de HTL/observado)
        self.y_bottom = float(np.nanmax(self.yr))
        self.y_top    = float(min(np.nanmin(self.yr), self.HTL)) - 1.0

        # X no cruzamento com a HTL (Y = HTL) será o X0 inicial e limite mínimo do slider
        self.Xhtl   = float(self._x_at_y_csv(self.HTL))
        self.x0_init = np.clip(self.Xhtl, self.xmin, self.Xdoc - 1e-6)

        # ---------------- figure ----------------
        self.fig, self.ax = plt.subplots(figsize=(12, 6.8), constrained_layout=False)
        # espaço para legenda externa (direita) e slider (embaixo)
        plt.subplots_adjust(right=0.80, bottom=0.16)

        # (1) Água — abaixo da HTL, no fundo
        self.water_fill = self.ax.fill_between([self.xmin, self.xmax],
                                               self.HTL, self.y_bottom,
                                               color=WATER, alpha=0.7, zorder=0)

        # (2) Linha da HTL
        (self.htl_line,) = self.ax.plot([self.xmin, self.xmax], [self.HTL, self.HTL],
                                        color="blue", lw=1.8, label="High Tide level", zorder=1)

        # (3) Areia clara — abaixo do OBSERVADO (preenche até o fundo)
        sort_idx = np.argsort(self.xr)
        self.sand_obs = self.ax.fill_between(self.xr[sort_idx], self.yr[sort_idx], self.y_bottom,
                                             color=LIGHTSAND, alpha=0.95, zorder=2)

        # (4) Linha preta do observado
        (self.obs_line,) = self.ax.plot(self.xr, self.yr, color="black", lw=1.8,
                                        label="Observed (CSV)", zorder=4)

        # (5) Areia escura sob o Dean (segmento) — criado na 1ª atualização
        self.sand_dean = None
        (self.dean_line,) = self.ax.plot([], [], "r--", lw=1.8, label="Dean Profile", zorder=5)

        # DoC marker
        self.doc_pt = self.ax.scatter([self.Xdoc], [self.doc], s=38, c="k", zorder=6)
        self.doc_anno = self.ax.annotate(f"DoC={self.doc:.1f} m", xy=(self.Xdoc, self.doc),
                                         xytext=(6, -4), textcoords="offset points", color="k", zorder=6)

        # faixa de calibração (X0→Xdoc) — atrás do Dean
        self.calib_span = None
        if SHOW_CALIB_SPAN:
            self.calib_span = self.ax.axvspan(self.x0_init, self.Xdoc, color="#C09048", alpha=0.25, zorder=3)

        # eixos
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.y_top, self.y_bottom)
        self.ax.invert_yaxis()  # profundidade positiva para baixo
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")
        self.ax.grid(True, linestyle=":", lw=0.5)

        # legenda (fora do gráfico, atualizada no update)
        self._rebuild_legend(Ahat=None, r2=None, rmse=None)

        # slider fora do gráfico (abaixo do eixo X)
        ax_sl = self.fig.add_axes([0.15, 0.04, 0.70, 0.035])
        self.slider = Slider(
            ax=ax_sl,
            label="X0 Drift (with recalibration)",
            valmin=self.Xhtl,            # <<< mínimo em X(HTL)
            valmax=self.Xdoc - 1e-6,
            valinit=self.x0_init,
        )
        self.slider.on_changed(self.update)

        # primeira atualização
        self.update(self.x0_init)

    # ---------------- helpers ----------------
    def _x_at_y_csv(self, yq: float) -> float:
        """Retorna o X do CSV onde y ~= yq (interpola; extrapola nas bordas se preciso)."""
        xr = self.xr
        yr = self.yr
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
        # sem cruzamento: extrapola pela borda mais próxima
        k = int(np.argmin(np.abs(d)))
        if k == 0:
            x0, x1 = xr[0], xr[1]; y0, y1 = yr[0], yr[1]
        else:
            x0, x1 = xr[-2], xr[-1]; y0, y1 = yr[-2], yr[-1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))

    def _rmse_on_segment(self, x_seg: np.ndarray, y_seg: np.ndarray) -> float:
        if x_seg.size < 3:
            return np.nan
        m = (self.xr >= x_seg.min()) & (self.xr <= x_seg.max())
        if not np.any(m):
            return np.nan
        y_csv = np.interp(x_seg, self.xr[m], self.yr[m])
        return float(np.sqrt(np.mean((y_csv - y_seg) ** 2)))

    def _r2_on_segment(self, x_seg: np.ndarray, y_seg: np.ndarray) -> float:
        if x_seg.size < 3:
            return np.nan
        m = (self.xr >= x_seg.min()) & (self.xr <= x_seg.max())
        if not np.any(m):
            return np.nan
        y_csv = np.interp(x_seg, self.xr[m], self.yr[m])
        ss_res = float(np.sum((y_csv - y_seg) ** 2))
        ss_tot = float(np.sum((y_csv - np.mean(y_csv)) ** 2))
        if ss_tot <= 0:
            return np.nan
        return 1.0 - ss_res / ss_tot

    def _rebuild_legend(self, Ahat=None, r2=None, rmse=None):
        # label do Dean com métricas em linhas separadas
        dean_label = "Dean Profile"
        extra = []
        if Ahat is not None and np.isfinite(Ahat): extra.append(f"A={Ahat:.4f}")
        if r2   is not None and np.isfinite(r2):   extra.append(f"R²={r2:.3f}")
        if rmse is not None and np.isfinite(rmse): extra.append(f"RMSE={rmse:.3f} m")
        if extra:
            dean_label += "\n" + "\n".join(extra)

        handles = [self.htl_line, self.obs_line, self.doc_pt, self.dean_line]
        labels  = ["High Tide level", "Observed (CSV)", f"DoC={self.doc:.1f} m", dean_label]

        # legenda fora do gráfico (direita)
        self.ax.legend(handles=handles, labels=labels,
                       loc="upper left", bbox_to_anchor=(1.02, 1.00),
                       frameon=False, borderaxespad=0.0)

    # ---------------- update ----------------
    def update(self, x0_abs):
        # recalibra o segmento (X0→DoC) e redesenha camadas dependentes dele
        try:
            x0 = float(x0_abs)
            x_seg, y_seg, Ahat, X0, Y0, Xdoc = self.model.calibrate_segment_x(x0)
        except Exception:
            # em erro, limpa Dean e areia Dean; mantém legenda sem métricas
            self.dean_line.set_data([], [])
            if self.sand_dean is not None:
                self.sand_dean.remove()
                self.sand_dean = None
            if self.calib_span is not None:
                self.calib_span.remove()
                self.calib_span = None
            if SHOW_CALIB_SPAN:
                self.calib_span = self.ax.axvspan(self.slider.val, self.Xdoc,
                                                  color="#C09048", alpha=0.25, zorder=3)
            self._rebuild_legend(Ahat=None, r2=None, rmse=None)
            self.fig.canvas.draw_idle()
            return

        # curva do Dean (só o segmento)
        self.dean_line.set_data(x_seg, y_seg)

        # areia escura sob o Dean (sempre para BAIXO, até y_bottom)
        # --- prolonga a curva até HTL SEM alterar a calibração (métricas seguem no segmento) ---
        # y_up: de HTL até Y0 (só se Y0 > HTL)
        if Y0 > self.HTL:
            # resolução proporcional ao tamanho do trecho
            n_up = max(20, int(200 * (Y0 - self.HTL) / max(1e-6, (self.doc - self.HTL))))
            y_up = np.linspace(self.HTL, Y0, n_up, endpoint=False)  # endpoint=False evita duplicar Y0
            x_up = X0 + dean1991_rev(y_up, Ahat, self.HTL) - dean1991_rev(Y0, Ahat, self.HTL)
            # concatena: [HTL→Y0) + [Y0→DoC]
            x_plot = np.concatenate([x_up, x_seg])
            y_plot = np.concatenate([y_up, y_seg])
        else:
            x_plot, y_plot = x_seg, y_seg

        # desenha curva completa (HTL→DoC) e o fill sob ela
        self.dean_line.set_data(x_plot, y_plot)

        if self.sand_dean is not None:
            self.sand_dean.remove()
        self.sand_dean = self.ax.fill_between(x_plot, y_plot, self.y_bottom,
                                            color=DARKSAND, alpha=0.95, zorder=3)

        # faixa de calibração atualizada (se habilitada)
        if self.calib_span is not None:
            self.calib_span.remove()
            self.calib_span = None
        if SHOW_CALIB_SPAN:
            self.calib_span = self.ax.axvspan(min(X0, Xdoc), max(X0, Xdoc),
                                              color="#C09048", alpha=0.25, zorder=3)

        # DoC marker/label
        self.doc_pt.set_offsets(np.c_[ [Xdoc], [self.doc] ])
        self.doc_anno.set_text(f"DoC={self.doc:.1f} m")
        self.doc_anno.xy = (Xdoc, self.doc)

        # métricas do segmento e legenda externa
        rmse = self._rmse_on_segment(x_seg, y_seg)
        r2   = self._r2_on_segment(x_seg, y_seg)
        self._rebuild_legend(Ahat=Ahat, r2=r2, rmse=rmse)

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    DeanTest()
    plt.show()
