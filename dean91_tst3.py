# dean91_tst3.py
# Teste interativo do perfil de Dean (segmento X0 -> DoC, ancorado em Y0 do CSV)
# Requer: calibration.py com o método calibrate_segment_x(x0_abs)

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import os
import matplotlib
# Se tentou Qt antes, limpe variável que força Wayland/XCB:
os.environ.pop("QT_QPA_PLATFORM", None)
# Se quiser forçar pela env var também (redundante mas ok):
os.environ["MPLBACKEND"] = "TkAgg"
matplotlib.use("TkAgg")          # <<< é 'matplotlib.use', não 'plt.use'
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- importa a classe de calibração (modo legacy -> pacote) ---
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
try:
    from calibration import cal_dean1991
except Exception:
    # fallback se estiver usando como pacote (IHSetDeans/.../calibration.py)
    from IHSetDean91.calibration import cal_dean1991  # pragma: no cover

class DeanTest:
    def __init__(self):
        # ---------------- user params ----------------
        self.HTL = -2.0
        self.doc = 8.0
        self.d50 = 0.30
        self.K = 0.51

        # CSV (no mesmo diretório do script)
        self.csv = str(root_path / "XY_PuertoChiquito_clean.csv")

        # ---------------- model ----------------
        self.model = cal_dean1991(HTL=self.HTL, doc=self.doc, d50=self.d50, K=self.K)
        # carrega CSV (apenas popula x_raw/y_raw; não vamos usar o retorno)
        self.model.add_data(self.csv)

        # dados brutos (absolutos) para desenhar
        self.xr = np.asarray(self.model.x_raw, dtype=float)
        self.yr = np.asarray(self.model.y_raw, dtype=float)

        # limites úteis
        self.xmin = float(np.nanmin(self.xr))
        self.xmax = float(np.nanmax(self.xr))
        self.Xdoc = float(self.model._x_doc_from_csv())

        # chute inicial do slider: um ponto dentro do trecho até o DoC
        self.x0_init = np.clip(self.xmin + 0.25 * (self.Xdoc - self.xmin), self.xmin, self.Xdoc - 1e-6)

        # ---------------- figure ----------------
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.18)

        # linha do CSV observado
        (self.obs_line,) = self.ax.plot(self.xr, self.yr, "k-", lw=2, label="Observed (CSV)")

        # linha horizontal da HTL (azul)
        (self.htl_line,) = self.ax.plot([self.xmin, self.xmax], [self.HTL, self.HTL], color="#3366ff",
                                        lw=2, label="High Tide level")

        # marcador do DoC (ponto preto)
        self.doc_pt = self.ax.scatter([self.Xdoc], [self.doc], c="k", s=40, zorder=5)
        self.ax.annotate(f"DoC={self.doc:.1f} m", xy=(self.Xdoc, self.doc),
                         xytext=(5, -5), textcoords="offset points", color="k")

        # curva de Dean (segmento) - será setada na 1ª atualização
        (self.dean_line,) = self.ax.plot([], [], "r--", lw=2, label="Dean Profile")

        # faixa de calibração [X0, Xdoc]
        self.calib_span = self.ax.axvspan(self.x0_init, self.Xdoc, color="#C09048", alpha=0.25, zorder=0)

        # estética dos eixos (profundidade positiva para baixo)
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")
        self.ax.set_xlim(self.xmin, self.xmax)
        # tenta um Y-range confortável
        ylo = float(np.nanmin(self.yr))
        yhi = float(np.nanmax(self.yr))
        # garante que HTL e DoC estejam visíveis
        ylo = min(ylo, self.doc + 1.0)
        yhi = max(yhi, self.HTL - 1.0)
        self.ax.set_ylim(yhi, ylo)  # invertido para ficar "positivo para baixo"

        # legenda à direita
        self.leg_text = self.ax.text(
            1.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        # ---------------- slider ----------------
        ax_sl = self.fig.add_axes([0.10, 0.07, 0.80, 0.04])
        self.slider = Slider(
            ax=ax_sl,
            label="drift (recalibration)",
            valmin=self.xmin,
            valmax=self.Xdoc - 1e-6,
            valinit=self.x0_init,
        )
        self.slider.on_changed(self.update)

        # 1ª atualização
        self.update(self.x0_init)

        # legenda “manual”
        self.ax.legend(loc="upper right")

    def update(self, x0_abs):
        """Recalibra e redesenha o segmento de Dean entre X0 e DoC."""
        try:
            x0 = float(x0_abs)
            x_seg, y_seg, Ahat, X0, Y0, Xdoc = self.model.calibrate_segment_x(x0)
        except Exception as e:
            # em caso de erro (slider muito à esquerda etc.), esvazia a curva
            self.dean_line.set_data([], [])
            self.calib_span.remove()
            self.calib_span = self.ax.axvspan(self.slider.val, self.Xdoc, color="#C09048", alpha=0.25, zorder=0)
            self.leg_text.set_text(f"Dean Profile\nA=—\n{e}")
            self.fig.canvas.draw_idle()
            return

        # curva do Dean (só o segmento)
        self.dean_line.set_data(x_seg, y_seg)

        # faixa de calibração atualizada
        self.calib_span.remove()
        self.calib_span = self.ax.axvspan(min(X0, Xdoc), max(X0, Xdoc),
                                          color="#C09048", alpha=0.25, zorder=0)

        # reposiciona/atualiza o marcador de DoC
        self.doc_pt.set_offsets(np.c_[ [Xdoc], [self.doc] ])

        # atualiza caixa de texto (A e métricas básicas do trecho)
        # RMSE em Y sobre o trecho (só para referência visual)
        if len(x_seg) > 2:
            # amostra o CSV no domínio [min(x_seg), max(x_seg)]
            m_dom = (self.xr >= x_seg.min()) & (self.xr <= x_seg.max())
            if np.any(m_dom):
                y_csv = np.interp(x_seg, self.xr[m_dom], self.yr[m_dom])
                rmse = float(np.sqrt(np.mean((y_csv - y_seg)**2)))
            else:
                rmse = np.nan
        else:
            rmse = np.nan

        self.leg_text.set_text(
            f"Dean Profile\nA={Ahat:.4f}\nRMSE={rmse:.3f} m"
        )

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    DeanTest()
    plt.show()

