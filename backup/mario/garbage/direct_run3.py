from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from calibration import cal_dean1991

"""
Interactive visualization of Dean's profile (1991).

• Horizontal slider controls water level (SL).
• Fills and blue line are updated in real time.
"""

# --------------------------------------------------------------
# CLI Arguments
# --------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--csv", help="CSV X,Y do perfil observado")
p.add_argument("--sl",  type=float, default=0.0, help="Sea level inicial [m]")
p.add_argument("--d50", type=float, default=None, help="D50 [mm]")
args = p.parse_args()

# --------------------------------------------------------------
# Fill colors
# --------------------------------------------------------------
SAND     = "#f4dcb8"   # light sand
DARKSAND = "#d6b07a"   # dark sand
WATER    = "#a6cee3"   # light blue

# --------------------------------------------------------------
# Call calibration and run model
# --------------------------------------------------------------
mod = cal_dean1991(Path(args.csv), sl=args.sl, d50=args.d50)
mod.calibrate().run_model()
print("A parameter calibrated:", mod.A)
print("Métricas:", mod.metrics())

x_obs, z_obs  = mod.x_obs,   mod.z_obs
x_den, z_den  = mod.x_dense, mod.z_model

# --------------------------------------------------------------
# Set X/Y limits for filling the graph
# --------------------------------------------------------------
x_min, x_max = 0.0, float(max(x_obs.max(), x_den.max()))
# bottom slightly below the deepest observed/modeled value
y_bottom     = float(max(z_obs.max(), z_den.max())) * 1.05

# -------------------------------------------------------------------
# Function to update the plot — DOES NOT recalibrate the model!
# -------------------------------------------------------------------
def make_plot(sl0: float) -> tuple[plt.Axes, Slider]:
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18)              # Define slider position

    # Define slider axes
    ax_sl = plt.axes([0.12, 0.05, 0.76, 0.05])    # [left, bottom, width, height]
    sldr  = Slider(ax_sl, "Sea level [m]", -2, 2, valinit=sl0, valstep=0.05)

    # Fills
    water_poly = ax.fill_between([x_min, x_max], sl0, y_bottom,
                                 color=WATER, alpha=0.7, zorder=0)
    sand_poly  = ax.fill_between(x_obs, z_obs, y_bottom,
                                 color=SAND, alpha=1.0, zorder=2)
    dark_poly  = ax.fill_between(x_den, z_den, y_bottom,
                                 color=DARKSAND, alpha=0.7, zorder=3)

    # Plot the lines
    sea_lvl, = ax.plot([x_min, x_max], [sl0, sl0],
                        color="blue", lw=1.5, label="Sea level", zorder=2)
    line_obs,  = ax.plot(x_obs, z_obs, color="black", lw=1.6, label="Observed")
    line_dean, = ax.plot(x_den, z_den, "r--", lw=1.5, label="Dean (1991)")

    # visual adjustments
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, -1)        # positive depths downwards
    ax.set_xlabel("Cross-shore distance X [m]")
    ax.set_ylabel("Elevation / Depth [m]")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right")

    # ---------------------------------------------------------------
    # Function called by the slider
    # ---------------------------------------------------------------
    def update(val):
        nonlocal water_poly, sand_poly, dark_poly
        sl = sldr.val

        # remove preenchimentos antigos
        water_poly.remove()
        sand_poly.remove()
        dark_poly.remove()

        # recria preenchimentos e guarda novas referências
        water_poly = ax.fill_between([x_min, x_max], sl, y_bottom,
                                    color=WATER, alpha=0.7, zorder=0)
        sand_poly  = ax.fill_between(x_obs, z_obs, y_bottom,
                                    color=SAND, alpha=1.0, zorder=1)
        dark_poly  = ax.fill_between(x_den, z_den, y_bottom,
                                    color=DARKSAND, alpha=0.7, zorder=2)

        # Plot the lines
        #ax.plot(x_obs, z_obs, color="black", linewidth=1.6, label="Observed",   zorder=4)
        #ax.plot(x_den, z_den, "r--", linewidth=1.5,      label="Dean (1991)", zorder=5)
        # Update sea level line
        sea_lvl.set_ydata([sl, sl])

        fig.canvas.draw_idle()
    
    sldr.on_changed(update)
    return ax, sldr

# --- exec --------------------------------------------------------
make_plot(args.sl)
plt.show()
