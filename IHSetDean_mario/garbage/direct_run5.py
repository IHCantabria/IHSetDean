# =============================================================
# run/direct_run4.py – 2 options
# =============================================================
"""Generates the Dean (1991) profile in two modes:

**Mode 1 - Calibration (with CSV)**
$ python direct_run4.py --csv profile.csv --sl 0.0 --d50 0.30
* Reads observed profile, calibrates *A*, plots observed x Dean.

**Mode 2 - Synthesis (without CSV)**
$ python direct_run2.py --length 300 --sl 0.0 --d50 0.30 --dx 1

* Generates a Dean profile from 0 to *length* with spacing *dx*,
using the empirical formula:
A = 0.067·(d50[m])^0.44

In both cases plots fills:
• Water (light blue) from SL to bottom.
• Light sand (observed) if present CSV.
• Dark translucent sand (Dean).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dean1991 import dean1991          
from calibration import cal_dean1991, dean_A_from_d50_no_calib    

# --- Fill colors ----------------------------------------------------
SAND = "#f4dcb8"
DARKSAND = "#d6b07a"
WATER = "#a6cee3"

# --------------------------------------------------------------------
# CLI – mutually exclusive CSV or length
# --------------------------------------------------------------------
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--csv",    type=str, help="CSV file: X,Y of the beach profile surveyed")
group.add_argument("--length", type=float, help="Active beach lenght profile (m)")

parser.add_argument("--sl",  type=float, default=0.0, help="Sea level [m]")
parser.add_argument("--d50", type=float, default=None, help="D50 [mm]")
parser.add_argument("--dx",  type=float, default=1.0,  help="Δx lenght resolution for beach profile [m]")
args = parser.parse_args()

# --------------------------------------------------------------------
# Mode 1 – CSV (with calibration)
# --------------------------------------------------------------------
if args.csv:
    csv_path = Path(args.csv)
    model = cal_dean1991(csv_path, sl=args.sl, d50=args.d50)
    model.calibrate().run_model()

    x_obs, z_obs = model.x_obs, model.z_obs
    x_dean, z_dean = model.x_dense, model.z_model

    print("A value calibrated:", model.A)
    print("Metrics:", model.metrics())

# --------------------------------------------------------------------
# Mode 2 – Only Dean profile without calibration
# --------------------------------------------------------------------
else:
    # generate axes x
    x_dean = np.arange(0.0, args.length + args.dx, args.dx)
    # Empirical value for "A"
    A_emp = dean_A_from_d50_no_calib(args.d50)
    z_dean = dean1991(x_dean, A_emp, args.sl)

    # empty arrays for plot conditional logic
    x_obs, z_obs = np.array([]), np.array([])

# --------------------------------------------------------------------
# Set Y limit (bottom) and X domain 
# --------------------------------------------------------------------
x_min, x_max = 0.0, float(x_dean.max() if x_obs.size == 0 else max(x_obs.max(), x_dean.max()))
#y_bottom = max(np.max(z_dean), args.sl) + 0.1 if z_obs.size == 0 else max(z_obs.max(), z_dean.max()) * 1.05
y_bottom = max(np.max(z_dean), args.sl) + 0.1 if z_obs.size == 0 else max(z_obs.max(), z_dean.max())

#print("y_bottom = ", y_bottom)
#print("z_dean max = ", z_dean.max())

# --------------------------------------------------------------------
# PLOT
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

# --- Fill parameters -------------------------------------------------
# Water - sea level (SL)
ax.fill_between([x_min, x_max], args.sl, y_bottom, color=WATER, alpha=0.7, zorder=0)

# Light Sand - observed profile (CSV)
if x_obs.size:
    ax.fill_between(x_obs, z_obs, y_bottom, color=SAND, zorder=2) #, label="Observed area")

# Dark Sand - empirical model (length)
ax.fill_between(x_dean, z_dean, y_bottom, color=DARKSAND, alpha=0.7, zorder=3) #, label="Dean area")

# --- Lines parameters ------------------------------------------------
# Waterline - sea level (SL)
ax.plot([x_min, x_max], [args.sl, args.sl], color="blue", lw=1.5, label="Sea level", zorder=1)

# Observed profile (CSV)
if x_obs.size:
    ax.plot(x_obs, z_obs, color="black", lw=1.6, label="Observed", zorder=4)

# Dean equilibrium profile
ax.plot(x_dean, z_dean, "r--", lw=1.5, label="Dean (1991)", zorder=5)

# --- Graph config ----------------------------------------------------
y_top    = args.sl * 1.5

# ajuste de limites – sem margens extras
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_top, y_bottom)
ax.set_xlabel("Cross-shore distance X [m]")
ax.set_ylabel("Elevation / Depth [m]")
ax.invert_yaxis()
ax.grid(True, linestyle=":", lw=0.5)
ax.legend(loc="upper right")
plt.tight_layout()

# legenda fora
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.tight_layout()
plt.show()

