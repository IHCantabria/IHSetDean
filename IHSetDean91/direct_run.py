# =============================================================
# direct_run.py   –   Dean (1991) | EqP  &  Cal  (DoC version)
# =============================================================
"""Gererate Dean equilibrium beach profiles:

1. **EqP** (empirical, NO CSV)  → requires *SL*, *DoC* **and** (*D50* **or** *A*).
2. **Cal** (com CSV)            → requires *CSV*, *SL* **and** *DoC*.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dean1991 import dean1991  # só para plot legend – não recalcula
from calibration import eqp_dean1991, cal_dean1991

# ------------------------------------------------------------------
# 1. PARSER – SL and DoC required | CSV, D50 and A are optional
# ------------------------------------------------------------------
# General parameters - both modes
parser = argparse.ArgumentParser() 
parser.add_argument("--doc", type=float, required=True, help="Depth of Closure DoC [m]")
parser.add_argument("--sl",  type=float, default=0.0, help="Sea Level reference [m]")

# NO CSV empirical parameters - class: eqp_dean1991
parser.add_argument("--d50", type=float, help="Median grain size D50 [mm]")
parser.add_argument("--A",   type=float, help="Dean parameter A [m^(1/3)] – overrides D50")

# CSV calibration parameters - class: cal_dean1991
parser.add_argument("--csv", help="CSV with beach profile (X,Y)")
args = parser.parse_args()

# ---------------------------------------------------------------
# 2. INPUT DATA VALIDATION (SL e DoC)
# ---------------------------------------------------------------
# (silte) 0.06 mm < sand < 4 mm (gravel)
if args.d50 is not None:
    if not (0.06 <= args.d50 <= 4.0):
        parser.error(
            "D50 grain size is outside acceptable range. "
            "Please provide a value between 0.06 mm and 4.0 mm."
        )

# Largest known coastal tidal range: Bay of Fundy, Canada = 16.3 m
if not (-17.0 <= args.sl <= 17.0):                                     
    parser.error(                                                      
        "Water level value outside acceptable limits for the model. "  
        "Please provide data within the range -17 +17 for SL.")        

# Greatest closure depths found in the literature for very energetic regions 18-20 m.
if not (0.5 <= args.doc <= 20.0):                                      
    parser.error(                                                      
        "Beach internal closure depth value outside the tolerated "    
        "limits for the model. Please provide a value within the "     
        "limits between 0.5 and 20m depth.")                           

# ------------------------------------------------------------------
# 3. DEAN MODE CHOOSER
# ------------------------------------------------------------------

csv_data: bool = args.csv is not None

if csv_data:
    # ---------- MODE 2 – CSV PROVIDED | CLASS Cal -----------------
    csv_path = Path(args.csv)
    model = cal_dean1991(csv_path, sl=args.sl, doc=args.doc)
    model.calibrate().run_model()
    print("A (calibrated) = ", model.A, "\nMetrics:", model.metrics())
    
    x_obs, y_obs = model.x_obs, model.y_obs
    x_dean, y_dean = model.x_dense, model.y_model

else:
    # ---------- MODE 1 – NO CSV | CLASS EQP -----------------------
    if args.A is None and args.d50 is None:
        raise SystemExit("For EqP mode, provide --A or --d50.")

    eqp = eqp_dean1991(sl=args.sl, doc=args.doc)

    if args.A is not None:
        x_dean, y_dean = eqp.dean_A(args.A)
        print(f"A (given)   = {args.A:.6f}")
    else:
        x_dean, y_dean = eqp.dean_d50(args.d50)
        print(f"A (empirical from D50={args.d50} mm) = {eqp.A:.6f}")
    
    x_obs, y_obs = np.array([]), np.array([])

# ------------------------------------------------------------------
# 4. DEAN BEACH PROFILE PLOT 
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

# --- Set plot limits ---------------------------------------------- 
x_min = float(x_dean.min() if x_obs.size == 0 else max(x_obs.min(), x_dean.min()))
x_max = float(x_dean.max() if x_obs.size == 0 else max(x_obs.max(), x_dean.max()))
y_bottom = max(np.max(y_dean), args.sl) if y_obs.size == 0 else max(y_obs.max(), y_dean.max())
y_top = args.sl
print("y_bottom = ", y_bottom)
print("y_top = ", y_top)

# --- Fill colors --------------------------------------------------
LIGHTSAND = "#f4dcb8"
DARKSAND = "#d6b07a"
WATER = "#a6cee3"

# --- Fill parameters -------------------------------------------------
# Water - sea level (SL)
ax.fill_between([x_min, x_max], args.sl, y_bottom, color=WATER, alpha=0.7, zorder=0)

# Light Sand - observed profile (CSV)
if x_obs.size:
    ax.fill_between(x_obs, y_obs, y_bottom, color=LIGHTSAND, zorder=2) #, label="Observed area")

# Dark Sand - empirical model (length)
ax.fill_between(x_dean, y_dean, y_bottom, color=DARKSAND, alpha=0.7, zorder=3) #, label="Dean area")

# --- Lines parameters ------------------------------------------------
# Waterline - sea level (SL)
ax.plot([x_min, x_max], [args.sl, args.sl], color="blue", lw=1.5, label="Sea level", zorder=1)

# Observed profile (CSV)
if x_obs.size:
    ax.plot(x_obs, y_obs, color="black", lw=1.6, label="Observed", zorder=4)

# Dean equilibrium profile
ax.plot(x_dean, y_dean, "r--", lw=1.5, label="Dean (1991)", zorder=5)

# --- Plot gen configs ------------------------------------------------
ax.set_xlabel("Cross-shore distance X [m]")
ax.set_ylabel("Elevation / Depth [m]")
ax.invert_yaxis()
ax.grid(True, linestyle=":", lw=0.5)
ax.legend(loc="upper right")

fig.subplots_adjust(right=0.80)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.tight_layout()
plt.show()
