from pathlib import Path
import numpy as np
from calibration import eqp_dean1991, cal_dean1991


def run_model(doc, sl, d50, A=None, csv=None):
    # ---------------------------------------------------------------
    # 1. Validaciones de entrada
    # ---------------------------------------------------------------
    
    if d50 is not None:
        if not (0.06 <= d50 <= 4.0):
            raise ValueError("D50 should be between 0.06 mm and 4.0 mm.")

    if not (-17.0 <= sl <= 17.0):
        raise ValueError("SL should be between -17 m and +17 m.")

    if not (0.5 <= doc <= 20.0):
        raise ValueError("DoC should be between 0.5 m and 20 m.")

    # ---------------------------------------------------------------
    # 2. Ejecutar modelo según modo
    # ---------------------------------------------------------------
    
    if csv:
        csv_path = Path(csv)
        model = cal_dean1991(csv_path, sl=sl, doc=doc)
        model.calibrate().run_model()
        print("A (calibrated) = ", model.A, "\nMetrics:", model.metrics())
        x_obs, y_obs = model.x_obs, model.y_obs
        x_dean, y_dean = model.x_dense, model.y_model

        return x_obs, y_obs, x_dean, y_dean

    else:
        if A is None and d50 is None:
            raise ValueError("For non-CSV mode, you must specify --A or --d50.")
        eqp = eqp_dean1991(sl=sl, doc=doc)
        if A is not None:
            x_dean, y_dean = eqp.dean_A(A)
            print(f"A = {A:.6f}")
            x_obs, y_obs = np.array([]), np.array([])  
            
        else:
            x_dean, y_dean = eqp.dean_d50(d50)
            print(f"A (D50={d50} mm) = {eqp.A:.6f}")
            x_obs, y_obs = np.array([]), np.array([])

        return x_obs, y_obs, x_dean, y_dean
