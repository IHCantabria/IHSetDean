from pathlib import Path
import numpy as np
from calibration import cal_dean1991

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
    
    model = cal_dean1991(sl=sl, doc=doc, d50=d50)

    if csv:
        # carrega o CSV, recalibra e retorna perfil calibrado
        x_dean, y_dean = model.add_data(csv)
        # observaçõe originais, recompondo coordenada absoluta
        x_obs = model.x_obs + model.x_drift
        y_obs = model.y_obs
    else:
        # sem dados medidos: arrays vazios
        x_obs = np.array([])
        y_obs = np.array([])
        # escolhe perfil teórico por d50 ou por A
        if d50 is not None:
            x_dean, y_dean = model.from_d50(d50)
        elif A is not None:
            x_dean, y_dean = model.from_A(A)
        else:
            raise ValueError("Você deve fornecer d50 ou A")
    return x_obs, y_obs, x_dean, y_dean