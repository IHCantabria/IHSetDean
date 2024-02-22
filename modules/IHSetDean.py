import numpy as np
from scipy.interpolate import interp1d      

def caida_grano(D50):
    ws = np.nan
    if D50 < 0.1:
        ws = 1.1e6 * (D50 * 0.001) ** 2
    elif 0.1 <= D50 <= 1:
        ws = 273 * (D50 * 0.001) ** 1.1
    elif D50 > 1:
        ws = 4.36 * D50 ** 0.5
    return ws

def RMSEq(Y, Y2t):
    return np.sqrt(np.mean((Y - Y2t) ** 2, axis=0))

def Dean(dp, zp, D50):
    z = zp - zp[0]
    d = dp - dp[0]
    
    # Profile with equidistant points
    dp = np.linspace(0, dp[-1], 500).reshape(-1, 1)  # 500 points
    interp_func = interp1d(d, z, kind='linear', fill_value='extrapolate')
    zp = interp_func(dp)
    zp = zp[1:]
    dp = dp[1:]
    
    ws = None
    if D50 is not None:
        ws = caida_grano(D50)
    
    Y = np.log(-zp)
    Y2 = 2 / 3 * np.log(dp)
    
    fc = np.arange(-20, 20, 0.001)
    Y2_grid, fc_grid = np.meshgrid(Y2, fc, indexing='ij')
    Y2t = fc_grid + Y2_grid
    
    out = RMSEq(Y, Y2t)
    I = np.argmin(out)

    A = np.exp(fc[I])
    kk = np.exp(fc[I] - np.log(ws**0.44)) if ws is not None else None

    hm = -A * dp ** (2 / 3)
    err = RMSEq(zp, hm)

    Para = {'model': 'Dean'}
    Para['formulation'] = ['h= Ax.^(2/3)', 'A=k ws^0.44']
    Para['name_coeffs'] = ['A', 'k']
    Para['coeffs'] = [A, kk]
    Para['RMSE'] = err

    model = {'D': np.array([0] + list(dp.flatten())),
             'Z': np.array([0] + list(hm.flatten()))}
    
    return Para, model
    