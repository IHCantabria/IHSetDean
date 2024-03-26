import numpy as np
from scipy.interpolate import interp1d
import xarray as xr

class cal_Dean(object):
    """
    cal_Dean
    
    Configuration to calibrate and run the Dean profile.
    
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, path):
        self.path = path
        
        cfg = xr.open_dataset(path+'config.nc')
        ens = xr.open_dataset(path+'ens.nc')
                
        self.Ymin = cfg['Ymin'].values
        self.Ymax = cfg['Ymax'].values
        self.dY = cfg['dy'].values
        self.D50 = ens['D50'].values
        self.dp = ens['d'].values
        self.zp = ens['z'].values
        
    def calibrate(self):
        self.zp = self.zp - self.zp[0]
        self.d = self.dp - self.dp[0]

        # Profile with equidistant points
        dp = np.linspace(0, self.dp[-1], 500).reshape(-1, 1)
        interp_func = interp1d(self.d, self.zp, kind="linear", fill_value="extrapolate")
        zp = interp_func(dp)
        zp = zp[1:]
        dp = dp[1:]

        ws = None
        if self.D50 is not None:
            ws = caida_grano(self.D50)

        Y = np.log(-zp)
        Y2 = 2 / 3 * np.log(dp)

        fc = np.arange(self.Ymin, self.Ymax, self.dY)
        Y2_grid, fc_grid = np.meshgrid(Y2, fc, indexing="ij")
        Y2t = fc_grid + Y2_grid

        out = RMSEq(Y, Y2t)
        I = np.argmin(out)

        self.A = np.exp(fc[I])
        self.kk = np.exp(fc[I] - np.log(ws**0.44)) if ws is not None else None
        self.dp_value = dp
        
        return self
    
def caida_grano(D50):
    ws = np.nan
    if D50 < 0.1:
        ws = 1.1e6 * (D50 * 0.001) ** 2
    elif 0.1 <= D50 <= 1:
        ws = 273 * (D50 * 0.001) ** 1.1
    elif D50 > 1:
        ws = 4.36 * D50**0.5
    return ws

def RMSEq(Y, Y2t):
    return np.sqrt(np.mean((Y - Y2t) ** 2, axis=0))
    
def Dean(self):
    
    self.hm = -self.A * self.dp ** (2 / 3)
    
    return self
