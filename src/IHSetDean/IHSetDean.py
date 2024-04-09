import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from IHSetUtils import wMOORE

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
                
        self.Dmin = cfg['Dmin'].values
        self.Dmax = cfg['Dmax'].values
        self.D50 = ens['D50'].values
        self.dp = ens['d'].values
        self.zp = ens['z'].values
        
    def calibrate(self):
        self.zp = self.zp - self.zp[0]
        self.d = self.dp - self.dp[0]

        # Profile with equidistant points
        dp = np.linspace(self.Dmin, self.Dmax, 500).reshape(-1, 1)
        interp_func = interp1d(self.d, self.zp, kind="linear", fill_value="extrapolate")
        zp = interp_func(dp)
        zp = zp[1:]
        dp = dp[1:]

        ws = None
        if self.D50 is not None:
            ws = wMOORE(self.D50)

        Y = np.log(-zp)
        Y2 = 2 / 3 * np.log(dp)

        fc = np.arange(-20, 20, 0.0001)
        Y2_grid, fc_grid = np.meshgrid(Y2, fc, indexing="ij")
        Y2t = fc_grid + Y2_grid
                
        def RMSEq(Y, Y2t):
            return np.sqrt(np.mean((Y - Y2t) ** 2, axis=0))

        out = RMSEq(Y, Y2t)
        I = np.argmin(out)

        self.A = np.exp(fc[I])
        self.kk = np.exp(fc[I] - np.log(ws**0.44)) if ws is not None else None
        self.dp_value = dp
        
        return self
        
def Dean(self):
    
    self.hm = -self.A * self.dp ** (2 / 3)
    
    return self
