import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
import pandas as pd
from IHSetUtils import wMOORE, ADEAN, Hs12Calc, depthOfClosure

class cal_Dean(object):
    """
    cal_Dean
    
    Configuration to calibrate and run the Dean profile.
    
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, path_prof, path_wav, Switch_Calibrate, Switch_Cal_DoC, **kwargs):
        self.path_prof = path_prof
        self.path_wav = path_wav
        prof = pd.read_csv(path_prof)
        self.D50 = kwargs['D50']                  # D50 is not necesarry if Switch_Calibrate = 1
        self.MSL = -kwargs['MSL']       
        self.xm = np.linspace(kwargs['Xm'][0], kwargs['Xm'][1], 1000).reshape(-1, 1)

        self.Switch_Calibrate = Switch_Calibrate
        # Calibrate Dean parameter using profile data [0: no without obs (using D50); 1: no with obs (using D50); 2: yes (using Obs)]
        if Switch_Calibrate == 1 or Switch_Calibrate == 2:
            self.xp = prof.iloc[:, 0]
            self.zp = prof.iloc[:, 1]
            self.zp = abs(self.zp)
            xp_inx = self.xp[(self.zp >= self.MSL)]
            self.xp = self.xp - min(xp_inx)
            
        if Switch_Calibrate == 2:
            self.Zmin = kwargs['Zmin']
            self.Zmax = kwargs['Zmax']
            
        self.Switch_Cal_DoC = Switch_Cal_DoC
        if Switch_Cal_DoC == 1:                   # Calculate Depth of Closure if you have wave data [0: no; 1: yes]
            wav = xr.open_dataset(path_wav)
            Hs = wav['Hs'].values
            Hs = Hs.reshape(-1, 1)
            Tp = wav['Tp'].values
            Tp = Tp.reshape(-1, 1)
            
            H12,T12 = Hs12Calc(Hs,Tp)
            self.DoC = depthOfClosure(H12,T12)
            # self.DoC = self.DoC[0]
                            
    def calibrate(self):
        
        if self.Switch_Calibrate == 2:
            self.xpp = self.xp[(self.zp >= self.MSL)]
            self.zpp = self.zp[(self.zp >= self.MSL)]
            
            if self.Zmin < np.min(self.zpp):
                self.Zmin = np.min(self.zpp)
            if self.Zmax > np.max(self.zpp):
                self.Zmax = np.max(self.zpp)
            zp = np.linspace(self.Zmin, self.Zmax, 100).reshape(-1, 1)
            interp_func = interp1d(self.zpp, self.xpp, kind="linear", fill_value="extrapolate")
            xp = interp_func(zp)
            zp = -zp[1:] + self.MSL
            xp = xp[1:]

            ws = None
            if self.D50 is not None:
                ws = wMOORE(self.D50)

            Y = np.log(-zp)
            Y2 = 2 / 3 * np.log(xp)

            fc = np.arange(-20, 20, 0.0001)
            Y2_grid, fc_grid = np.meshgrid(Y2, fc, indexing="ij")
            Y2t = fc_grid + Y2_grid
                
            def RMSEq(Y, Y2t):
                return np.sqrt(np.mean((Y - Y2t) ** 2, axis=0))

            out = RMSEq(Y, Y2t)
            I = np.argmin(out)

            self.A = np.exp(fc[I])
            self.kk = np.exp(fc[I] - np.log(ws**0.44)) if ws is not None else None
            
            # self.zp = self.zp - self.MSL
        
        if self.Switch_Calibrate == 0 or self.Switch_Calibrate == 1:
            D50 = self.D50 / 1000
            self.A = ADEAN(D50)
            
        return self
        
def Dean(self):
    self.zm = self.A * self.xm ** (2 / 3) + self.MSL
    if self.Switch_Cal_DoC == 1:
        self.xm_DoC = np.mean(self.xm[(self.zm <= self.DoC + 0.05) & (self.zm >= self.DoC - 0.05)])
        self.zm_DoC = np.mean(self.zm[(self.zm <= self.DoC + 0.05) & (self.zm >= self.DoC - 0.05)])
    
    return self
