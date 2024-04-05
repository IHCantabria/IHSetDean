from IHSetDean import IHSetDean
import xarray as xr
import os
import matplotlib.pyplot as plt

config = xr.Dataset(coords={'dy': 0.001,            # Calibrate 
                            'Ymin': -20,            # Calibrate the minimum initial distance value
                            'Ymax': 20,              # Calibrate the maximum initial distance value
                            'Dmin': 0,              # Calibrate the minimum observed distnace value
                            'Dmax': 20,              # Calibrate the maximum observed distance value
                            })

wrkDir = os.getcwd()
config.to_netcdf(wrkDir+'/data/config.nc', engine='netcdf4')
model = IHSetDean.cal_Dean(wrkDir+'/data/')
self = IHSetDean.Dean(model.calibrate())

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

hk = []
hk.append(plt.plot(self.dp, self.zp, '--k')[0])
hk.append(plt.plot(self.dp, self.hm, linewidth=2)[0])
plt.xlabel('Distance [m]', fontdict=font)
plt.ylabel('Water depth [m]', fontdict=font)
plt.show()