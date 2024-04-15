from IHSetDean import IHSetDean
import os
import matplotlib.pyplot as plt

wrkDir = os.getcwd()
model = IHSetDean.cal_Dean(wrkDir+'/data/prof.csv',wrkDir+'/data/wav.nc', 2, 1, Xm = [0, 500], Zmin = 0.0, Zmax = 1.0, D50 = 0.5, MSL = 0.0)
self = IHSetDean.Dean(model.calibrate())

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

if self.Switch_Cal_DoC == 0:
        plt.plot(self.xm, self.zm, '-', color=[0.8, 0.8, 0], linewidth=2, label='Dean profile')[0]
        plt.fill(self.xm.tolist() + [min(self.xm), min(self.xm)],
                 self.zm.tolist() + [max(self.zm), min(self.zm)], color='yellow', alpha=0.5)
        xLim = self.xm[self.zm <= 5]
        plt.xlim([self.xm[0]-20,xLim[-1]+20])
        plt.ylim([self.MSL-1,5+0.5])
        plt.plot([xLim[-1],xLim[-1]-5,xLim[-1]+5,xLim[-1]],
                 [self.MSL,self.MSL-0.25,self.MSL-0.25,self.MSL], 'b', linewidth=2)
        plt.text(xLim[-1], self.MSL-0.4,'MSL', fontdict=font, horizontalalignment='center', verticalalignment='center')

if self.Switch_Cal_DoC == 1:
        plt.plot(self.xm, self.zm, '-', color=[0.8, 0.8, 0], linewidth=2, label='Dean profile')[0]
        xm_DoC_fill = self.xm[self.zm <= self.DoC]
        zm_DoC_fill = self.zm[self.zm <= self.DoC]
        plt.fill(xm_DoC_fill.tolist() + [min(xm_DoC_fill), min(xm_DoC_fill)],
                 zm_DoC_fill.tolist() + [max(zm_DoC_fill), min(zm_DoC_fill)], color='yellow', alpha=0.5)
        plt.plot(self.xm_DoC, self.zm_DoC, 'ro', markersize=8)
        plt.text(self.xm_DoC, self.zm_DoC-0.2,'h*', fontdict=font, horizontalalignment='center', verticalalignment='center')
        plt.plot([xm_DoC_fill[-1],xm_DoC_fill[-1]-5,xm_DoC_fill[-1]+5,xm_DoC_fill[-1]],
                 [self.MSL,self.MSL-0.25,self.MSL-0.25,self.MSL], 'b', linewidth=2)
        plt.text(xm_DoC_fill[-1], self.MSL-0.4,'MSL', fontdict=font, horizontalalignment='center', verticalalignment='center')
        plt.xlim([self.xm[0]-20,xm_DoC_fill[-1]+20])
        plt.ylim([self.MSL-1,self.DoC+0.5])
        
if self.Switch_Calibrate == 2:
        plt.fill([min(self.xm), min(self.xm), max(self.xm), max(self.xm)],
                 [self.Zmin, self.Zmax, self.Zmax, self.Zmin], color=[0.5, 0.5, 0.5], alpha=0.25)
if self.Switch_Calibrate == 1 or self.Switch_Calibrate == 2:
        plt.plot(self.xp, self.zp, '--k', linewidth=2, label='Observed profile')[0]
plt.plot([min(self.xm), max(self.xm)], [self.MSL, self.MSL], '--b', linewidth=2)[0]

plt.xlabel('Offshore distance [m]', fontdict=font)
plt.ylabel('Water depth [m]', fontdict=font)

plt.grid(True)
plt.gca().invert_yaxis()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
plt.show()
