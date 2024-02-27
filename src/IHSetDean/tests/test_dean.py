from scipy import io
import numpy as np
from IHSetDean import IHSetDean
from scipy.interpolate import interp1d


def test_Dean():

    perfil = io.loadmat("./data/perfiles_cierre.mat")
    p = 0
    d = perfil["perfil"]["d"][0][p].flatten()
    d = d - d[0]
    z = perfil["perfil"]["z"][0][p].flatten()
    CM = perfil["perfil"]["CM_95"][0][p].flatten()
    z = z - CM
    di = np.linspace(d[0], d[-1], 100)
    z = interp1d(d, z, kind="linear", fill_value="extrapolate")(di)
    d = di

    D50 = perfil["perfil"]["D50"][0][p].flatten()

    # Assuming the 'ajuste_perfil' function is defined as in the previous code
    pDeank, mDeank = IHSetDean.Dean(d, z, D50)
    assert pDeank["RMSE"][0] == 0.4840710371023019
