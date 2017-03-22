# -*- coding: utf-8 -*-


from typing import Tuple

import numpy as np
import scipy.optimize

from bag.tech.mos import MosCharDB


def get_iv_fun(env, pdb, w, ibias, vdd, vcm, vstar_targ, num_points=200, vtol=1e-6):
    # type: (str, MosCharDB, float, float, float, float, float, int, float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    ids_load = pdb.get_scalar_function('ids', env=env)

    # find nominal pmos gate bias
    def f1(vin1):
        return (-ids_load(np.array([w, 0, vcm - vdd, vin1 - vdd])) - ibias) / 1e-6

    vbias = scipy.optimize.brentq(f1, 0, vdd, xtol=vtol)  # type: float

    # find maximum output voltage
    vgs = vbias - vdd

    def f2(vin2):
        imin = -ids_load(np.array([w, 0, vin2 - vdd, vgs]))
        imax = -ids_load(np.array([w, 0, vin2 - vstar_targ - vdd, vgs]))
        return ((imax + imin) / 2 - ibias) / 1e-6

    vout_max = scipy.optimize.brentq(f2, vstar_targ, vdd, xtol=vtol)  # type: float

    # calculate iin_diff to vout_cm/diff transfer function
    vout_vec = np.linspace(vout_max - vstar_targ, vout_max, num_points, endpoint=True)  # type: np.ndarray
    arg = np.zeros((num_points, 4))
    arg[:, 0] = w
    arg[:, 2] = vout_vec - vdd
    arg[:, 3] = vgs
    iin = -ids_load(arg) - ibias  # type: np.ndarray
    iin = iin[::-1]

    # resample so that iin_diff has fixed step size
    iin_resample = np.linspace(iin[0], iin[-1], num_points, endpoint=True)  # type: np.ndarray
    vout_vec = np.interp(iin_resample, iin, vout_vec[::-1])  # type: np.ndarray

    vout_flip = vout_vec[::-1]
    arg[:, 2] = vout_flip - vdd
    vod = vout_flip - vout_vec
    voc = (vout_vec + vout_flip) / 2

    return vod, voc, iin_resample


def test(w=4, ibias=50e-6, vstar_targ=0.3, vdd=0.9, vcm=0.75, env='tt'):
    root_dir = 'tsmc16_FFC/mos_data'

    pdb = MosCharDB(root_dir, 'pch', ['intent', 'l'], [env], intent='ulvt', l=16e-9, method='spline')

    vod, voc, iin = get_iv_fun(env, pdb, w, ibias, vdd, vcm, vstar_targ, num_points=200, vtol=1e-6)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(iin / 1e-6, vod, '-bo')
    plt.figure(2)
    plt.plot(iin / 1e-6, voc, '-ro')
    plt.show()
