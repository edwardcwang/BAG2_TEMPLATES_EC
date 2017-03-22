# -*- coding: utf-8 -*-


from typing import Tuple
from itertools import product

import numpy as np
import scipy.optimize

from bag.tech.mos import MosCharDB
from bag.math.dfun import DiffFunction


def find_vgs(ids_load, w, ibias, vdd, vcm, vtol=1e-6):
    # type: (DiffFunction, float, float, float, float, float) -> float

    # find nominal pmos gate bias
    def f1(vin1):
        return (-ids_load(np.array([w, 0, vcm - vdd, vin1 - vdd])) - ibias) / 1e-6

    vbias = scipy.optimize.brentq(f1, 0, vdd, xtol=vtol)  # type: float

    # find maximum output voltage
    return vbias - vdd


def get_iv_fun(ids_load, w, vgs, ibias, vdd, vstar_targ, num_points=400, vtol=1e-6):
    # type: (DiffFunction, float, float, float, float, float, int, float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

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


def test(vstar_targ=0.3, vdd=0.9, env='tt'):
    lch = 16e-9
    w = 4
    intent_range = ['ulvt', 'svt']
    ibias_range = np.arange(20, 61, 5) * 1e-6
    vcm_range = [0.65, 0.7, 0.75]
    num_points = 200
    root_dir = 'tsmc16_FFC/mos_data'

    videal = np.linspace(-vstar_targ, vstar_targ, num_points, endpoint=True)  # type: np.ndarray

    for intent in intent_range:
        pdb = MosCharDB(root_dir, 'pch', ['intent', 'l'], [env], intent=intent, l=lch, method='spline')
        ids_load = pdb.get_scalar_function('ids', env=env)

        for ibias, vcm in product(ibias_range, vcm_range):
            try:
                vgs = find_vgs(ids_load, w, ibias, vdd, vcm)
            except ValueError:
                print('failed to find vgs for ibias = %.4g uA' % (ibias * 1e6))
                continue
            vg = vdd + vgs
            vod, voc, iin = get_iv_fun(ids_load, w, vgs, ibias, vdd, vstar_targ, num_points=num_points)
            verr = np.max(np.abs(vod - videal))
            print('ibias = %.4g uA, vcm = %.4g, vg = %.4g, verr = %.4g mV' % (ibias * 1e6, vcm, vg, verr * 1e3))
