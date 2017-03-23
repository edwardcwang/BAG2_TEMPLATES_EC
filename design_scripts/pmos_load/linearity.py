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


def get_iv_fun(ids_load, w, vgs, ibias, vdd, vout_fullscale, num_points=400, vtol=1e-6):
    # type: (DiffFunction, float, float, float, float, float, int, float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

    def f2(vin2):
        imin = -ids_load(np.array([w, 0, vin2 - vdd, vgs]))
        imax = -ids_load(np.array([w, 0, vin2 - vout_fullscale - vdd, vgs]))
        return ((imax + imin) / 2 - ibias) / 1e-6

    vout_max = scipy.optimize.brentq(f2, vout_fullscale, vdd, xtol=vtol)  # type: float

    # calculate iin_diff to vout_cm/diff transfer function
    vout_vec = np.linspace(vout_max - vout_fullscale, vout_max, num_points, endpoint=True)  # type: np.ndarray
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


def test(vout_fullscale=0.25, verr_max=10e-3, ratio_min_targ=0.75, vdd=1.0):
    lch = 16e-9
    w = 4
    intent_range = ['ulvt', 'lvt', 'svt']
    ibias_range = np.arange(80, 121, 2) * 1e-6
    vcm_range = np.arange(800, 876, 25) * 1e-3
    env_range = ['tt', 'ff', 'ss', 'fs', 'sf', 'ff_hot', 'ss_hot', 'ss_cold']
    num_points = 200
    # root_dir = 'tsmc16_FFC/mos_data'
    root_dir = 'mos_data'

    def fit_fun(xval, scale):
        return scale * xval

    verr_max_mv = verr_max * 1e3
    for intent in intent_range:
        pdb = MosCharDB(root_dir, 'pch', ['intent', 'l'], env_range, intent=intent, l=lch, method='spline')

        for ibias, vcm in product(ibias_range, vcm_range):
            verr_worst = None
            res_list = []
            vg_list = []
            verr_list = []
            imax_list = []
            # find worst case error across process
            for env in env_range:
                ids_load = pdb.get_function('ids', env=env)
                try:
                    vgs = find_vgs(ids_load, w, ibias, vdd, vcm)
                except ValueError:
                    verr_worst = None
                    break

                vod, _, iin = get_iv_fun(ids_load, w, vgs, ibias, vdd, vout_fullscale, num_points=num_points)

                # fit a line that passes through origin to the transfer function
                res_val = scipy.optimize.curve_fit(fit_fun, iin, vod, p0=1)[0]
                verr_cur = 1e3 * np.max(np.abs(vod - res_val[0] * iin))
                res_list.append(res_val[0])
                verr_list.append(verr_cur)
                vg_list.append(vgs + vdd)
                imax_list.append(iin[-1])
                if verr_worst is None:
                    verr_worst = verr_cur
                else:
                    verr_worst = max(verr_worst, verr_cur)

            ibias_ua = ibias * 1e6
            if verr_worst is None:
                print('failed to find vgs for ibias = %.4g uA, vcm = %.4g V' % (ibias_ua, vcm))
                continue

            imax_list = np.array(imax_list)
            ratio_min = np.min(imax_list) / ibias

            if verr_worst <= verr_max_mv and ratio_min >= ratio_min_targ:
                print('intent = %s, ibias = %.4g uA, vcm = %.4g V, '
                      'verr = %.4g mV, ratio_min = %.4g' % (intent, ibias_ua, vcm, verr_worst, ratio_min))
                print('rl: ', res_list)
                print('vg: ', vg_list)
                print('verr: ', verr_list)
                print('imax: ', imax_list)
