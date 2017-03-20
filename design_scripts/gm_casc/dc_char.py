# -*- coding: utf-8 -*-

from typing import List, Union

import numpy as np
import scipy.optimize

from bag.tech.mos import MosCharDB


def solve_nmos_dc(db_list, w_list, fg_list, vg_list, vb_list, vbot, vtop, inorm=1e-6, itol=1e-9):
    # type: (List[MosCharDB], List[Union[float, int]], List[int], List[float], List[float], float, float) -> np.ndarray
    ifun_list = [db.get_scalar_function('ids') for db in db_list]
    info_list = list(zip(ifun_list, w_list, fg_list, vg_list, vb_list))
    num_mos = len(w_list)
    vstack = np.zeros(num_mos + 1)
    istack = np.zeros(num_mos)
    vstack[0] = vbot
    vstack[num_mos] = vtop

    def fun(varr):
        for idx in range(1, num_mos):
            vstack[idx] = vstack[idx - 1] + varr[idx - 1]
        for idx, (ifun, w, fg, vg, vb) in enumerate(info_list):
            vs = vstack[idx]
            arg = np.array([w, vb - vs, vstack[idx + 1] - vs, vg - vs])
            istack[idx] = ifun(arg) * fg / inorm
        return istack[1:] - istack[:num_mos - 1]

    xguess = np.linspace(vbot, vtop, num_mos + 1, endpoint=True)  # type: np.ndarray
    x0 = xguess[1:num_mos] - xguess[:num_mos - 1]

    result = scipy.optimize.root(fun, x0, tol=itol / inorm)
    if not result.success:
        raise ValueError('solution failed.')
    return result.x


def solve_casc_gm_dc(db_list, w_list, fg_list, vdd, vcm, vstar_targ, vgs_min=0.1,
                     inorm=1e-6, itol=1e-9, vtol=1e-6):
    # find vbias to achieve target vstar
    vg_gm_list = [vcm, vdd]
    vb_gm_list = [0, 0]
    vstar_in = db_list[0].get_scalar_function('vstar')
    varr = np.zeros(1)
    win = w_list[0]

    def fun1(vin1):
        varr[:] = solve_nmos_dc(db_list, w_list, fg_list, vg_gm_list, vb_gm_list, vin1, vcm,
                                inorm=inorm, itol=itol)
        return vstar_in(np.array([win, 0 - vin1, varr[0], vcm - vin1])) - vstar_targ

    vtail = scipy.optimize.brentq(fun1, 0, vcm - vgs_min, xtol=vtol)  # type: float
    arg_in = np.array([win, 0 - vtail, varr[0], vcm - vtail])
    vtest = vstar_in(arg_in) - vstar_targ
    if abs(vtest) > vtol:
        raise ValueError('vstar is not correct.')
    vmid = vtail + varr[0]
    ids_in = db_list[0].get_scalar_function('ids')
    ibias = ids_in(arg_in) * fg_list[0]

    return ibias, vtail, vmid


def solve_load_bias(pdb, w_load, fg_load, vdd, vcm, ibias, vtol=1e-6):
    # find load bias voltage
    ids_load = pdb.get_scalar_function('ids')

    def fun2(vin2):
        return (-fg_load * ids_load(np.array([w_load, 0, vcm - vdd, vin2 - vdd])) - ibias) / 1e-6

    vload = scipy.optimize.brentq(fun2, 0, vdd, xtol=vtol)  # type: float

    return vload


def test():
    root_dir = 'tsmc16_FFC/mos_data'
    vdd = 0.8
    vcm = 0.65
    vstar_targ = 0.3
    cw = 5e-15
    rw = 200
    fanout = 2

    w_list = [4, 4, 4, 4]
    fg_in = 8
    # fg_casc_swp = [4]
    # fg_load_swp = [4]
    fg_casc_swp = list(range(4, 17, 2))
    fg_load_swp = list(range(4, 17, 2))

    ndb = MosCharDB(root_dir, 'nch', ['intent', 'l'], ['tt'], intent='ulvt', l=16e-9, method='spline')
    pdb = MosCharDB(root_dir, 'pch', ['intent', 'l'], ['tt'], intent='ulvt', l=16e-9, method='spline')
    db_gm_list = [ndb, ndb]
    w_gm_list = w_list[1:3]
    w_load = w_list[3]

    db_char_list = [ndb, ndb, pdb]
    w_char_list = w_list[1:]

    for fg_casc in fg_casc_swp:
        fg_gm_list = [fg_in, fg_casc]
        try:
            ibias, vtail, vmid = solve_casc_gm_dc(db_gm_list, w_gm_list, fg_gm_list, vdd, vcm, vstar_targ)
        except ValueError:
            continue
        for fg_load in fg_load_swp:
            try:
                vload = solve_load_bias(pdb, w_load, fg_load, vdd, vcm, ibias)
            except ValueError:
                continue

            fg_char_list = [fg_in, fg_casc, fg_load]

