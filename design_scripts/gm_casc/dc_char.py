# -*- coding: utf-8 -*-

from typing import List, Union, Dict, Tuple
import pprint

import numpy as np
import scipy.optimize
import scipy.signal

from bag.tech.mos import MosCharDB
from bag.data.lti import LTICircuit


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


def char_cascode(db_list, w_list, fg_list, vdd, vcm, vload, vtail, vmid, rw, cw, fanout, ton, num_points=2000):
    # type: (List[MosCharDB], List[float], List[int], float, float, float, float, float) -> Dict[str, float]
    in_params = db_list[0].query(w=w_list[0], vbs=-vtail, vds=vmid - vtail, vgs=vcm - vtail)
    casc_params = db_list[1].query(w=w_list[1], vbs=-vmid, vds=vcm - vmid, vgs=vdd - vmid)
    load_params = db_list[2].query(w=w_list[2], vbs=0, vds=vcm - vdd, vgs=vload - vdd)

    builder = LTICircuit(['in', 'mid', 'd', 'out'])
    builder.add_transistor(in_params, 'in', 'mid', 'gnd', fg=fg_list[0])
    builder.add_transistor(casc_params, 'gnd', 'd', 'mid', fg=fg_list[1])
    builder.add_transistor(load_params, 'gnd', 'd', 'gnd', fg=fg_list[2])
    builder.add_cap(cw / 2, 'd', 'gnd')
    builder.add_cap(cw / 2, 'out', 'gnd')
    builder.add_res(rw, 'd', 'out')

    # get input capacitance and add it as load
    fin = 1 / (2 * ton)
    win = 2 * np.pi * fin
    zin = builder.get_impedance('in', fin)
    cin = (1 / zin).imag / win
    builder.add_cap(cin * fanout, 'out', 'gnd')

    # get DC gain/percent settling
    sys = builder.get_voltage_gain_system('in', 'out')
    _, gain_vec = sys.freqresp(w=[0.1])
    gain_sgn = 1 if gain_vec[0].real > 0 else -1
    dc_gain = abs(gain_vec[0])

    tvec = np.linspace(0, ton, num_points, endpoint=True)
    tvec, yvec = scipy.signal.step(sys, T=tvec)  # type: Tuple[np.ndarray, np.ndarray]
    k_settle = 1 - abs(yvec[num_points - 1] - gain_sgn * dc_gain) / dc_gain

    # calculate remaining parameters of interest
    fg_in, fg_casc, fg_load = fg_list
    ibias = in_params['ids'] * fg_in
    gm_in = in_params['gm'] * fg_in
    ro_in = 1 / (fg_in * in_params['gds'])
    gm_casc = casc_params['gm'] * fg_casc
    ro_casc = 1 / (fg_casc * casc_params['gds'])
    ro_load = 1 / (fg_load * load_params['gds'])

    return dict(
        ibias=ibias,
        dc_gain=dc_gain,
        k_settle=k_settle,
        cin=cin,
        gm_in=gm_in,
        ro_in=ro_in,
        gm_casc=gm_casc,
        ro_casc=ro_casc,
        casc_gain=gm_casc * ro_casc,
        ro_load=ro_load,
        vload=vload,
        vtail=vtail,
        vmid=vmid,
    )


def test(vdd=0.8, vcm=0.65, env='tt'):
    root_dir = 'mos_data'
    vstar_targ = 0.3
    cw = 6e-15
    rw = 200
    ton = 50e-12
    fanout = 2
    k_settle_targ = 0.95
    gain_targ = 1.0

    w_list = [4, 4, 4, 4]
    fg_in = 8
    # fg_casc_swp = [4]
    # fg_load_swp = [4]
    fg_casc_swp = list(range(4, 17, 2))
    fg_load_swp = list(range(4, 17, 2))

    ndb = MosCharDB(root_dir, 'nch', ['intent', 'l'], [env], intent='ulvt', l=16e-9, method='spline')
    pdb = MosCharDB(root_dir, 'pch', ['intent', 'l'], [env], intent='ulvt', l=16e-9, method='spline')
    db_gm_list = [ndb, ndb]
    w_gm_list = w_list[1:3]
    w_load = w_list[3]

    db_char_list = [ndb, ndb, pdb]
    w_char_list = w_list[1:]

    ibias_opt = float('inf')
    char_dict_opt = None
    fg_opt = None

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
            char_dict = char_cascode(db_char_list, w_char_list, fg_char_list, vdd, vcm,
                                     vload, vtail, vmid, rw, cw, fanout, ton)
            if char_dict['k_settle'] >= k_settle_targ and char_dict['dc_gain'] >= gain_targ:
                ibias_cur = char_dict['ibias']
                if ibias_cur < ibias_opt:
                    ibias_opt = ibias_cur
                    char_dict_opt = char_dict
                    fg_opt = fg_char_list

    if char_dict_opt is None:
        raise ValueError('No solution found.')
    else:
        pprint.pprint(fg_opt)
        pprint.pprint(char_dict_opt)
