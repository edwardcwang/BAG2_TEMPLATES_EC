# -*- coding: utf-8 -*-

from typing import List, Union, Dict, Tuple
import pprint

import numpy as np
import scipy.optimize
import scipy.signal

from bag.tech.mos import MosCharDB
from bag.data.lti import LTICircuit


def solve_nmos_dc(env,  # type: str
                  db_list,  # type: List[MosCharDB]
                  w_list,  # type: List[Union[float, int]]
                  fg_list,  # type: List[int]
                  vg_list,  # type: List[float]
                  vb_list,  # type: List[float]
                  vbot,  # type: float
                  vtop,  # type: float
                  inorm=1e-6,  # type: float
                  itol=1e-9  # type: float
                  ):
    # type: (...) -> np.ndarray
    ifun_list = [db.get_scalar_function('ids', env=env) for db in db_list]
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


def solve_casc_diff_dc(env_list,  # type: List[str]
                       db_list,  # type: List[MosCharDB]
                       w_list,  # type: List[Union[float, int]]
                       fg_list,  # type: List[int]
                       vbias_list,  # type: List[float]
                       vload_list,  # type: List[float]
                       vtail_list,  # type: List[float]
                       vmid_list,  # type: List[float]
                       vdd,  # type: float
                       vcm,  # type: float
                       vin_max,  # type: float
                       num_points=20,
                       inorm=1e-6,  # type: float
                       itol=1e-9  # type: float
                       ):
    # type: (...) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]
    vin_vec = np.linspace(0, vin_max, num_points, endpoint=True)  # type: np.ndarray
    fg_tail, fg_in, fg_casc, fg_load = fg_list
    w_tail, w_in, w_casc, w_load = w_list

    vout_info_list = []
    for env, vbias, vload, vt, vm in zip(env_list, vbias_list, vload_list, vtail_list, vmid_list):
        ifun_tail = db_list[0].get_scalar_function('ids', env=env)
        ifun_in = db_list[1].get_scalar_function('ids', env=env)
        ifun_casc = db_list[2].get_scalar_function('ids', env=env)
        ifun_load = db_list[3].get_scalar_function('ids', env=env)
        xguess = np.array([vt, vm - vt, vm - vt, vcm - vm, vcm - vm])

        vop_list, von_list = [], []
        for vin_diff in vin_vec:
            vinp = vcm + vin_diff / 2
            vinn = vcm - vin_diff / 2

            def fun(varr):
                vtail, vd1, vd2, vd3, vd4 = varr
                vmidp = vtail + vd1
                vmidn = vtail + vd2
                voutp = vmidp + vd3
                voutn = vmidn + vd4

                itail = 2 * fg_tail * ifun_tail(np.array([w_tail, 0, vtail, vbias])) / inorm
                iinp = fg_in * ifun_in(np.array([w_in, -vtail, vmidn - vtail, vinp])) / inorm
                iinn = fg_in * ifun_in(np.array([w_in, -vtail, vmidp - vtail, vinn])) / inorm
                icascp = fg_casc * ifun_casc(np.array([w_casc, -vmidn, voutn - vmidn, vdd - vmidn])) / inorm
                icascn = fg_casc * ifun_casc(np.array([w_casc, -vmidp, voutp - vmidp, vdd - vmidp])) / inorm
                iloadp = fg_load * -ifun_load(np.array([w_load, 0, voutn - vdd, vload - vdd])) / inorm
                iloadn = fg_load * -ifun_load(np.array([w_load, 0, voutp - vdd, vload - vdd])) / inorm

                return np.array([iinp + iinn - itail, icascn - iinn, icascp - iinp, iloadn - icascn, iloadp - icascp])

            result = scipy.optimize.root(fun, xguess, tol=itol / inorm)

            if not result.success:
                raise ValueError('solution failed.')

            vts, vd1s, vd2s, vd3s, vd4s = result.x
            vmps = vts + vd1s
            vmns = vts + vd2s
            vop_list.append(vmps + vd3s)
            von_list.append(vmns + vd4s)

        vout_info_list.append((np.array(vop_list), np.array(von_list)))

    return vin_vec, vout_info_list


def solve_casc_gm_dc(env_list,  # type: List[str]
                     db_list,  # type: List[MosCharDB]
                     w_list,  # type: List[Union[float, int]]
                     fg_list,  # type: List[int]
                     vdd,  # type: float
                     vcm,  # type: float
                     vstar_targ,  # type: float
                     vgs_min=0.1,  # type: float
                     inorm=1e-6,  # type: float
                     itol=1e-9,  # type: float
                     vtol=1e-6  # type: float
                     ):
    # type: (...) -> Tuple[List[Dict[str, float]], List[float], List[float]]
    vtail_list = []
    vmid_list = []
    in_params_list = []

    for env in env_list:
        # find vbias to achieve target vstar
        vg_gm_list = [vcm, vdd]
        vb_gm_list = [0, 0]
        vstar_in = db_list[0].get_scalar_function('vstar', env=env)
        varr = np.zeros(1)
        win = w_list[0]

        def fun1(vin1):
            varr[:] = solve_nmos_dc(env, db_list, w_list, fg_list, vg_gm_list, vb_gm_list, vin1, vcm,
                                    inorm=inorm, itol=itol)
            return vstar_in(np.array([win, 0 - vin1, varr[0], vcm - vin1])) - vstar_targ

        vtail = scipy.optimize.brentq(fun1, 0, vcm - vgs_min, xtol=vtol)  # type: float
        arg_in = np.array([win, 0 - vtail, varr[0], vcm - vtail])
        vtest = vstar_in(arg_in) - vstar_targ
        if abs(vtest) > vtol:
            raise ValueError('vstar is not correct.')

        vmid = vtail + varr[0]
        in_params = db_list[0].query(w=win, vbs=-vtail, vds=vmid - vtail, vgs=vcm - vtail)
        vtail_list.append(vtail)
        vmid_list.append(vmid)
        in_params_list.append(in_params)

    return in_params_list, vtail_list, vmid_list


def solve_load_bias(env_list, pdb, w_load, fg_load, vdd, vcm, ibias_list, vtol=1e-6):
    # type: (List[str], MosCharDB, float, int, float, float, List[float], float) -> List[float]
    # find load bias voltage

    vload_list = []
    for env, ibias in zip(env_list, ibias_list):
        ids_load = pdb.get_scalar_function('ids', env=env)

        def fun2(vin2):
            return (-fg_load * ids_load(np.array([w_load, 0, vcm - vdd, vin2 - vdd])) - ibias) / 1e-6

        vload_list.append(scipy.optimize.brentq(fun2, 0, vdd, xtol=vtol))

    return vload_list


def solve_tail_bias(env, ndb, w_tail, fg_tail, vdd, vtail, ibias, vtol=1e-6):
    # type: (str, MosCharDB, float, int, float, float, float, float) -> float
    # find load bias voltage
    ids_tail = ndb.get_scalar_function('ids', env=env)

    def fun2(vin2):
        return (fg_tail * ids_tail(np.array([w_tail, 0, vtail, vin2])) - ibias) / 1e-6

    vbias = scipy.optimize.brentq(fun2, 0, vdd, xtol=vtol)  # type: float

    return vbias


def design_tail(env_list, ndb, fg_in, in_params_list, vtail_list, fg_swp, w_tail, vdd, tau_max):
    fg_opt = None
    ro_opt = 0
    vbias_list_opt = None
    ro_list_opt = None
    for fg_tail in fg_swp:
        tau_worst = 0
        ro_worst = float('inf')
        ro_list = []
        vbias_list = []
        for env, vtail, in_params in zip(env_list, vtail_list, in_params_list):
            ibias = in_params['ids'] * fg_in
            try:
                vbias = solve_tail_bias(env, ndb, w_tail, fg_tail, vdd, vtail, ibias)
            except ValueError:
                tau_worst = None
                break

            vbias_list.append(vbias)
            tail_params = ndb.query(w=w_tail, vbs=0, vds=vtail, vgs=vbias)
            ro_tail = 1 / (fg_tail * tail_params['gds'])
            gm_in = fg_in * in_params['gm']
            cdd_tail = fg_tail * tail_params['cdd']
            css_gm = fg_in * in_params['css']
            tau = (css_gm + cdd_tail) / (1 / ro_tail + gm_in)
            ro_list.append(ro_tail)
            if tau > tau_worst:
                tau_worst = tau
            if ro_tail < ro_worst:
                ro_worst = ro_tail

        if tau_worst is not None:
            if tau_worst <= tau_max and ro_worst > ro_opt:
                ro_opt = ro_worst
                fg_opt = fg_tail
                vbias_list_opt = vbias_list
                ro_list_opt = ro_list

    if fg_opt is None:
        raise ValueError('No solution for tail current source.')

    return fg_opt, ro_list_opt, vbias_list_opt


def calc_linearity(env_list, fg_list, w_list, db_list, vbias_list, vload_list,
                   vtail_list, vmid_list, vcm, vdd, vin_max):
    vin_vec, vout_info_listsolve_casc_diff_dc(env_list, db_list, w_list, fg_list, vbias_list, vload_list,
                                              vtail_list, vmid_list, vdd, vcm, vin_max, num_points=20)


def test(vstar_targ=0.3, vdd=1.0, vcm=0.7):
    root_dir = 'tsmc16_FFC/mos_data'
    env_range = ['tt', 'ff', 'ss', 'fs', 'sf', 'ff_hot', 'ss_hot', 'ss_cold']
    cw = 6e-15
    rw = 200
    ton = 50e-12
    fanout = 2
    ibias_load_unit = 55e-6
    k_settle_targ = 0.95
    gain_range = [1.0, 6]
    tau_tail_max = ton / 20

    w_list = [4, 4, 4, 6]
    fg_in = 4
    # fg_casc_swp = [4]
    # fg_load_swp = [4]
    fg_casc_range = list(range(6, 7, 2))
    fg_tail_swp = list(range(8, 9, 2))

    ndb = MosCharDB(root_dir, 'nch', ['intent', 'l'], env_range, intent='ulvt', l=16e-9, method='spline')
    pdb = MosCharDB(root_dir, 'nch', ['intent', 'l'], env_range, intent='svt', l=16e-9, method='spline')

    db_list = [ndb, ndb, ndb, pdb]
    db_gm_list = [ndb, ndb]
    w_gm_list = w_list[1:3]
    w_load = w_list[3]
    w_tail = w_list[0]

    for fg_casc in fg_casc_range:
        fg_gm_list = [fg_in, fg_casc]
        try:
            in_params_list, vtail_list, vmid_list = solve_casc_gm_dc(env_range, db_gm_list, w_gm_list, fg_gm_list,
                                                                     vdd, vcm, vstar_targ)
            fg_tail, rtail_list_opt, vbias_list = design_tail(env_range, ndb, fg_in, in_params_list, vtail_list,
                                                              fg_tail_swp, w_tail, vdd, tau_tail_max)

        except ValueError:
            print('failed to solve with fg_casc = %d' % fg_casc)
            continue

        ibias_list = [fg_in * in_params['ids'] for in_params in in_params_list]
        ibias_max = max(ibias_list)
        fg_load = int(np.ceil(ibias_max / ibias_load_unit))

        vload_list = solve_load_bias(env_range, pdb, w_load, fg_load, vdd, vcm, ibias_list)

        fg_list = [fg_tail, fg_in, fg_casc, fg_load]
        calc_linearity(env_range, fg_list, w_list, db_list, vbias_list, vload_list, vtail_list,
                       vmid_list, vcm, vdd, vstar_targ)
