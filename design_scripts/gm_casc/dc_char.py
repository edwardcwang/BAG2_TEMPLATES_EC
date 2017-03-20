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


def cascode_char(db_list, w_list, fg_list, vdd, vcm, vstar_targ):
    # find vtail to achieve target vstar
    w_top_list = w_list[1:]
    fg_top_list = fg_list[1:]
    db_top_list = db_list[1:]
    vg_list = [vcm, vdd]
    vb_list = [0, 0]
    vstar_in = db_list[1].get_scalar_function('vstar')
    fg_tail, fg_in, fg_cas = fg_list

    def fun1(vin1):
        vds1 = solve_nmos_dc(db_top_list, w_top_list, fg_top_list, vg_list, vb_list, vin1, vcm)[0]
        return vstar_in(np.array([w_top_list[0], 0 - vin1, vds1, vcm - vin1])) - vstar_targ

    vtail = scipy.optimize.brentq(fun1, 0, vcm - 0.1)  # type: float
    vds = solve_nmos_dc(db_top_list, w_top_list, fg_top_list, vg_list, vb_list, vtail, vcm)[0]
    vmid = vtail + vds
    in_par = db_list[1].query(w=w_list[1], vbs=-vtail, vds=vds, vgs=vcm - vtail)
    cas_par = db_list[2].query(w=w_list[2], vbs=-vmid, vds=vcm - vmid, vgs=vdd - vmid)
    ibias = in_par['ids'] * fg_in

    # find vbias to achieve vtail
    ids_tail = db_list[0].get_scalar_function('ids')

    def fun2(vin2):
        return (fg_tail * ids_tail(np.array([w_list[0], 0, vtail, vin2])) - ibias) / 1e-6

    vbias = scipy.optimize.brentq(fun2, 0, vdd)
    tail_par = db_list[0].query(w=w_list[0], vbs=0, vds=vtail, vgs=vbias)

    # compute small signal parameters
    gm_in = in_par['gm'] * fg_in
    ro_in = 1 / (in_par['gds'] * fg_in)
    cdd_in = in_par['cdd'] * fg_in
    css_in = in_par['css'] * fg_in
    gm_cas = cas_par['gm'] * fg_cas
    ro_cas = 1 / (cas_par['gds'] * fg_cas)
    cdd_cas = cas_par['cdd'] * fg_cas
    css_cas = cas_par['css'] * fg_cas
    ro_tail = 1 / (tail_par['gds'] * fg_tail)
    cdd_tail = tail_par['cdd'] * fg_tail

    return dict(
        ibias=ibias,
        gm=gm_in,
        ro_gm=ro_in + ro_cas + gm_cas * ro_in * ro_cas,
        ro_tail=ro_tail,
        cdd_gm=cdd_cas,
        tau_tail=(cdd_tail + css_in) * ro_tail / (1 + gm_in * ro_tail),
        tau_casc=(cdd_in + css_cas) * ro_in / (1 + gm_cas * ro_in),
        vbias=vbias,
        vmid=vmid,
        vtail=vtail,
    )


def load_char(pdb, pw, pfg, gm_params, vdd, vcm):
    # find vbias to achieve vtail
    ids = pdb.get_scalar_function('ids')
    ibias = gm_params['ibias']

    def fun2(vin2):
        return (-pfg * ids(np.array([pw, 0, vcm - vdd, vin2 - vdd])) - ibias) / 1e-6

    vgp = scipy.optimize.brentq(fun2, 0, vdd)  # type: float
    load_par = pdb.query(w=pw, vbs=0, vds=vcm - vdd, vgs=vgp - vdd)

    amp_params = gm_params.copy()

    amp_params['vload'] = vgp
    amp_params['cdd_load'] = load_par['cdd'] * pfg
    amp_params['cdd_tot'] = amp_params['cdd_load'] + amp_params['cdd_gm']
    amp_params['ro_load'] = 1 / (pfg * load_par['gds'])
    amp_params['ro_tot'] = 1 / (1 / amp_params['ro_load'] + 1 / amp_params['ro_gm'])
    amp_params['gain'] = amp_params['ro_tot'] * amp_params['gm']
    amp_params['tau_out'] = amp_params['ro_tot'] * amp_params['cdd_tot']

    return amp_params


def test():
    root_dir = 'tsmc16_FFC/mos_data'
    vdd = 0.8
    vcm = 0.65
    vstar_targ = 0.3
    w_list = [6, 4, 6]
    fg_list = [4, 4, 4]
    ndb = MosCharDB(root_dir, 'nch', ['intent', 'l'], ['tt'], intent='ulvt', l=16e-9, method='spline')
    db_list = [ndb] * 3

    gm_params = cascode_char(db_list, w_list, fg_list, vdd, vcm, vstar_targ)

    pdb = MosCharDB(root_dir, 'pch', ['intent', 'l'], ['tt'], intent='ulvt', l=16e-9, method='spline')
    pw = 4
    pfg = 4
    amp_params = load_char(pdb, pw, pfg, gm_params, vdd, vcm)
    return amp_params
