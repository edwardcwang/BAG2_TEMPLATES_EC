# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import scipy.signal


class LTICircuitBuilder(object):
    def __init__(self, n):
        # type: (int) -> None
        self._n = n
        self._gmat = np.zeros((n, n))
        self._cmat = np.zeros((n, n))

    def add_res(self, res, node_p, node_n):
        # type: (float, int, int) -> None
        if node_p == node_n:
            return
        if node_p < node_n:
            node_p, node_n = node_n, node_p
        g = 1 / res
        self._gmat[node_p, node_p] += g
        if node_n >= 0:
            self._gmat[node_p, node_n] -= g
            self._gmat[node_n, node_n] += g
            self._gmat[node_n, node_p] -= g

    def add_gm(self, gm, node_p, node_n, node_cp, node_cn=-1):
        # type: (float, int, int, int, int) -> None
        if node_p == node_n or node_cp == node_cn:
            return

        if node_cp >= 0:
            if node_p >= 0:
                self._gmat[node_p, node_cp] += gm
            if node_n >= 0:
                self._gmat[node_n, node_cp] -= gm
        if node_cn >= 0:
            if node_p >= 0:
                self._gmat[node_p, node_cn] -= gm
            if node_n >= 0:
                self._gmat[node_n, node_cn] += gm

    def add_cap(self, cap, node_p, node_n):
        # type: (float, int, int) -> None
        if node_p == node_n:
            return
        if node_p < node_n:
            node_p, node_n = node_n, node_p

        self._cmat[node_p, node_p] += cap
        if node_n >= 0:
            self._cmat[node_p, node_n] -= cap
            self._cmat[node_n, node_n] += cap
            self._cmat[node_n, node_p] -= cap

    def add_transistor(self, tran_info, fg, node_g, node_d, node_s, node_b=-1):
        # type: (Dict[str, float], int, int, int, int, int) -> None
        gm = tran_info['gm'] * fg
        ro = 1 / (tran_info['gds'] * fg)
        gb = tran_info['gb'] * fg
        cgd = tran_info['cgd'] * fg
        cgs = tran_info['cgs'] * fg
        cgb = tran_info['cgb'] * fg
        cds = tran_info['cds'] * fg
        cdb = tran_info['cdb'] * fg
        csb = tran_info['csb'] * fg

        self.add_gm(gm, node_d, node_s, node_g, node_s)
        self.add_res(ro, node_d, node_s)
        self.add_gm(gb, node_d, node_s, node_b, node_s)
        self.add_cap(cgd, node_g, node_d)
        self.add_cap(cgs, node_g, node_s)
        self.add_cap(cgb, node_g, node_b)
        self.add_cap(cds, node_d, node_s)
        self.add_cap(cdb, node_d, node_b)
        self.add_cap(csb, node_s, node_b)

    def get_gain_system(self, node_in, node_out):
        # type: (int, int) -> scipy.signal.StateSpace
        new_gmat = np.delete(self._gmat, node_in, axis=0)
        new_cmat = np.delete(self._cmat, node_in, axis=0)

        print(self._gmat)
        print(self._cmat)

        col_core = [idx for idx in range(self._n) if idx != node_in]
        cmat_core = new_cmat[:, col_core]
        mat_rank = np.linalg.matrix_rank(cmat_core)
        if mat_rank != cmat_core.shape[0]:
            raise ValueError('cap matrix is singular.')
        cvec_in = new_cmat[:, node_in:node_in + 1]

        inv_mat = np.linalg.inv(cmat_core)
        gmat_core = new_gmat[:, col_core]
        gvec_in = new_gmat[:, node_in:node_in + 1]

        if node_out > node_in:
            node_out -= 1

        if np.count_nonzero(cvec_in) > 0:
            # modify state space representation so we don't have input derivative term
            weight_vec = np.dot(inv_mat, cvec_in)
            print(weight_vec)
            gvec_in -= np.dot(gmat_core, weight_vec)
            dmat = np.ones((1, 1)) * -weight_vec[node_out, 0]
        else:
            dmat = np.zeros((1, 1))

        amat = np.dot(inv_mat, -gmat_core)
        bmat = np.dot(inv_mat, -gvec_in)
        cmat = np.zeros((1, self._n - 1))
        cmat[0, node_out] = 1
        print(amat)
        print(bmat)
        print(cmat)
        print(dmat)
        return scipy.signal.lti(amat, bmat, cmat, dmat)


def test():
    cgs1 = 1e-15
    gm1 = 1e-3
    ro1 = 10e3
    cm = 5e-15
    gm2 = 1e-3
    ro2 = 10e3
    cds2 = 1e-15
    cd = 3e-15
    ro3 = 5e3
    rw = 200
    cl = 5e-15

    builder = LTICircuitBuilder(4)
    builder.add_cap(cgs1, 0, 1)
    builder.add_gm(gm1, 1, -1, 0)
    builder.add_res(ro1, 1, -1)
    builder.add_cap(cm, 1, -1)
    builder.add_gm(gm2, 1, 2, 1)
    builder.add_res(ro2, 1, 2)
    builder.add_cap(cds2, 1, 2)
    builder.add_res(ro3, 2, -1)
    builder.add_cap(cd, 2, -1)
    builder.add_res(rw, 2, 3)
    builder.add_cap(cl, 3, -1)
    sys = builder.get_gain_system(0, 3)

    tvec = np.linspace(0, 1e-9, 1001)
    _, yvec = scipy.signal.step(sys, T=tvec)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(tvec, yvec)
    plt.show()

    return sys
