# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################


"""This script tests that AnalogBase draws rows of transistors properly."""

from typing import Dict, Any, Set

import yaml

from bag import BagProject
from bag.layout.routing import RoutingGrid, TrackID
from bag.layout.template import TemplateDB

from abs_templates_ec.laygo.core import LaygoBase


class StrongArmLatch(LaygoBase):
    """A single diff amp.

    Parameters
    ----------
    temp_db : TemplateDB
            the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(StrongArmLatch, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            config='laygo configuration dictionary.',
            threshold='transistor threshold flavor.',
            draw_boundaries='True to draw boundaries.',
            num_nblk='number of nmos blocks, single-ended.',
            num_pblk='number of pmos blocks, single-ended.',
            num_nand_blk='number of nand blocks.',
            num_dblk='number of dummy blocks on both sides of latch.',
            show_pins='True to draw pin geometries.',
            w_n='nmos width.',
            w_p='pmos width.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """

        if not self.fg2d_s_short:
            raise ValueError('This template current only works if source wires of fg2d are shorted.')

        threshold = self.params['threshold']
        draw_boundaries = self.params['draw_boundaries']
        num_pblk = self.params['num_pblk']
        num_nblk = self.params['num_nblk']
        num_dblk = self.params['num_dblk']
        num_nand_blk = self.params['num_nand_blk']
        show_pins = self.params['show_pins']
        w_n = self.params['w_n']
        w_p = self.params['w_p']
        w_sub = self.params['config']['w_sub']
        wire_sp = 2
        wire_nand_sp = 2
        nand_sp_blk = 1

        # error checking
        if num_nand_blk % 2 != 0 or num_nand_blk <= 0:
            raise ValueError('num_nand_blk must be even and positive.')

        row_list = ['ptap', 'nch', 'nch', 'nch', 'pch', 'ntap']
        orient_list = ['R0', 'R0', 'MX', 'MX', 'R0', 'MX']
        thres_list = [threshold] * 6
        w_list = [w_sub, w_n, w_n, w_n, w_p, w_sub]
        num_g_tracks = [0, 1, 2, 2, 2, 0]
        num_gb_tracks = [0, 1, 0, 2, 2, 0]
        num_ds_tracks = [2, 0, 0, 0, 0, 2]
        options = {}
        row_kwargs = [{}, options, options, options, options, {}]
        if draw_boundaries:
            end_mode = 15
        else:
            end_mode = 0

        # specify row types
        self.set_row_types(row_list, w_list, orient_list, thres_list, draw_boundaries, end_mode,
                           num_g_tracks, num_gb_tracks, num_ds_tracks, guard_ring_nf=0,
                           row_kwargs=row_kwargs)

        # determine total number of blocks
        tot_pblk = num_pblk + 2
        tot_nblk = num_nblk
        tot_blk_single = max(tot_pblk, tot_nblk)
        tot_latch_blk = 1 + 2 * (tot_blk_single + num_dblk)

        colp = num_dblk + tot_blk_single - tot_pblk
        coln = num_dblk + tot_blk_single - tot_nblk

        laygo_info = self.laygo_info
        # compute NAND gate location
        # step 1: compute ym wire indices
        x_latch_mid = laygo_info.col_to_coord(coln + num_nblk, 'd', unit_mode=True)
        hm_layer = self.conn_layer + 1
        ym_layer = hm_layer + 1
        clk_idx = self.grid.coord_to_nearest_track(ym_layer, x_latch_mid, half_track=True,
                                                   mode=1, unit_mode=True)
        if num_nblk % 2 == 1:
            ds_type = 'd'
            outp_mid_col = coln + (num_nblk - 1) // 2
        else:
            ds_type = 's'
            outp_mid_col = coln + num_nblk // 2
        x_outp_mid = laygo_info.col_to_coord(outp_mid_col, ds_type, unit_mode=True)
        op_idx = self.grid.coord_to_nearest_track(ym_layer, x_outp_mid, half_track=True,
                                                  mode=1, unit_mode=True)
        op_idx = min(op_idx, clk_idx - wire_sp)
        on_idx = clk_idx + (clk_idx - op_idx)
        nand_op_idx = on_idx + 3 * wire_sp + wire_nand_sp
        # based on nand outp track index, compute nand gate column index.
        nand_op_x = self.grid.track_to_coord(ym_layer, nand_op_idx, unit_mode=True)
        nand_col, _ = laygo_info.coord_to_nearest_col(nand_op_x, ds_type='D', mode=1, unit_mode=True)
        num_sp_blk = nand_col - tot_latch_blk - num_nand_blk // 2
        tot_nand_blk = num_sp_blk + 2 * num_nand_blk + nand_sp_blk
        tot_blk = tot_latch_blk + tot_nand_blk

        # add blocks
        pdum_list, ndum_list = [], []
        blk_type = 'fg2d'

        # nwell tap
        cur_col, row_idx = 0, 5
        nw_tap = self.add_laygo_primitive('sub', loc=(cur_col, row_idx), nx=tot_blk, spx=1)

        # pmos inverter row
        cur_col, row_idx = 0, 4
        pdum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=colp, spx=1), 0))
        cur_col += colp
        rst_midp = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx))
        cur_col += 1
        rst_outp = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx))
        cur_col += 1
        invp_outp = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_pblk, spx=1)
        cur_col += num_pblk
        pdum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx)), 0))
        cur_col += 1
        invp_outn = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_pblk, spx=1)
        cur_col += num_pblk
        rst_outn = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx))
        cur_col += 1
        rst_midn = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx))
        cur_col += 1
        pdum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=colp, spx=1), 0))
        cur_col += colp + num_sp_blk
        nandpl, nandpr = {'gb': [], 'gt': [], 'd': [], 's': []}, {'gb': [], 'gt': [], 'd': [], 's': []}
        for nand_dict in [nandpl, nandpr]:
            for idx in range(num_nand_blk):
                inst = self.add_laygo_primitive('fg2s', loc=(cur_col + idx, row_idx), flip=idx % 2 == 1)
                nand_dict['gb'].extend(inst.get_all_port_pins('g0'))
                nand_dict['gt'].extend(inst.get_all_port_pins('g1'))
                nand_dict['d'].extend(inst.get_all_port_pins('d'))
                nand_dict['s'].extend(inst.get_all_port_pins('s'))
            cur_col += num_nand_blk + nand_sp_blk

        # nmos inverter row
        cur_col, row_idx = 0, 3
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=coln - 1, spx=1), 0))
        cur_col += coln - 1
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), split_s=True), 1))
        cur_col += 1
        invn_outp = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_nblk, spx=1)
        cur_col += num_nblk
        tail_sw1 = self.add_laygo_primitive('stack2d', loc=(cur_col, row_idx))
        cur_col += 1
        invn_outn = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_nblk, spx=1)
        cur_col += num_nblk
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), split_s=True), -1))
        cur_col += 1
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=coln - 1, spx=1), 0))
        cur_col += coln - 1 + num_sp_blk
        nandnl, nandnr = {'gb': [], 'gt': [], 'd': [], 's': []}, {'gb': [], 'gt': [], 'd': [], 's': []}
        for nand_dict in [nandnl, nandnr]:
            for idx in range(num_nand_blk):
                flip = idx % 2 == 1
                inst = self.add_laygo_primitive('stack2s', loc=(cur_col + idx, row_idx), flip=flip)
                nand_dict['gb'].extend(inst.get_all_port_pins('g0'))
                nand_dict['gt'].extend(inst.get_all_port_pins('g1'))
                nand_dict['d'].extend(inst.get_all_port_pins('d'))
                nand_dict['s'].extend(inst.get_all_port_pins('s'))
            cur_col += num_nand_blk + nand_sp_blk

        # nmos input row
        cur_col, row_idx = 0, 2
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=coln - 1, spx=1), 0))
        cur_col += coln - 1
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), split_s=True), 1))
        cur_col += 1
        inn = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_nblk, spx=1)
        cur_col += num_nblk
        tail_sw2 = self.add_laygo_primitive('stack2d', loc=(cur_col, row_idx))
        cur_col += 1
        inp = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_nblk, spx=1)
        cur_col += num_nblk
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), split_s=True), -1))
        cur_col += 1
        ndum_list.append(
            (self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=coln - 1 + tot_nand_blk, spx=1), 0))

        # nmos tail row
        cur_col, row_idx = 0, 1
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=coln, spx=1), 0))
        cur_col += coln
        tailn = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_nblk, spx=1)
        cur_col += num_nblk
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx)), 0))
        cur_col += 1
        tailp = self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=num_nblk, spx=1)
        cur_col += num_nblk
        ndum_list.append((self.add_laygo_primitive(blk_type, loc=(cur_col, row_idx), nx=coln + tot_nand_blk, spx=1), 0))

        # pwell tap
        cur_col, row_idx = 0, 0
        pw_tap = self.add_laygo_primitive('sub', loc=(cur_col, row_idx), nx=tot_blk, spx=1)

        # compute overall block size
        self.set_laygo_size(num_col=tot_blk)
        self.fill_space()
        # draw boundaries and get guard ring power rail tracks
        self.draw_boundary_cells()

        # connect ground
        source_vss = pw_tap.get_all_port_pins('VSS') + tailn.get_all_port_pins('s') + tailp.get_all_port_pins('s')
        source_vss.extend(nandnl['s'])
        source_vss.extend(nandnr['s'])
        drain_vss = []
        for inst, mode in ndum_list:
            if mode == 0:
                source_vss.extend(inst.get_all_port_pins('s'))
            drain_vss.extend(inst.get_all_port_pins('d'))
            drain_vss.extend(inst.get_all_port_pins('g'))
        source_vss_tid = self.make_track_id(0, 'ds', 0, width=1)
        drain_vss_tid = self.make_track_id(0, 'ds', 1, width=1)
        source_vss_warrs = self.connect_to_tracks(source_vss, source_vss_tid)
        drain_vss_warrs = self.connect_to_tracks(drain_vss, drain_vss_tid)
        self.add_pin('VSS', source_vss_warrs, show=show_pins)
        self.add_pin('VSS', drain_vss_warrs, show=show_pins)

        # connect tail
        tail = []
        for inst in (tailp, tailn, inp, inn):
            tail.extend(inst.get_all_port_pins('d'))
        tail_tid = self.make_track_id(1, 'gb', 0)
        self.connect_to_tracks(tail, tail_tid)

        # connect tail clk
        clk_list = []
        tclk_tid = self.make_track_id(1, 'g', 0)
        tclk = tailp.get_all_port_pins('g') + tailn.get_all_port_pins('g')
        clk_list.append(self.connect_to_tracks(tclk, tclk_tid))

        # connect inputs
        in_tid = self.make_track_id(2, 'g', 1)
        inp_warr = self.connect_to_tracks(inp.get_all_port_pins('g'), in_tid)
        inn_warr = self.connect_to_tracks(inn.get_all_port_pins('g'), in_tid)
        self.add_pin('inp', inp_warr, show=show_pins)
        self.add_pin('inn', inn_warr, show=show_pins)

        # connect tail switch clk
        tsw = self.make_track_id(2, 'g', 0)
        tsw_list = tail_sw2.get_all_port_pins('g') + tail_sw1.get_all_port_pins('g')
        clk_list.append(self.connect_to_tracks(tsw_list, tsw, min_len_mode=0))

        # get output/mid horizontal track id
        nout_idx = self.get_track_index(3, 'gb', 0)
        mid_idx = self.get_track_index(3, 'gb', 1)
        nout_tid = TrackID(hm_layer, nout_idx)
        mid_tid = TrackID(hm_layer, mid_idx)

        # connect nmos mid
        nmidp = inn.get_all_port_pins('s') + invn_outp.get_all_port_pins('s')
        nmidn = inp.get_all_port_pins('s') + invn_outn.get_all_port_pins('s')
        nmidp = self.connect_wires(nmidp)[0].to_warr_list()
        nmidn = self.connect_wires(nmidn)[0].to_warr_list()
        # exclude last wire to avoid horizontal line-end DRC error.
        nmidp = self.connect_to_tracks(nmidp[:-1], mid_tid)
        nmidn = self.connect_to_tracks(nmidn[1:], mid_tid)

        # connect pmos mid
        mid_tid = self.make_track_id(4, 'gb', 1)
        pmidp = self.connect_to_tracks(rst_midp.get_all_port_pins('d'), mid_tid, min_len_mode=-1)
        pmidn = self.connect_to_tracks(rst_midn.get_all_port_pins('d'), mid_tid, min_len_mode=1)

        # connect nmos output
        noutp = self.connect_to_tracks(invn_outp.get_all_port_pins('d'), nout_tid)
        noutn = self.connect_to_tracks(invn_outn.get_all_port_pins('d'), nout_tid)

        # connect pmos output
        pout_tid = self.make_track_id(4, 'gb', 0)
        poutp = invp_outp.get_all_port_pins('d') + rst_outp.get_all_port_pins('d')
        poutn = invp_outn.get_all_port_pins('d') + rst_outn.get_all_port_pins('d')
        poutp = self.connect_to_tracks(poutp, pout_tid)
        poutn = self.connect_to_tracks(poutn, pout_tid)

        # connect clock in inverter row
        pclk = []
        for inst in (rst_midp, rst_midn, rst_outp, rst_outn):
            pclk.extend(inst.get_all_port_pins('g'))
        pclk_tid = self.make_track_id(4, 'g', 1)
        clk_list.append(self.connect_to_tracks(pclk, pclk_tid))

        # connect inverter gate
        invg_tid = self.make_track_id(3, 'g', 1)
        invgp = invn_outp.get_all_port_pins('g') + invp_outp.get_all_port_pins('g')
        invgp = self.connect_to_tracks(invgp, invg_tid)
        invgn = invn_outn.get_all_port_pins('g') + invp_outn.get_all_port_pins('g')
        invgn = self.connect_to_tracks(invgn, invg_tid)

        # connect vdd
        source_vdd = nw_tap.get_all_port_pins('VDD')
        source_vdd.extend(invp_outp.get_all_port_pins('s'))
        source_vdd.extend(invp_outn.get_all_port_pins('s'))
        source_vdd.extend(rst_midp.get_all_port_pins('s'))
        source_vdd.extend(rst_midn.get_all_port_pins('s'))
        source_vdd.extend(nandpl['s'])
        source_vdd.extend(nandpr['s'])
        drain_vdd = []
        for inst, _ in pdum_list:
            source_vdd.extend(inst.get_all_port_pins('s'))
            drain_vdd.extend(inst.get_all_port_pins('d'))
            drain_vdd.extend(inst.get_all_port_pins('g'))
        source_vdd_tid = self.make_track_id(5, 'ds', 0, width=1)
        drain_vdd_tid = self.make_track_id(5, 'ds', 1, width=1)
        source_vdd_warrs = self.connect_to_tracks(source_vdd, source_vdd_tid)
        drain_vdd_warrs = self.connect_to_tracks(drain_vdd, drain_vdd_tid)
        self.add_pin('VDD', source_vdd_warrs, show=show_pins)
        self.add_pin('VDD', drain_vdd_warrs, show=show_pins)

        # connect nand
        nand_gbl_tid = self.make_track_id(3, 'g', 1)
        nand_gtl_id = self.get_track_index(3, 'g', 0)
        nand_gtr_id = self.get_track_index(4, 'g', 0)
        nand_gbr_tid = self.make_track_id(4, 'g', 1)
        nand_nmos_out_tid = self.make_track_id(3, 'gb', 0)
        nand_outnl = self.connect_to_tracks(nandnl['d'], nand_nmos_out_tid, min_len_mode=0)
        nand_outnr = self.connect_to_tracks(nandnr['d'], nand_nmos_out_tid, min_len_mode=0)

        nand_gtl = nandnl['gt'] + nandpl['gt']
        nand_gtl.extend(nandpr['d'])
        nand_gtr = nandnr['gt'] + nandpr['gt']
        nand_gtr.extend(nandpl['d'])
        nand_outpl, nand_outpr = self.connect_differential_tracks(nand_gtr, nand_gtl, hm_layer,
                                                                  nand_gtr_id, nand_gtl_id)

        nand_gbl = self.connect_to_tracks(nandnl['gb'] + nandpl['gb'], nand_gbl_tid)
        nand_gbr = self.connect_to_tracks(nandnr['gb'] + nandpr['gb'], nand_gbr_tid)

        # connect nand ym wires
        nand_outl = [nand_outnl, nand_outpl]
        nand_outr = [nand_outnr, nand_outpr]
        nand_outl_id = self.grid.coord_to_nearest_track(ym_layer, nand_outnl.middle, half_track=True, mode=1)
        nand_outr_id = self.grid.coord_to_nearest_track(ym_layer, nand_outnr.middle, half_track=True, mode=-1)
        nand_gbr_yt = self.grid.get_wire_bounds(hm_layer, nand_gbr_tid.base_index, unit_mode=True)[1]
        ym_via_ext = self.grid.get_via_extensions(hm_layer, 1, 1, unit_mode=True)[1]
        out_upper = nand_gbr_yt + ym_via_ext
        nand_outl, nand_outr = self.connect_differential_tracks(nand_outl, nand_outr, ym_layer,
                                                                nand_outl_id, nand_outr_id, track_upper=out_upper,
                                                                unit_mode=True)
        self.add_pin('outp', nand_outl, show=show_pins)
        self.add_pin('outn', nand_outr, show=show_pins)

        nand_inn_tid = nand_outl_id - wire_sp
        nand_inp_tid = nand_outr_id + wire_sp
        self.connect_differential_tracks(nand_gbl, nand_gbr, ym_layer, nand_inn_tid, nand_inp_tid)

        # connect ym wires
        clk_tid = TrackID(ym_layer, clk_idx)
        clk_warr = self.connect_to_tracks(clk_list, clk_tid)
        self.add_pin('clk', clk_warr, show=show_pins)

        op_tid = TrackID(ym_layer, op_idx)
        outp1 = self.connect_to_tracks([poutp, noutp], op_tid)
        on_tid = TrackID(ym_layer, on_idx)
        outn1 = self.connect_to_tracks([poutn, noutn], on_tid)
        op_tid = TrackID(ym_layer, on_idx + wire_sp)
        on_tid = TrackID(ym_layer, op_idx - wire_sp)
        outp2 = self.connect_to_tracks(invgn, op_tid)
        outn2 = self.connect_to_tracks(invgp, on_tid)

        mn_tid = TrackID(ym_layer, on_idx + 2 * wire_sp)
        mp_tid = TrackID(ym_layer, op_idx - 2 * wire_sp)
        self.connect_to_tracks([nmidn, pmidn], mn_tid)
        self.connect_to_tracks([nmidp, pmidp], mp_tid)

        xm_layer = ym_layer + 1
        om_idx = self.grid.coord_to_nearest_track(xm_layer, outp1.middle, half_track=True)
        outp, outn = self.connect_differential_tracks([outp1, outp2], [outn1, outn2], xm_layer,
                                                      om_idx + wire_sp / 2, om_idx - wire_sp / 2)
        self.add_pin('midp', outp, show=show_pins)
        self.add_pin('midn', outn, show=show_pins)
        self.connect_differential_tracks(outn, outp, ym_layer, nand_inn_tid, nand_inp_tid)


def make_tdb(prj, target_lib, specs):
    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate(prj, specs):
    lib_name = 'AAAFOO'

    params = specs['params']

    temp_db = make_tdb(prj, lib_name, specs)

    template = temp_db.new_template(params=params, temp_cls=StrongArmLatch, debug=False)
    name = 'STRONGARM_LATCH'
    print('create layout')
    temp_db.batch_layout(prj, [template], [name])
    print('done')


if __name__ == '__main__':

    with open('test_specs/strongarm_latch.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

        generate(bprj, block_specs)
    else:
        print('loading BAG project')
