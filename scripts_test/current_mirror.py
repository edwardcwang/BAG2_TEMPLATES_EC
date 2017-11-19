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


from typing import Dict, Any, Set, Union, List

import yaml

from bag import BagProject
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB

from abs_templates_ec.analog_core import AnalogBase


class PMOSCurrentMirror(AnalogBase):
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
        super(PMOSCurrentMirror, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w='transistor width.',
            th='transistor threshold.',
            fg='reference number of fingers.',
            fg_out_list='list of output number of fingers.',
            fg_cap='decap fingers on both sides.',
            hm_width='width of horizontal track wires.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            show_pins='True to draw pin geometry.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: PMOSCurrentMirror
                            lch,  # type: float
                            ptap_w,  # type: Union[float, int]
                            ntap_w,  # type: Union[float, int]
                            w,  # type: Union[float, int]
                            th,  # type: str
                            fg,  # type: int
                            fg_out_list,  # type: List[int]
                            fg_cap,  # type: int
                            hm_width,  # type: int
                            guard_ring_nf,  # type: int
                            show_pins,  # type: bool
                            **kwargs
                            ):
        # type: (...) -> None

        # calculate total number of fingers.
        fg_tot = sum(fg_out_list) + fg + fg_cap * (len(fg_out_list) + 2)

        gds_space = 1

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params = dict(
            lch=lch,
            fg_tot=fg_tot,
            ptap_w=ptap_w,
            ntap_w=ntap_w,
            nw_list=[],
            nth_list=[],
            pw_list=[w, w],
            pth_list=[th, th],
            ng_tracks=[],
            pg_tracks=[1, hm_width],
            nds_tracks=[],
            pds_tracks=[1, gds_space + hm_width],
            guard_ring_nf=guard_ring_nf,
            p_orientations=['MX', 'R0'],
        )
        self.draw_base(**draw_params)

        out_idx = self.make_track_id('pch', 1, 'ds', gds_space + (hm_width - 1) / 2, width=hm_width)
        ref_idx = self.make_track_id('pch', 1, 'g', (hm_width - 1) / 2, width=hm_width)

        self.draw_mos_decap('pch', 1, 0, fg_cap, 2)
        ref_ports = self.draw_mos_conn('pch', 1, fg_cap, fg, 2, 0, diode_conn=True, gate_ext_mode=3)
        self.connect_to_substrate('ntap', ref_ports['s'])
        gate_list = [ref_ports['g']]

        cursor = fg + fg_cap
        for idx, fg_cur in enumerate(fg_out_list):
            self.draw_mos_decap('pch', 1, cursor, fg_cap, 3)
            cursor += fg_cap
            cur_ports = self.draw_mos_conn('pch', 1, cursor, fg_cur, 2, 2, gate_ext_mode=3, gate_pref_loc='d')
            cursor += fg_cur

            self.connect_to_substrate('ntap', cur_ports['s'])
            iout = self.connect_to_tracks(cur_ports['d'], out_idx)
            self.add_pin('iout<%d>' % idx, iout, show=show_pins)
            gate_list.append(cur_ports['g'])

        self.draw_mos_decap('pch', 1, cursor, fg_cap, 1)

        bot_decap_ports = self.draw_mos_decap('pch', 0, 0, fg_tot, 0, export_gate=True, sdir=2, ddir=0, inner=True)
        gate_list.append(bot_decap_ports['g'])

        _, vdd_warrs = self.fill_dummy()
        self.add_pin('VDD', vdd_warrs, show=show_pins)

        iin = self.connect_to_tracks(gate_list, ref_idx)
        self.add_pin('iin', iin, show=show_pins)


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
    cell_name = 'CURRENT_MIRROR'

    params = specs['params']
    temp_db = make_tdb(prj, lib_name, specs)
    template = temp_db.new_template(params=params, temp_cls=PMOSCurrentMirror, debug=True)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)


if __name__ == '__main__':

    with open('specs_test/current_mirror.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
