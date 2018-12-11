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


from typing import Dict, Any, Set, Union

import yaml
import pprint

from bag import BagProject
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB

from abs_templates_ec.analog_core import AnalogBase


class CSAmp(AnalogBase):
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
        super(CSAmp, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._num_fg = -1

    @property
    def num_fingers(self):
        # type: () -> int
        return self._num_fg

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            th_dict={},
            nduml=4,
            ndumr=4,
            gds_space=1,
            hm_width=1,
            hm_cur_width=-1,
            guard_ring_nf=0,
            show_pins=True,
        )

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
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            fg_dict='NMOS/PMOS number of fingers dictionary.',
            nduml='Number of left dummy fingers.',
            ndumr='Number of right dummy fingers.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            show_pins='True to draw pin geometry.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: CSAmp
                            lch,  # type: float
                            ptap_w,  # type: Union[float, int]
                            ntap_w,  # type: Union[float, int]
                            w_dict,  # type: Dict[str, Union[float, int]]
                            th_dict,  # type: Dict[str, str]
                            fg_dict,  # type: Dict[str, int]
                            nduml,  # type: int
                            ndumr,  # type: int
                            hm_width,  # type: int
                            hm_cur_width,  # type: int
                            guard_ring_nf,  # type: int
                            show_pins,  # type: bool
                            **kwargs
                            ):
        # type: (...) -> None

        # calculate total number of fingers.
        fg_tot = max(fg_dict.values()) + nduml + ndumr
        self._num_fg = fg_tot

        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params = dict(
            lch=lch,
            fg_tot=fg_tot,
            ptap_w=ptap_w,
            ntap_w=ntap_w,
            nw_list=[w_dict['n']],
            nth_list=[th_dict['n']],
            pw_list=[w_dict['p']],
            pth_list=[th_dict['p']],
            ng_tracks=[hm_width],
            pg_tracks=[hm_width],
            nds_tracks=[1],
            pds_tracks=[hm_cur_width],
            guard_ring_nf=guard_ring_nf,
        )
        self.draw_base(**draw_params)
        print('Size tuple: %s' % str(self.size))
        print('Overall bounding box: %s' % self.bound_box)
        print('Array box: %s' % self.array_box)
        self.set_size_from_array_box(self.mos_conn_layer + 1)

        # determine nmos/pmos source/drain
        nfg = fg_dict['n']
        pfg = fg_dict['p']
        nfg_eff = nfg // self.num_fg_per_sd
        pfg_eff = pfg // self.num_fg_per_sd
        n_sup, n_out = 's', 'd'
        nsdir, nddir = 0, 2
        if (nfg_eff - pfg_eff) % 4 == 0:
            p_sup, p_out = n_sup, n_out
        else:
            p_sup, p_out = n_out, n_sup

        if p_sup == 's':
            psdir, pddir = 2, 0
        else:
            psdir, pddir = 0, 2

        # draw nmos connection
        nmos_col = nduml + max((pfg - nfg) // 2, 0)
        nmos_ports = self.draw_mos_conn('nch', 0, nmos_col, nfg, nsdir, nddir)
        print('nmos ports: ')
        pprint.pprint(nmos_ports)

        # draw pmos connection
        pmos_col = nduml + max((nfg - pfg) // 2, 0)
        pmos_ports = self.draw_mos_conn('pch', 0, pmos_col, pfg, psdir, pddir)
        print('pmos ports: ')
        pprint.pprint(pmos_ports)

        # connect to supplies
        self.connect_to_substrate('ptap', nmos_ports[n_sup])
        self.connect_to_substrate('ntap', pmos_ports[p_sup])

        # connect output
        # use pmos row 0, track index (hm_cur_width - 1) / 2, with width hm_cur_width
        out_tid = self.make_track_id('pch', 0, 'ds', (hm_cur_width - 1) / 2, width=hm_cur_width)
        out_warr = self.connect_to_tracks([pmos_ports[p_out], nmos_ports[n_out]], out_tid)
        self.add_pin('out', out_warr, show=show_pins)

        # connect nmos/pmos gates
        ng_tid = self.make_track_id('nch', 0, 'g', (hm_width - 1) / 2, width=hm_width)
        ng_warr = self.connect_to_tracks(nmos_ports['g'], ng_tid, min_len_mode=0)
        self.add_pin('in', ng_warr, show=show_pins)
        pg_tid = self.make_track_id('pch', 0, 'g', (hm_width - 1) / 2, width=hm_width)
        pg_warr = self.connect_to_tracks(pmos_ports['g'], pg_tid)
        self.add_pin('bias', pg_warr, show=show_pins)

        # connect dummies and draw supply wires
        vss_warrs, vdd_warrs = self.fill_dummy()
        self.add_pin('VSS', vss_warrs, label='VSS:', show=show_pins)
        self.add_pin('VDD', vdd_warrs, show=show_pins)


def make_tdb(prj, target_lib, specs):
    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']
    width_override = grid_specs.get('width_override', None)

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir, width_override=width_override)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate(prj, specs):
    lib_name = 'AAAFOO'
    cell_name = 'CSAMP_TEST'

    params = specs['params']
    temp_db = make_tdb(prj, lib_name, specs)
    template = temp_db.new_template(params=params, temp_cls=CSAmp, debug=True)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)


if __name__ == '__main__':

    with open('specs_test/cs_amp.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
