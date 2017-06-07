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
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB

from abs_templates_ec.laygo.core import LaygoBase


class NAND(LaygoBase):
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
        super(NAND, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            num_blk='number of driver segments.',
            show_pins='True to draw pin geometries.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """

        if not self.fg2d_s_short:
            raise ValueError('This template current only works if source wires of fg2d are shorted.')

        threshold = self.params['threshold']
        draw_boundaries = self.params['draw_boundaries']
        num_blk = self.params['num_blk']
        show_pins = self.params['show_pins']

        row_list = ['ptap', 'nch', 'pch', 'ntap']
        orient_list = ['R0', 'MX', 'R0', 'MX']
        thres_list = [threshold] * 4

        # compute number of tracks
        # note: because we're using thick wires, we need to compute space needed to
        # satisfy DRC rules
        hm_layer = self.conn_layer + 1
        num_g_tracks = [0, 1, 1, 0]
        num_gb_tracks = [0, 1, 1, 0]
        num_ds_tracks = [1, 0, 0, 1]

        # to draw special stack driver primitive, we need to enable dual_gate options.
        options = dict()
        row_kwargs = [{}, options, options, {}]
        if draw_boundaries:
            end_mode = 15
        else:
            end_mode = 0

        # specify row types
        self.set_row_types(row_list, orient_list, thres_list, draw_boundaries, end_mode,
                           num_g_tracks, num_gb_tracks, num_ds_tracks, guard_ring_nf=0,
                           row_kwargs=row_kwargs)

        # determine total number of blocks
        # draw nwell tap
        row_idx = 3
        nw_tap = self.add_laygo_primitive('sub', loc=(0, row_idx), nx=num_blk, spx=1)

        # draw pmos row
        row_idx = 2
        pmos = self.add_laygo_primitive('fg2s', loc=(0, row_idx), nx=num_blk, spx=1)

        # draw nmos row
        row_idx = 1
        nmos = self.add_laygo_primitive('stack2s', loc=(0, row_idx), nx=num_blk, spx=1)

        # draw pwell tap
        row_idx = 0
        pw_tap = self.add_laygo_primitive('sub', loc=(0, row_idx), nx=num_blk, spx=1)

        # compute overall block size
        self.set_laygo_size(num_col=num_blk + 4)
        self.fill_space()
        # draw boundaries and get guard ring power rail tracks
        self.draw_boundary_cells()


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

    template = temp_db.new_template(params=params, temp_cls=NAND, debug=False)
    name = 'NAND_GATE'
    print('create layout')
    temp_db.batch_layout(prj, [template], [name])
    print('done')


if __name__ == '__main__':

    with open('test_specs/nand.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

        generate(bprj, block_specs)
    else:
        print('loading BAG project')
