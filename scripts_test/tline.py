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


"""This script tests transmission line drawing."""


import math
import yaml
from itertools import chain

import numpy as np

import bag
from bag.layout import RoutingGrid, TemplateDB
from bag.layout.template import TemplateBase
from bag.layout.objects import TLineBus, Path
from bag.layout.util import BBox


class TLine(TemplateBase):
    """A transmission line template.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(TLine, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            layer_id='signal layer ID',
            bridge_layer_id='bridge layer ID',
            w_dict='width dictionary.',
            sp_dict='space dictionary.',
            sep_bridge='Approximate separation between bridge.',
            points_list='list of center points of transmission line.',
        )

    def draw_layout(self):
        layer_id = self.params['layer_id']
        bridge_layer_id = self.params['bridge_layer_id']
        w_dict = self.params['w_dict']
        sp_dict = self.params['sp_dict']
        sep_bridge = self.params['sep_bridge']
        points_list = self.params['points_list']

        res = self.grid.resolution

        w_sig = int(round(w_dict['sig'] / res))
        w_gnd = int(round(w_dict['gnd'] / res))
        w_bridge = int(round(w_dict['bridge'] / res))
        sp_sig = int(round(sp_dict['sig'] / res))
        sp_gnd = int(round(sp_dict['gnd'] / res))
        sep_bridge = int(round(sep_bridge / res))
        points_list = [(int(round(x / res)), int(round(y / res))) for x, y in points_list]

        self._draw_tline_bus(points_list, w_sig, w_gnd, w_bridge, sp_sig, sp_gnd,
                             sep_bridge, layer_id, bridge_layer_id)

    def _draw_tline_bus(self, points_list, w_sig, w_gnd, w_bridge, sp_sig, sp_gnd,
                        sep_bridge, layer_id, bridge_layer_id):
        w_list = [w_gnd, w_sig, w_sig, w_gnd]
        sp_list = [sp_gnd, sp_sig, sp_gnd]

        lay_name = self.grid.get_layer_name(layer_id, 0)
        bus = TLineBus(self.grid.resolution, lay_name, points_list, w_list, sp_list, unit_mode=True)
        for path in bus.paths_iter():
            self.add_path(path)

        # draw bridges on ground wires for every straight segment
        # also draw keep-out layers
        gap = w_sig + sp_gnd + (sp_sig + w_gnd) // 2
        pnext = iter(points_list)
        next(pnext)
        for (x0, y0), (x1, y1) in zip(points_list, pnext):
            if x1 == x0 or y0 == y1:
                self.draw_bridges(x0, y0, x1, y1, w_gnd, w_bridge, sep_bridge, gap,
                                  layer_id, bridge_layer_id)

    def draw_bridges(self, x0, y0, x1, y1, w_gnd, w_bridge, sep_bridge, gap,
                     layer_id, bridge_layer_id):
        res = self.grid.resolution

        dir_vec = np.array((x1 - x0, y1 - y0), dtype=int)
        seg_length = np.max(np.abs(dir_vec))  # type: int
        dir_unit = dir_vec // seg_length
        ortho_unit = np.array((-dir_unit[1], dir_unit[0]), dtype=int)

        if x1 == x0:
            # vertical wire
            wire_dir = 'y'
            box = BBox(-w_gnd // 2, -w_bridge // 2, w_gnd // 2, w_bridge // 2, res, unit_mode=True)
        else:
            wire_dir = 'x'
            box = BBox(-w_bridge // 2, -w_gnd // 2, w_bridge // 2, w_gnd // 2, res, unit_mode=True)

        num_seg = seg_length // sep_bridge
        if num_seg > 0:
            values = np.linspace(0.0, seg_length, num_seg + 2, endpoint=True)
            bridge_layer = self.grid.tech_info.get_layer_name(bridge_layer_id)
            orig = np.array((x0, y0), dtype=int)
            for sel in values[1:-1]:
                sel = int(round(sel))
                mp0 = orig + dir_unit * sel - ortho_unit * gap
                box0 = box.move_by(dx=mp0[0], dy=mp0[1], unit_mode=True)
                delta = ortho_unit * 2 * gap
                box1 = box0.move_by(dx=delta[0], dy=delta[1], unit_mode=True)
                wbox = box0.merge(box1)
                self.add_rect(bridge_layer, wbox)
                for idx in range(bridge_layer_id, layer_id):
                    bot_layer = self.grid.tech_info.get_layer_name(idx)
                    top_layer = self.grid.tech_info.get_layer_name(idx + 1)
                    bot_via_dir = self.grid.get_direction(idx)
                    self.add_via(box0, bot_layer, top_layer, bot_via_dir)
                    self.add_via(box1, bot_layer, top_layer, bot_via_dir)


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
    lib_name = 'AAAFOO_TLINE'
    params = specs['params']

    temp_db = make_tdb(prj, lib_name, specs)
    name_list = ['TLINE_TEST']
    temp_list = [temp_db.new_template(params=params, temp_cls=TLine)]

    print('creating layouts')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('layout done')


if __name__ == '__main__':

    with open('specs_test/tline.yaml', 'r') as f:
        block_specs = yaml.load(f)
    
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = bag.BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
