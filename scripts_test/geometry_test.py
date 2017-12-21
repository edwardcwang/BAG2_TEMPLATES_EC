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


"""This script tests that layout primitives geometries work properly."""

from typing import Dict, Any, Set

from bag import BagProject
from bag.layout.util import BBox
from bag.layout.routing import RoutingGrid
from bag.layout.objects import Path, Blockage, Boundary, Polygon, TLineBus
from bag.layout.template import TemplateBase, TemplateDB


class Test1(TemplateBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(Test1, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict()

    def draw_layout(self):
        res = self.grid.resolution

        # simple rectangle
        self.add_rect('M1', BBox(100, 60, 180, 80, res, unit_mode=True))

        # a path
        width = 20
        points = [(0, 0), (2000, 0), (3000, 1000), (3000, 3000)]
        path = Path(res, 'M2', width, points, 'truncate', 'round', unit_mode=True)
        self.add_path(path)

        # set top layer and bounding box so parent can query those
        self.prim_top_layer = 3
        self.prim_bound_box = BBox(0, 0, 400, 400, res, unit_mode=True)


class Test2(TemplateBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(Test2, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict()

    def draw_layout(self):
        res = self.grid.resolution

        # instantiate Test1
        master = self.template_db.new_template(params={}, temp_cls=Test1)
        self.add_instance(master, 'X0', loc=(-100, -100), orient='MX', unit_mode=True)

        # add via, using BAG's technology DRC calculator
        self.add_via(BBox(0, 0, 100, 100, res, unit_mode=True),
                     'M1', 'M2', 'x')

        # add via, specify all parameters
        # note: the via name may be different in each technology.
        self.add_via_primitive('M2_M1', [300, 300], num_rows=2, num_cols=2,
                               sp_rows=50, sp_cols=100, enc1=[20, 20, 30, 30],
                               enc2=[30, 30, 40, 40], unit_mode=True)

        # add a primitive pin
        self.add_pin_primitive('mypin', 'M1', BBox(-100, 0, 0, 20, res, unit_mode=True))

        # add a polygon
        points = [(0, 0), (300, 200), (100, 400)]
        p = Polygon(res, 'M3', points, unit_mode=True)
        self.add_polygon(p)

        # add a blockage
        points = [(-1000, -1000), (-1000, 1000), (1000, 1000), (1000, -1000)]
        b = Blockage(res, 'placement', '', points, unit_mode=True)
        self.add_blockage(b)

        # add a boundary
        points = [(-500, -500), (-500, 500), (500, 500), (500, -500)]
        b = Boundary(res, 'PR', points, unit_mode=True)
        self.add_boundary(b)

        # add a parallel path bus
        widths = [100, 50, 100]
        spaces = [80, 80]
        points = [(0, -3000), (-3000, -3000), (-4000, -2000), (-4000, 0)]
        bus = TLineBus(res, ('M2', 'drawing'), points, widths, spaces, end_style='round', unit_mode=True)
        for p in bus.paths_iter():
            self.add_path(p)

        self.prim_top_layer = 3
        self.prim_bound_box = BBox(-10000, -10000, 10000, 10000, res, unit_mode=True)


def make_tdb(prj, target_lib):
    layers = [3, 4, 5]
    spaces = [0.1, 0.1, 0.2]
    widths = [0.1, 0.1, 0.2]
    bot_dir = 'y'

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate(prj):
    lib_name = 'AAAFOO_GEOTEST'

    temp_db = make_tdb(prj, lib_name)
    name_list, temp_list = [], []
    name_list.append('TEST2')
    temp_list.append(temp_db.new_template(params={}, temp_cls=Test2))

    print('creating layouts')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('layout done.')


if __name__ == '__main__':

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj)
