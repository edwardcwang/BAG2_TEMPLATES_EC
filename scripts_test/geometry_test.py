# -*- coding: utf-8 -*-

"""This script tests that layout primitives geometries work properly."""

from typing import Dict, Any, Set

import yaml

from bag import BagProject
from bag.layout.util import BBox
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
        bus = TLineBus(res, ('M2', 'drawing'), points, widths, spaces, end_style='round',
                       unit_mode=True)
        for p in bus.paths_iter():
            self.add_path(p)

        self.prim_top_layer = 3
        self.prim_bound_box = BBox(-10000, -10000, 10000, 10000, res, unit_mode=True)


if __name__ == '__main__':
    with open('specs_test/geometry_test.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    bprj.generate_cell(block_specs, Test2, gen_lay=True, gen_sch=False, run_lvs=False, 
                       debug=True)
