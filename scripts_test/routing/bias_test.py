# -*- coding: utf-8 -*-

"""This script tests that layout primitives geometries work properly."""

from typing import Dict, Any, Set

import yaml

from bag import BagProject
from bag.layout.template import TemplateBase, TemplateDB

from abs_templates_ec.routing.bias import BiasShield


class Test(TemplateBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **kwargs) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict()

    def draw_layout(self):
        layer = 4

        warr_list2 = [
            self.add_wires(layer + 1, 10, 500, 2200, unit_mode=True),
            self.add_wires(layer - 1, 8, 0, 300, unit_mode=True),
            self.add_wires(layer - 1, 12, 2000, 2200, width=2, unit_mode=True),
        ]

        nwire = len(warr_list2)
        blk_w, blk_h = BiasShield.get_block_size(self.grid, layer, nwire)
        result = BiasShield.connect_bias_shields(self, layer, warr_list2, 2 * blk_h, lu_end_mode=1)
        print(result)


if __name__ == '__main__':
    with open('specs_test/abs_templates_ec/routing/bias_shield.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    block_specs['impl_cell'] = 'BIAS_TEST'
    block_specs['params'] = {}
    bprj.generate_cell(block_specs, Test, gen_lay=True, debug=True)
