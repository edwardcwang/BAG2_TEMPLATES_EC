# -*- coding: utf-8 -*-

import yaml

from bag.core import BagProject

from abs_templates_ec.routing.fill import DecapFill

if __name__ == '__main__':
    with open('specs_test/abs_templates_ec/routing/decap.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    bprj.generate_cell(block_specs, DecapFill, debug=True)
    # bprj.generate_cell(block_specs, DecapFill, gen_sch=True, debug=True)
