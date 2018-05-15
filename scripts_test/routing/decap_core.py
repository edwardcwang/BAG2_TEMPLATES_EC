# -*- coding: utf-8 -*-

import yaml

from bag.core import BagProject

from abs_templates_ec.routing.fill import DecapFillCore

if __name__ == '__main__':
    with open('specs_test/abs_templates_ec/routing/decap_core.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    bprj.generate_cell(block_specs, DecapFillCore, debug=True)
    # bprj.generate_cell(block_specs, DecapFillCore, gen_sch=True, debug=True)
