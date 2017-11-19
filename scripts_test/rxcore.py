# -*- coding: utf-8 -*-

import yaml

from bag.core import BagProject
from bag.layout import RoutingGrid, TemplateDB

from abs_templates_ec.serdes.rxcore_samp import RXCore


def make_tdb(prj, target_lib, specs):
    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate_layout(prj, specs):
    temp_db = make_tdb(prj, impl_lib, specs)
    params = specs['rxcore_params']
    layout_params = specs['rxcore_layout_params']
    params.update(layout_params)

    template = temp_db.new_template(params=params, temp_cls=RXCore, debug=False)
    name_list = ['RXCORE_SAMP']
    temp_db.batch_layout(prj, [template], name_list)
    print('done')

    return template.sch_params


if __name__ == '__main__':

    impl_lib = 'AAAFOO'

    with open('specs_test/rxcore.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    sch_params = generate_layout(bprj, block_specs)
