# -*- coding: utf-8 -*-

import yaml

from bag import BagProject
from abs_templates_ec.serdes.rxcore_samp import RXCore
# from abs_templates_ec.serdes.rxtop_samp import RXFrontendCore
from bag.layout import RoutingGrid, TemplateDB


def make_tdb(prj, target_lib, specs):
    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def rxcore(prj, specs):
    temp_db = make_tdb(prj, impl_lib, specs)
    rxcore_params = specs['rxcore_params']
    rxcore_layout_params = specs['rxcore_layout_params']

    rxcore_layout_params.update(rxcore_params)

    print('creating layout')
    template = temp_db.new_template(params=rxcore_layout_params, temp_cls=RXCore, debug=False)
    print('total number of fingers: %d' % template.num_fingers)
    print('instantiating layout')
    temp_db.instantiate_layout(prj, template, top_cell_name=top_cell_name, debug=True)

    print('rxcore layout done.')

    return template.sch_params


def rxcore_sch(prj, sch_params):
    print('creating design module')
    dsn = prj.create_design_module(lib_name, cell_name)
    print('designing schematics')
    dsn.design_specs(**sch_params)
    print('creating rxcore schematics')
    dsn.implement_design(impl_lib, top_cell_name=top_cell_name, erase=True)

    print('rxcore schematic done')


if __name__ == '__main__':

    lib_name = 'serdes_bm_templates'
    cell_name = 'rxcore_ffe1_dfe4_v2'
    top_cell_name = 'RXCORE_TOP'
    impl_lib = 'AAAFOO'

    with open('specs_test/rxfrontend.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    rxcore_sch_params = rxcore(bprj, block_specs)
    rxcore_sch(bprj, rxcore_sch_params)
    # rxfrontend(bprj, tdb)

    """
    print('running lvs')
    lvs_passed, lvs_log = bprj.run_lvs(impl_lib, cell_name)
    if not lvs_passed:
        raise Exception('oops lvs died.  See LVS log file %s' % lvs_log)
    print('lvs passed')

    print('running rcx')
    rcx_passed, rcx_log = bprj.run_rcx(impl_lib, cell_name)
    if not rcx_passed:
        raise Exception('oops rcx died.  See RCX log file %s' % rcx_log)
    print('rcx passed')
    """