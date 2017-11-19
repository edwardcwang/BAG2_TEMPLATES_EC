# -*- coding: utf-8 -*-

import yaml

from bag import float_to_si_string
from bag.core import BagProject
from bag.layout import RoutingGrid, TemplateDB

from abs_templates_ec.serdes.amplifier import DiffAmp


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
    temp_db = make_tdb(prj, impl_lib, specs)
    params = specs['params']
    lch_list = specs['swp_params']['lch']
    gr_nf_list = specs['swp_params']['guard_ring_nf']

    temp_list = []
    name_list = []
    name_fmt = 'DIFFAMP_DIODE_DECAP_L%s_gr%d'
    for gr_nf in gr_nf_list:
        for lch in lch_list:
            params['lch'] = lch
            params['guard_ring_nf'] = gr_nf
            temp_list.append(temp_db.new_template(params=params, temp_cls=DiffAmp, debug=False))
            name_list.append(name_fmt % (float_to_si_string(lch), gr_nf))
    temp_db.batch_layout(prj, temp_list, name_list)
    print('done')


if __name__ == '__main__':

    impl_lib = 'AAAFOO'

    with open('specs_test/diffamp_diode_decap.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
