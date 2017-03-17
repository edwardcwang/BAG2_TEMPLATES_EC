# -*- coding: utf-8 -*-

import pprint

import bag
from abs_templates_ec.serdes.rxcore import RXCore
from bag.layout import RoutingGrid, TemplateDB

impl_lib = 'AAAFOO_serdes'
# impl_lib = 'craft_io_ec'


def rxcore(prj, temp_db):

    cell_name = 'rxcore_ffe1_dfe4'

    layout_params = dict(
        lch=16e-9,
        ptap_w=6,
        ntap_w=6,
        hm_width=1,
        hm_cur_width=2,
        diff_space=1,
        sig_widths=[1, 2],
        sig_spaces=[3, 2],
        min_fg_sep=4,
        w_dict={'load': 6, 'casc': 6, 'in': 4, 'sw': 4, 'tail': 6},
        th_dict={'load': 'ulvt', 'casc': 'ulvt', 'in': 'lvt', 'sw': 'ulvt', 'tail': 'ulvt'},
        integ_params={'load': 8, 'casc': 8, 'in': 4, 'sw': 4, 'tail': 8},
        alat_params_list=[
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
            {'load': 10, 'casc': 10, 'in': 6, 'sw': 6, 'tail': 10},
        ],
        dlat_params_list=[
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
        ],
        intsum_params=dict(
            fg_load=36,
            gm_fg_list=[
                {'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
                {'casc': 8, 'in': 4, 'sw': 4, 'tail': 8},
                {'casc': 4, 'in': 4, 'sw': 4, 'tail': 4},
                {'casc': 4, 'in': 4, 'sw': 4, 'tail': 4},
                {'casc': 8, 'in': 4, 'sw': 4, 'tail': 8},
            ],
            sgn_list=[1, -1, -1, -1, -1],
        ),
        summer_params=dict(
            fg_load=16,
            gm_fg_list=[
                {'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
                {'casc': 10, 'in': 6, 'sw': 6, 'tail': 10},
            ],
            sgn_list=[1, -1],
        ),
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=RXCore, debug=False)
    print('total number of fingers: %d' % template.num_fingers)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True, flatten=False)


def rxcore_sch(prj):
    lib_name = 'serdes_bm_templates'
    cell_name = 'rxcore_ffe1_dfe4'

    params = dict(
        lch=16e-9,
        w_dict={'load': 6, 'casc': 6, 'in': 4, 'sw': 4, 'tail': 6},
        th_dict={'load': 'ulvt', 'casc': 'ulvt', 'in': 'lvt', 'sw': 'ulvt', 'tail': 'ulvt'},
        integ_params={'load': 8, 'casc': 8, 'in': 4, 'sw': 4, 'tail': 8},
        alat_params_list=[
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
            {'load': 10, 'casc': 10, 'in': 6, 'sw': 6, 'tail': 10},
        ],
        dlat_params_list=[
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
            {'load': 12, 'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
        ],
        intsum_params=dict(
            fg_load=36,
            gm_fg_list=[
                {'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
                {'casc': 8, 'in': 4, 'sw': 4, 'tail': 8},
                {'casc': 4, 'in': 4, 'sw': 4, 'tail': 4},
                {'casc': 4, 'in': 4, 'sw': 4, 'tail': 4},
                {'casc': 8, 'in': 4, 'sw': 4, 'tail': 8},
            ],
            sgn_list=[1, -1, -1, -1, -1],
        ),
        summer_params=dict(
            fg_load=16,
            gm_fg_list=[
                {'casc': 12, 'in': 8, 'sw': 8, 'tail': 12},
                {'casc': 10, 'in': 6, 'sw': 6, 'tail': 10},
            ],
            sgn_list=[1, -1],
        ),
    )
    pprint.pprint(params)
    dsn = prj.create_design_module(lib_name, cell_name)
    dsn.design_specs(**params)
    dsn.implement_design(impl_lib, top_cell_name=cell_name, erase=True)


if __name__ == '__main__':

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = bag.BagProject()
        temp = 70.0
        layers = [4, 5, 6]
        spaces = [0.084, 0.080, 0.084]
        widths = [0.060, 0.100, 0.060]
        bot_dir = 'x'

        routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=True)

        rxcore(bprj, tdb)
        # rxcore_sch(bprj)
    else:
        print('loading BAG project')
