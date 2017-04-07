# -*- coding: utf-8 -*-

import pprint

import bag
from abs_templates_ec.serdes.rxcore import RXCore
from abs_templates_ec.serdes.rxtop import RXFrontend
from bag.layout import RoutingGrid, TemplateDB

impl_lib = 'AAAFOO_serdes'
# impl_lib = 'craft_io_ec'


def rxfrontend(prj, temp_db):

    cell_name = 'rx_frontend'

    params = dict(
        lch=16e-9,
        w_dict={'load': 6, 'casc': 6, 'in': 4, 'sw': 4, 'tail': 4},
        th_dict={'load': 'ulvt', 'casc': 'ulvt', 'in': 'ulvt', 'sw': 'ulvt', 'tail': 'svt'},
        integ_params={'load': 4, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
        alat_params_list=[
            {'load': 2, 'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
            {'load': 2, 'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
        ],
        intsum_params=dict(
            fg_load=12,
            fg_offset=12,
            gm_fg_list=[
                {'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
                {'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
                {'casc': 4, 'in': 4, 'tail': 2},
                {'casc': 4, 'in': 4, 'tail': 2},
                {'casc': 8, 'in': 8, 'tail': 4},
            ],
            sgn_list=[1, -1, -1, -1, -1],
        ),
        summer_params=dict(
            fg_load=4,
            gm_fg_list=[
                {'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
                {'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
            ],
            sgn_list=[1, -1],
        ),
        dlat_params_list=[
            {'load': 6, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
            {'load': 6, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
            {'load': 6, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
        ],
    )

    rcore_params = dict(
        ptap_w=6,
        ntap_w=6,
        hm_width=1,
        hm_cur_width=2,
        diff_space=1,
        sig_widths=[1, 2],
        sig_spaces=[2, 2],
        clk_widths=[2, 3, 4],
        clk_spaces=[2, 3, 6],
        sig_clk_spaces=[2, 3],
        min_fg_sep=4,
    )

    rcore_params.update(params)

    rxclk_params = dict(
        passive_params=dict(
            l=10e-6,
            w=0.36e-6,
            cap_edge_margin=0.25,
            num_seg=2,
            num_cap_layer=3,
            io_width=[1, 1, 3],
            sub_lch=16e-9,
            sub_w=6,
            sub_type='ntap',
            threshold='ulvt',
            res_type='standard',
        ),
        io_width=3,
        clk_names=['nmos_integ', 'nmos_analog', 'pmos_analog', 'nmos_intsum', 'pmos_digital',
                   'nmos_digital', 'pmos_summer', 'nmos_summer', 'nmos_tap1'],
        clk_locs=[0, 1, 1, 0, 1, 0, 1, 0, 0],
    )

    ctle_params = dict(
        l=0.72e-6,
        w=0.36e-6,
        cap_edge_margin=0.2,
        num_cap_layer=4,
        cap_port_widths=[2, 1, 2, 2],
        cap_port_offset=3,
        num_r1=4,
        num_r2=6,
        num_dumr=1,
        num_dumc=4,
        io_width=2,
        sub_type='ntap',
        threshold='ulvt',
        res_type='standard',
        sup_width=2,
        sub_lch=16e-9,
        sub_w=6,
    )

    layout_params = dict(
        core_params=rcore_params,
        rxclk_params=rxclk_params,
        ctle_params=ctle_params,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=RXFrontend, debug=False)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)


def rxcore(prj, temp_db):

    cell_name = 'rxcore_ffe1_dfe4'

    params = dict(
        lch=16e-9,
        w_dict={'load': 6, 'casc': 6, 'in': 4, 'sw': 4, 'tail': 4},
        th_dict={'load': 'ulvt', 'casc': 'ulvt', 'in': 'ulvt', 'sw': 'ulvt', 'tail': 'svt'},
        integ_params={'load': 4, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
        alat_params_list=[
            {'load': 2, 'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
            {'load': 2, 'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
        ],
        intsum_params=dict(
            fg_load=12,
            fg_offset=12,
            gm_fg_list=[
                {'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
                {'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
                {'casc': 4, 'in': 4, 'tail': 2},
                {'casc': 4, 'in': 4, 'tail': 2},
                {'casc': 8, 'in': 8, 'tail': 4},
            ],
            sgn_list=[1, -1, -1, -1, -1],
        ),
        summer_params=dict(
            fg_load=4,
            gm_fg_list=[
                {'casc': 8, 'in': 6, 'sw': 6, 'tail': 4},
                {'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
            ],
            sgn_list=[1, -1],
        ),
        dlat_params_list=[
            {'load': 6, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
            {'load': 6, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
            {'load': 6, 'casc': 4, 'in': 4, 'sw': 4, 'tail': 2},
        ],
    )

    layout_params = dict(
        ptap_w=6,
        ntap_w=6,
        hm_width=1,
        hm_cur_width=2,
        diff_space=1,
        sig_widths=[1, 2],
        sig_spaces=[2, 2],
        clk_widths=[2, 3, 4],
        clk_spaces=[2, 3, 6],
        sig_clk_spaces=[2, 3],
        min_fg_sep=4,
    )

    layout_params.update(params)

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=RXCore, debug=False)
    print('total number of fingers: %d' % template.num_fingers)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)
    return params, template.num_fingers


def rxcore_sch(prj, sch_params, tot_fg):
    lib_name = 'serdes_bm_templates'
    cell_name = 'rxcore_ffe1_dfe4'

    dsn = prj.create_design_module(lib_name, cell_name)
    dsn.design_specs(fg_tot=2 * tot_fg, **sch_params)
    dsn.implement_design(impl_lib, top_cell_name=cell_name, erase=True)


if __name__ == '__main__':

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = bag.BagProject()
        temp = 70.0
        layers = [3, 4, 5, 6, 7, 8]
        spaces = [0.05, 0.084, 0.080, 0.084, 0.080, 0.36]
        widths = [0.04, 0.060, 0.100, 0.060, 0.100, 0.36]
        bot_dir = 'y'

        routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=True)

        # core_params, fg_tot = rxcore(bprj, tdb)
        # rxcore_sch(bprj, core_params, fg_tot)
        rxfrontend(bprj, tdb)
    else:
        print('loading BAG project')
