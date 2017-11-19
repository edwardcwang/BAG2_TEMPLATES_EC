# -*- coding: utf-8 -*-

import pprint

from bag import BagProject
from abs_templates_ec.passives.hp_filter import HighPassFilter
from abs_templates_ec.passives.cap import MOMCap, MOMCapUnit
from abs_templates_ec.serdes.rxpassive import RXClkArray, CTLE
from bag.layout import RoutingGrid, TemplateDB

impl_lib = 'serdes_rx_top'


def ctle(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None

    layout_params = dict(
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
        sub_type='ptap',
        threshold='ulvt',
        res_type='standard',
        sup_width=2,
        sub_lch=16e-9,
        sub_w=6,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=CTLE, debug=False)
    temp_db.instantiate_layout(prj, template, 'ctle', debug=True)


def rxclk(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None

    layout_params = dict(
        passive_params=dict(
            l=10e-6,
            w=0.36e-6,
            cap_edge_margin=0.3,
            num_seg=2,
            num_cap_layer=3,
            io_width=2,
            sub_lch=16e-9,
            sub_w=6,
            sub_type='ntap',
            threshold='ulvt',
            res_type='standard',
        ),
        out_width=3,
        in_width=1,
        clk_names=['nmos_integ', 'nmos_analog', 'pmos_analog', 'nmos_intsum', 'pmos_digital',
                   'nmos_digital', 'pmos_summer', 'nmos_summer', 'nmos_tap1'],
        clk_locs=[0, 1, 1, 0, 1, 0, 1, 0, 0],
        parity=1,
        show_pins=True,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=RXClkArray, debug=False)
    temp_db.instantiate_layout(prj, template, 'rxclkarr', debug=True)


def hpf(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None

    layout_params = dict(
        l=10e-6,
        w=0.36e-6,
        cap_edge_margin=0.3,
        num_seg=2,
        num_cap_layer=3,
        io_width=2,
        sub_lch=16e-9,
        sub_w=6,
        sub_type='ntap',
        threshold='ulvt',
        res_type='standard',
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=HighPassFilter, debug=False)
    temp_db.instantiate_layout(prj, template, 'hpfilter', debug=True)


def mom(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None

    layout_params = dict(
        cap_bot_layer=4,
        cap_top_layer=7,
        cap_width=3.0,
        cap_height=6.0,
        sub_lch=16e-9,
        sub_w=6,
        sub_type='ptap',
        threshold='ulvt',
        show_pins=True,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=MOMCap, debug=False)
    temp_db.instantiate_layout(prj, template, 'momcap', debug=True)


def mom_unit(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None

    layout_params = dict(
        cap_bot_layer=4,
        cap_top_layer=7,
        cap_width=5.0,
        cap_height=5.0,
        port_width=3,
        show_pins=True,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=MOMCapUnit, debug=False)
    temp_db.instantiate_layout(prj, template, 'momcap', debug=True)


if __name__ == '__main__':

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
        temp = 70.0
        layers = [3, 4, 5, 6, 7, 8]
        spaces = [0.04, 0.084, 0.080, 0.084, 0.080, 0.36]
        widths = [0.05, 0.060, 0.100, 0.060, 0.100, 0.36]
        bot_dir = 'y'

        routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=True)

        # hpf(bprj, tdb)
        # mom(bprj, tdb)
        # rxclk(bprj, tdb)
        # ctle(bprj, tdb)
        mom_unit(bprj, tdb)
    else:
        print('loading BAG project')
