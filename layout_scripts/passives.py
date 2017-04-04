# -*- coding: utf-8 -*-

import pprint

from bag import BagProject
from abs_templates_ec.passives.hp_filter import HighPassFilterDiff
from abs_templates_ec.passives.cap import MOMCap
from bag.layout import RoutingGrid, TemplateDB

impl_lib = 'AAAFOO_cap'


def hpf(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None

    layout_params = dict(
        l=10e-6,
        w=0.36e-6,
        cap_port_width=2,
        cap_edge_margin=0.5e-6,
        cap_diff_margin=0.5e-6,
        num_seg=2,
        num_cap_layer=4,
        tr_idx_list=[6.5, 3.5],
        io_width=2,
        sub_lch=16e-9,
        sub_w=6,
        sub_type='ntap',
        threshold='ulvt',
        res_type='standard',
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=HighPassFilterDiff, debug=False)
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


if __name__ == '__main__':

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
        temp = 70.0
        layers = [4, 5, 6, 7]
        spaces = [0.084, 0.080, 0.084, 0.080]
        widths = [0.060, 0.100, 0.060, 0.100]
        bot_dir = 'x'

        routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=True)

        # hpf(bprj, tdb)
        mom(bprj, tdb)
    else:
        print('loading BAG project')