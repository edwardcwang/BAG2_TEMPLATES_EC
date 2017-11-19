# -*- coding: utf-8 -*-

import pprint

from bag import BagProject
from abs_templates_ec.serdes.tx_cml import CMLLoadSingle, CMLCorePMOS, CMLDriverPMOS
from bag.layout import RoutingGrid, TemplateDB


def txload(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None
    cell_name = 'txload'

    layout_params = dict(
        res_params=dict(
            l=1.3e-6,
            w=1.8e-6,
            nx=9,
            ndum=1,
            sub_type='ptap',
            threshold='ulvt',
            res_type='high_speed',
            em_specs={'idc': 0.738e-3, 'dc_temp': 105},
        ),
        sub_params=dict(
            lch=16e-9,
            w=6,
            sub_type='ptap',
            threshold='ulvt',
        ),
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=CMLLoadSingle, debug=False)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)

    print('tot_width = %.6g' % template.array_box.width)
    print('output tracks = %s' % template.output_tracks)


def txcore(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None
    cell_name = 'txcore'

    layout_params = dict(
        lch=16e-9,
        w=4,
        fg=24,
        fg_ref=2,
        output_tracks=[31, 44, 57, 70, 83, 96, 109, 122, 135],
        em_specs={'idc': 0.738e-3, 'dc_temp': 105},
        threshold='ulvt',
        input_width=2,
        input_space=1,
        ntap_w=6,
        guard_ring_nf=2,
        tot_width=30.06,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=CMLCorePMOS, debug=False)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)


def generate(prj, temp_db):
    # type: (BagProject, TemplateDB) -> None
    cell_name = 'tx_driver'

    layout_params = dict(
        res_params=dict(
            l=1.3e-6,
            w=1.8e-6,
            nx=9,
            ndum=1,
            sub_type='ptap',
            threshold='ulvt',
            res_type='high_speed',
            em_specs={'idc': 1.333e-3, 'dc_temp': 105},
        ),
        lch=16e-9,
        w=4,
        fg=24,
        fg_ref=2,
        threshold='ulvt',
        input_width=2,
        input_space=2,
        ntap_w=10,
        guard_ring_nf=2,
        top_layer=8,
    )

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=CMLDriverPMOS, debug=False)
    print('num fingers = %d' % template.num_fingers)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)


if __name__ == '__main__':

    impl_lib = 'serdes_tx_driver'

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
        temp = 70.0
        layers = [3, 4, 5, 6, 7, 8, 9]
        spaces = [0.04, 0.084, 0.080, 0.084, 0.080, 0.36, 0.36]
        widths = [0.05, 0.060, 0.100, 0.060, 0.100, 0.36, 0.36]
        bot_dir = 'y'

        routing_grid = RoutingGrid(bprj.tech_info, layers, spaces, widths, bot_dir)

        tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=True)

        generate(bprj, tdb)
    else:
        print('loading BAG project')
