# -*- coding: utf-8 -*-

import pprint

from bag import BagProject
from abs_templates_ec.serdes.amplifier import DiffAmp
from bag.layout import RoutingGrid, TemplateDB

impl_lib = 'AAAFOO_diffamp2'


def diffamp(prj, temp_db, run_lvs=False, run_rcx=False):
    # type: (BagProject, TemplateDB) -> None
    lib_name = 'serdes_bm_templates'
    cell_name = 'diffamp_casc'

    params = dict(
        lch=16e-9,
        w_dict={'load': 6, 'casc': 4, 'in': 4, 'tail': 4},
        th_dict={'load': 'ulvt', 'casc': 'ulvt', 'in': 'ulvt', 'tail': 'ulvt'},
        fg_dict={'load': 2, 'casc': 4, 'in': 4, 'tail': 4},
    )

    layout_params = dict(
        ptap_w=6,
        ntap_w=6,
        nduml=4,
        ndumr=4,
        min_fg_sep=4,
        gds_space=1,
        diff_space=1,
        hm_width=1,
        hm_cur_width=2,
        show_pins=True,
        guard_ring_nf=0,
    )

    layout_params.update(params)

    pprint.pprint(layout_params)
    template = temp_db.new_template(params=layout_params, temp_cls=DiffAmp, debug=False)
    fg_tot = template.num_fingers
    print('total number of fingers: %d' % fg_tot)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)

    run_lvs = run_lvs or run_rcx

    if run_lvs:
        dsn = prj.create_design_module(lib_name, cell_name)
        dsn.design_specs(fg_tot=fg_tot, **params)
        dsn.implement_design(impl_lib, top_cell_name=cell_name, erase=True)
        print('run lvs')
        success, log_fname = prj.run_lvs(impl_lib, cell_name)
        if not success:
            raise ValueError('lvs failed.  Check log file: %s' % log_fname)
        else:
            print('lvs passed')

    if run_rcx:
        print('run rcx')
        success, log_fname = prj.run_rcx(impl_lib, cell_name)
        if not success:
            raise ValueError('rcx failed.  Check log file: %s' % log_fname)
        else:
            print('rcx passed')


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
    else:
        print('loading BAG project')
