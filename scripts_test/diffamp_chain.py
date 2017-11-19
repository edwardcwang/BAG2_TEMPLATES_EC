# -*- coding: utf-8 -*-

import yaml

from bag import float_to_si_string
from bag.core import BagProject
from bag.layout import RoutingGrid, TemplateDB
from bag.layout.template import TemplateBase

from abs_templates_ec.serdes.amplifier import DiffAmp


class AmpChain(TemplateBase):
    """A single diff amp.

    Parameters
    ----------
    temp_db : TemplateDB
            the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(AmpChain, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            amp_params='amplifier parameters.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        # make DiffAmp layout master
        amp_params = self.params['amp_params']
        amp_master = self.new_template(params=amp_params, temp_cls=DiffAmp)
        # add first instance at origin
        inst0 = self.add_instance(amp_master, loc=(0, 0), unit_mode=True)
        # abut second instance to the right of amp_master
        # note: if you want to change parameters for second instance, just create a different
        # master.
        xr = inst0.bound_box.right_unit
        inst1 = self.add_instance(amp_master, loc=(xr, 0), unit_mode=True)

        # get inst0 output and inst 1 input ports.  We know there are only one pin on one layer.
        outp = inst0.get_all_port_pins('outp')[0]
        outn = inst0.get_all_port_pins('outn')[0]
        inp = inst1.get_all_port_pins('inp')[0]
        inn = inst1.get_all_port_pins('inn')[0]
        # get horizontal/vertical layer number
        hm_layer = outp.layer_id
        vm_layer = hm_layer + 1

        # get total bounding box
        bnd_box = inst0.bound_box.merge(inst1.bound_box)
        # set size and top routing layer of this template.
        # round up because bounding box may not be quantized to vertical track pitch.
        self.set_size_from_bound_box(vm_layer, bnd_box, round_up=True)
        # add PR boundary
        self.add_cell_boundary(self.bound_box)

        # get vertical tracks used to connect input/output
        mid_index = self.grid.coord_to_nearest_track(vm_layer, xr, half_track=True, mode=1, unit_mode=True)
        left_idx = mid_index - 1
        right_idx = mid_index + 1

        # connect horizontal tracks to vertical tracks in a differential manner.
        midp, midn = self.connect_differential_tracks([outp, inp], [outn, inn], vm_layer, left_idx, right_idx)
        # export middle vertical wires
        self.add_pin('midp', midp, show=True)
        self.add_pin('midn', midn, show=True)

        # connect supplies
        vdd_list = inst0.get_all_port_pins('VDD') + inst1.get_all_port_pins('VDD')
        vdd = self.connect_wires(vdd_list)
        self.add_pin('VDD', vdd, show=True)

        # reexport port, renaming is optional
        self.reexport(inst0.get_port('VSS'), net_name='VSS2', show=True)


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
    name_fmt = 'AMPCHAIN_L%s_gr%d'
    for gr_nf in gr_nf_list:
        for lch in lch_list:
            params['lch'] = lch
            params['guard_ring_nf'] = gr_nf
            p = dict(amp_params=params)
            temp_list.append(temp_db.new_template(params=p, temp_cls=AmpChain, debug=False))
            name_list.append(name_fmt % (float_to_si_string(lch), gr_nf))
    print('creating layout')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('done')


if __name__ == '__main__':

    impl_lib = 'AAAFOO'

    with open('specs_test/diffamp.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
