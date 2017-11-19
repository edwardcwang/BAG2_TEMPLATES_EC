# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################


"""This script tests that AnalogBase draws rows of transistors properly."""


from typing import Dict, Any, Set

import yaml

from bag import BagProject
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB, TemplateBase

from abs_templates_ec.serdes.amplifier import DiffAmp
from abs_templates_ec.analog_core.substrate import DeepNWellRing


class DNWRingTest(TemplateBase):
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
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(DNWRingTest, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            sub_params='substrate ring parameters.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        amp_params = self.params['amp_params']
        sub_params = self.params['sub_params']

        # make masters
        amp_master = self.new_template(params=amp_params, temp_cls=DiffAmp)
        sub_params['top_layer'] = amp_master.top_layer
        sub_params['bound_box'] = amp_master.bound_box
        sub_master = self.new_template(params=sub_params, temp_cls=DeepNWellRing)

        # place instances
        self.add_instance(sub_master, 'XS')
        self.add_instance(amp_master, 'XA', loc=sub_master.blk_loc_unit, unit_mode=True)

        # set size
        if sub_master.size is None:
            self.prim_bound_box = sub_master.prim_bound_box
            self.prim_top_layer = sub_master.prim_top_layer
        else:
            self.set_size_from_bound_box(sub_master.top_layer, sub_master.bound_box)
        self.array_box = sub_master.array_box
        self.add_cell_boundary(self.bound_box)


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
    lib_name = 'AAAFOO_DNWRING'
    amp_params = specs['amp_params']
    sub_params = specs['sub_params']
    lch1, lch2 = specs['lch']
    top_lay1, top_lay2 = specs['top_layer']

    temp_db = make_tdb(prj, lib_name, specs)
    name_list, temp_list = [], []
    amp_params1 = amp_params.copy()
    amp_params1['lch'] = lch1
    amp_params1['top_layer'] = top_lay1
    name_list.append('ANALOGBASE_TEST1')
    temp_list.append(temp_db.new_template(params=amp_params1, temp_cls=DiffAmp))

    params3 = dict(amp_params=amp_params1, sub_params=sub_params)
    name_list.append('SUBRING_TEST1')
    temp_list.append(temp_db.new_template(params=params3, temp_cls=DNWRingTest))

    print('creating layouts')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('layout done.')


if __name__ == '__main__':

    with open('specs_test/dnw_ring.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
