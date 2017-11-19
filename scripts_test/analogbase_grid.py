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
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
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
            params1='Amp1 parameters.',
            params2='Amp2 parameters.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        params1 = self.params['params1']
        params2 = self.params['params2']

        master1 = self.new_template(params=params1, temp_cls=DiffAmp)  # type: DiffAmp
        master2 = self.new_template(params=params2, temp_cls=DiffAmp)  # type: DiffAmp

        # place inst1
        inst1 = self.add_instance(master1, 'X1')
        if master1.top_layer == master2.top_layer:
            x0 = inst1.array_box.right_unit - master2.array_box.left_unit
        else:
            x0 = inst1.bound_box.right_unit

        # place inst2
        if master2.size is not None:
            xblk, _ = self.grid.get_block_size(master2.top_layer, unit_mode=True)
            x0 = -(-x0 // xblk) * xblk
        inst2 = self.add_instance(master2, 'X2', loc=(x0, 0), unit_mode=True)

        # set size
        my_top_layer = max(master1.mos_conn_layer + 2, master1.top_layer, master2.top_layer)
        bbox = inst1.bound_box.merge(inst2.bound_box)
        self.set_size_from_bound_box(my_top_layer, bbox, round_up=True)


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
    lib_name = 'AAAFOO_ANALOGBASE'
    params = specs['params']
    lch1, lch2 = specs['lch']
    top_lay1, top_lay2 = specs['top_layer']

    temp_db = make_tdb(prj, lib_name, specs)
    name_list, temp_list = [], []
    params1 = params.copy()
    params1['lch'] = lch1
    params1['top_layer'] = top_lay1
    name_list.append('ANALOGBASE_TEST1')
    temp_list.append(temp_db.new_template(params=params1, temp_cls=DiffAmp))

    params2 = params1.copy()
    params2['lch'] = lch2
    params2['top_layer'] = top_lay2
    name_list.append('ANALOGBASE_TEST2')
    temp_list.append(temp_db.new_template(params=params2, temp_cls=DiffAmp))

    params3 = dict(params1=params1, params2=params1)
    name_list.append('ANALOGBASE_CHAIN_TEST1')
    temp_list.append(temp_db.new_template(params=params3, temp_cls=AmpChain))

    params4 = dict(params1=params1, params2=params2)
    name_list.append('ANALOGBASE_CHAIN_TEST2')
    temp_list.append(temp_db.new_template(params=params4, temp_cls=AmpChain))

    params5 = dict(params1=params2, params2=params2)
    name_list.append('ANALOGBASE_CHAIN_TEST3')
    temp_list.append(temp_db.new_template(params=params5, temp_cls=AmpChain))

    print('creating layouts')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('layout done.')


if __name__ == '__main__':

    with open('specs_test/analogbase_grid.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
