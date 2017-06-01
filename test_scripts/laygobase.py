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
import copy

from bag import BagProject, float_to_si_string
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB

from abs_templates_ec.laygo.core import LaygoBase


class Test(LaygoBase):
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
        super(Test, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            config='laygo configuration dictionary.',
            guard_ring_nf='number of guard ring fingers.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self.set_row_types(['nch'], ['R0'], ['lvt'], True, 15, guard_ring_nf=self.params['guard_ring_nf'])
        self.add_laygo_primitive('fg2d', loc=(0, 0), nx=3, spx=1)
        self.add_laygo_primitive('fg2d', loc=(3, 0), split_s=True)
        self.add_laygo_primitive('stack2d', loc=(4, 0), nx=2, spx=1)
        self.add_laygo_primitive('fg2d', loc=(6, 0), split_s=True)
        self.add_laygo_primitive('fg2d', loc=(7, 0), nx=3, spx=1)
        self.set_laygo_size()
        self.fill_space()
        self.draw_boundary_cells()


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
    lib_name = 'AAAFOO'

    params = specs['params']
    lch_vmsp_list = specs['swp_params']['lch_vmsp']
    gr_nf_list = specs['swp_params']['guard_ring_nf']

    temp_db = make_tdb(prj, lib_name, specs)

    temp_list = []
    name_list = []
    name_fmt = 'LAYGOBASE_L%s_gr%d'
    for gr_nf in gr_nf_list:
        params['guard_ring_nf'] = gr_nf
        for lch, vm_sp in lch_vmsp_list:
            config = copy.deepcopy(params['config'])
            config['lch'] = lch
            config['tr_spaces'][1] = vm_sp
            params['config'] = config
            temp_list.append(temp_db.new_template(params=params, temp_cls=Test, debug=False))
            name_list.append(name_fmt % (float_to_si_string(lch), gr_nf))
    print('creating layout')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('done')


if __name__ == '__main__':

    with open('test_specs/laygobase.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

        generate(bprj, block_specs)
    else:
        print('loading BAG project')
