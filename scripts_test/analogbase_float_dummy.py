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
from bag.layout.template import TemplateDB

from abs_templates_ec.analog_core import AnalogBase


class AmpBase(AnalogBase):
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
        super(AmpBase, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

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
            lch='channel length, in meters.',
            fg_tot='Total number of fingers.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            nw_list='NMOS width list',
            nth_list='NMOS threshold list',
            pw_list='PMOS width list',
            pth_list='PMOS threshold list',
            ng_tracks='NMOS gate tracks list',
            nds_tracks='NMOS drain/source tracks list',
            pg_tracks='PMOS gate tracks list',
            pds_tracks='PMOS gate tracks list',
            n_orientations='NMOS orientations.',
            p_orientations='PMOS orientations',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            top_layer='top layer ID.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self.draw_base(**self.params)

        fg_tot = self.params['fg_tot']
        fg_test = fg_tot // 2 - 1
        self.draw_mos_conn('nch', 0, 0, fg_tot, 0, 2)
        self.draw_mos_conn('nch', 1, 0, fg_test, 0, 2)
        self.draw_mos_conn('nch', 1, fg_tot - fg_test, fg_test, 0, 2)

        self.fill_dummy()


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

    temp_db = make_tdb(prj, lib_name, specs)
    name_list, temp_list = [], []
    name_list.append('ANALOGBASE_TEST')
    temp_list.append(temp_db.new_template(params=params, temp_cls=AmpBase))

    print('creating layouts')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('layout done.')


if __name__ == '__main__':

    with open('specs_test/analogbase_float_dummy.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj, block_specs)
