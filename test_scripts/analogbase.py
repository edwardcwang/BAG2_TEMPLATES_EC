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


from typing import Dict, Any, Set, Union

import yaml

from bag import BagProject
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB

from abs_templates_ec.serdes.base import SerdesRXBase, SerdesRXBaseInfo


class AmpBase(SerdesRXBase):
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
        self._num_fg = -1

    @property
    def num_fingers(self):
        # type: () -> int
        return self._num_fg

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            th_dict={},
            nduml=4,
            ndumr=4,
            min_fg_sep=0,
            gds_space=1,
            diff_space=1,
            hm_width=1,
            hm_cur_width=-1,
            guard_ring_nf=0,
        )

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
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            fg_dict='NMOS/PMOS number of fingers dictionary.',
            nduml='Number of left dummy fingers.',
            ndumr='Number of right dummy fingers.',
            min_fg_sep='Minimum separation between transistors.',
            gds_space='number of tracks reserved as space between gate and drain/source tracks.',
            diff_space='number of tracks reserved as space between differential tracks.',
            hm_width='width of horizontal track wires.',
            hm_cur_width='width of horizontal current track wires. If negative, defaults to hm_width.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
        )

    def draw_layout(self):
        """Draw the layout of a dynamic latch chain.
        """
        self._draw_layout_helper(**self.params)

    def _draw_layout_helper(self,  # type: AmpBase
                            lch,  # type: float
                            ptap_w,  # type: Union[float, int]
                            ntap_w,  # type: Union[float, int]
                            w_dict,  # type: Dict[str, Union[float, int]]
                            th_dict,  # type: Dict[str, str]
                            fg_dict,  # type: Dict[str, int]
                            nduml,  # type: int
                            ndumr,  # type: int
                            min_fg_sep,  # type: int
                            gds_space,  # type: int
                            diff_space,  # type: int
                            hm_width,  # type: int
                            hm_cur_width,  # type: int
                            guard_ring_nf,  # type: int
                            **kwargs
                            ):
        # type: (...) -> None

        serdes_info = SerdesRXBaseInfo(self.grid, lch, guard_ring_nf, min_fg_sep=min_fg_sep)
        diffamp_info = serdes_info.get_diffamp_info(fg_dict)
        fg_tot = diffamp_info['fg_tot'] + nduml + ndumr
        self._num_fg = fg_tot

        if hm_cur_width < 0:
            hm_cur_width = hm_width  # type: int

        # draw AnalogBase rows
        # compute pmos/nmos gate/drain/source number of tracks
        draw_params = dict(
            lch=lch,
            fg_tot=fg_tot,
            ptap_w=ptap_w,
            ntap_w=ntap_w,
            w_dict=w_dict,
            th_dict=th_dict,
            gds_space=gds_space,
            pg_tracks=[hm_width],
            pds_tracks=[2 * hm_cur_width + diff_space],
            min_fg_sep=min_fg_sep,
            guard_ring_nf=guard_ring_nf,
        )
        ng_tracks = []
        nds_tracks = []
        for row_name in ['tail', 'en', 'sw', 'in', 'casc']:
            if w_dict.get(row_name, -1) > 0:
                if row_name == 'in':
                    ng_tracks.append(2 * hm_width + diff_space)
                else:
                    ng_tracks.append(hm_width)
                nds_tracks.append(hm_cur_width + gds_space)
        draw_params['ng_tracks'] = ng_tracks
        draw_params['nds_tracks'] = nds_tracks

        self.draw_rows(**draw_params)


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
    cell_name = 'ANALOGBASE_TEST'

    params = specs['params']
    temp_db = make_tdb(prj, lib_name, specs)
    template = temp_db.new_template(params=params, temp_cls=AmpBase, debug=True)
    temp_db.instantiate_layout(prj, template, cell_name, debug=True)


if __name__ == '__main__':

    with open('test_specs/analogbase.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

        generate(bprj, block_specs)
    else:
        print('loading BAG project')
