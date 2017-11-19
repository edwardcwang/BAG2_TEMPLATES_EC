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

import pprint
from typing import Dict, Any, Set

from bag import BagProject
from bag.layout.routing import RoutingGrid, TrackID
from bag.layout.template import TemplateBase, TemplateDB


class WireTest(TemplateBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        super(WireTest, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict()

    def draw_layout(self):
        warr1 = self.add_wires(3, -0.5, 0, 0.5)
        warr2 = self.add_wires(3, 11.5, 0, 0.5)
        warr3 = self.add_wires(5, -0.5, 5.0, 6.0, width=4)

        res1 = self.connect_with_via_stack(warr1, TrackID(6, 5))
        res2 = self.connect_with_via_stack(warr2, TrackID(6, -5, width=3), tr_w_list=[2, 2, -1])
        res3 = self.connect_with_via_stack(warr3, TrackID(3, -0.5))

        pprint.pprint(res1)
        pprint.pprint(res2)
        pprint.pprint(res3)

        warr1 = self.add_wires(9, -0.5, -2, 2, width=2)
        self.connect_to_tracks(warr1, TrackID(10, 0, width=1), min_len_mode=0)
        self.connect_to_tracks(warr1, TrackID(10, 4, width=2), min_len_mode=0)
        self.connect_to_tracks(warr1, TrackID(10, 8, width=3), min_len_mode=0)
        self.connect_to_tracks(warr1, TrackID(10, 12, width=4), min_len_mode=0)
        self.connect_to_tracks(warr1, TrackID(10, 20, width=5), min_len_mode=0)


def make_tdb(prj, target_lib):
    layers = [3, 4, 5, 6, 7, 8, 9, 10]
    spaces = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4]
    widths = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4]
    bot_dir = 'y'

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate(prj):
    lib_name = 'AAAFOO_WIRETEST'

    temp_db = make_tdb(prj, lib_name)
    name_list, temp_list = [], []
    name_list.append('ANALOGBASE_TEST')
    temp_list.append(temp_db.new_template(params={}, temp_cls=WireTest))

    print('creating layouts')
    temp_db.batch_layout(prj, temp_list, name_list)
    print('layout done.')


if __name__ == '__main__':

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate(bprj)
