#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2020
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@kth.se>
# 	Peter Göransson <pege@kth.se>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#

import platform
import time

import numpy as np
from pyPLANES.classes.fem_model import FemModel
from pyPLANES.classes.mesh import FemMesh
from pyPLANES.classes.calculus import FemCalculus
from pyPLANES.gmsh.tools.load_msh_file import load_msh_file
from pyPLANES.model.preprocess import create_lists, activate_dofs, desactivate_dofs_dimension, desactivate_dofs_BC, renumber_dofs, affect_dofs_to_elements, periodicity_initialisation, check_model, elementary_matrices
from pyPLANES.classes.entity_classes import PwFem


class FemProblem(FemMesh, FemModel, FemCalculus):
    def __init__(self, **kwargs):
        self.name_server = platform.node()
        if self.name_server in ["oliviers-macbook-pro.home", "Oliviers-MacBook-Pro.local"]:
            self.verbose = True
        else:
            self.verbose = False
        self.verbose = kwargs.get("verbose", True)
        FemCalculus.__init__(self, **kwargs)
        self.initialisation_out_files()
        FemMesh.__init__(self, **kwargs)
        FemModel.__init__(self, **kwargs)

        self.order = kwargs.get("order", 2)

        self.dim = 2

        for _ent in self.model_entities:
            if isinstance(_ent, PwFem):
                _ent.theta_d = self.theta_d
        create_lists(self)
        activate_dofs(self)
        desactivate_dofs_dimension(self)
        desactivate_dofs_BC(self)
        renumber_dofs(self)
        affect_dofs_to_elements(self)
        periodicity_initialisation(self)
        check_model(self)
        self.duration_importation = time.time() - self.start_time
        self.info_file.write("Duration of importation ={} s\n".format(self.duration_importation))
        elementary_matrices(self)
        self.duration_assembly = time.time() - self.start_time - self.duration_importation
        self.info_file.write("Duration of assembly ={} s\n".format(self.duration_assembly))

    def resolution(self):
        return FemModel.resolution(self)

