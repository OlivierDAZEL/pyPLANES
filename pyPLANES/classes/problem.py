#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
#
# This file is part of pymls, a software distributed under the MIT license.
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

import numpy as np
from pyPLANES.classes.model import FemModel
from pyPLANES.gmsh.load_msh_file import load_msh_file
from pyPLANES.model.preprocess import init_vec_frequencies, create_lists, activate_dofs, desactivate_dofs_dimension, desactivate_dofs_BC, renumber_dofs, affect_dofs_to_elements, periodicity_initialisation, check_model, elementary_matrices
from pyPLANES.classes.entity_classes import PwFem

class Mesh():
    def __init__(self, **kwargs):
        self.entities = [] # List of all GMSH Entities
        self.model_entities = [] # List of Entities used in the Model
        self.vertices = []
        self.elements = []
        self.materials_directory = kwargs.get("materials_directory", "")
        self.reference_elements = dict() # dictionary of reference_elements
        load_msh_file(self, **kwargs)


class Calculus():
    def __init__(self, **kwargs):
        frequencies = kwargs.get("frequencies", np.array([440]))
        self.theta_d = kwargs.get("theta_d", 0.)
        self.name_project = kwargs.get("name_project", "unnamed_project")
        self.outfiles_directory = kwargs.get("outfiles_directory", False)


        self.frequencies = init_vec_frequencies(frequencies)
        self.out_file = self.name_project + "_out.txt"
        self.info_file = self.name_project + "_info.txt"

        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.A_i_c, self.A_j_c, self.A_v_c = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        self.modulus_reflex, self.modulus_trans, self.abs = 0, 0, 1


class FemProblem(Mesh, FemModel, Calculus):
    def __init__(self, **kwargs):
        self.name_server = platform.node()
        if self.name_server in ["oliviers-macbook-pro.home", "Oliviers-MacBook-Pro.local"]:
            self.verbose = True
        else:
            self.verbose = False
        self.verbose = kwargs.get("verbose", True)
        Mesh.__init__(self, **kwargs)

        self.order = kwargs.get("order", 5)
        FemModel.__init__(self, **kwargs)
        self.dim = 2
        Calculus.__init__(self, **kwargs)
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
        elementary_matrices(self)

