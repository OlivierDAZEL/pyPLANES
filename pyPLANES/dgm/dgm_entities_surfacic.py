#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# entity_classes.py
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

import numpy as np
import numpy.linalg as LA
from numpy import pi
from itertools import chain

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from mediapack import Air
Air = Air()

from pyPLANES.generic.entities_generic import DgmEntity
from pyPLANES.fem.elements_surfacic import imposed_pw_elementary_vector, fsi_elementary_matrix, fsi_elementary_matrix_incompatible, imposed_Neumann
from pyPLANES.pw.utils_TM import weak_orth_terms
from pyPLANES.fem.utils_fem import dof_p_element, dof_u_element, dof_ux_element, dof_uy_element, orient_element
from pyPLANES.fem.utils_fem import dof_p_linear_system_to_condense, dof_p_linear_system_master, dof_up_linear_system_to_condense, dof_up_linear_system_master, dof_up_linear_system, dof_u_linear_system_master, dof_ux_linear_system_master, dof_uy_linear_system_master,dof_u_linear_system, dof_u_linear_system_to_condense
from pyPLANES.pw.utils_TM import ZOD_terms


class ImposedDisplacementDgm(DgmEntity):
    def __init__(self, **kwargs):
        DgmEntity.__init__(self, **kwargs)
    
    def __str__(self):
        out = "Imposed Displacement"
        return out




class RigidWallDgm(DgmEntity):
    def __init__(self, **kwargs):
        DgmEntity.__init__(self, **kwargs)

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "RigidWall" + DgmEntity.__str__(self)
        return out

