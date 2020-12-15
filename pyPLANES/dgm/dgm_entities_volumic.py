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

from pyPLANES.generic.entities_generic import DgmEntity
from pyPLANES.fem.elements_volumic import fluid_elementary_matrices, pem98_elementary_matrices, pem01_elementary_matrices, elas_elementary_matrices

from pyPLANES.pw.utils_TM import weak_orth_terms
from pyPLANES.fem.utils_fem import dof_p_element, dof_u_element, dof_ux_element, dof_uy_element, orient_element
from pyPLANES.fem.utils_fem import dof_p_linear_system_to_condense, dof_p_linear_system_master, dof_up_linear_system_to_condense, dof_up_linear_system_master, dof_up_linear_system, dof_u_linear_system_master, dof_ux_linear_system_master, dof_uy_linear_system_master,dof_u_linear_system, dof_u_linear_system_to_condense

class FluidDgm(DgmEntity):
    def __init__(self, **kwargs):
        DgmEntity.__init__(self, **kwargs)
        self.mat = kwargs.get("mat", "Air")


    def __str__(self):
        out = "Fluid" + DgmEntity.__str__(self)
        return out

    def elementary_matrices(self, _el):
        # Elementary matrices
        H, Q = fluid_elementary_matrices(_el)
        orient_p = orient_element(_el)
        _el.H = orient_p @ H @ orient_p
        _el.Q = orient_p @ Q @ orient_p

        _el.dof_p_m = dof_p_linear_system_master(_el)
        _el.dof_p_c = dof_p_linear_system_to_condense(_el)

    def update_frequency(self, omega):
        self.mat.update_frequency(omega)

    def update_system(self, omega):
        A_i, A_j, A_v, T_i, T_j, T_v, F_i, F_v =[], [], [], [], [], [], [], []

        if self.mat.MEDIUM_TYPE == "eqf":
            rho = self.mat.rho_eq_til
            K = self.mat.K_eq_til
        else: 
            rho = self.mat.rho
            K = self.mat.K 

        for _el in self.elements:
            nb_m_SF = _el.reference_element.nb_m_SF
            nb_SF = _el.reference_element.nb_SF

            pp = _el.H/(rho*omega**2)- _el.Q/K

            l_p_m = slice(nb_m_SF)
            l_p_c = slice(nb_m_SF, nb_SF)

            mm = pp[l_p_m, l_p_m]
            cm = pp[l_p_c, l_p_m]
            mc = pp[l_p_m, l_p_c]
            cc = pp[l_p_c, l_p_c]

            t = -LA.inv(cc)@cm
            mm += mc@t

            T_i.extend(list(chain.from_iterable([[_d]*(nb_m_SF) for _d in _el.dof_p_c])))
            T_j.extend(list(_el.dof_p_m)*((nb_SF-nb_m_SF)))
            T_v.extend(t.flatten())

            A_i.extend(list(chain.from_iterable([[_d]*(nb_m_SF) for _d in _el.dof_p_m])))
            A_j.extend(list(_el.dof_p_m)*(nb_m_SF))
            A_v.extend(mm.flatten())
        return A_i, A_j, A_v, T_i, T_j, T_v, F_i, F_v
