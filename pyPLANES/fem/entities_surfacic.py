#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# entity_classes.py
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

import numpy as np
import numpy.linalg as LA
from numpy import pi
from itertools import chain

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from mediapack import Air


from pyPLANES.fem.entities_plain import FemEntity
from pyPLANES.fem.elements_surfacic import imposed_pw_elementary_vector, fsi_elementary_matrix, fsi_elementary_matrix_incompatible
from pyPLANES.pw.utils_TM import weak_orth_terms
from pyPLANES.fem.utils_fem import dof_p_element, dof_u_element, dof_ux_element, dof_uy_element, orient_element
from pyPLANES.fem.utils_fem import dof_p_linear_system_to_condense, dof_p_linear_system_master, dof_up_linear_system_to_condense, dof_up_linear_system_master, dof_up_linear_system, dof_u_linear_system_master, dof_ux_linear_system_master, dof_uy_linear_system_master,dof_u_linear_system, dof_u_linear_system_to_condense

class InterfaceFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.ml = kwargs.get("ml", False)
        self.side = kwargs.get("side", False)
        self.neighbour = False # Neighbouring interface
        self.nodes = None # Bounding nodes
        self.delta = None # vector of geometrical shift with the other interface


    def __str__(self):
        out = "Interface" + FemEntity.__str__(self)
        out += "Neighbour Entity tag={}\n".format(self.neighbour.tag)
        return out

    def elementary_matrices(self, _el):
        M = fsi_elementary_matrix_incompatible(_el)
        orient_ = orient_element(_el)
        for i, neigh in enumerate(_el.neighbours):
            orient__ = orient_element((neigh._elem))
            M[i] = orient_ @ M[i] @ orient__


    def create_dynamical_matrices(self, omega, n_m):
        phi = coo_matrix((n_m, self.nb_dofs), dtype=complex)
        self.eta_TM = coo_matrix((self.nb_dofs, self.nb_dofs), dtype=complex)
        tau = coo_matrix((self.nb_dofs, self.nb_dofs), dtype=complex)
        Omega_0 = np.array([-1j*self.ky[0]/(Air.rho*omega**2), 1]).reshape((2, 1))
        Omega_0_weak, self.Omega_0_orth = weak_orth_terms(omega, self.kx[0], Omega_0, self.ml, self.typ)
        self.Omega_0_orth = self.Omega_0_orth[:, 0]
        Omega_0_weak = Omega_0_weak[:, 0]
        qdsqsddsq


class FluidStructureFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.fluid_neighbour = None
        self.struc_neighbour = None

    def elementary_matrices(self, _el):
        # Elementary matrices
        M = fsi_elementary_matrix(_el)
        orient_ = orient_element(_el)

        _el.M = orient_ @ M @ orient_

    def __str__(self):
        out = "FluidStructure" + FemEntity.__str__(self)
        return out

    def append_linear_system(self, omega):
        A_i, A_j, A_v =[], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []
        for _el in self.elements:
            dof_ux = dof_ux_linear_system_master(_el)
            dof_uy = dof_uy_linear_system_master(_el)
            dof_p = dof_p_linear_system_master(_el)

            v = (_el.M).flatten()

            if _el.normal_fluid[0] != 0:
                A_i.extend(list(chain.from_iterable([[_d]*len(dof_ux) for _d in dof_p])))
                A_j.extend(list(dof_ux)*len(dof_p))
                A_v.extend(-_el.normal_fluid[0]*v)

            if _el.normal_fluid[1] != 0:
                A_i.extend(list(chain.from_iterable([[_d]*len(dof_uy) for _d in dof_p])))
                A_j.extend(list(dof_uy)*len(dof_p))
                A_v.extend(-_el.normal_fluid[1]*v)
            if _el.normal_struc[0] != 0:
                A_i.extend(list(chain.from_iterable([[_d]*len(dof_p) for _d in dof_ux])))
                A_j.extend(list(dof_p)*len(dof_ux))
                A_v.extend(_el.normal_struc[0]*v)
            if _el.normal_struc[1] != 0:
                A_i.extend(list(chain.from_iterable([[_d]*len(dof_p) for _d in dof_uy])))
                A_j.extend(list(dof_p)*len(dof_uy))
                A_v.extend(_el.normal_struc[1]*v)

        return A_i, A_j, A_v, T_i, T_j, T_v

class RigidWallFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "RigidWall" + FemEntity.__str__(self)
        return out

class PeriodicityFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Periodicity" + FemEntity.__str__(self)
        return out

