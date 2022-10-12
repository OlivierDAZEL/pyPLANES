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

from pyPLANES.generic.entities_generic import FemEntity
from pyPLANES.fem.elements_volumic import fluid_elementary_matrices, pem98_elementary_matrices, pem01_elementary_matrices, elas_elementary_matrices

from pyPLANES.pw.utils_TM import weak_orth_terms
from pyPLANES.fem.utils_fem import dof_p_element, dof_u_element, dof_ux_element, dof_uy_element, orient_element
from pyPLANES.fem.utils_fem import dof_p_linear_system_to_condense, dof_p_linear_system_master, dof_up_linear_system_to_condense, dof_up_linear_system_master, dof_up_linear_system, dof_u_linear_system_master, dof_ux_linear_system_master, dof_uy_linear_system_master,dof_u_linear_system, dof_u_linear_system_to_condense

class FluidFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs.get("mat", "Air")
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []

    def __str__(self):
        out = "Fluid" + FemEntity.__str__(self)
        return out

    def elementary_matrices(self, _el):
        # Elementary matrices
        H, Q = fluid_elementary_matrices(_el)
        
        orient_p = orient_element(_el)
        _el.H = (orient_p @ H) @ orient_p
        _el.Q = (orient_p @ Q) @ orient_p

        _el.dof_p_m = dof_p_linear_system_master(_el)
        _el.dof_p_c = dof_p_linear_system_to_condense(_el)
        _el.dof_p = np.append(_el.dof_p_m, _el.dof_p_c)

    def update_frequency(self, omega):
        self.mat.update_frequency(omega)

    def update_Q(self):
        A_i, A_j, A_v, T_i, T_j, T_v, F_i, F_v =[], [], [], [], [], [], [], []
        for _el in self.elements:
            nb_m_SF = _el.reference_element.nb_m_SF
            nb_SF = _el.reference_element.nb_SF
            pp = _el.Q

            A_i.extend(list(chain.from_iterable([[_d]*(nb_SF) for _d in _el.dof_p])))
            A_j.extend(list(_el.dof_p)*(nb_SF))
            A_v.extend(pp.flatten())

        return A_i, A_j, A_v

    def update_H(self):
        A_i, A_j, A_v, T_i, T_j, T_v, F_i, F_v =[], [], [], [], [], [], [], []
        for _el in self.elements:
            nb_m_SF = _el.reference_element.nb_m_SF
            nb_SF = _el.reference_element.nb_SF
            pp = _el.H

            A_i.extend(list(chain.from_iterable([[_d]*(nb_SF) for _d in _el.dof_p])))
            A_j.extend(list(_el.dof_p)*(nb_SF))
            A_v.extend(pp.flatten())

        return A_i, A_j, A_v

    def update_system(self, omega):
        A_i, A_j, A_v, = [], [], []
        if self.condensation == True:
            T_i, T_j, T_v = [], [], []
        if self.mat.MEDIUM_TYPE == "eqf":
            rho = self.mat.rho_eq_til
            K = self.mat.K_eq_til
        else: 
            rho = self.mat.rho
            K = self.mat.K 


        if self.condensation:
            for _el in self.elements:
                pp = _el.H/(rho*omega**2)- _el.Q/K
                nb_SF = _el.reference_element.nb_SF
                nb_m_SF = _el.reference_element.nb_m_SF
                l_p_m = slice(nb_m_SF)
                l_p_c = slice(nb_m_SF, nb_SF)

                mm = pp[l_p_m, l_p_m]
                cm = pp[l_p_c, l_p_m]
                mc = pp[l_p_m, l_p_c]
                cc = pp[l_p_c, l_p_c]

                t = -LA.solve(cc, cm)
                mm += mc@t

                T_i.extend(list(chain.from_iterable([[_d]*(nb_m_SF) for _d in _el.dof_p_c])))
                T_j.extend(list(_el.dof_p_m)*((nb_SF-nb_m_SF)))
                T_v.extend(t.flatten())

                A_i.extend(list(chain.from_iterable([[_d]*(nb_m_SF) for _d in _el.dof_p_m])))
                A_j.extend(list(_el.dof_p_m)*(nb_m_SF))
                A_v.extend(mm.T.flatten())
            return A_i, A_j, A_v, [], [], T_i, T_j, T_v

        else:
            for _el in self.elements:
                pp = _el.H/(rho*omega**2)- _el.Q/K
                nb_SF = pp.shape[0]

                A_i.extend(list(chain.from_iterable([[_d]*(nb_SF) for _d in _el.dof_p])))
                A_j.extend(list(_el.dof_p)*(nb_SF))
                A_v.extend(pp.flatten())

            
            return A_i, A_j, A_v, [], []

class PemFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
        self.condensation = kwargs.get("condensation", True)
        self.formulation98 = None
    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Pem" + FemEntity.__str__(self)
        out += "formulation98 = {}\n".format(self.formulation98)
        return out

    def update_frequency(self, omega):
        self.mat.update_frequency(omega)

    def elementary_matrices(self, _el):
        orient_p = orient_element(_el)
        orient_u = orient_element(_el, "u")
        if self.formulation98:
            M, K_0, K_1, H, Q, C = pem98_elementary_matrices(_el)
        else:
            M, K_0, K_1, H, Q, C, C2 = pem01_elementary_matrices(_el)
        # Orientation of the matrices elementary matrices
        _el.M = orient_u @ M   @ orient_u
        _el.K_0 = orient_u @ K_0 @ orient_u
        _el.K_1 = orient_u @ K_1 @ orient_u
        _el.H = orient_p @ H   @ orient_p
        _el.Q = orient_p @ Q   @ orient_p
        _el.C = orient_u @ C   @ orient_p
        if not self.formulation98:
            _el.C2 = orient_u @ C2   @ orient_p
        # Separate of master and condensed dofs
        nb_m_SF = _el.reference_element.nb_m_SF
        nb_SF = _el.reference_element.nb_SF
        _ = list(range(nb_m_SF))
        _ += [d+nb_SF for d in _]
        _ += list(range(nb_m_SF, nb_SF)) +list(range(nb_SF+nb_m_SF, 2*nb_SF))
        # Renumbering of the elementary matrices
        _el.M = _el.M[:, _][_]
        _el.K_0 = _el.K_0[:, _][_]
        _el.K_1 = _el.K_1[:, _][_]
        _el.C = _el.C[:, :][_]
        if not self.formulation98:
            _el.C2 = _el.C2[:, :][_]

    def update_system(self, omega):
        A_i, A_j, A_v, T_i, T_j, T_v, =[], [], [], [], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []
        for _el in self.elements:
            nb_m_SF = _el.reference_element.nb_m_SF
            nb_SF = _el.reference_element.nb_SF
            # With condensation
            # Based on reordered matrix
            uu = self.mat.P_hat*_el.K_0+self.mat.N*_el.K_1-omega**2*self.mat.rho_til*_el.M
            if self.formulation98:
                up = -self.mat.gamma_til*_el.C
                pu = -self.mat.gamma_til*(_el.C.T)
            else:
                up = -(self.mat.gamma_til+1)*_el.C-_el.C2
                pu = -(self.mat.gamma_til+1)*(_el.C.T)-_el.C2.T
            pp = _el.H/(self.mat.rho_eq_til*omega**2)-_el.Q/(self.mat.K_eq_til)

            l_u_m = slice(2*nb_m_SF)
            l_p_m = slice(nb_m_SF)
            l_u_c = slice(2*nb_m_SF, 2*nb_SF)
            l_p_c = slice(nb_m_SF, nb_SF)

            mm = np.block([[uu[l_u_m, l_u_m], up[l_u_m, l_p_m]], [pu[l_p_m, l_u_m], pp[l_p_m, l_p_m]]])
            cm = np.block([[uu[l_u_c, l_u_m], up[l_u_c, l_p_m]], [pu[l_p_c, l_u_m], pp[l_p_c, l_p_m]]])
            mc = np.block([[uu[l_u_m, l_u_c], up[l_u_m, l_p_c]], [pu[l_p_m, l_u_c], pp[l_p_m, l_p_c]]])
            cc = np.block([[uu[l_u_c, l_u_c], up[l_u_c, l_p_c]], [pu[l_p_c, l_u_c], pp[l_p_c, l_p_c]]])

            t = -LA.solve(cc, cm)
            mm += mc@t

            dof_up_m = dof_up_linear_system_master(_el)
            dof_up_c = dof_up_linear_system_to_condense(_el)

            T_i.extend(list(chain.from_iterable([[_d]*(3*nb_m_SF) for _d in dof_up_c])))
            T_j.extend(list(dof_up_m)*(3*(nb_SF-nb_m_SF)))
            T_v.extend(t.flatten())

            A_i.extend(list(chain.from_iterable([[_d]*(3*nb_m_SF) for _d in dof_up_m])))
            A_j.extend(list(dof_up_m)*(3*nb_m_SF))
            A_v.extend(mm.flatten())

        return A_i, A_j, A_v, [], [], T_i, T_j, T_v

class ElasticFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
        self.condensation = kwargs.get("condensation", True)

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Elastic" + FemEntity.__str__(self)
        return out

    def update_frequency(self, omega):
        self.mat.update_frequency(omega)

    def elementary_matrices(self, _el):
        orient_u = orient_element(_el, "u")
        M, K_0, K_1 = elas_elementary_matrices(_el)
        # Orientation of the matrices elementary matrices
        _el.M = orient_u @ M   @ orient_u
        _el.K_0 = orient_u @ K_0 @ orient_u
        _el.K_1 = orient_u @ K_1 @ orient_u
        # Renumbering of the elementary matrices to separate master and condensed dofs
        nb_m_SF = _el.reference_element.nb_m_SF
        nb_SF = _el.reference_element.nb_SF
        _ = list(range(nb_m_SF))
        _ += [d+nb_SF for d in _]
        _ += list(range(nb_m_SF, nb_SF)) +list(range(nb_SF+nb_m_SF, 2*nb_SF))
        _el.M = _el.M[:, _][_]
        _el.K_0 = _el.K_0[:, _][_]
        _el.K_1 = _el.K_1[:, _][_]

    def update_system(self, omega):
        A_i, A_j, A_v, T_i, T_j, T_v, F_i, F_v =[], [], [], [], [], [], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []
        for _el in self.elements:
            nb_m_SF = _el.reference_element.nb_m_SF
            nb_SF = _el.reference_element.nb_SF
            # With condensation
            # Based on reordered matrix
            uu = (self.mat.lambda_+2*self.mat.mu)*_el.K_0+self.mat.mu*_el.K_1 - omega**2*self.mat.rho*_el.M

            l_u_m = slice(2*nb_m_SF)
            l_u_c = slice(2*nb_m_SF, 2*nb_SF)

            mm = uu[l_u_m, l_u_m]
            cm = uu[l_u_c, l_u_m]
            mc = uu[l_u_m, l_u_c]
            cc = uu[l_u_c, l_u_c]

            t = -LA.solve(cc,cm)
            mm += mc@t

            dof_u_m = dof_u_linear_system_master(_el)
            dof_u_c = dof_u_linear_system_to_condense(_el)

            T_i.extend(list(chain.from_iterable([[_d]*(2*nb_m_SF) for _d in dof_u_c])))
            T_j.extend(list(dof_u_m)*(2*(nb_SF-nb_m_SF)))
            T_v.extend(t.flatten())

            A_i.extend(list(chain.from_iterable([[_d]*(2*nb_m_SF) for _d in dof_u_m])))
            A_j.extend(list(dof_u_m)*(2*nb_m_SF))
            A_v.extend(mm.flatten())

        return A_i, A_j, A_v, F_i, F_v, T_i, T_j, T_v
