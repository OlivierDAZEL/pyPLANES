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


from itertools import product
from scipy.sparse import csr_matrix

from mediapack import Air

from pyPLANES.fem.elements.volumic_elements import fluid_elementary_matrices, pem98_elementary_matrices
from pyPLANES.fem.elements.surfacic_elements import imposed_pw_elementary_vector
from pyPLANES.utils.utils_fem import dof_p_element, dof_u_element
from pyPLANES.utils.utils_fem import dof_p_linear_system_to_condense, dof_p_linear_system_master, dof_up_linear_system_to_condense, dof_up_linear_system_master, dof_up_linear_system


class GmshEntity():
    def __init__(self, **kwargs):
        self.dim = kwargs["dim"]
        self.tag = kwargs["tag"]
        self.physical_tags = kwargs["physical_tags"]
        if "condition" not in list(self.physical_tags.keys()):
            self.physical_tags["condition"] = None
        if "model" not in list(self.physical_tags.keys()):
            self.physical_tags["model"] = None
        if self.dim == 2:
            self.bounding_curves = kwargs["bounding_curves"]
        elif self.dim == 1:
            self.bounding_points = kwargs["bounding_points"]
        elif self.dim == 0:
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.z = kwargs["z"]
    def __str__(self):
        out = "Entity / tag={} / dim= {}\n".format(self.tag, self.dim)
        out += "Physical tags={}\n".format(self.physical_tags)
        return out

class FemEntity(GmshEntity):
    def __init__(self, **kwargs):
        GmshEntity.__init__(self, **kwargs)
        self.order = kwargs["p"].order
        self.elements = []
    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Fem" + GmshEntity.__str__(self)
        out += "order:{}\n".format(self.order)
        related_elements = [_el.tag for _el in self.elements]
        out  += "related elements={}\n".format(related_elements)
        return out

    def condensation(self, omega):
        return [], [], [], [], [], []

    def update_frequency(self, omega):
        pass

    def create_elementary_matrices(self, _elem):
        pass

    def append_linear_system(self, omega):
        return [], [], [], [], [], []

    def link_elem(self,n):
        self.elements.append(n)

class AirFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = Air()
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []

    def __str__(self):
        out = "Air" + FemEntity.__str__(self)
        return out

    def append_global_matrices(self, _elem):
        # Elementary matrices
        H, Q = fluid_elementary_matrices(_elem)
        dof_p, orient_p, elem_dof = dof_p_element(_elem)
        nb_dof = len(dof_p)
        # Orientation of the elementary matrices
        H = orient_p @ H @ orient_p
        Q = orient_p @ Q @ orient_p

        dof_m, dof_c = elem_dof["dof_m"], elem_dof["dof_c"]

        _elem.H_mm = H[elem_dof["dof_m"], elem_dof["dof_m"]]
        _elem.H_cm = H[elem_dof["dof_c"], elem_dof["dof_m"]]
        _elem.H_mc = H[elem_dof["dof_m"], elem_dof["dof_c"]]
        _elem.H_cc = H[elem_dof["dof_c"], elem_dof["dof_c"]]
        _elem.Q_mm = Q[elem_dof["dof_m"], elem_dof["dof_m"]]
        _elem.Q_cm = Q[elem_dof["dof_c"], elem_dof["dof_m"]]
        _elem.Q_mc = Q[elem_dof["dof_m"], elem_dof["dof_c"]]
        _elem.Q_cc = Q[elem_dof["dof_c"], elem_dof["dof_c"]]

        # self.H_i.extend(list(chain.from_iterable([[_d]*nb_dof for _d in dof_p])))
        # self.H_j.extend(list(dof_p)*nb_dof)
        # self.H_v.extend(list(H[dof_m, dof_m].flatten()))
        # self.Q_i.extend(list(chain.from_iterable([[_d]*nb_dof for _d in dof_p])))
        # self.Q_j.extend(list(dof_p)*nb_dof)
        # self.Q_v.extend(list(Q[dof_m, dof_m].flatten()))


    def append_linear_system(self, omega):
        A_i, A_j, A_v =[], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []
        for _e in self.elements:
            Mat = _e.H_mm/(self.mat.rho*omega**2) - _e.Q_mm/self.mat.K
            dof_master = dof_p_linear_system_master(_e)
            dof_condense = dof_p_linear_system_to_condense(_e)
            n_m, n_c = len(dof_master), len(dof_condense)
            Di = LA.inv((_e.H_cc/(self.mat.rho*omega**2))-(_e.Q_cc/(self.mat.K)))
            CC = (_e.H_cm/(self.mat.rho*omega**2))-(_e.Q_cm/(self.mat.K)).reshape((n_c, n_m))
            BB = (_e.H_mc/(self.mat.rho*omega**2))-(_e.Q_mc/(self.mat.K)).reshape((n_m, n_c))
            T_i.extend(list(chain.from_iterable([[_d]*n_m for _d in dof_condense])))
            T_j.extend(list(dof_master)*n_c)
            _ = np.array(-Di.dot(CC))
            T_v.extend(_.flatten())
            A_i.extend(list(chain.from_iterable([[_d]*n_m for _d in dof_master])))
            A_j.extend(list(dof_master)*n_m)
            _ = Mat-np.array(BB.dot(_))
            A_v.extend(_.flatten())
        return A_i, A_j, A_v, T_i, T_j, T_v

class Pem98Fem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
        self.K_0_i, self.K_0_j, self.K_0_v = [], [], []
        self.K_1_i, self.K_1_j, self.K_1_v = [], [], []
        self.M_i, self.M_j, self.M_v = [], [], []
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []
        self.C_i, self.C_j, self.C_v = [], [], []

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Pem98" + FemEntity.__str__(self)
        return out

    def update_frequency(self, omega):
        self.mat.update_frequency(omega)

    def create_elementary_matrices(self, _elem):
        dof_u, orient_u, elem_dof_u = dof_u_element(_elem)
        dof_p, orient_p, elem_dof_p = dof_p_element(_elem)
        M, K_0, K_1, H, Q, C = pem98_elementary_matrices(_elem)

        _elem.M =   orient_u @ M   @ orient_u
        _elem.K_0 = orient_u @ K_0 @ orient_u
        _elem.K_1 = orient_u @ K_1 @ orient_u
        _elem.H =   orient_p @ H   @ orient_p
        _elem.Q =   orient_p @ Q   @ orient_p
        _elem.C =   orient_u @ C   @ orient_p
        dof_up_m = dof_up_linear_system_master(_elem)
        dof_up_c = dof_up_linear_system_to_condense(_elem)

        dof_up = dof_up_linear_system(_elem)

        n_m, n_c = len(dof_up_m), len(dof_up_c)
        n_u_m, n_p_m = int(2*n_m/3), int(n_m/3)
        n_u_c, n_p_c = int(2*n_c/3), int(n_c/3)
        n_p = n_p_m +n_p_c
        n_up = n_m + n_c
        # print(_e.K_0.shape)
        # print("n_u_m={}".format(n_u_m))
    # Renumbering of the elementary matrices to separate master and condensed dofs
        # new_order = list(range(n_p_m))
        # new_order += [d+n_p for d in new_order]
        # new_order += list(range(n_p_m, n_p)) +list(range(n_p+n_p_m,2*n_p))
        # _elem.M = _elem.M[:, new_order][new_order]
        # _elem.K_0 = _elem.K_0[:, new_order][new_order]
        # _elem.K_1 = _elem.K_1[:, new_order][new_order]
        # _elem.C = _elem.C[:, :][new_order]
        print(np.diag(_elem.M))



    def append_linear_system(self, omega):

        A_i, A_j, A_v =[], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []
        for _e in self.elements:

            dof_up_m = dof_up_linear_system_master(_e)
            dof_up_c = dof_up_linear_system_to_condense(_e)

            dof_up = dof_up_linear_system(_e)

            n_m, n_c = len(dof_up_m), len(dof_up_c)
            n_u_m, n_p_m = int(2*n_m/3), int(n_m/3)
            n_u_c, n_p_c = int(2*n_c/3), int(n_c/3)
            n_p = n_p_m +n_p_c
            n_up = n_m + n_c

            l_u_m = slice(n_u_m)
            l_p_m = slice(n_p_m)
            l_u_c = slice(n_u_m, n_u_m+n_u_c)
            l_p_c = slice(n_p_m, n_p_m+n_p_c)

            # Dynamic matrices
            uu = self.mat.P_hat*_e.K_0+self.mat.N*_e.K_1-omega**2*self.mat.rho_til*_e.M
            up = -self.mat.gamma_til*_e.C
            pu = -self.mat.gamma_til*(_e.C.T)
            pp = _e.H/(self.mat.rho_eq_til*omega**2)- _e.Q/(self.mat.K_eq_til)

            mm = np.block([[uu[l_u_m, l_u_m], up[l_u_m, l_p_m]], [pu[l_p_m, l_u_m], pp[l_p_m, l_p_m]]])
            cm = np.block([[uu[l_u_c, l_u_m], up[l_u_c, l_p_m]], [pu[l_p_c, l_u_m], pp[l_p_c, l_p_m]]])
            mc = np.block([[uu[l_u_m, l_u_c], up[l_u_m, l_p_c]], [pu[l_p_m, l_u_c], pp[l_p_m, l_p_c]]])
            cc = np.block([[uu[l_u_c, l_u_c], up[l_u_c, l_p_c]], [pu[l_p_c, l_u_c], pp[l_p_c, l_p_c]]])

            t = -LA.inv(cc)@cm
            mm += mc@t

            T_i.extend(list(chain.from_iterable([[_d]*n_m for _d in dof_up_c])))
            T_j.extend(list(dof_up_m)*n_c)
            T_v.extend(t.flatten())

            # A_i.extend(list(chain.from_iterable([[_d]*n_m for _d in dof_up_m])))
            # A_j.extend(list(dof_up_m)*n_m)
            # A_v.extend(mm.flatten())

            m = np.block([[uu, up],[pu, pp]])

            A_i.extend(list(chain.from_iterable([[_d]*n_up for _d in dof_up])))
            A_j.extend(list(dof_up)*n_up)
            A_v.extend(m.flatten())


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

class EquivalentFluidFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "EquivalentFluid" + FemEntity.__str__(self)
        return out

    def update_frequency(self,omega):
        self.mat.update_frequency(omega)
    def append_global_matrices(self, _elem, Reference_Element):
        H, Q = fluid_elem(_elem.coorde, Reference_Element)
        dof_p = dof_pr_triangle(_elem)
        orient_p = orient_triangle(_elem)
        for ii, jj in product(range(len(dof_p)), range(len(dof_p))):
            self.H_i.append(dof_p[ii])
            self.H_j.append(dof_p[jj])
            self.H_v.append(orient_p[ii]*orient_p[jj]*H[ii, jj])
            self.Q_i.append(dof_p[ii])
            self.Q_j.append(dof_p[jj])
            self.Q_v.append(orient_p[ii]*orient_p[jj]*Q[ii, jj])
    def append_linear_system(self,p,_elem=None):
        A_i = self.H_i.copy()
        A_j = self.H_j.copy()
        A_v = np.array(self.H_v)/(self.mat.rho*p.omega**2)
        A_i.extend(self.Q_i)
        A_j.extend(self.Q_j)
        A_v.extend(-np.array(self.Q_v)/(self.mat.K))
        return A_i, A_j, A_v

class Pem01Fem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
        self.K_0_i, self.K_0_j, self.K_0_v = [], [], []
        self.K_1_i, self.K_1_j, self.K_1_v = [], [], []
        self.M_i, self.M_j, self.M_v = [], [], []
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []
        self.C_i, self.C_j, self.C_v = [], [], []

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Pem01" + FemEntity.__str__(self)
        return out

    def update_frequency(self,omega):
        self.mat.update_frequency(omega)
    def append_global_matrices(self, _elem, Reference_Element):
        dof_u = dof_u_triangle(_elem)
        dof_p = dof_pr_triangle(_elem)
        orient_p = orient_triangle(_elem)
        orient_u = 3*orient_p
        M, K_0, K_1, H, Q, C = pem01_elem(_elem.coorde, Reference_Element)
        for ii, jj in product(range(len(dof_u)), range(len(dof_u))):
            self.K_0_i.append(dof_u[ii])
            self.K_0_j.append(dof_u[jj])
            self.K_0_v.append(orient_u[ii]*orient_u[jj]*K_0[ii, jj])
            self.K_1_i.append(dof_u[ii])
            self.K_1_j.append(dof_u[jj])
            self.K_1_v.append(orient_u[ii]*orient_u[jj]*K_1[ii, jj])
            self.M_i.append(dof_u[ii])
            self.M_j.append(dof_u[jj])
            self.M_v.append(orient_u[ii]*orient_u[jj]*M[ii, jj])
        for ii, jj in product(range(len(dof_p)), range(len(dof_p))):
            self.H_i.append(dof_p[ii])
            self.H_j.append(dof_p[jj])
            self.H_v.append(orient_p[ii]*orient_p[jj]*H[ii, jj])
            self.Q_i.append(dof_p[ii])
            self.Q_j.append(dof_p[jj])
            self.Q_v.append(orient_p[ii]*orient_p[jj]*Q[ii, jj])
        for ii, jj in product(range(len(dof_u)), range(len(dof_p))):
            self.C_i.append(dof_u[ii])
            self.C_j.append(dof_p[jj])
            self.C_v.append(orient_u[ii]*orient_p[jj]*C[ii, jj])
    def append_linear_system(self, omega):
        A_i = self.K_0_i.copy()
        A_j = self.K_0_j.copy()
        A_v = list(self.mat.P_hat*np.array(self.K_0_v))
        A_i.extend(self.K_1_i)
        A_j.extend(self.K_1_j)
        A_v.extend(self.mat.N*np.array(self.K_1_v))
        A_i.extend(self.M_i)
        A_j.extend(self.M_j)
        A_v.extend(-omega**2*self.mat.rho_til*np.array(self.M_v))
        A_i.extend(self.H_i)
        A_j.extend(self.H_j)
        A_v.extend(np.array(self.H_v)/(self.mat.rho_eq_til*omega**2))
        A_i.extend(self.Q_i)
        A_j.extend(self.Q_j)
        A_v.extend(-np.array(self.Q_v)/(self.mat.K_eq_til))
        A_i.extend(self.C_i)
        A_j.extend(self.C_j)
        A_v.extend(-self.mat.gamma_til*np.array(self.C_v))
        A_i.extend(self.C_j)
        A_j.extend(self.C_i)
        A_v.extend(-self.mat.gamma_til*np.array(self.C_v))
        return A_i, A_j, A_v

class ElasticSolidFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.K_0_i, self.K_0_j, self.K_0_v = [], [], []
        self.K_1_i, self.K_1_j, self.K_1_v = [], [], []
        self.M_i, self.M_j, self.M_v = [], [], []

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "ElaticSolid" + FemEntity.__str__(self)
        return out

class UnitDisplacementFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.dim = 1
        self.mat = "Unit_displacement"
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "UnitDisplacement" + FemEntity.__str__(self)
        return out

    def append_global_matrices(self, _elem, Reference_Element):
        F = imposed_Neumann(_elem.coorde, self.Ref_Elem[(_elem.geo, _elem.order)])
        dof_p = _elem.dofs[3][0:2]+_elem.dofs[3][2]
        _orient = [1]*2 # Orientation of the vertices
        for k in range(_elem.order-1): # Orientation of the edges
            _orient.append(_elem.edges_orientation[0]**k)
        for ii, _dof in enumerate(dof_p):
            self.F_i.append(_dof)
            self.F_v.append(_orient[ii]*F[ii])

class PwFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []
        self.dofs = []
        self.theta_d = kwargs["p"].theta_d
        # self.period = kwargs["p"].period
        self.kx, self.ky = [],[]
        self.rho_i, self.rho_j, self.rho_v = [], [], []
        self.nb_dofs = 0
        self.Omega = None

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Pw" + FemEntity.__str__(self)
        return out

    def update_frequency(self, omega):
        k_air = omega/Air.c
        k_x = k_air*np.sin(self.theta_d*np.pi/180.)
        nb_bloch_waves = int(np.ceil((self.period/(2*pi))*(3*np.real(k_air)-k_x))+10)
        nb_bloch_waves = 3
        # print("nb_bloch_waves ={}".format(nb_bloch_waves))
        _ = np.arange(-nb_bloch_waves, nb_bloch_waves+1)
        self.kx = k_x+_*(2*pi/self.period)
        k_y = np.sqrt(k_air**2-self.kx**2+0*1j)
        self.ky = np.real(k_y)-1j*np.imag(k_y)
        self.dofs = np.arange(1+2*nb_bloch_waves)
        self.nb_dofs = 1+2*nb_bloch_waves

    def create_dynamical_matrices(self, omega):
        self.rho_i, self.rho_j, self.rho_v = [], [], []
        _ = np.diag(1j*self.ky/(Air.rho*omega**2))
        # print(self.nb_dofs)
        self.Omega = csr_matrix(_, shape=(self.nb_dofs, self.nb_dofs), dtype=complex)
        for i_w, kx in enumerate(self.kx):
            for _elem in self.elements:
                F = imposed_pw_elementary_vector(_elem, kx)
                dof_FEM, orient, _ = dof_p_element(_elem)
                dof_pw = [self.dofs[i_w]]*len(dof_FEM)
                _ = orient@F
                self.rho_i.extend([d-1 for d in dof_FEM])
                self.rho_j.extend(dof_pw)
                self.rho_v.extend(_)

class IncidentPwFem(PwFem):
    def __init__(self, **kwargs):
        PwFem.__init__(self, **kwargs)
        self.dof_spec = None

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Imposed" + PwFem.__str__(self)
        return out

    def update_frequency(self, omega):
        PwFem.update_frequency(self, omega)
        self.dof_spec = int((self.nb_dofs-1)/2)

class TransmissionPwFem(PwFem):
    def __init__(self, **kwargs):
        PwFem.__init__(self, **kwargs)

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Imposed" + PwFem.__str__(self)
        return out



if __name__ == "__main__":
    pass