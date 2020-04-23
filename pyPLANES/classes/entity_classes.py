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
from numpy import pi
from itertools import product

from mediapack import Air

from pyPLANES.fem.elements.volumic_elements import fluid_elementary_matrices, pem98_elementary_matrices
from pyPLANES.fem.elements.surfacic_elements import imposed_pw_elementary_vector
from pyPLANES.utils.utils_fem import dof_p_element, dof_u_element

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

    def update_frequency(self, omega):
        pass

    def append_global_matrices(self, _elem):
        pass

    def append_linear_system(self, omega):
        pass

    def link_elem(self,n):
        self.elements.append(n)

class AirFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = Air()
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Air" + FemEntity.__str__(self)
        return out

    def append_global_matrices(self, _elem):
        H, Q = fluid_elementary_matrices(_elem)
        dof_p, orient_p = dof_p_element(_elem)
        for ii, jj in product(range(len(dof_p)), range(len(dof_p))):
            self.H_i.append(dof_p[ii])
            self.H_j.append(dof_p[jj])
            self.H_v.append(orient_p[ii]*orient_p[jj]*H[ii, jj])
            self.Q_i.append(dof_p[ii])
            self.Q_j.append(dof_p[jj])
            self.Q_v.append(orient_p[ii]*orient_p[jj]*Q[ii, jj])
    def append_linear_system(self, omega):
        A_i = self.H_i.copy()
        A_j = self.H_j.copy()
        A_v = list(np.array(self.H_v)/(self.mat.rho*omega**2))
        A_i.extend(self.Q_i)
        A_j.extend(self.Q_j)
        A_v.extend(-np.array(self.Q_v)/(self.mat.K))
        return A_i, A_j, A_v

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

    def append_global_matrices(self, _elem):
        dof_u, orient_u = dof_u_element(_elem)
        dof_p, orient_p = dof_p_element(_elem)
        M, K_0, K_1, H, Q, C = pem98_elementary_matrices(_elem)
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
        # print(self.mat.name)
        # print("K_eq_til={}".format(self.mat.K_eq_til))
        # print("P_tilde={}".format(self.mat.P_til))
        # print("Q_tilde={}".format(self.mat.Q_til))
        # print("R_tilde={}".format(self.mat.R_til))
        # print("N={}".format(self.mat.N))
        # print("rho_11_tilde={}".format(self.mat.rho_11_til))
        # print("rho_12_tilde={}".format(self.mat.rho_12_til))
        # print("rho_22_tilde={}".format(self.mat.rho_22_til))
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

class IncidentPwFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []
        self.dofs, self.sol = [], []
        self.theta_d = kwargs["p"].theta_d
        self.kx, self.ky = [],[]

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "ImposedPw" + FemEntity.__str__(self)
        return out

    def update_frequency(self, omega):
        k_air = omega/Air.c
        k_x = k_air*np.sin(self.theta_d*np.pi/180.)
        nb_bloch_waves = np.ceil((self.period/(2*pi))*(3*np.real(k_air)-k_x))+5
        # nb_bloch_waves = 0
        _ = np.arange(-nb_bloch_waves, nb_bloch_waves+1)
        self.kx = k_x+_*(2*pi/self.period)
        k_y = np.sqrt(k_air**2-self.kx**2+0*1j)
        self.ky = np.real(k_y)-1j*np.imag(k_y)
        print(self.ky)

    def append_linear_system(self, omega):
        A_i, A_j, A_v, F_i, F_v = [], [], [], [], []
        i_spec = int((len(self.dofs)-1)/2)
        for i_w, kx in enumerate(self.kx):
            for _elem in self.elements:
                F = imposed_pw_elementary_vector(_elem, kx)
                dof_FEM, orient = dof_p_element(_elem)
                dof_pw = [self.dofs[i_w]]*len(dof_FEM)
                Omega_u = 1j*self.ky[i_w]/(Air.rho*omega**2)
                _ = np.array(orient)*F
                A_i.extend(dof_FEM)
                A_j.extend(dof_pw)
                A_v.extend(Omega_u*_)
                A_i.extend(dof_pw)
                A_j.extend(dof_FEM)
                A_v.extend(np.conj(_))
                if i_w == i_spec:
                    F_i.extend(dof_FEM)
                    F_v.extend(Omega_u*_)
        # append_orthogonality(self):
            A_i.append(self.dofs[i_w])
            A_j.append(self.dofs[i_w])
            A_v.append(-self.period)

        F_i.append(self.dofs[i_spec])
        F_v.append(self.period)
        return A_i, A_j, A_v, F_i, F_v

class TransmissionPwFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []
        self.dofs = []
        self.theta_d = kwargs["p"].theta_d
        # self.period = kwargs["p"].period
        self.kx, self.ky = [],[]

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "TransmissionPw" + FemEntity.__str__(self)
        return out

    def update_frequency(self, omega):
        k_air = omega/Air.c
        k_x = k_air*np.sin(self.theta_d*np.pi/180.)
        nb_bloch_waves = np.ceil((self.period/(2*pi))*(3*np.real(k_air)-k_x))+5
        # nb_bloch_waves = 0

        _ = np.arange(-nb_bloch_waves, nb_bloch_waves+1)
        self.kx = k_x+_*(2*pi/self.period)
        k_y = np.sqrt(k_air**2-self.kx**2+0*1j)
        self.ky = np.real(k_y)-1j*np.imag(k_y)

    def append_linear_system(self, omega):
        A_i, A_j, A_v, F_i, F_v = [], [], [], [], []
        i_spec = int((len(self.dofs)-1)/2)
        for i_w, kx in enumerate(self.kx):
            for _elem in self.elements:
                F = imposed_pw_elementary_vector(_elem, kx)
                dof_FEM, orient = dof_p_element(_elem)
                dof_pw = [self.dofs[i_w]]*len(dof_FEM)
                Omega_u = 1j*self.ky[i_w]/(Air.rho*omega**2)
                _ = np.array(orient)*F
                A_i.extend(dof_FEM)
                A_j.extend(dof_pw)
                A_v.extend(Omega_u*_)
                A_i.extend(dof_pw)
                A_j.extend(dof_FEM)
                A_v.extend(np.conj(_))
        # append_orthogonality(self):
            A_i.append(self.dofs[i_w])
            A_j.append(self.dofs[i_w])
            A_v.append(-self.period)

        return A_i, A_j, A_v, F_i, F_v

if __name__ == "__main__":
    pass