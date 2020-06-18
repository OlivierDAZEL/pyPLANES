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

from pyPLANES.fem.elements.volumic_elements import fluid_elementary_matrices, pem98_elementary_matrices, pem01_elementary_matrices, elas_elementary_matrices
from pyPLANES.fem.elements.surfacic_elements import imposed_pw_elementary_vector, fsi_elementary_matrix
from pyPLANES.utils.utils_TM import weak_orth_terms
from pyPLANES.utils.utils_fem import dof_p_element, dof_u_element, dof_ux_element, dof_uy_element, orient_element
from pyPLANES.utils.utils_fem import dof_p_linear_system_to_condense, dof_p_linear_system_master, dof_up_linear_system_to_condense, dof_up_linear_system_master, dof_up_linear_system, dof_u_linear_system_master, dof_ux_linear_system_master, dof_uy_linear_system_master,dof_u_linear_system, dof_u_linear_system_to_condense

class GmshEntity():
    def __init__(self, **kwargs):
        self.dim = kwargs["dim"]
        self.tag = kwargs["tag"]
        self.physical_tags = kwargs["physical_tags"]
        entities = kwargs["entities"]
        if "condition" not in list(self.physical_tags.keys()):
            self.physical_tags["condition"] = None
        if "model" not in list(self.physical_tags.keys()):
            self.physical_tags["model"] = None
        if self.dim == 0:
            self.neighbouring_curves = []
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.z = kwargs["z"]
            self.neighbours = []
        elif self.dim == 1:
            self.neighbouring_surfaces = []
            self.bounding_points = [next((e for e in entities if e.tag == abs(t)), None) for t in kwargs["bounding_points"]]
            self.center = np.array([0., 0., 0.])
            for p in self.bounding_points:
                self.center += [p.x, p.y, p.z]
            self.center /= len(self.bounding_points)
            for _e in self.bounding_points:
                _e.neighbouring_curves.append(self)
        elif self.dim == 2:
            self.neighbouring_surfaces = [] # Neighbouring 2D entities, will be completed in preprocess
            self.bounding_curves = [next((e for e in entities if e.tag == abs(t)), None) for t in kwargs["bounding_curves"]]
            for _e in self.bounding_curves:
                _e.neighbouring_surfaces.append(self)
            self.center = np.array([0., 0., 0.])
            for c in self.bounding_curves:
                self.center += c.center
            self.center /= len(self.bounding_curves)


    def __str__(self):
        out = "Entity / tag={} / dim= {}\n".format(self.tag, self.dim)
        out += "Physical tags={}\n".format(self.physical_tags)
        if self.dim == 0:
            out += "Belongs to curves "
            for _c in self.neighbouring_curves:
                out += "{} ({}) ".format(_c.tag,_c.physical_tags["condition"])
            out += "\n"
        if self.dim == 1:
            out += "Related points "
            for _b in self.bounding_points:
                out += "{} ({}) ".format(_b.tag,_b.physical_tags["condition"])
            out += "\n"
        if self.dim == 2:
            out += "Related curves "
            for _c in self.bounding_curves:
                out += "{} ({}) ".format(_c.tag,_c.physical_tags["condition"])
            out += "\n"
        return out

class FemEntity(GmshEntity):
    def __init__(self, **kwargs):
        GmshEntity.__init__(self, **kwargs)
        self.order = kwargs["order"]
        self.elements = []
    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Fem" + GmshEntity.__str__(self)
        out += "order:{}\n".format(self.order)
        # related_elements = [_el.tag for _el in self.elements]
        # out  += "related elements={}\n".format(related_elements)
        return out

    def condensation(self, omega):
        return [], [], [], [], [], []

    def update_frequency(self, omega):
        pass

    def elementary_matrices(self, _elem):
        pass

    def append_linear_system(self, omega):
        return [], [], [], [], [], []

    def link_elem(self,n):
        self.elements.append(n)

class InterfaceFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.ml = kwargs.get("ml", False)
        self.side = kwargs.get("side", False)
        self.
        self.fluid_neighbour = None
        self.struc_neighbour = None

    def __str__(self):
        out = "Interface" + FemEntity.__str__(self)
        return out

    def elementary_matrices(self, _el):
        pass

    def append_linear_system(self, omega):
        A_i, A_j, A_v =[], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []


        return A_i, A_j, A_v, T_i, T_j, T_v



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

class FluidFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = Air()
        self.H_i, self.H_j, self.H_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []

    def __str__(self):
        out = "Air" + FemEntity.__str__(self)
        return out

    def elementary_matrices(self, _el):
        # Elementary matrices
        H, Q = fluid_elementary_matrices(_el)
        orient_p = orient_element(_el)
        _el.H = orient_p @ H @ orient_p
        _el.Q = orient_p @ Q @ orient_p

    def append_linear_system(self, omega):
        A_i, A_j, A_v =[], [], []
        # Translation matrix to compute internal dofs
        T_i, T_j, T_v =[], [], []
        for _el in self.elements:
            nb_m_SF = _el.reference_element.nb_m_SF
            nb_SF = _el.reference_element.nb_SF
            # With condensation
            # Based on reordered matrix
            pp = _el.H/(self.mat.rho*omega**2)- _el.Q/(self.mat.K)

            l_p_m = slice(nb_m_SF)
            l_p_c = slice(nb_m_SF, nb_SF)

            mm = pp[l_p_m, l_p_m]
            cm = pp[l_p_c, l_p_m]
            mc = pp[l_p_m, l_p_c]
            cc = pp[l_p_c, l_p_c]

            t = -LA.inv(cc)@cm
            mm += mc@t

            dof_p_m = dof_p_linear_system_master(_el)
            dof_p_c = dof_p_linear_system_to_condense(_el)

            T_i.extend(list(chain.from_iterable([[_d]*(nb_m_SF) for _d in dof_p_c])))
            T_j.extend(list(dof_p_m)*((nb_SF-nb_m_SF)))
            T_v.extend(t.flatten())

            A_i.extend(list(chain.from_iterable([[_d]*(nb_m_SF) for _d in dof_p_m])))
            A_j.extend(list(dof_p_m)*(nb_m_SF))
            A_v.extend(mm.flatten())

        return A_i, A_j, A_v, T_i, T_j, T_v

class PemFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
        self.formulation98 = True
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

    def append_linear_system(self, omega):
        A_i, A_j, A_v =[], [], []
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

            t = -LA.inv(cc)@cm
            mm += mc@t

            dof_up_m = dof_up_linear_system_master(_el)
            dof_up_c = dof_up_linear_system_to_condense(_el)

            T_i.extend(list(chain.from_iterable([[_d]*(3*nb_m_SF) for _d in dof_up_c])))
            T_j.extend(list(dof_up_m)*(3*(nb_SF-nb_m_SF)))
            T_v.extend(t.flatten())

            A_i.extend(list(chain.from_iterable([[_d]*(3*nb_m_SF) for _d in dof_up_m])))
            A_j.extend(list(dof_up_m)*(3*nb_m_SF))
            A_v.extend(mm.flatten())

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

class ElasticFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.mat = kwargs["mat"]
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

    def append_linear_system(self, omega):
        A_i, A_j, A_v =[], [], []
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

            t = -LA.inv(cc)@cm
            mm += mc@t

            dof_u_m = dof_u_linear_system_master(_el)
            dof_u_c = dof_u_linear_system_to_condense(_el)

            T_i.extend(list(chain.from_iterable([[_d]*(2*nb_m_SF) for _d in dof_u_c])))
            T_j.extend(list(dof_u_m)*(2*(nb_SF-nb_m_SF)))
            T_v.extend(t.flatten())

            A_i.extend(list(chain.from_iterable([[_d]*(2*nb_m_SF) for _d in dof_u_m])))
            A_j.extend(list(dof_u_m)*(2*nb_m_SF))
            A_v.extend(mm.flatten())

        return A_i, A_j, A_v, T_i, T_j, T_v

class PwFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []
        self.dofs = []
        self.theta_d = None
        # self.period = kwargs["p"].period
        self.kx, self.ky = [], []
        self.phi_i, self.phi_j, self.phi_v = [], [], []
        self.nb_dofs = None
        self.Omega_orth = None
        self.ny = None
        self.ml = None

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Pw" + FemEntity.__str__(self)
        return out

    def update_frequency(self, omega):
        k_air = omega/Air.c
        k_x = k_air*np.sin(self.theta_d*np.pi/180.)
        nb_bloch_waves = int(np.ceil((self.period/(2*pi))*(3*np.real(k_air)-k_x))+5)
        # nb_bloch_waves = 0
        print("nb_bloch_waves ={}".format(nb_bloch_waves))
        _ = np.array([0] + list(range(-nb_bloch_waves, 0)) + list(range(1, nb_bloch_waves+1)))
        self.nb_waves = 1+2*nb_bloch_waves
        self.kx = k_x+_*(2*pi/self.period)
        k_y = np.sqrt(k_air**2-self.kx**2+0*1j)
        self.ky = np.real(k_y)-1j*np.imag(k_y)
        self.dofs = np.arange(self.nb_R*(1+2*nb_bloch_waves))
        self.nb_dofs = self.nb_R*(1+2*nb_bloch_waves)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []

    def apply_periodicity(self, nb_dof_m, dof_left, dof_right, delta):
        for i_left, _dof_left in enumerate(dof_left):
            # Corresponding dof
            _dof_right = dof_right[i_left]
            index = np.where(self.phi_i == _dof_right)
            self.phi_i[index] = _dof_left
            for _i in index:
                self.phi_v[_i] /= delta
        self.phi = coo_matrix((self.phi_v, (self.phi_i, self.phi_j)), shape=(nb_dof_m, self.nb_dofs)).tocsr()

class IncidentPwFem(PwFem):
    def __init__(self, **kwargs):
        PwFem.__init__(self, **kwargs)
        self.ny = -1.

    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Imposed" + PwFem.__str__(self)
        return out

    def get_wave_dofs(self, _l):
        _ = list(range(self.nb_R*_l, self.nb_R*(_l+1)))
        dof_r = [__ for __ in _ for j in range(self.nb_R)]
        dof_c = _ * self.nb_R
        return dof_r, dof_c

    def get_tau_eta(self, kx, ky, om):
        # State vector S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}^t, 4:p, 5:u_x^s}'''

        Omega = np.array([1j*ky/(Air.rho*om**2), 1]).reshape((2, 1))
        Omega_l_weak, Omega_l_orth = weak_orth_terms(om, kx, Omega, self.ml, self.typ)

        eta_l = LA.inv(Omega_l_orth)
        tau_l = np.dot(Omega_l_weak, eta_l)
        return tau_l, eta_l

    def create_dynamical_matrices(self, omega, n_m):
        phi = coo_matrix((n_m, self.nb_dofs), dtype=complex)
        self.eta_TM = coo_matrix((self.nb_dofs, self.nb_dofs), dtype=complex)
        tau = coo_matrix((self.nb_dofs, self.nb_dofs), dtype=complex)
        Omega_0 = np.array([-1j*self.ky[0]/(Air.rho*omega**2), 1]).reshape((2, 1))
        Omega_0_weak, self.Omega_0_orth = weak_orth_terms(omega, self.kx[0], Omega_0, self.ml, self.typ)
        self.Omega_0_orth = self.Omega_0_orth[:, 0]
        Omega_0_weak = Omega_0_weak[:, 0]

        for _l in range(self.nb_waves):
            tau_l, eta_l = self.get_tau_eta(self.kx[_l], self.ky[_l], omega)

            if _l == 0:
                tau_0 = tau_l
            # dofs for eta and tau matrices
            dof_r, dof_c = self.get_wave_dofs(_l)
            self.eta_TM += coo_matrix((eta_l.flatten(), (dof_r, dof_c)), shape=(self.nb_dofs, self.nb_dofs))
            tau += coo_matrix((tau_l.flatten(), (dof_r, dof_c)), shape=(self.nb_dofs, self.nb_dofs))
            for _elem in self.elements:
                phi_l = imposed_pw_elementary_vector(_elem, self.kx[_l])
                if self.typ == "fluid":
                    dof_p, orient_p, _ = dof_p_element(_elem)
                    dof_1 = [self.dofs[self.nb_R*_l]]*len(dof_p)
                    _ = orient_p@phi_l
                    print(n_m)
                    print(self.nb_dofs)
                    print(dof_p)
                    print(dof_1)
                    phi += coo_matrix((_, (dof_p, dof_1)), shape=(n_m, self.nb_dofs))
                elif self.typ == "elastic":
                    dof_ux, orient_ux = dof_ux_element(_elem)
                    dof_uy, orient_uy = dof_uy_element(_elem)
                    dof_1 = [self.dofs[self.nb_R*_l]]*len(dof_ux)
                    dof_2 = [self.dofs[self.nb_R*_l+1]]*len(dof_ux)

                    _ = orient_ux@phi_l
                    phi += coo_matrix((_, (dof_ux, dof_1)), shape=(n_m, self.nb_dofs))
                    phi += coo_matrix((_, (dof_uy, dof_2)), shape=(n_m, self.nb_dofs))
                elif self.typ in ["Biot98", "Biot01"]:
                    dof_ux, orient_ux = dof_ux_element(_elem)
                    dof_uy, orient_uy = dof_uy_element(_elem)
                    dof_p, orient_p, _ = dof_p_element(_elem)
                    dof_1 = [self.dofs[self.nb_R*_l]]*len(dof_uy)
                    dof_2 = [self.dofs[self.nb_R*_l+1]]*len(dof_uy)
                    dof_3 = [self.dofs[self.nb_R*_l+2]]*len(dof_uy)
                    _ = orient_uy@phi_l
                    phi += coo_matrix((_, (dof_ux, dof_1)), shape=(n_m, self.nb_dofs))
                    phi += coo_matrix((_, (dof_uy, dof_2)), shape=(n_m, self.nb_dofs))
                    phi += coo_matrix((_, (dof_p, dof_3)), shape=(n_m, self.nb_dofs))
                else:
                    raise ValueError("Unknown typ")
                # print(len(dof_ux))
        A_TM = -(phi@tau@phi.H/self.period).tocoo()
        F_TM = np.zeros(self.nb_dofs, dtype=complex)
        F_TM[:self.nb_R] = Omega_0_weak-tau_0@self.Omega_0_orth

        F_TM = coo_matrix((phi@F_TM).reshape(n_m, 1))
        # print(F_TM)
        _ = phi.tocoo()
        self.phi_i, self.phi_j, self.phi_v = _.row, _.col, _.data
        return A_TM*self.ny, F_TM*self.ny

class TransmissionPwFem(PwFem):
    def __init__(self, **kwargs):
        PwFem.__init__(self, **kwargs)
        self.ny = 1.

    def __str__(self):
        out = "Transmitted " + PwFem.__str__(self)
        return out


    def get_wave_dofs(self, _l):
        _ = list(range(self.nb_R*_l, self.nb_R*(_l+1)))
        dof_r = [__ for __ in _ for j in range(self.nb_R)]
        dof_c = _ * self.nb_R
        return dof_r, dof_c

    def get_tau_eta(self, kx, ky, om):
        # State vector S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}^t, 4:p, 5:u_x^s}'''
        Omega = np.array([-1j*ky/(Air.rho*om**2), 1]).reshape((2,1))
        Omega_l_weak, Omega_l_orth = weak_orth_terms(om, kx, Omega, self.ml, self.typ)

        eta_l = LA.inv(Omega_l_orth)
        tau_l = np.dot(Omega_l_weak, eta_l)

        return tau_l, eta_l

    def create_dynamical_matrices(self, omega, n_m):
        phi = coo_matrix((n_m, self.nb_dofs), dtype=complex)
        self.eta_TM = coo_matrix((self.nb_dofs, self.nb_dofs), dtype=complex)
        tau = coo_matrix((self.nb_dofs, self.nb_dofs), dtype=complex)

        for _l in range(self.nb_waves):
            tau_l, eta_l = self.get_tau_eta(self.kx[_l], self.ky[_l], omega)

            # dofs for eta and tau matrices
            dof_r, dof_c = self.get_wave_dofs(_l)
            self.eta_TM += coo_matrix((eta_l.flatten(), (dof_r, dof_c)), shape=(self.nb_dofs, self.nb_dofs))
            tau += coo_matrix((tau_l.flatten(), (dof_r, dof_c)), shape=(self.nb_dofs, self.nb_dofs))
            for _elem in self.elements:
                phi_l = imposed_pw_elementary_vector(_elem, self.kx[_l])
                if self.typ == "fluid":
                    dof_p, orient_p, _ = dof_p_element(_elem)
                    dof_1 = [self.dofs[self.nb_R*_l]]*len(dof_p)
                    _ = orient_p@phi_l
                    phi += coo_matrix((_, (dof_p, dof_1)), shape=(n_m, self.nb_dofs))
                elif self.typ == "elastic":
                    dof_ux, orient_ux = dof_ux_element(_elem)
                    dof_uy, orient_uy = dof_uy_element(_elem)
                    dof_1 = [self.dofs[self.nb_R*_l]]*len(dof_ux)
                    dof_2 = [self.dofs[self.nb_R*_l+1]]*len(dof_ux)
                    _ = orient_ux@phi_l
                    phi += coo_matrix((_, (dof_ux, dof_1)), shape=(n_m, self.nb_dofs))
                    phi += coo_matrix((_, (dof_uy, dof_2)), shape=(n_m, self.nb_dofs))
                elif self.typ in ["Biot98", "Biot01"]:
                    dof_ux, orient_ux = dof_ux_element(_elem)
                    dof_uy, orient_uy = dof_uy_element(_elem)
                    dof_p, orient_p, _ = dof_p_element(_elem)
                    dof_1 = [self.dofs[self.nb_R*_l]]*len(dof_uy)
                    dof_2 = [self.dofs[self.nb_R*_l+1]]*len(dof_uy)
                    dof_3 = [self.dofs[self.nb_R*_l+2]]*len(dof_uy)
                    _ = orient_uy@phi_l
                    phi += coo_matrix((_, (dof_ux, dof_1)), shape=(n_m, self.nb_dofs))
                    phi += coo_matrix((_, (dof_uy, dof_2)), shape=(n_m, self.nb_dofs))
                    phi += coo_matrix((_, (dof_p, dof_3)), shape=(n_m, self.nb_dofs))
                else:
                    raise ValueError("Unknown typ")
                # print(len(dof_ux))
        A_TM = -(phi@tau@phi.H/self.period).tocoo()

        F_TM = coo_matrix((n_m, 1))

        _ = phi.tocoo()
        self.phi_i, self.phi_j, self.phi_v = _.row, _.col, _.data

        return A_TM*self.ny, F_TM*self.ny

