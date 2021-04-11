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
from pyPLANES.fem.elements_surfacic import imposed_pw_elementary_vector
from pyPLANES.pw.utils_TM import weak_orth_terms
from pyPLANES.fem.utils_fem import *


class PwFem(FemEntity):
    def __init__(self, **kwargs):
        FemEntity.__init__(self, **kwargs)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.F_i, self.F_v = [], []
        self.dofs = []
        self.theta_d = None
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
        nb_bloch_waves = 0
        # print("nb_bloch_waves ={}".format(nb_bloch_waves))
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

    def update_system(self, omega, n_m):

        phi = coo_matrix((n_m, self.nb_dofs), dtype=complex)
        phi_i, phi_j, phi_v = [], [], []
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
                    phi_i.extend(dof_p)
                    phi_j.extend(dof_1)
                    phi_v.extend(_)
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


        # dsqdqsqdsqsd
        F_TM = np.zeros(self.nb_dofs, dtype=complex)
        F_TM[:self.nb_R] = Omega_0_weak-tau_0@self.Omega_0_orth

        F_TM = coo_matrix((phi@F_TM).reshape(n_m, 1))
        # print(F_TM)
        _ = phi.tocoo()
        self.phi_i, self.phi_j, self.phi_v = _.row, _.col, _.data
        A_TM *= self.ny
        F_TM *= self.ny

        # print(F_TM)
        # print(coo_matrix(F_TM))
        F_TM = coo_matrix(F_TM)
        return A_TM.row, A_TM.col, A_TM.data, [], [], [], F_TM.row, F_TM.data

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

    def update_system(self, omega, n_m):
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

        A_TM *=self.ny
        F_TM *=self.ny        

        F_TM = coo_matrix(F_TM)


        return A_TM.row, A_TM.col, A_TM.data, [], [], [], F_TM.row, F_TM.data

