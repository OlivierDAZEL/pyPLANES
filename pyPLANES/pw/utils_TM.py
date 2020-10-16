#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_fem.py
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
from scipy.linalg import expm
from numpy import sqrt

def convert_Omega(Om_m, typ_minus, typ_plus):
    # fluid {0:u_y , 1:p}
    # elastic {0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
    # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
    # We have S = Omega @ [R,x,y]

    if typ_minus == typ_plus:
        Om_plus = Om_m
    else:
        if typ_minus == "pem":
            if typ_plus == "fluid":
                Om_plus = np.zeros((2, 1), dtype=complex)
                # The two in vacuo stresses are zero
                M = np.array([[Om_m[0, 1], Om_m[0, 2]], [Om_m[3, 1], Om_m[3, 2]]])
                F = np.array([Om_m[0, 0], Om_m[3, 0]])
                X = LA.solve(M, F)
                # print(Om_m[0,0]-x[0]*Om_m[0, 1]-x[1]*Om_m[0, 2])
                Om_plus[0, 0] = Om_m[2, 0]-x[0]*Om_m[2, 1]-x[1]*Om_m[2, 2]
                Om_plus[1, 0] = Om_m[4, 0]-x[0]*Om_m[4, 1]-x[1]*Om_m[4, 2]
            elif typ_plus == "elastic":
                Om_plus = np.zeros((4,2), dtype=complex)
                # We have to impose that u_y^s = u_y^t
                c_R = Om_m[2, 0] - Om_m[1, 0]
                c_1 = Om_m[2, 1] - Om_m[1, 1]
                c_2 = Om_m[2, 2] - Om_m[1, 2]
                if abs(c_2) == 0:
                    if abs(c_1) == 0:
                        raise ValueError("c_1 and c_2 are both zero")
                    else:
                        print("swapping c_1 and c_2")
                        c_1, c_2 = c_2, c_1
                        Om_m[:, [1, 2]]= Om_m[:, [2, 1]]
                Om_plus[0, 0] = Om_m[0, 0] - (c_R/c_2)*Om_m[0, 2]
                Om_plus[0, 1] = Om_m[0, 1] - (c_1/c_2)*Om_m[0, 2]
                Om_plus[1, 0] = Om_m[1, 0] - (c_R/c_2)*Om_m[1, 2]
                Om_plus[1, 1] = Om_m[1, 1] - (c_1/c_2)*Om_m[1, 2]
                Om_plus[2, 0] = (Om_m[3, 0]-Om_m[4,0]) - (c_R/c_2)*(Om_m[3, 2]-Om_m[4, 2])
                Om_plus[2, 1] = (Om_m[3, 1]-Om_m[4,1]) - (c_1/c_2)*(Om_m[3, 2]-Om_m[4, 2])
                Om_plus[3, 0] = Om_m[5, 0] - (c_R/c_2)*Om_m[5, 2]
                Om_plus[3, 1] = Om_m[5, 1] - (c_1/c_2)*Om_m[5, 2]
            elif typ_plus in ("Biot98", "Biot01"):
                Om_plus = Om_m
        elif typ_minus == "elastic":
            if typ_plus in ("pem", "Biot98", "Biot01"):
                Om_plus = np.zeros((6, 3), dtype=complex)
                Om_plus[0, :] = [Om_m[0, 0], Om_m[0, 1], 0]
                Om_plus[1, :] = [Om_m[1, 0], Om_m[1, 1], 0]
                Om_plus[2, :] = [Om_m[1, 0], Om_m[1, 1], 0]
                Om_plus[3, :] = [0 , 0, 1] # \hat{sigma}_yy is the new unknwon
                Om_plus[4, :] = [-Om_m[2, 0], -Om_m[2, 1], 1] # p = \hat{sigma}_yy-sigma^t_yy
                Om_plus[5, :] = [Om_m[3, 0], Om_m[3, 1], 0]
            elif typ_plus == "fluid":
                Om_plus = np.zeros((2, 1), dtype=complex)
                _ = Om_m[0, 0]/ Om_m[0, 1]
                Om_plus[0, 0] = Om_m[1, 0]-_*Om_m[1, 1]
                Om_plus[1, 0] = -(Om_m[2, 0]-_*Om_m[2, 1])
        elif typ_minus == "fluid":
            if typ_plus in ("pem", "Biot98", "Biot01"):
                Om_plus = np.zeros((6, 3), dtype=complex)
                Om_plus[1, :] = [0, 0, 1]
                Om_plus[2, :] = [Om_m[0, 0], 0, 0]
                Om_plus[4, :] = [Om_m[1, 0], 0, 0]
                Om_plus[5, :] = [0, 1, 0]
            elif typ_plus == "elastic":
                Om_plus = np.zeros((4,2), dtype=complex)
                Om_plus[1, :] = [Om_m[0, 0], 0]
                Om_plus[2, :] = [-Om_m[1, 0], 0] #sigma_yy = -p
                Om_plus[3, :] = [0, 1]
    return Om_plus

def TM_fluid(layer, kx, om):
    """
    TM_fluid returns the 2x2 Transfer Matrix of a fluid layer

    Parameters
    ----------
    layer : Layer
    kx : transversal wave number
    om : circular frequency

    Returns
    -------
    2x2 numpy array
    """

    h = layer.thickness
    rho = layer.medium.rho
    K = layer.medium.K
    k = om*np.sqrt(rho/K)
    ky = np.sqrt(k**2-kx**2)
    T = np.zeros((2, 2), dtype=complex)
    T[0, 0] = np.cos(ky*h)
    T[1, 0] = (om**2*rho/ky)*np.sin(ky*h)
    T[0, 1] = -(ky/(om**2*rho))*np.sin(ky*h)
    T[1, 1] = np.cos(ky*h)
    return T

def TM_elastic(layer, kx, om):
    rho = layer.medium.rho
    lam = layer.medium.lambda_
    mu = layer.medium.mu
    d = layer.thickness

    P_mat = lam + 2*mu
    delta_p = om*np.sqrt(rho/P_mat)
    delta_s = om*np.sqrt(rho/mu)

    beta_p = np.sqrt(delta_p**2-kx**2+0j)
    beta_s = np.sqrt(delta_s**2-kx**2+0j)

    alpha_p = -1j*lam*delta_p**2 - 2j*mu*beta_p**2
    alpha_s = 2j*mu*beta_s*kx

    Phi_0 = np.zeros((4, 4), dtype=np.complex)
    Phi_0[0,0] = -2j*mu*beta_p*kx
    Phi_0[0,1] = 2j*mu*beta_p*kx
    Phi_0[0,2] = 1j*mu*(beta_s**2-kx**2)
    Phi_0[0,3] = 1j*mu*(beta_s**2-kx**2)

    Phi_0[1,0] = beta_p
    Phi_0[1,1] = -beta_p
    Phi_0[1,2] = kx
    Phi_0[1,3] = kx

    Phi_0[2,0] = alpha_p
    Phi_0[2,1] = alpha_p
    Phi_0[2,2] = -alpha_s
    Phi_0[2,3] = alpha_s

    Phi_0[3,0] = kx
    Phi_0[3,1] = kx
    Phi_0[3,2] = -beta_s
    Phi_0[3,3] = beta_s

    V_0 = np.diag([
        np.exp(-1j*beta_p*d),
        np.exp(1j*beta_p*d),
        np.exp(-1j*beta_s*d),
        np.exp(1j*beta_s*d)
    ])

    T = Phi_0@V_0@LA.inv(Phi_0)

    return T

def TM_pem(layer, kx, om):

    medium = layer.medium
    d = layer.thickness

    medium.update_frequency(om)
    beta_1 = sqrt(medium.delta_1**2-kx**2)
    beta_2 = sqrt(medium.delta_2**2-kx**2)
    beta_3 = sqrt(medium.delta_3**2-kx**2)
    alpha_1 = -1j*medium.A_hat*medium.delta_1**2 - 2j*medium.N*beta_1**2
    alpha_2 = -1j*medium.A_hat*medium.delta_2**2 - 2j*medium.N*beta_2**2
    alpha_3 = 2j*medium.N*beta_3*kx

    Phi_0 = np.zeros((6,6), dtype=np.complex)
    Phi_0[0 ,0] = -2j*medium.N*beta_1*kx
    Phi_0[0 ,1] = 2j*medium.N*beta_1*kx
    Phi_0[0 ,2] = -2j*medium.N*beta_2*kx
    Phi_0[0 ,3] = 2j*medium.N*beta_2*kx
    Phi_0[0 ,4] = 1j*medium.N*(beta_3**2-kx**2)
    Phi_0[0 ,5] = 1j*medium.N*(beta_3**2-kx**2)

    Phi_0[1, 0] = beta_1
    Phi_0[1, 1] = -beta_1
    Phi_0[1, 2] = beta_2
    Phi_0[1, 3] = -beta_2
    Phi_0[1, 4] = kx
    Phi_0[1, 5] = kx

    Phi_0[2, 0] = medium.mu_1*beta_1
    Phi_0[2, 1] = -medium.mu_1*beta_1
    Phi_0[2, 2] = medium.mu_2*beta_2
    Phi_0[2, 3] = -medium.mu_2*beta_2
    Phi_0[2, 4] = medium.mu_3*kx
    Phi_0[2, 5] = medium.mu_3*kx

    Phi_0[3, 0] = alpha_1
    Phi_0[3, 1] = alpha_1
    Phi_0[3, 2] = alpha_2
    Phi_0[3, 3] = alpha_2
    Phi_0[3, 4] = -alpha_3
    Phi_0[3, 5] = alpha_3

    Phi_0[4, 0] = 1j*medium.delta_1**2*medium.K_eq_til*medium.mu_1
    Phi_0[4, 1] = 1j*medium.delta_1**2*medium.K_eq_til*medium.mu_1
    Phi_0[4, 2] = 1j*medium.delta_2**2*medium.K_eq_til*medium.mu_2
    Phi_0[4, 3] = 1j*medium.delta_2**2*medium.K_eq_til*medium.mu_2
    Phi_0[4, 4] = 0
    Phi_0[4, 5] = 0

    Phi_0[5, 0] = kx
    Phi_0[5, 1] = kx
    Phi_0[5, 2] = kx
    Phi_0[5, 3] = kx
    Phi_0[5, 4] = -beta_3
    Phi_0[5, 5] = beta_3

    V_0 = np.diag([
        np.exp(-1j*beta_1*d),
        np.exp(1j*beta_1*d),
        np.exp(-1j*beta_2*d),
        np.exp(1j*beta_2*d),
        np.exp(-1j*beta_3*d),
        np.exp(1j*beta_3*d)
    ])

    T = Phi_0@V_0@LA.inv(Phi_0)

    return T

def weak_orth_terms(om, kx, Omega, layers, typ_end):
    # fluid {0:u_y , 1:p}
    # elastic {0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
    # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
    typ = "fluid"
    if layers:
        for _l in layers:
            # print("converting {} to {}".format(typ, _l.medium.MEDIUM_TYPE))
            Omega = convert_Omega(Omega, typ, _l.medium.MEDIUM_TYPE)
            if _l.medium.MEDIUM_TYPE == "fluid":
                Omega = TM_fluid(_l, kx, om)@Omega
            elif _l.medium.MEDIUM_TYPE == "elastic":
                Omega = TM_elastic(_l, kx, om)@Omega
            elif _l.medium.MEDIUM_TYPE == "pem":
                Omega = TM_pem(_l, kx, om)@Omega
        # print("converting {} to {}".format(layers[-1].medium.MEDIUM_TYPE, typ_end))
        Omega = convert_Omega(Omega, layers[-1].medium.MEDIUM_TYPE, typ_end)
    else:
        Omega = convert_Omega(Omega, "fluid", typ_end)
    if typ_end == "fluid":
        # u_y^t
        weak = np.array([Omega[0, :]])
        # p
        orth = np.array([Omega[1, :]])
    elif typ_end == "elastic":
        # sigma_xy and sigma_yy = -p
        weak = np.array([Omega[0, :], Omega[2, :]])
        # u_x^s and u_y
        orth = np.array([Omega[3, :], Omega[1, :]])
    elif typ_end == "Biot98":
        # sigma_xy, sigma_yy and u_y^t
        weak = np.array([Omega[0, :], Omega[3, :], Omega[2, :]])
        # u_x, u_y and p
        orth = np.array([Omega[5, :], Omega[1, :], Omega[4, :]])
    elif typ_end == "Biot01":
        # sigma_xy^t and sigma_yy^t and u_t-u_s
        weak = np.array([Omega[0, :], Omega[3, :]-Omega[4, :], Omega[2, :]-Omega[1, :]])
        # u_x^s and u_y^s and p
        orth = np.array([Omega[5, :], Omega[1, :], Omega[4, :]])
    else:
        raise ValueError("Unknown typ")
    return weak, orth
