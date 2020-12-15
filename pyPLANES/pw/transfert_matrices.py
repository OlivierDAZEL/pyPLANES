#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# transfert_matrices.py
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

    h = layer.d
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
