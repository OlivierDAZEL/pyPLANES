#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# surfacic_elements.py
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


def imposed_Neumann(_elem):
    coord_e = _elem.get_coordinates()
    K_ref = _elem.reference_element
    n, m = K_ref.Phi.shape
    F = np.zeros(n)
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        F += K_ref.w[ipg]*_Phi
    F *= LA.norm(coord_e[:, 1]-coord_e[:, 0])/2.
    return F


def imposed_pw_elementary_vector(_elem, k):
    ''' Calculus of I = int_{\Omega_e} e^{-jkx} Phi(x) dx
        On the reference element:
            I = (h/2)e^{-jkx-mid} * \int_{-1}^{1} e^{-jkhxi/2} \Phi(xi) dxi
    '''
    # Geometrical datas
    coord_e = _elem.get_coordinates()
    h = LA.norm(coord_e[:, 1]-coord_e[:, 0])
    x_mid = min(coord_e[0, :]) + h/2.
    k_prime = k*h/2.
    K_ref = _elem.reference_element
    # Integration on the reference element
    n, m = K_ref.Phi.shape
    F = np.zeros(n, dtype=complex)
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        F += K_ref.w[ipg]*_Phi*np.exp(-1j*k_prime*K_ref.xi[ipg])
    # print("Validation")
    # print("k_prime={}".format(k_prime))
    # print("F_numerical= {}".format(F))
    F_analytical = _elem.reference_element.int_lobatto_exponential(k_prime)
    # print("F_analytical={}".format(F_analytical))
    # print("Error on F ={}".format(np.linalg.norm(F-F_analytical)))
    # Integral on the real element
    # F *= (h/2.)*np.exp(-1j*k*x_mid)
    return (h/2.)*np.exp(-1j*k*x_mid)*F_analytical
    # return F

def fluid_structure_interaction_elementary_matrix(_elem):
    coord_e = _elem.get_coordinates()
    K_ref = _elem.reference_element
    n, m = K_ref.Phi.shape
    M = np.zeros((n, n))
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        M += K_ref.w[ipg]*np.dot(_Phi.reshape((n, 1)), _Phi.reshape((1, n)))
    M *= LA.norm(coord_e[:, 1]-coord_e[:, 0])/2.

    return M