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
from pyPLANES.fem.lobatto_polynomials import lobatto as l

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
    F_analytical = _elem.reference_element.int_lobatto_exponential(k_prime)
    return (h/2.)*np.exp(-1j*k*x_mid)*F_analytical

def fsi_elementary_matrix(_elem):
    coord_e = _elem.get_coordinates()
    K_ref = _elem.reference_element
    n, m = K_ref.Phi.shape
    M = np.zeros((n, n))
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        M += K_ref.w[ipg]*np.dot(_Phi.reshape((n, 1)), _Phi.reshape((1, n)))
    M *= LA.norm(coord_e[:, 1]-coord_e[:, 0])/2.
    return M

def fsi_elementary_matrix_incompatible(_elem):
    """
    fsi_elementary_matrix_incompatible: Create the elementary "mass" matrix in the case of an incompatible mesh

    Parameters
    ----------
    _elem : Element instance

    Returns
    -------
    [type]
        [description]

    """
    coord_e = _elem.get_coordinates()
    K_ref = _elem.reference_element
    order = K_ref.order

    print("element tag = {}".format(_elem.tag))

    node_0 = np.array(_elem.vertices[0].coord)
    node_1 = np.array(_elem.vertices[1].coord)
    print("node_0={}".format(node_0))
    print("node_1={}".format(node_1))
    Matrices = []
    for neigh in _elem.neighbours:
        Node_0 = neigh._elem.vertices[0].coord
        Node_1 = neigh._elem.vertices[1].coord
        print("Node_0={}".format(Node_0))
        print("Node_1={}".format(Node_1))
        print("neighbour_element_tag ={}".format(neigh._elem.tag))
        # Gauss points of both elements
        xi = (-1+2*neigh.s[0])+ (K_ref.xi+1)*(neigh.s[1]-neigh.s[0])
        Xi = (-1+2*neigh.S[0])+ (K_ref.xi+1)*(neigh.S[1]-neigh.S[0])
        # Check that they coincide
        gp = node_0[0]+((xi+1)/2)*(node_1[0]-node_0[0])+1j*(node_0[1]+((xi+1)/2)*(node_1[1]-node_0[1]))
        GP = Node_0[0]+((Xi+1)/2)*(Node_1[0]-Node_0[0])+1j*(Node_0[1]+((Xi+1)/2)*(Node_1[1]-Node_0[1]))
        GP -= _elem.delta[0]+1j*_elem.delta[1]
        if not np.allclose(gp, GP):
            raise ValueError(" Gauss points do not coincide ")


        n, m = 1+order, len(K_ref.w)
        phi = np.zeros((n, m))
        Phi = np.zeros((n, m))
        for _o in range(order+1):
            phi[_o, :] = l(_o, xi)[0]
            Phi[_o, :] = l(_o, Xi)[0]

        M = np.zeros((n, n))
        for ipg in range(m):
            _phi = phi[:, ipg].reshape(n)
            _Phi = Phi[:, ipg].reshape(n)
            M += K_ref.w[ipg]*np.dot(_Phi.reshape((n, 1)), _phi.reshape((1, n)))
        M *= LA.norm(node_1-node_0)*abs(neigh.s[1]-neigh.s[0])/2.
        Matrices.append(M)
    return Matrices