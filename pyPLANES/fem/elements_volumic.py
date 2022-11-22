#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# volumic_elements.py
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

def fluid_elementary_matrices(elem):

    K_ref = elem.reference_element
    n, m = K_ref.Phi.shape
    vh = np.zeros((n ,n))
    vq = np.zeros((n, n))
    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = elem.get_jacobian_matrix(K_ref.xi_1[ipg],K_ref.xi_2[ipg])
        _weight = K_ref.w[ipg] * np.abs(LA.det(J))
        Gd = LA.solve(J, _dPhi)

        vh += _weight*np.dot(Gd.T, Gd)
        vq += _weight*np.dot(_Phi.T, _Phi)

    return vh, vq

def elas_elementary_matrices(elem):

    K_ref = elem.reference_element
    n, m = K_ref.Phi.shape
    vm = np.zeros((2*n, 2*n))
    vk0 = np.zeros((2*n, 2*n))
    vk1 = np.zeros((2*n, 2*n))

    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = elem.get_jacobian_matrix(K_ref.xi_1[ipg],K_ref.xi_2[ipg])
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)
        eps = np.zeros((3, 2*n))
        eps[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        eps[1, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}
        eps[2, 0:n] = Gd[1, :]
        eps[2, n:2*n] = Gd[0, :] # eps_yz = \dpar{uz}{y}+ \dpar{uy}{z}

        Phi_u = np.zeros((2, 2*n))
        Phi_u[0, 0:n] = _Phi
        Phi_u[1, n:2*n] = _Phi

        D_0, D_1 = np.zeros((3, 3)), np.zeros((3, 3))
        D_0[0, 0:2] = 1.
        D_0[1, 0:2] = 1.
        D_1[0, 1] = -2.
        D_1[1, 0] = -2.
        D_1[2, 2] = 1.

        vk0 += _weight*LA.multi_dot([eps.T, D_0, eps])
        vk1 += _weight*LA.multi_dot([eps.T, D_1, eps])
        vm += _weight*np.dot(Phi_u.T, Phi_u)

    return vm, vk0, vk1

def pem98_elementary_matrices(elem):

    K_ref = elem.reference_element

    n, m = K_ref.Phi.shape
    vm = np.zeros((2*n, 2*n))
    vk0 = np.zeros((2*n, 2*n))
    vk1 = np.zeros((2*n, 2*n))
    vc = np.zeros((2*n, n))
    vh = np.zeros((n, n))
    vq = np.zeros((n, n))

    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        elem.get_jacobian_matrix(K_ref.xi_1[ipg],K_ref.xi_2[ipg])
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)
        eps = np.zeros((3, 2*n))
        eps[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        eps[1, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}
        eps[2, 0:n] = Gd[1, :]
        eps[2, n:2*n] = Gd[0, :] # eps_yz = \dpar{uz}{y}+ \dpar{uy}{z}

        Phi_u = np.zeros((2, 2*n))
        Phi_u[0, 0:n] = _Phi
        Phi_u[1, n:2*n] = _Phi

        D_0, D_1 = np.zeros((3, 3)), np.zeros((3, 3))
        D_0[0, 0:2] = 1.
        D_0[1, 0:2] = 1.
        D_1[0, 1] = -2.
        D_1[1, 0] = -2.
        D_1[2, 2] = 1.

        vk0 += _weight*LA.multi_dot([eps.T, D_0, eps])
        vk1 += _weight*LA.multi_dot([eps.T, D_1, eps])
        vm += _weight*np.dot(Phi_u.T, Phi_u)
        vc += _weight*np.dot(Phi_u.T, Gd)
        vh += _weight*np.dot(Gd.T, Gd)
        vq += _weight*np.dot(_Phi.T, _Phi)

    return vm, vk0, vk1, vh, vq, vc

def pem01_elementary_matrices(elem):


    K_ref = elem.reference_element
    n, m = K_ref.Phi.shape
    vm = np.zeros((2*n, 2*n))
    vk0 = np.zeros((2*n, 2*n))
    vk1 = np.zeros((2*n, 2*n))
    vc = np.zeros((2*n, n))
    vc2 = np.zeros((2*n, n))
    vh = np.zeros((n, n))
    vq = np.zeros((n, n))

    # X1X2 = coord_e[:,1]- coord_e[:,0]
    # e_x = X1X2/LA.norm(X1X2)
    # X1X3 = coord_e[:,2]- coord_e[:,0]
    # e_z = np.cross(X1X2, X1X3)
    # e_z /= LA.norm(e_z)
    # e_y = np.cross(e_z, e_x)

    # Coord_e = np.zeros((2,3))
    # Coord_e[0, 1] = X1X2.dot(e_x)
    # Coord_e[0, 2] = X1X3.dot(e_x)
    # Coord_e[1, 2] = X1X3.dot(e_y)



    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = elem.get_jacobian_matrix(K_ref.xi_1[ipg],K_ref.xi_2[ipg])
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)
        eps = np.zeros((3, 2*n))
        eps[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        eps[1, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}
        eps[2, 0:n] = Gd[1, :]
        eps[2, n:2*n] = Gd[0, :] # eps_yz = \dpar{uz}{y}+ \dpar{uy}{z}

        Phi_u = np.zeros((2, 2*n))
        Phi_u[0, 0:n] = _Phi
        Phi_u[1, n:2*n] = _Phi

        div_u = np.zeros((1, 2*n))
        div_u[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        div_u[0, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}

        D_0, D_1 = np.zeros((3, 3)), np.zeros((3, 3))
        D_0[0, 0:2] = 1.
        D_0[1, 0:2] = 1.
        D_1[0, 1] = -2.
        D_1[1, 0] = -2.
        D_1[2, 2] = 1.

        vk0 += _weight*LA.multi_dot([eps.T, D_0, eps])
        vk1 += _weight*LA.multi_dot([eps.T, D_1, eps])
        vm += _weight*np.dot(Phi_u.T, Phi_u)
        vc += _weight*np.dot(Phi_u.T, Gd)
        vc2 += _weight*np.dot(div_u.T, _Phi)
        vh += _weight*np.dot(Gd.T, Gd)
        vq += _weight*np.dot(_Phi.T, _Phi)

    return vm, vk0, vk1, vh, vq, vc, vc2