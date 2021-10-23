#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# reference_elements.py
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
import importlib

from pyPLANES.fem.lobatto_polynomials import lobatto as l
from pyPLANES.fem.lobatto_polynomials import lobatto_kernels as kernel
from pyPLANES.fem.utils_fem import create_legendre_table


import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from numpy.polynomial.legendre import leggauss
import quadpy as quadpy


class Ka:
    def __init__(self, order, p):

        self.order = order
        self.xi, self.w = leggauss(p)

        # Number of Shape Functions
        self.nb_v = 2
        self.nb_e = order-1
        self.nb_SF = self.nb_v+self.nb_e
        # Initialisation of Shape Functions and derivatives
        self.Phi = np.zeros((self.nb_SF, len(self.w)))
        self.dPhi = np.zeros((self.nb_SF, len(self.w)))
        # Shape Functions
        for _o in range(order+1):
            self.Phi[_o, :], self.dPhi[_o, :] = l(_o, self.xi)

    def __str__(self):
        out = "K_a of order {}".format(self.order)
        return out

class KaPw(Ka):
    def __init__(self, order, p):
        Ka.__init__(self, order, 15)
        # legendre table corresponds to values of L_n^(j)(1)
        self.legendre_table = create_legendre_table(self.order)

    def __str__(self):
        out = "KaPw of order {}".format(self.order)
        return out

    def int_lobatto_exponential(self, k):
        ''' Returns the vector of \dint{-1}{1} ln(xi)e^{-ik xi} dxi'''

        n, m = self.Phi.shape
        out = np.zeros(n, dtype=complex)
        # print(k)
        ik = 1j*k
        if np.abs(k) > (-0.1) :
            for i_w, w in enumerate(self.w):
                out += (np.exp(-ik*self.xi[i_w])*w)* self.Phi[:, i_w].reshape(n)
        else:
            out[0] = (np.exp(ik)/ik)+(1j*np.sin(k))/k**2
            out[1] =-(np.exp(-ik)/ik)-(1j*np.sin(k))/k**2
            for _n in range(2, n):
                _ = [self.legendre_table[_n-1, j]*(np.exp(-ik)-np.exp(ik)*(-1)**(_n-1+j))/(ik)**j for j in range(_n) ]
                out[_n] = (np.sum(_)/(k**2*np.sqrt(2./(2*_n-1))))
        return out

class Kt:
    def __init__(self, order=2):
        self.order = order
        # Integration scheme
        scheme = quadpy.t2.schemes["dunavant_{:02d}".format(2*order)]()
        _points = (scheme.points.T).dot(np.array([[-1, -1], [1, -1], [-1, 1]]))
        self.xi_1, self.xi_2, self.w = _points[:, 0], _points[:,1], 2*scheme.weights

        # Number of Shape Functions
        self.nb_v = 3
        self.nb_e = 3*(order-1)
        self.nb_f = int(((order-1)*(order-2))/2)
        # Number of master shape functions
        self.nb_m_SF = self.nb_v+ self.nb_e

        # Number of slave shape functions (to condense)
        self.nb_s_SF = self.nb_f
        self.nb_SF = self.nb_m_SF+self.nb_s_SF

        self.Phi, self.dPhi = shape_functions_Kt(self.xi_1, self.xi_2, self.order)

        # For plots
        tri_Kt = mtri.Triangulation(np.asarray([-1,1,-1]), np.asarray([-1,-1,1]))
        tri_Kt = mtri.UniformTriRefiner(tri_Kt).refine_triangulation(subdiv=5)
        xi_1 = tri_Kt.x
        xi_2 = tri_Kt.y
        self.phi_plot = shape_functions_Kt(xi_1, xi_2, order)[0]


    def __str__(self):
        out = "K_t of order {}".format(self.order)
        return out

class PlotKt:
    def __init__(self, order=2):
        self.order = order
        x = np.asarray([-1,1,-1])
        y = np.asarray([-1,-1,1])
        tri = mtri.Triangulation(x, y)

        refiner = mtri.UniformTriRefiner(tri)
        fine_tri = refiner.refine_triangulation(subdiv=9)

        xi_1 = fine_tri.x
        xi_2 = fine_tri.y

        self.Phi, self.dPhi = shape_functions_Kt(xi_1, xi_2, self.order)

        for i_SF in range(self.Phi.shape[0]):
            plt.figure()
            plt.tricontourf(fine_tri, self.Phi[i_SF,:],cmap=cm.jet,levels=10)
            plt.colorbar()
        plt.show()

def shape_functions_Kt(xi_1, xi_2, order):
    ''' Return Lobatto Shape functions on Kt'''
    nb_SF = 3*order+int(((order-1)*(order-2))/2)

    lamda = np.zeros((3,len(xi_1)))
    dlamda = [np.zeros((3, len(xi_1))),np.zeros((3, len(xi_1)))]
    # Equations (2.17) of Solin 
    lamda[0,:] = (xi_2 +1.)/2.
    dlamda[1][0, :] = 1./2.
    lamda[1, :] = -(xi_1 +xi_2)/2.
    dlamda[0][1, :] = -1./2.
    dlamda[1][1, :] = -1./2.
    lamda[2, :] = (xi_1 +1.)/2.
    dlamda[0][2, :] = 1./2.

    # Initialisation of Shape Functions and derivatives
    Phi = np.zeros((nb_SF, len(xi_1)))
    dPhi = [np.zeros((nb_SF, len(xi_1))),np.zeros((nb_SF, len(xi_1)))]

    # Vertices Shape Functions Eq (2.20) of Solin
    Phi[0, :], Phi[1, :], Phi[2, :] = lamda[1, :], lamda[2, :], lamda[0, :]
    for _xi in range(2):
        dPhi[_xi][0, :], dPhi[_xi][1, :], dPhi[_xi][2, :] = dlamda[_xi][1, :], dlamda[_xi][2, :], dlamda[_xi][0, :]
    # Edge Shape Functions Eq (2.21) of Solin 
    for _o in range(order-1):
        _index = 3+_o
        Phi[_index, :] = lamda[1, :]*lamda[2, :]*kernel(_o, lamda[2, :]-lamda[1, :])[0]
        for _xi in range(2):
            dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[2, :]*kernel(_o, lamda[2, :]-lamda[1, :])[0]
            dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[1, :]*kernel(_o, lamda[2, :]-lamda[1, :])[0]
            dPhi[_xi][_index, :] += lamda[1, :]*lamda[2, :]*kernel(_o, lamda[2, :]-lamda[1,:])[1]*(dlamda[_xi][2, :]-dlamda[_xi][1, :])

        _index = 3+order-1 + _o
        Phi[_index, :] = lamda[2, :]*lamda[0, :]*kernel(_o, lamda[0, :]-lamda[2, :])[0]
        for _xi in range(2):
            dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[0, :]*kernel(_o, lamda[0, :]-lamda[2, :])[0]
            dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[2, :]*kernel(_o, lamda[0, :]-lamda[2, :])[0]
            dPhi[_xi][_index, :] += lamda[0, :]*lamda[2, :]*kernel(_o, lamda[0, :]-lamda[2,:])[1]*(dlamda[_xi][0, :]-dlamda[_xi][2, :])
        _index = 3+2*(order-1)+ _o
        Phi[_index, :] = lamda[0, :]*lamda[1, :]*kernel(_o, lamda[1, :]-lamda[0, :])[0]
        for _xi in range(2):
            dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[1, :]*kernel(_o, lamda[1, :]-lamda[0, :])[0]
            dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[0, :]*kernel(_o, lamda[1, :]-lamda[0, :])[0]
            dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*kernel(_o, lamda[1, :]-lamda[0,:])[1]*(dlamda[_xi][1, :]-dlamda[_xi][0, :])

    # Face Shape Functions Eq (2.23) of Solin
    _index = 3*order
    for n_1 in range(1, order):
        for n_2 in range(1, order-n_1):
            Phi[_index, :] = lamda[0, :]*lamda[1, :]*lamda[2, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
            for _xi in range(2):
                dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[1, :]*lamda[2, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[2, :]*lamda[0, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[0, :]*lamda[1, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[1]*(dlamda[_xi][2, :]-dlamda[_xi][1, :])*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[1]*(dlamda[_xi][1, :]-dlamda[_xi][0, :])*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]
            _index += 1

    return Phi, dPhi
