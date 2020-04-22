#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# reference_elements.py
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
import importlib

from pyPLANES.fem.elements.lobatto_polynomials import lobatto as l
from pyPLANES.fem.elements.lobatto_polynomials import lobatto_kernels as phi

import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from numpy.polynomial.legendre import leggauss
import quadpy as quadpy





class Ka:
    def __init__(self, order=2, p= 4):
        self.order = order
        self.xi, self.w = leggauss(p)
        # Number of Shape Functions
        self.nb_v = 2
        self.nb_e = (order-1)
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

class Kt:
    def __init__(self, order=2, p= 4):
        self.order = order
        scheme = eval("quadpy.triangle.dunavant_"+'{:02}'.format(2*order) +"()")
        _points = scheme.points.dot(np.array([[-1, -1], [1, -1], [-1, 1]]))
        self.xi_1, self.xi_2, self.w = _points[:,0],_points[:,1],2*scheme.weights

        # Number of Shape Functions
        self.nb_v = 3
        self.nb_e = 3*(order-1)
        self.nb_f = int(((order-1)*(order-2))/2)
        # print("self.nb_v,e,f")
        # print("({},{},{})".format(self.nb_v, self.nb_e, self.nb_f))
        self.nb_SF = self.nb_v+self.nb_e+self.nb_f
        self.Phi, self.dPhi = shape_functions_Kt(self.xi_1,self.xi_2,self.order)

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

        lamda = np.zeros((3,len(xi_1)))
        dlamda = [np.zeros((3, len(xi_1))),np.zeros((3, len(xi_1)))]

        lamda[0,:] = (xi_2 +1.)/2.
        dlamda[1][0, :] = 1./2.
        lamda[1, :] = -(xi_1 +xi_2)/2.
        dlamda[0][1, :] = -1./2.
        dlamda[1][1, :] = -1./2.
        lamda[2, :] = (xi_1 +1.)/2.
        dlamda[0][2, :] = 1./2.

        # Number of Shape Functions
        self.nb_v = 3
        self.nb_e = 3*(order-1)
        self.nb_f = int(((order-1)*(order-2))/2)
        self.nb_SF = self.nb_v+self.nb_e+self.nb_f
        # Initialisation of Shape Functions and derivatives
        self.Phi = np.zeros((self.nb_SF, len(xi_1)))
        self.dPhi = [np.zeros((self.nb_SF, len(xi_1))),np.zeros((self.nb_SF, len(xi_1)))]

        # Vertices Shape Functions
        self.Phi[0, :], self.Phi[1, :], self.Phi[2, :] = lamda[1, :], lamda[2, :], lamda[0, :]
        for _xi in range(2):
            self.dPhi[_xi][0, :], self.dPhi[_xi][1, :], self.dPhi[_xi][2, :] = dlamda[_xi][1, :], dlamda[_xi][2, :], dlamda[_xi][0, :]
        # Edge Shape Functions
        for _o in range(order-1):
            _index_edge = [3+_o, 3+(order-1)+_o, 3+2*(order-1)+_o]
            for i_e in range(3):
                self.Phi[_index_edge[i_e], :] = lamda[(i_e+1)%3, :]*lamda[(i_e+2)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[0]
                for _xi in range(2):
                    self.dPhi[_xi][_index_edge[i_e], :] += dlamda[_xi][(i_e+1)%3, :]*lamda[(i_e+2)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[0]
                    self.dPhi[_xi][_index_edge[i_e], :] += dlamda[_xi][(i_e+2)%3, :]*lamda[(i_e+1)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[0]
                    self.dPhi[_xi][_index_edge[i_e], :] += lamda[(i_e+1)%3, :]*lamda[(i_e+2)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[1]*(dlamda[_xi][(i_e+2)%3, :]-dlamda[_xi][(i_e+1)%3, :])
        # Face Shape Functions
        _index = self.nb_v+self.nb_e
        for n_1 in range(1, order-1):
            for n_2 in range(1, n_1+1):
                self.Phi[_index, :] = lamda[0, :]*lamda[1, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                for _xi in range(2):
                    self.dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[1, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                    self.dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[0, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                    self.dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[1, :]*lamda[0, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                    self.dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[1]*(dlamda[_xi][2, :]-dlamda[_xi][1, :])*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                    self.dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*phi(n_2-1, lamda[1, :]-lamda[0, :])[1]*(dlamda[_xi][1, :]-dlamda[_xi][0, :])*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]
                _index += 1

        i_SF = 14
        plt.figure()
        plt.tricontourf(fine_tri, self.Phi[i_SF,:],cmap=cm.jet,levels=10)
        plt.colorbar()
        plt.show()

def shape_functions_Kt(xi_1, xi_2, order):
    ''' Return Lobatto Shape functions on Kt'''
    lamda = np.zeros((3,len(xi_1)))
    dlamda = [np.zeros((3, len(xi_1))),np.zeros((3, len(xi_1)))]

    lamda[0,:] = (xi_2 +1.)/2.
    dlamda[1][0, :] = 1./2.
    lamda[1, :] = -(xi_1 +xi_2)/2.
    dlamda[0][1, :] = -1./2.
    dlamda[1][1, :] = -1./2.
    lamda[2, :] = (xi_1 +1.)/2.
    dlamda[0][2, :] = 1./2.
    nb_SF = 3*order+int(((order-1)*(order-2))/2)
    # Initialisation of Shape Functions and derivatives
    Phi = np.zeros((nb_SF, len(xi_1)))
    dPhi = [np.zeros((nb_SF, len(xi_1))),np.zeros((nb_SF, len(xi_1)))]

    # Vertices Shape Functions
    Phi[0, :], Phi[1, :], Phi[2, :] = lamda[1, :], lamda[2, :], lamda[0, :]
    for _xi in range(2):
        dPhi[_xi][0, :], dPhi[_xi][1, :], dPhi[_xi][2, :] = dlamda[_xi][1, :], dlamda[_xi][2, :], dlamda[_xi][0, :]
    # Edge Shape Functions
    for _o in range(order-1):
        _index_edge = [3+_o, 3+(order-1)+_o, 3+2*(order-1)+_o]
        for i_e in range(3):
            Phi[_index_edge[i_e], :] = lamda[(i_e+1)%3, :]*lamda[(i_e+2)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[0]
            for _xi in range(2):
                dPhi[_xi][_index_edge[i_e], :] += dlamda[_xi][(i_e+1)%3, :]*lamda[(i_e+2)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[0]
                dPhi[_xi][_index_edge[i_e], :] += dlamda[_xi][(i_e+2)%3, :]*lamda[(i_e+1)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[0]
                dPhi[_xi][_index_edge[i_e], :] += lamda[(i_e+1)%3, :]*lamda[(i_e+2)%3, :]*phi(_o, lamda[(i_e+2)%3, :]-lamda[(i_e+1)%3, :])[1]*(dlamda[_xi][(i_e+2)%3, :]-dlamda[_xi][(i_e+1)%3, :])
    # Face Shape Functions
    _index = 3*order
    for n_1 in range(1, order-1):
        for n_2 in range(1, n_1+1):
            Phi[_index, :] = lamda[0, :]*lamda[1, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
            for _xi in range(2):
                dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[1, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[0, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[1, :]*lamda[0, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*phi(n_1-1, lamda[2, :]-lamda[1, :])[1]*(dlamda[_xi][2, :]-dlamda[_xi][1, :])*phi(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*phi(n_2-1, lamda[1, :]-lamda[0, :])[1]*(dlamda[_xi][1, :]-dlamda[_xi][0, :])*phi(n_1-1, lamda[2, :]-lamda[1, :])[0]
            _index += 1
    return Phi, dPhi