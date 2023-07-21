#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# lobatto_polynomials.py
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


def lagrange_on_Ka(order, xi):
    if order == 1:
        d = np.vstack(((1-xi)/2, (1+xi)/2))
        p = np.vstack((-1/2, 1/2))
    elif order ==2:
        d = np.vstack((xi*(xi-1)/2, xi*(1+xi)/2, 1-xi**2))
        p = np.vstack(((2*xi-1)/2, (2*xi+1)/2, -2*xi))
    return d, p 


def lagrange_on_Kt(order, xi_1, xi_2):
    """
    Lagrange shape functions on Ka

    Parameters
    ----------
    order : integer
        degree of the polynomial
    x_1, x_2 : ndarray
        coordinate in the reference element
    Returns
    -------
    p : ndarray
        Shape_Functions at x_1, x_2 
    dp_xi_1 : ndarray
        first derivative with respect to x_1      
    dp_xi_2 : ndarray
        first derivative with respect to x_3
    """
    if order == 1:
        d = [-(xi_1+xi_2)/2, (xi_1+1)/2, (xi_2+1)/2]
        dp_xi_1 = [- 1/2, 1/2, 0]
        dp_xi_2 = [-1/2, 0, 1/2]      

    elif order == 2:
        d = [(xi_1**2+xi_2**2+2*xi_1*xi_2+xi_1+xi_2)/2]
        dp_xi_1 = [(2*xi_1+2*xi_2+1)/2]
        dp_xi_2 = [(2*xi_2+2*xi_1+1)/2]
        d.append((xi_1**2+xi_1)/2)
        dp_xi_1.append((2*xi_1+1)/2)
        dp_xi_2.append(0)
        d.append((xi_2**2+xi_2)/2)
        dp_xi_1.append(0)
        dp_xi_2.append((2*xi_2+1)/2)
        d.append(-(xi_1**2+xi_1*xi_2+xi_1+xi_2))
        dp_xi_1.append(-(2*xi_1+xi_2+1))
        dp_xi_2.append(-(xi_1+1))
        d.append(xi_1*xi_2+xi_1+xi_2+1)
        dp_xi_1.append(xi_2+1)
        dp_xi_2.append(xi_1+1)
        d.append(-(xi_1*xi_2+xi_2**2+xi_1+xi_2))
        dp_xi_1.append(-(xi_2+1))
        dp_xi_2.append(-(xi_1+2*xi_2+1))
    return d, np.vstack((dp_xi_1, dp_xi_2))




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib import cm

    x = np.asarray([-1,1,-1])
    y = np.asarray([-1,-1,1])
    tri = mtri.Triangulation(x, y)

    refiner = mtri.UniformTriRefiner(tri)
    fine_tri = refiner.refine_triangulation(subdiv=9)

    xi_1 = fine_tri.x
    xi_2 = fine_tri.y

    order = 2
    Phi = lagrange_on_Kt(order,xi_1,xi_2)

    if order == 1 :
        for i_SF in range(3):
            v_1 = lagrange_on_Kt(order,-1,-1)[i_SF]
            v_2 = lagrange_on_Kt(order,1,-1)[i_SF]
            v_3 = lagrange_on_Kt(order,-1,1)[i_SF]
            print("SF[{}] -> {} / {} / {}".format(i_SF, v_1, v_2, v_3) )
    elif order == 2 :
        for i_SF in range(6):
            v_1 = lagrange_on_Kt(order,-1,-1)[i_SF]
            v_2 = lagrange_on_Kt(order,1,-1)[i_SF]
            v_3 = lagrange_on_Kt(order,-1,1)[i_SF]
            v_4 = lagrange_on_Kt(order,0,-1)[i_SF]
            v_5 = lagrange_on_Kt(order,0,0)[i_SF]
            v_6 = lagrange_on_Kt(order,-1,0)[i_SF]
            print("SF[{}] -> {} / {} / {} / {} / {} / {}".format(i_SF, v_1, v_2, v_3,v_4, v_5, v_6) )

    # for i_SF in range(len(Phi)):
    #     plt.figure()
    #     plt.tricontourf(fine_tri, Phi[i_SF],cmap=cm.jet,levels=10)
    #     plt.colorbar()
    # plt.show()
