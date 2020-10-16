#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_interfaces.py
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

from mediapack import Air
import numpy as np
import matplotlib.pyplot as plt


class PwLayer():
    """
    Layer for Plane Wave Solver
    
    Attributes :
    ------------------------

    mat : mediapack material 

    d : thickness of the layer 

    x : list with the abscissa of the layers 

    interfaces : list with the interfaces of the layer
    
    """
    def __init__(self, mat, d, x_0=0):
        """
        Parameters
        ----------
        mat : mediapack medium 
            Name of the material 
        d : float
            thickness of the layer
        dofs : ndarray list of dofs        
        """
        self.medium = mat 
        self.d = d
        # pymls layer constructor 
        self.x = [x_0, x_0+self.d]  # 
        self.interfaces = [None, None]  # 
        self.dofs = None
        self.ky = None
        self.SV = None
        self.SVp = None

    def __str__(self):
        pass


    def update_frequency(self, f, k, kx):
        self.medium.update_frequency(f, k, kx)


class FluidLayer(PwLayer):
    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)

    def __str__(self):
        out = "\t Fluid Layer / " + self.medium.name
        return out

    def update_frequency(self, f, k, kx):
        PwLayer.update_frequency(self, f, k, kx)
        self.ky = np.sqrt(k**2-kx**2)
        self.SV = np.array([[self.ky/(1j*self.medium.K*k**2), -self.ky/(1j*self.medium.K*k**2)],[1, 1]], dtype=complex)

    def plot_sol(self, plot, X, nb_points=200):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        pr = self.SV[1, 0]*np.exp(-1j*self.ky*x_f)*X[0]
        pr += self.SV[1, 1]*np.exp( 1j*self.ky*x_b)*X[1]
        ut = self.SV[0, 0]*np.exp(-1j*self.ky*x_f)*X[0]
        ut += self.SV[0, 1]*np.exp( 1j*self.ky*x_b)*X[1]
        print(plot)
        if plot[2]:
            plt.figure(2)
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r')
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm')
            plt.title("Pressure")






