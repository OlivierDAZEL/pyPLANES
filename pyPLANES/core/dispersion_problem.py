#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_classes.py
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
from termcolor import colored
import numpy as np
import numpy.linalg as LA

from numpy import pi

import matplotlib.pyplot as plt

from mediapack import Air, Fluid

# from pyPLANES.utils.io import initialisation_out_files_plain
from pyPLANES.core.calculus import Calculus
from pyPLANES.pw.periodic_layer import PeriodicLayer

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *


class DispersionProblem(Calculus, PeriodicLayer):
    """
        eTMM Problem
    """ 
    def __init__(self, **kwargs):
        assert "name_mesh" in kwargs
        name_mesh = kwargs.get("name_mesh")
        self.condensation = kwargs.get("condensation", True)
        Calculus.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        if self.theta_d == 0:
            self.theta_d = 1e-15 
        self.order = kwargs.get("order", 2)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)

        self.Result.order = self.order
        self.Result.Solver = type(self).__name__
        self.Result.k =[]

        # Out files
        self.out_file_method = "dispersion"
        if isinstance(name_mesh, str):
            PeriodicLayer.__init__(self, name_mesh=name_mesh, theta_d= self.theta_d, verbose=self.verbose, order=self.order, plot=self.plot, condensation=self.condensation)
            self.Result.period = self.period
        elif isinstance(name_mesh, list):
        # Calculus variable (for pylint)
            self.kx, self.ky, self.k = None, None, None


    def preprocess(self):
        Calculus.preprocess(self)
        self.info_file.write("Periodic Plane Wave solver // Recursive method\n")

    def update_frequency(self, omega):
        Calculus.update_frequency(self, omega)
        self.k_air = omega/Air.c
        k_x = self.k_air*np.sin(self.theta_d*np.pi/180.)
        if self.period:
            if self.nb_bloch_waves is not False:
                nb_bloch_waves = self.nb_bloch_waves
            else: 
                nb_bloch_waves = int(np.floor((self.period/(2*pi))*(3*np.real(self.k_air)-k_x))+3)
            self.nb_waves = 1+2*nb_bloch_waves
            _ = np.zeros(self.nb_waves)
            for i in range(nb_bloch_waves):
                _[1+2*i]= i+1
                _[2+2*i]= -(i+1)
            self.kx = k_x+_*(2*pi/self.period)
            k_y = np.sqrt(self.k_air**2-self.kx**2+0*1j)
        else:
            self.nb_waves = 1
            self.kx = np.array([k_x])
            k_y = np.sqrt(self.k_air**2-self.kx**2+0*1j)
        # print("ky={}".format(k_y))
        self.ky = np.real(k_y)-1j*np.imag(k_y) # ky is either real or imaginary to have the good sign
        # print(self.ky)
        PeriodicLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        pass


    def solve(self):
        Calculus.solve(self)
        print(self.TM.shape)
        lam, v  = LA.eig(self.TM)
        k = np.log(lam)/(1j*self.period)
        k = k[np.argsort(np.real(k))] 
        self.Result.k.append(k)
        # print(k)

        # from mediapack import Air
        # k_a = 2*np.pi*self.f/Air.c
        # k_x = np.arange(-2, 3)*2*np.pi/self.period
        # print(np.sqrt(k_a**2-k_x**2+0*1j))
        # fdsffds





    def plot_solution(self):
        x_minus = self.X_0_minus # Information vector at incident interface  x^-
        # print("x_minus={}".format(x_minus))
        if not isinstance (x_minus, np.ndarray):
            x_minus = np.array([x_minus])
        for i, _l in enumerate(self.layers):
            x_plus = self.interfaces[i].Tau @ x_minus # Transfert through the interface x^+
            # print(x_plus)
            if isinstance(_l, PeriodicLayer):
                S_b = _l.Omega_plus @ x_plus
                S_t = LA.solve(_l.TM, S_b)
                _l.plot_solution(S_b, S_t)
            else: # Homogeneous layer
                q = LA.solve(_l.SV, _l.Omega_plus@x_plus)
                _l.plot_solution_recursive(self.plot, q)
            x_minus = LA.inv(_l.Xi)@x_plus # Transfert through the layer x^-_{+1}
