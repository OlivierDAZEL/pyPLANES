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

from pyPLANES.pw.periodic_multilayer import PeriodicMultiLayer

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *


class PeriodicPwProblem(Calculus, PeriodicMultiLayer):
    """
        eTMM Problem
    """ 
    def __init__(self, **kwargs):
        assert "ml" in kwargs
        ml = kwargs.get("ml")
        self.condensation = kwargs.get("condensation", True)
        Calculus.__init__(self, **kwargs)
        self.termination = kwargs.get("termination", "rigid")
        self.theta_d = kwargs.get("theta_d", 0.0)
        if self.theta_d == 0:
            self.theta_d = 1e-15 
        self.order = kwargs.get("order", 2)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)
        self.Result.order = self.order
        self.Result.Solver = type(self).__name__

        # Out files
        self.out_file_method = "eTMM"
        PeriodicMultiLayer.__init__(self, ml, theta_d=self.theta_d, order=self.order, plot=self.plot, condensation=self.condensation)
        # self.period = 5e-2
        # print(self.period)
        self.add_excitation_and_termination(self.termination)
        # Calculus variable (for pylint)
        self.kx, self.ky, self.k = None, None, None
        self.R, self.T = None, None

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
            # print("nb_bloch_waves={}".format(nb_bloch_waves))
            self.nb_waves = 1+2*nb_bloch_waves
            _ = np.array([0] + list(range(-nb_bloch_waves, 0)) + list(range(1, nb_bloch_waves+1)))
            self.kx = k_x+_*(2*pi/self.period)
            k_y = np.sqrt(self.k_air**2-self.kx**2+0*1j)
        else:
            self.nb_waves = 1
            self.kx = np.array([k_x])
            k_y = np.sqrt(self.k_air**2-self.kx**2+0*1j)
        # print("ky={}".format(k_y))
        self.ky = np.real(k_y)+1j*np.imag(k_y) # ky is either real or imaginary to have the good sign
        # print(self.ky)
        PeriodicMultiLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        if self.termination == "transmission":
            self.Omega, self.back_prop = self.interfaces[-1].Omega(self.nb_waves)
            for i, _l in enumerate(self.layers[::-1]):
                next_interface = self.interfaces[-i-2]
                _l.Omega_plus, _l.Xi = _l.transfert(self.Omega)
                self.back_prop = self.back_prop@_l.Xi
                self.Omega, next_interface.Tau = next_interface.transfert(_l.Omega_plus)
                self.back_prop = self.back_prop@next_interface.Tau
        else: # Rigid backing
            self.Omega = self.interfaces[-1].Omega(self.nb_waves)
            for i, _l in enumerate(self.layers[::-1]):                
                next_interface = self.interfaces[-i-2]
                _l.Omega_plus, _l.Xi = _l.transfert(self.Omega)
                self.Omega, next_interface.Tau = next_interface.transfert(_l.Omega_plus)

    def solve(self):
        Calculus.solve(self)
        
        if self.nb_waves == 0:
            self.Omega = self.Omega.reshape(2)
            _ = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            det = -self.Omega[0]+_*self.Omega[1]
            R0 = (self.Omega[0]+_*self.Omega[1])/det
            self.Result.R0.append(R0)
            # print(R0)
            abs = 1-np.abs(R0)**2
            self.X_0_minus = 2*_/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.Result.T0.append(Omega_end[0])

        else:
            M = np.zeros((2*self.nb_waves, 2*self.nb_waves), dtype=complex)
            
            M[:,:self.nb_waves] = self.Omega[:2*self.nb_waves,:self.nb_waves]

            _ = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            Omega_0 = np.array([_, 1], dtype=complex).reshape((2,1))
            E_0 = np.zeros(2*self.nb_waves, dtype=complex)
            E_0[:2] = np.array([-_, 1]).reshape((2))

            for _w in range(self.nb_waves):
                # print(self.ky[_w])
                _ = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
                Omega_0 = np.array([_, 1], dtype=complex).reshape((2,1))
                M[2*_w:2*(_w+1),self.nb_waves+_w] = -Omega_0.reshape(2)
            # import matplotlib.pyplot as plt
            # plt.matshow(np.log(np.abs(M)))
            # plt.show()
            # qsd
            X = LA.solve(M, E_0)
            R = X[self.nb_waves:]
            self.Result.R0.append(R[0])
            self.Result.R.append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))
            abs = 1-self.Result.R[-1]
            self.X_0_minus = X[:self.nb_waves]
            if self.termination == "transmission":
                T = (self.back_prop@self.X_0_minus)[::self.interfaces[-1].len_X]
                self.Result.T0.append(T[0])
                self.Result.T.append(np.sum(np.real(self.ky)*np.abs(T**2))/np.real(self.ky[0]))
                abs -= self.Result.T[-1]
            self.Result.abs.append(abs)


    def plot_solution(self):
        x_minus = self.X_0_minus # Information vector at incident interface  x^-
        if not isinstance (x_minus, np.ndarray):
            x_minus = np.array([x_minus])
        for i, _l in enumerate(self.layers):
            x_plus = self.interfaces[i].Tau @ x_minus # Transfert through the interface x^+
            
            if isinstance(_l, PeriodicLayer):
                S_b = _l.Omega_plus @ x_plus
                S_t = LA.solve(_l.TM, S_b)
                _l.plot_solution(S_b, S_t)
            else: # Homogeneous layer
                q = LA.solve(_l.SV, _l.Omega_plus@x_plus)
                _l.plot_solution_recursive(self.plot, q)
            x_minus = _l.Xi@x_plus # Transfert through the layer x^-_{+1}
