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
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.method = kwargs.get("method", "jap")
        # print(self.method)
        if self.method.lower() in ["recursive", "jap", "recursive method"]:
            self.method = "Recursive Method"
            if self.theta_d == 0:
                self.theta_d = 1e-12
        elif self.method.lower() in ["tmm", "transfer matrix method"]:
            self.method = "TMM"
            if self.theta_d == 0:
                self.theta_d = 1e-12
        else: 
            raise NameError("Invalid method in PeriodicPwProblem")
        
        self.method_TM = kwargs.get("method_TM", False)
        
        if self.method_TM in ["cheb_1", "cheb_2"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        self.termination = kwargs.get("termination", "rigid")

        self.order = kwargs.get("order", 2)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)
        self.Result.order = self.order
        self.Result.Solver = type(self).__name__

        # Out files
        self.out_file_method = "eTMM"

        PeriodicMultiLayer.__init__(self, ml, theta_d=self.theta_d, order=self.order, plot=self.plot, condensation=self.condensation)
     

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
        self.ky = np.real(k_y)-1j*np.imag(k_y) # ky is either real or imaginary // - is to impose the good sign
        PeriodicMultiLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        # print(self.verbose)
        Calculus.create_linear_system(self, omega)
        if self.termination == "transmission":
            self.Omega, self.back_prop = self.interfaces[-1].Omega(self.nb_waves)
            for i, _l in enumerate(self.layers[::-1]):
                next_interface = self.interfaces[-i-2]
                _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                # print("_l.Xi=\n{}".format(_l.Xi))
                self.back_prop = self.back_prop@_l.Xi
                self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)
                # print("tau=\n{}.".format(next_interface.Tau))
                self.back_prop = self.back_prop@next_interface.Tau
            # print("backprop=\n{}".format(self.back_prop))
        else: # Rigid backing
            self.Omega = self.interfaces[-1].Omega(self.nb_waves)
            for i, _l in enumerate(self.layers[::-1]):
                next_interface = self.interfaces[-i-2]
                # print(_l)
                _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)


    def solve(self):
        Calculus.solve(self)
        if self.nb_waves == 0:
            self.Omega = self.Omega.reshape(2)
            _ = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            det = -self.Omega[0]+_*self.Omega[1]
            R0 = (self.Omega[0]+_*self.Omega[1])/det
            self.Result.R0.append(R0)
            if self.verbose:
                print("R_0={}".format(R0))
            abs = 1-np.abs(R0)**2
            self.X_0_minus = 2*_/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.Result.T0.append(Omega_end[0])
                if self.verbose:
                    print("T_0={}".format(Omega_end[0]))
        else:
            # Excitation (10) and (11) JAP
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            E_0 = np.zeros(2*self.nb_waves, dtype=complex)
            E_0[:2] = np.array([-alpha, 1]).reshape((2))

            # Matrix inverted eq (11) in JAP
            M = np.zeros((2*self.nb_waves, 2*self.nb_waves), dtype=complex)
            # [Omega_0^-] of JAP
            M[:,:self.nb_waves] = self.Omega[:,:self.nb_waves]
            # [-Omega_0] second part of JAP
            for _w in range(self.nb_waves):
                alpha_w = 1j*(self.ky[_w]/self.k_air)/(2*pi*self.f*Air.Z)
                Omega_0 = np.array([alpha_w, 1], dtype=complex).reshape((2,1))
                M[2*_w:2*(_w+1),self.nb_waves+_w] = -Omega_0.reshape(2)

            X = LA.solve(M, E_0)
            # R is second part of the solution vector
            R = X[self.nb_waves:]
            # print("R={}".format(R))
            if self.verbose:
                print("R={}".format(R))
            print("R={}".format(R))
            self.Result.R0.append(R[0])

            self.Result.R.append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))
            abs = 1-self.Result.R[-1]

            self.X_0_minus = X[:self.nb_waves]
            # print("XO={}".format(self.X_0_minus))
            if self.termination == "transmission":
                # print(self.back_prop.shape)
                # print(self.back_prop)
                # print(self.back_prop@self.X_0_minus)
                T = (self.back_prop@self.X_0_minus)[::self.interfaces[-1].len_X]  
                self.Result.T0.append(T[0])
                if self.verbose:
                    print("T={}".format(T))
                self.Result.T.append(np.sum(np.real(self.ky)*np.abs(T**2))/np.real(self.ky[0]))
                abs -= self.Result.T[-1]
            self.Result.abs.append(abs)
            # if self.verbose:
            #     print("abs={}".format(abs))





    def plot_solution(self):
        x_minus = self.X_0_minus # Information vector at incident interface  x^-
        # print("x_minus={}".format(x_minus))
        if not isinstance (x_minus, np.ndarray):
            x_minus = np.array([x_minus])
        for i, _l in enumerate(self.layers):
            x_plus = self.interfaces[i].Tau @ x_minus # Transfert through the interface x^+
            # print("x_plus={}".format(x_plus))
            if isinstance(_l, PeriodicLayer):
                S_b = _l.Omega_plus @ x_plus
                # S_b[0] *= -1
                S_t = LA.solve(_l.TM, S_b)
                # S_t[0] *= -1
                # print("S_b={}".format(S_b))
                # print("S_t={}".format(S_t))
                _l.plot_solution(S_b, S_t)
            else: # Homogeneous layer
                q = LA.solve(_l.SV, _l.Omega_plus@x_plus)
                _l.plot_solution_recursive(self.plot, q)
            x_minus = LA.inv(_l.Xi)@x_plus # Transfert through the layer x^-_{+1}
