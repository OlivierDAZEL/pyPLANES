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
import numpy as np
import numpy.linalg as LA

from numpy import pi
import matplotlib.pyplot as plt
from mediapack import Air, Fluid


from pyPLANES.core.calculus import Calculus
from pyPLANES.pw.multilayer import MultiLayer


from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *

class PwProblem(Calculus, MultiLayer):
    """
        Plane Wave Problem 
    """
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        self.Result.Solver = type(self).__name__
        # self.Results["R0"], self.Results["T0"] = [], [] 
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.method = kwargs.get("method", "Global Method")
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
            self.method = "Global Method"
        self.method_TM = kwargs.get("method_TM", "diag")
        if self.method_TM in ["cheb_1"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        
        assert "ml" in kwargs
        ml = kwargs.get("ml")

        MultiLayer.__init__(self, ml, self.method_TM)
        if self.method_TM in ["cheb_1"]:
            for l in self.layers:
                l.order_chebychev = self.order_chebychev 
        self.termination = kwargs.get("termination", "rigid")
        self.add_excitation_and_termination(self.method, self.termination)

        # Calculus variable (for pylint)
        self.kx, self.ky, self.k = None, None, None
        self.R, self.T = None, None

    def update_frequency(self, omega):
        Calculus.update_frequency(self, omega)
        self.kx = np.array([omega*np.sin(self.theta_d*np.pi/180)/Air.c])
        self.ky = np.array([omega*np.cos(self.theta_d*np.pi/180)/Air.c])
        self.k_air = omega/Air.c
        MultiLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        if self.method in ["Recursive Method", "TMM"]:
            if self.termination == "transmission":
                self.Omega, self.back_prop = self.interfaces[-1].Omega()
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                    self.back_prop = self.back_prop@_l.Xi
                    self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)
                    self.back_prop = self.back_prop@next_interface.Tau

            else: # Rigid backing
                self.Omega = self.interfaces[-1].Omega()
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                    self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)

        elif self.method == "Global Method":
            self.A = np.zeros((self.nb_PW-1, self.nb_PW),dtype=complex)
            i_eq = 0
            # Loop on the interfaces
            for _int in self.interfaces:
                if self.method == "Global Method":
                    i_eq = _int.update_M_global(self.A,i_eq)
            self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
            self.A = np.delete(self.A, 0, axis=1)


    def solve(self):
        Calculus.solve(self)
        if self.method in ["Recursive Method", "TMM"]:
            self.Omega = self.Omega.reshape(2)
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            det = -self.Omega[0]+alpha*self.Omega[1]
            self.Result.R0.append((self.Omega[0]+alpha*self.Omega[1])/det)
            if self.verbose:
                print("R_0={}".format(self.Result.R0))
            print("R_0={}".format(self.Result.R0))
            self.Result.abs.append(1-np.abs(self.Result.R0[-1])**2)
            self.X_0_minus = 2*alpha/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.Result.T0.append(Omega_end[0])
                self.Result.abs[-1] -= np.abs(self.Result.T0[-1])**2
        elif self.method == "Global Method":
            self.X = LA.solve(self.A, self.F)
            self.Result.R0.append(self.X[0])
            if self.verbose:
                print("R_0={}".format(self.Result.R0))
            self.Result.abs.append(1-np.abs(self.Result.R0[-1])**2)
            if self.termination == "transmission":
                self.Result.T0.append(self.X[-1])
                self.Result.abs[-1] -= np.abs(self.Result.T0[-1])**2

    def plot_solution(self):
        if self.method == "Global Method":
            for _l in self.layers[1:]:
                _l.plot_solution_global(self.plot,self.X[_l.dofs-1])  
        else:
            x = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            for i, _l in enumerate(self.layers):
                x = self.interfaces[i].Tau @ x # Transfert through the interface x^+
                q = LA.solve(_l.SV, _l.Omega_plus@x)
                _l.plot_solution_recursive(self.plot, q)
                x = _l.Xi@x # Transfert through the layer x^-_{+1}

    def load_results(self):

        name_file_with_method = self.out_file_name.split(".")
        name_file_with_method.insert(1, self.out_file_extension)
        name_file_with_method = ".".join(name_file_with_method)

        data = np.loadtxt(name_file_with_method)
        f  = data[:, 0]
        R  = data[:, 1] + 1j*data[:, 2]
        if self.termination == "transmission":
            T = data[:,3] + 1j*data[:,4]
        else:
            T= False
        return f, R, T

    def plot_results(self):
        if self.method == "Recursive Method":
            method = " RM"
            marker = "."
        elif self.method == "Global Method":
            method = " RM"
            marker = "+"
        f, R, T = self.load_results()
        plt.figure(self.name_project + "/ Reflection coefficient")
        plt.plot(f, np.real(R),"r"+marker,label="Re(R)"+method)
        plt.plot(f, np.imag(R),"b"+marker,label="Im(R)"+method)
        plt.legend()
        if self.termination == "transmission":
            plt.figure(self.name_project + "/ Transmission coefficient")
            plt.plot(f, np.real(T),"r"+marker,label="Re(T)"+method)
            plt.plot(f, np.imag(T),"b"+marker,label="Im(T)"+method)
            plt.legend()