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

        self.Results["R0"], self.Results["T0"] = [], []
        self.Results["n_dof"] = []
        self.Results["order"] = self.order

        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)
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
                nb_bloch_waves = int(np.ceil((self.period/(2*pi))*(3*np.real(self.k_air)-k_x))+10)
            nb_bloch_waves = 0
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
            self.Results["R0"].append(R0)
            abs = 1-np.abs(R0)**2
            self.X_0_minus = 2*_/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.Results["T0"].append(Omega_end[0])

        else:
            M = np.zeros((2*self.nb_waves, 2*self.nb_waves), dtype=complex)
            M[:,:self.nb_waves] = self.Omega[:2*self.nb_waves,:self.nb_waves]

            _ = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            Omega_0 = np.array([_, 1], dtype=complex).reshape((2,1))
            E_0 = np.zeros(2*self.nb_waves, dtype=complex)
            E_0[:2] = np.array([-_, 1]).reshape((2))

            for _w in range(self.nb_waves):
                _ = 1j*(self.ky[_w]/self.k_air)/(2*pi*self.f*Air.Z)
                Omega_0 = np.array([_, 1], dtype=complex).reshape((2,1))
                M[2*_w:2*(_w+1),self.nb_waves+_w] = -Omega_0.reshape(2)

            X = LA.solve(M, E_0)
            R = X[self.nb_waves:]
            self.Results["R0"].append(R[0])
            # print(R[0])
            # self.Results["R"].append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))
            # abs = 1-self.result.R
            self.X_0_minus = X[:self.nb_waves]
            if self.termination == "transmission":
                T = (self.back_prop@self.X_0_minus)[::self.interfaces[-1].len_X]
                self.Results["T0"].append(T[0])
                # self.Results["T"].append(np.sum(np.real(self.ky)*np.abs(T**2))/np.real(self.ky[0]))
            #     abs -= result.T
            # self.Results["abs"].append(abs)

    def results_to_json(self):
        self.Results["real(R0)"] = np.real(self.Results["R0"]).tolist()
        self.Results["imag(R0)"] = np.imag(self.Results["R0"]).tolist()
        del self.Results["R0"]
        self.Results["real(T0)"] = np.real(self.Results["T0"]).tolist()
        self.Results["imag(T0)"] = np.imag(self.Results["T0"]).tolist()
        del self.Results["T0"]


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

    def load_results(self):
        data = np.loadtxt(self.out_file_name)
        f  = data[:, 0]
        R  = data[:, 1] + 1j*data[:, 2]
        if self.termination == "transmission":
            T = data[:,3] + 1j*data[:,4]
        else:
            T= False
        return f, R, T

    def compute_error(self, f_ref, R_ref, T_ref, plot_RT=False, eps=1e-8):
        # self.frequencies = f_ref
        self.resolution()
        R, T = self.load_results()[1:]
        error_R = LA.norm(R-R_ref)
        error_T = LA.norm(T-T_ref)
        _ = np.sqrt(error_R**2+error_T**2)/len(f_ref)
        if plot_RT:
            self.plot_results()

        if _< eps:
            print("Overall error = {}".format(_) + "\t"*6 + "["+ colored("OK", "green")  +"]")
        else:
            print("Overall error = {}".format(_) + "\t"*6 + "["+ colored("Fail", "red")  +"]")
        return error_R, error_T

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