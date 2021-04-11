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
from pyPLANES.core.periodic_multilayer import PeriodicMultiLayer

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *


class PeriodicPwProblem(Calculus, PeriodicMultiLayer):
    """
        eTMM Problem
    """ 
    def __init__(self, **kwargs):
        assert "ml" in kwargs
        ml = kwargs.get("ml")
        Calculus.__init__(self, **kwargs)
        PeriodicMultiLayer.__init__(self, ml)
        self.termination = kwargs.get("termination", "rigid")
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.method = "Recursive Method"
        self.add_excitation_and_termination(self.method, self.termination)

        # Out files
        self.out_file_name = self.file_names + ".eTMM.txt"
        self.info_file_name = self.file_names + ".info.eTMM.txt"     
        # Calculus variable (for pylint)
        self.kx, self.ky, self.k = None, None, None
        self.R, self.T = None, None

    def preprocess(self):
        Calculus.preprocess(self)
        if self.method == "Global Method":
            self.info_file.write("Plane Wave solver // Global method\n")        
        elif self.method == "Recursive Method":
            self.info_file.write("Plane Wave solver // Recursive method\n")

    def update_frequency(self, omega):
        Calculus.update_frequency(self, omega)
        self.kx = omega*np.sin(self.theta_d*np.pi/180)/Air.c
        self.ky = omega*np.cos(self.theta_d*np.pi/180)/Air.c
        self.k = omega/Air.c
        PeriodicMultiLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        if self.termination == "transmission":
            self.Omega, self.back_prop = self.interfaces[-1].Omega()
            for i, _l in enumerate(self.layers[::-1]):
                self.Omega, Xi = _l.transfert(self.Omega)
                self.back_prop = self.back_prop@Xi
                self.Omega, Tau = self.interfaces[-i-2].transfert(self.Omega)
                self.back_prop = self.back_prop@Tau
        else: # Rigid backing
            self.Omega = self.interfaces[-1].Omega()
            for i, _l in enumerate(self.layers[::-1]):
                self.Omega = _l.transfert(self.Omega)[0]
                self.Omega = self.interfaces[-i-2].transfert(self.Omega)[0]

    def solve(self):
        Calculus.solve(self)
        self.Omega = self.Omega.reshape(2)
        _ = (self.ky/self.k)/(1j*2*pi*self.f*Air.Z) 
        detM = -self.Omega[0]+_*self.Omega[1]
        self.R = (self.Omega[0]+_*self.Omega[1])/detM
        if self.termination == "transmission":
            X_0_minus = 2*_/detM
            self.Omega = (self.back_prop*X_0_minus).flatten()
            self.T = self.Omega[0]
        print(np.abs(self.R))


    def plot_solution(self):
         for _layer in self.layers[1:]:
            _layer.plot_solution(self.plot, self.X[_layer.dofs-1])  

    def write_out_files(self):
        self.out_file.write("{:.12e}\t".format(self.f))
        self.out_file.write("{:.12e}\t".format(self.R.real))
        self.out_file.write("{:.12e}\t".format(self.R.imag))
        if self.termination == "transmission":
            self.out_file.write("{:.12e}\t".format(self.T.real))
            self.out_file.write("{:.12e}\t".format(self.T.imag))
        self.out_file.write("\n")

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