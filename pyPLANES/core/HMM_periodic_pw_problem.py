#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# HMM_periodic_pw_problem.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@univ-lemans.fr>
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

from pyPLANES.pw.periodic_multilayer_HMM import PeriodicMultiLayer_HMM
from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *



class HMMPeriodicPwProblem(Calculus, PeriodicMultiLayer_HMM):
    """
        Periodic recursive Problem
    """ 
    def __init__(self, **kwargs):
        assert "ml" in kwargs
        ml = kwargs.get("ml")
        self.condensation = kwargs.get("condensation", True)
        Calculus.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.method_TM = kwargs.get("method_TM", False)
        
        if self.method_TM in ["cheb_1", "cheb_2"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        self.termination = kwargs.get("termination", "rigid")

        self.order = kwargs.get("order", 2)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)
        self.result.order = self.order
        self.result.Solver = type(self).__name__
        # Read periodic multilayer
        PeriodicMultiLayer_HMM.__init__(self, ml, theta_d=self.theta_d, order=self.order, plot=self.plot,method="HMM",  condensation=self.condensation)
        

        self.add_excitation_and_termination(self.termination)
        # Calculus variable (for pylint)
        self.kx, self.ky, self.k = None, None, None
        self.R, self.T = None, None

    def preprocess(self):
        Calculus.preprocess(self)
        self.info_file.write("Periodic Plane Wave solver // Recursive method\n")

    def update_frequency(self, omega):
        Calculus.update_frequency(self, omega)
        self.omega = omega
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
        self.ky = np.real(k_y)-1j*np.imag(k_y) # ky is either real or imaginary // - is to impose the good sign
        PeriodicMultiLayer_HMM.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        self.H = self.interfaces[-1].HMM_update().astype(complex) 

        for i, _l in enumerate(self.layers[::-1]):
            print(f"Layer {len(self.layers)-i-1} // {type(_l).__name__}")
            _l.H_top = self.H.copy()
            self.H = _l.HMM_update(self.H)
            self.H = self.interfaces[-i-2].HMM_update(self.H)
            self.H = self.interfaces[-i-2].HMM_update(self.H)

        
        
        
    def solve(self):
        Calculus.solve(self)
        H = self.H[0,0]
        self.result.R0 = [(np.cos(self.theta_d*pi/180)-Air.Z*H)/(np.cos(self.theta_d*pi/180)+Air.Z*H)]
        self.resultabs = 1-np.abs(self.result.R0)**2        
        if self.termination == "transmission":
            self.result.T0 = np.array([1+self.R0]).reshape((1,1))
            for i, _int in enumerate(self.interfaces[:-1]):
                self.result.T0 = _int.I_cal@self.T0
                self.result.T0 = self.layers[i].L_cal@self.T0
            self.result.T0 = self.T0.flatten()[0]
            self.result.abs -= np.abs(self.result.T0)**2
        
        
        

    def plot_solution(self):
        if self.method in ["Recursive Method", "TMM"]:
            if not(isinstance(self.X_0_minus,np.ndarray)):
                X_minus = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            else:
                X_minus=self.X_0_minus
            for i, _l in enumerate(self.layers):
                X_plus = self.interfaces[i].Tau @ X_minus # Transfert through the interface x^+
                X_minus = _l.Xi@X_plus
                if isinstance(_l, PeriodicLayer):
                    S_b = _l.Omega_plus @ X_plus
                    S_t = _l.Omega_minus @ X_minus
                    _l.plot_solution(S_b, S_t)
                else:   
                    q = LA.solve(_l.SV, _l.Omega_plus@X_plus)
                    _l.plot_solution_recursive(self.plot, q)
        elif self.method == "characteristics":
            if not(isinstance(self.X_0_minus,np.ndarray)):
                q_minus = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            else:
                q_minus=self.X_0_minus
            for i, _l in enumerate(self.layers):
                q_plus = self.interfaces[i].Tau @ q_minus # Transfert through the interface x^+
                q_minus = _l.Xi@q_plus # Transfert through the layer x^-_{+1}
                if isinstance(_l, PeriodicLayer):
                    S_b = _l.Omega_plus @ q_plus
                    S_t = _l.Omega_minus @ q_minus
                    _l.plot_solution(S_b, S_t)
                else:                
                    _l.plot_solution_characteristics(self.plot, _l.Omega_minus@q_minus)
        elif self.method == "Global Method":
            for _l in self.layers[1:]:
                if isinstance(_l, PeriodicLayer):
                    S_b = self.X[_l.dofs_bottom-self.nb_waves]
                    S_t = self.X[_l.dofs_top-self.nb_waves]
                    _l.plot_solution(S_b, S_t)
                    # _l.plot_solution_global(self.plot,self.X[_l.dofs-1])  