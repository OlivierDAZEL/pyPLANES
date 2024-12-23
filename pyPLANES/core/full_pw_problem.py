#! /usr/bin/env python
# -*- coding:utf-8 -*-
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
from pyPLANES.pw.general.general_multilayer import GeneralMultiLayer
from pyPLANES.pw.window import Window



class FullPwProblem(Calculus, GeneralMultiLayer):
    def __init__(self, **kwargs):
        # General variables
        Calculus.__init__(self, **kwargs)
        self.result.Solver = type(self).__name__

        # Angles
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.angles = kwargs.get("angles", None)
        if self.angles is not None:
            self.theta_d = self.angles[0]
            self.phi_d = self.angles[1]

        # Windowing
        self.window = kwargs.get("window", False)
        if self.window is not False:
            self.window = Window(self.window[0], self.window[1])

        # Computation method
        self.method = kwargs.get("method", "Global Method")            
        if self.method.lower() in ["recursive", "jap", "recursive method"]:
            self.method = "Recursive Method"
        elif self.method.lower() in ["tmm", "transfer matrix method"]:
            self.method = "TMM"
        elif self.method.lower() in ["characteristics", "characteristic", "carac"]:
            self.method = "characteristics"
        else: 
            self.method = "Global Method"
        # Put a non zero angle for some methods
        if self.method in [ "TMM", "Recursive Method"]:
            if self.theta_d == 0:
                self.theta_d = 1e-12

        # Creation of the Multilayer  
        assert "ml" in kwargs
        GeneralMultiLayer.__init__(self, ml=kwargs.get("ml"), method=self.method, material_database=self.material_database)
        self.termination = kwargs.get("termination", "rigid")
        self.add_excitation_and_termination(self.termination)

        # Calculus variable (for pylint)
        self.kx, self.ky, self.kz, self.k = None, None, None, None
        self.R, self.T = None, None
        


    def update_frequency(self, omega):
        Calculus.update_frequency(self, omega)
        self.k_air = omega/Air.c
        self.kx = self.k_air*np.array([np.sin(self.theta_d*np.pi/180)])*np.array([np.cos(self.phi_d*np.pi/180)])
        self.kz = self.k_air*          np.sin(self.theta_d*np.pi/180)            *np.sin(self.phi_d*np.pi/180)
        self.ky = self.k_air*np.cos(self.theta_d*np.pi/180)

        # print(self.kx, self.ky, self.kz)
        # print(self.kx**2+self.ky**2+self.kz**2)
        GeneralMultiLayer.update_frequency(self, omega, self.kx, self.kz)

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
        elif self.method == "characteristics":
            if self.termination == "transmission":
                self.Omega, self.back_prop = self.interfaces[-1].Omegac()
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_minus = self.Omega
                    _l.Omega_plus, _l.Xi = _l.update_Omegac(self.Omega, omega, self.method)
                    self.back_prop = self.back_prop@_l.Xi
                    self.Omega, next_interface.Tau = next_interface.update_Omegac(_l.Omega_plus)
                    self.back_prop = self.back_prop@next_interface.Tau
            else: # Rigid backing
                self.Omega = self.interfaces[-1].Omegac()
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_minus = self.Omega
                    _l.Omega_plus, _l.Xi = _l.update_Omegac(self.Omega, omega, self.method)
                    self.Omega, next_interface.Tau = next_interface.update_Omegac(_l.Omega_plus)
        elif self.method == "Global Method":
            self.A = np.zeros((self.nb_PW-1, self.nb_PW),dtype=complex)
            i_eq = 0
            # Loop on the interfaces
            for _int in self.interfaces:
                if self.method == "Global Method":
                    i_eq = _int.update_M_global(self.A,i_eq)
            self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
            self.A = np.delete(self.A, 0, axis=1)
        else:
            raise NameError("Unknow method")
    def solve(self):
        Calculus.solve(self)
        if self.method in ["Recursive Method", "TMM", "characteristics"]:
            self.Omega = self.Omega.reshape(2)
            if self.method == "characteristics":
                self.Omega = self.interfaces[0].carac_bottom.P@self.Omega
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            det = -self.Omega[0]+alpha*self.Omega[1]
            self.result.R0.append((self.Omega[0]+alpha*self.Omega[1])/det)
            self.result.abs.append(1-np.abs(self.result.R0[-1])**2)
            self.X_0_minus = 2*alpha/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.result.T0.append(Omega_end[0])
                self.result.abs[-1] -= np.abs(self.result.T0[-1])**2
        elif self.method == "Global Method":
            self.X = LA.solve(self.A, self.F)
            # print(f"X={self.X}")
            R = self.X[0]
            # print("-----")
            # print(f"R={R}")
            # q = self.X[1:2*self.layers[1].nb_waves_in_medium+1]
            
            # self.n_b = self.layers[0].nb_waves_in_medium 
            # self.n_t = self.layers[1].nb_waves_in_medium
            # SV_b = self.layers[0].SV
            # SV_t = self.layers[1].SV
            # d_b = ([self.layers[0].d]*self.n_b+[0]*self.n_b)
            # d_t = ([0]*self.n_t +[-self.layers[1].d]*self.n_t)
            # delta_b = np.diag(np.exp(self.layers[0].lam*d_b))
            # delta_t = np.diag(np.exp(self.layers[1].lam*d_t))
            # print(f"SV_b={SV_b@delta_b@(np.array([np.exp(1j*self.ky*self.layers[0].d),R]).reshape((2,1)))}")
            # print(f"SV_t={SV_t@delta_t@q}")
            # d_t = ([self.layers[1].d]*self.n_t+[0]*self.n_t)
            # delta_t = np.diag(np.exp(self.layers[1].lam*d_t))
            
            # print(f"SV_0={SV_t@delta_t@q}")

                        
            # exit() 
             
            
            
            
            
            # exit()
            
            
            self.result.R0.append(self.X[0])
            self.result.abs.append(1-np.abs(self.result.R0[-1])**2)
            if self.termination == "transmission":
                self.result.T0.append(self.X[-1])
                if self.window:
                    self.window.update_frequency(2*pi*self.f)
                    sigma = self.window.sigma_average_Yu(self.k_air*np.sin(self.theta_d*pi/180))
                else:
                    sigma = 1/np.cos(self.theta_d*pi/180)

                self.tau_c = self.X[-1]
                self.win = np.cos(self.theta_d*pi/180)*sigma
                self.result.tau.append((np.abs(self.X[-1])**2)*np.cos(self.theta_d*pi/180)*sigma)
                self.result.abs[-1] -= np.abs(self.result.T0[-1])**2
        self.result.Z_prime.append((self.result.R0[-1]+1)/(1-self.result.R0[-1]))

    def plot_solution(self):
        if self.method == "Global Method":
            for _l in self.layers[1:]:
                _l.plot_solution_global(self.plot,self.X[_l.dofs-1])  
        elif self.method == "Recursive Method":
            x = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            for i, _l in enumerate(self.layers):
                x = self.interfaces[i].Tau @ x # Transfert through the interface x^+
                q = LA.solve(_l.SV, _l.Omega_plus@x)
                _l.plot_solution_recursive(self.plot, q)
                x = _l.Xi@x # Transfert through the layer x^-_{+1}
        elif self.method == "characteristics":
            q = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            for i, _l in enumerate(self.layers):
                q = self.interfaces[i].Tau @ q # Transfert through the interface x^+
                q = _l.Xi@q # Transfert through the layer x^-_{+1}
                _l.plot_solution_characteristics(self.plot, _l.Omega_minus@q)

        else: 
            raise NameError("No method")
