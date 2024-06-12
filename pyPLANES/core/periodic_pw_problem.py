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

from pyPLANES.pw.periodic_multilayer import PeriodicMultiLayer
from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *


class PeriodicPwProblem(Calculus, PeriodicMultiLayer):
    """
        Periodic recursive Problem
    """ 
    def __init__(self, **kwargs):
        assert "ml" in kwargs
        ml = kwargs.get("ml")
        self.condensation = kwargs.get("condensation", True)
        Calculus.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.method = kwargs.get("method", "Global Method")
        

        if self.method.lower() in ["recursive", "jap", "recursive method"]:
            self.method = "Recursive Method"
            if self.theta_d == 0:
                self.theta_d = 1e-12
        elif self.method.lower() in ["tmm", "transfer matrix method"]:
            self.method = "TMM"
            if self.theta_d == 0:
                self.theta_d = 1e-12
        elif self.method.lower() in ["characteristics", "characteristic", "carac"]:
            self.method = "characteristics"
        elif self.method.lower() in ["global", "characteristic", "carac"]:
            self.method = "Global Method"
        else: 
            raise NameError("Invalid method name: " + self.method)
        

        self.method_TM = kwargs.get("method_TM", False)
        
        if self.method_TM in ["cheb_1", "cheb_2"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        self.termination = kwargs.get("termination", "rigid")

        self.order = kwargs.get("order", 2)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)
        self.result.order = self.order
        self.result.Solver = type(self).__name__
        self.result.Method = self.method
        

        # Read periodic multilayer
        PeriodicMultiLayer.__init__(self, ml, theta_d=self.theta_d, order=self.order, plot=self.plot,method=self.method,  condensation=self.condensation)



        self.add_excitation_and_termination(self.termination)
        
        if self.method == "characteristics":
            for i_l, _layer in enumerate(self.layers):
                if isinstance(_layer, PeriodicLayer):
                    _layer.characteristics[0] = self.interfaces[i_l].carac_top
                    _layer.characteristics[1] = self.interfaces[i_l+1].carac_bottom

        
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
        self.ky = np.real(k_y)-1j*np.imag(k_y) # ky is either real or imaginary // - is to impose the good sign
        PeriodicMultiLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        if self.method in ["Recursive Method", "TMM"]:
            if self.termination == "transmission":
                self.Omega, self.back_prop = self.interfaces[-1].Omega(self.nb_waves)
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                    self.back_prop = self.back_prop@_l.Xi
                    self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)
                    self.back_prop = self.back_prop@next_interface.Tau
            else: # Rigid backing
                self.Omega = self.interfaces[-1].Omega(self.nb_waves)
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                    self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)
        elif self.method == "characteristics":
            if self.termination == "transmission":
                self.Omega, self.back_prop = self.interfaces[-1].Omegac(self.nb_waves)
                for i, _l in enumerate(self.layers[::-1]):
                    # print(i)
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_minus = self.Omega
                    _l.Omega_plus, _l.Xi = _l.update_Omegac(self.Omega, omega)
                    self.back_prop = self.back_prop@_l.Xi
                    self.Omega, next_interface.Tau = next_interface.update_Omegac(_l.Omega_plus)
                    self.back_prop = self.back_prop@next_interface.Tau
            else: # Rigid backing
                self.Omega = self.interfaces[-1].Omegac(self.nb_waves)
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_minus = self.Omega
                    _l.Omega_plus, _l.Xi = _l.update_Omegac(self.Omega, omega)
                    self.Omega, next_interface.Tau = next_interface.update_Omegac(_l.Omega_plus)
        elif self.method == "Global Method":
            if hasattr(self.result, "n_dof"):
                nb_dof_FEM = self.result.n_dof
            else:
                nb_dof_FEM = 0
            
            
            self.A = np.zeros((self.nb_PW-self.nb_waves, self.nb_PW),dtype=complex)

            i_eq = 0
            for _int in self.interfaces:
                if self.method == "Global Method":
                    i_eq = _int.update_M_global(self.A,i_eq)
            self.F = -self.A[:, 0]*np.exp(1j*self.ky[0]*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
            for i in range(self.nb_waves):
                self.A = np.delete(self.A, 2*(self.nb_waves-i-1), axis=1)
        else:
            raise NameError("Unknow method")
        
    def solve(self):
        Calculus.solve(self)
        if self.method == "Global Method":
            # plt.spy(self.A)
            # plt.show()
            self.X = LA.solve(self.A, self.F)
                
            R = self.X[:self.nb_waves]
            self.result.R0.append(R[0])
            self.result.R.append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))

            self.result.abs.append(1-np.abs(self.result.R0[-1])**2)
            if self.termination == "transmission":
                self.result.T0.append(self.X[-1])
                if self.window:
                    self.window.update_frequency(2*pi*self.f)
                    sigma = self.window.sigma_average_Yu(self.k_air*np.sin(self.theta_d*pi/180))
                else:
                    sigma = 1/np.cos(self.theta_d*pi/180)
        else:
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            E_0 = np.array([-alpha, 1]).reshape((2,1))
            Omega_0 = [np.array([1j*(self.ky[_w]/self.k_air)/(2*pi*self.f*Air.Z),1]).reshape(2,1) for _w in range(self.nb_waves)]
            Omega_0 = block_diag(*Omega_0)

            if self.method == "characteristics":
                self.Omega = np.kron(np.eye(self.nb_waves), self.interfaces[0].carac_bottom.P)@self.Omega

            A, A_minus = Omega_0[:2, :1], self.Omega[:2, :1]
            B, B_minus = Omega_0[:2, 1:], self.Omega[:2, 1:]
            C, C_minus = Omega_0[2:, :1], self.Omega[2:, :1]
            D, D_minus = Omega_0[2:, 1:], self.Omega[2:, 1:]

            M_temp = np.hstack((D_minus, -D))
            F_temp = np.hstack((-C_minus, C))
            E = LA.solve(M_temp, F_temp)
            
            M_temp = np.hstack((A_minus, -A))+np.hstack((B_minus, -B))@E
            X = LA.solve(M_temp, E_0)
            R = E@X 
            self.X_0_minus = np.append([X[0]], R[:self.nb_waves-1]) 
            # R is second part of the solution vector
            R = np.append([X[1]], R[self.nb_waves-1:])

            self.result.R0.append(R[0])
            self.result.R.append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))
            abs = 1-self.result.R[-1]
            if self.termination == "transmission":
                T = (self.back_prop@self.X_0_minus)
                if self.method != "characteristics":
                    T = (self.back_prop@self.X_0_minus)[::self.interfaces[-1].carac_bottom.n_w]     
                self.result.T0.append(T[0])
                self.result.T.append(np.sum(np.real(self.ky)*np.abs(T**2))/np.real(self.ky[0]))
                abs -= self.result.T[-1]
            self.result.abs.append(abs)

                    
    def plot_solution(self):
        if self.method == "Recursive Method":
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