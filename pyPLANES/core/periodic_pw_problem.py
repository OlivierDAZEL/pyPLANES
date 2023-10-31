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
        self.method = kwargs.get("method", "jap")
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
        else: 
            raise NameError("Invalid method name" + self.method)
        self.method_TM = kwargs.get("method_TM", False)
        
        if self.method_TM in ["cheb_1", "cheb_2"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        self.termination = kwargs.get("termination", "rigid")

        self.typ_solver = kwargs.get("typ_solver", "direct")
        self.order = kwargs.get("order", 2)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", False)
        self.result.order = self.order
        self.result.Solver = type(self).__name__
        self.result.Method = self.method
        

        # Out files

        PeriodicMultiLayer.__init__(self, ml, theta_d=self.theta_d, order=self.order, plot=self.plot, condensation=self.condensation)

        self.add_excitation_and_termination(self.termination)
        
        
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
            self.ky = np.real(k_y)-1j*np.imag(k_y) # ky is either real or imaginary // - is to impose the good sign
            
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
                    _l.Omega_plus, _l.Xi = _l.update_Omega(self.Omega, omega, self.method)
                    self.Omega, next_interface.Tau = next_interface.update_Omega(_l.Omega_plus)
        elif self.method == "characteristics":
            if self.termination == "transmission":
                self.Omega, self.back_prop = self.interfaces[-1].Omegac()
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_minus = self.Omega
                    _l.Omega_plus, _l.Xi = _l.update_Omegac(self.Omega, omega)
                    self.back_prop = self.back_prop@_l.Xi
                    self.Omega, next_interface.Tau = next_interface.update_Omegac(_l.Omega_plus)
                    self.back_prop = self.back_prop@next_interface.Tau
            else: # Rigid backing
                self.Omega = self.interfaces[-1].Omegac(self.nb_waves)
                # print(self.Omega)
                for i, _l in enumerate(self.layers[::-1]):
                    next_interface = self.interfaces[-i-2]
                    _l.Omega_minus = self.Omega
                    _l.Omega_plus, _l.Xi = _l.update_Omegac(self.Omega, omega)
                    self.Omega, next_interface.Tau = next_interface.update_Omegac(_l.Omega_plus)
                    
    def solve(self):

        Calculus.solve(self)
        if self.nb_waves == 1:
            self.Omega = self.Omega.reshape(2)
            if self.method == "characteristics":
                self.Omega = self.interfaces[0].carac_bottom.P@self.Omega
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            det = -self.Omega[0]+alpha*self.Omega[1]
            self.result.R0.append((self.Omega[0]+alpha*self.Omega[1])/det)
            if self.verbose:
                print("R_0={}".format(self.result.R0))
            abs = 1-np.abs(self.result.R0)**2
            self.X_0_minus = 2*alpha/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.result.T0.append(Omega_end[0])
                if self.verbose:
                    print("T_0={}".format(Omega_end[0]))
        else:
            # Excitation (10) and (11) JAP
            if self.typ_solver == "direct":
                if self.method == "characteristics":
                    self.Omega = np.kron(np.eye(self.nb_waves), self.interfaces[0].carac_bottom.P)@self.Omega
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
                # print("R={}".format(R))
                self.result.R0.append(R[0])

                self.result.R.append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))
                abs = 1-self.result.R[-1]

                self.X_0_minus = X[:self.nb_waves]
                if self.termination == "transmission":
                    T = (self.back_prop@self.X_0_minus)[::self.interfaces[-1].len_X]  
                    self.result.T0.append(T[0])
                    if self.verbose:
                        print("T={}".format(T))
                    self.result.T.append(np.sum(np.real(self.ky)*np.abs(T**2))/np.real(self.ky[0]))
                    abs -= self.result.T[-1]
                self.result.abs.append(abs)
            elif self.typ_solver == "characteristics":
                alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
                E_0 = np.array([-alpha, 1]).reshape((2,1))

                Omega_0_minus = self.Omega
                # [-Omega_0] second part of JAP
             
                Omega_0 = np.zeros((2*self.nb_waves, self.nb_waves), dtype=complex)
                for _w in range(self.nb_waves):
                    alpha_w = 1j*(self.ky[_w]/self.k_air)/(2*pi*self.f*Air.Z)
                    omega_0 = np.array([alpha_w, 1], dtype=complex).reshape((2,1))
                    Omega_0[2*_w:2*(_w+1),_w] = omega_0.reshape(2)

                A_minus = Omega_0_minus[:2, :1]
                B_minus = Omega_0_minus[:2, 1:]
                C_minus = Omega_0_minus[2:, :1]
                D_minus = Omega_0_minus[2:, 1:]

                A = Omega_0[:2, :1]
                B = Omega_0[:2, 1:]
                C = Omega_0[2:, :1]
                D = Omega_0[2:, 1:]

                M_temp = np.hstack((D_minus, -D))
                F_temp = np.hstack((-C_minus, C))
                E = LA.solve(M_temp, F_temp)

                E_X = E[:,:self.nb_waves-1]
                E_R = E[:,self.nb_waves-1:]
                
                
                M_temp = np.hstack((A_minus, -A))+np.hstack((B_minus, -B))@E
                X = LA.solve(M_temp, E_0)
                R = E@X 
                # R is second part of the solution vector
                R = np.append([X[1]], R[self.nb_waves-1:])
                self.X_0_minus = np.append([X[0]], R[:self.nb_waves])
                # print("R={}".format(R))
                if self.verbose:
                    print("R={}".format(R))
                # print("R={}".format(R))
                self.result.R0.append(R[0])

                self.result.R.append(np.sum(np.real(self.ky)*np.abs(R**2))/np.real(self.ky[0]))
                abs = 1-self.result.R[-1]
                if self.termination == "transmission":
                    T = (self.back_prop@self.X_0_minus)[::self.interfaces[-1].len_X]  
                    self.result.T0.append(T[0])
                    if self.verbose:
                        print("T={}".format(T))
                    self.result.T.append(np.sum(np.real(self.ky)*np.abs(T**2))/np.real(self.ky[0]))
                    abs -= self.result.T[-1]
                self.result.abs.append(abs)
            else:
                raise NameError("No solver for " + __name__)
            # if self.verbose:
            #     print("abs={}".format(abs))

    def plot_solution(self):
        x_minus = self.X_0_minus # Information vector at incident interface  x^-
        # print("x_minus={}".format(x_minus))
        if not isinstance (x_minus, np.ndarray):
            x_minus = np.array([x_minus])
        for i, _l in enumerate(self.layers):
            x_plus = self.interfaces[i].Tau @ x_minus # Transfert through the interface x^+
            x_minus = _l.Xi@x_plus # Transfert through the layer x^-_{+1}
            if isinstance(_l, PeriodicLayer):
                S_b = _l.Omega_plus @ x_plus
                S_t = _l.Omega_minus @ x_minus
                # print(f"S_b={S_b}")
                # print(f"S_t={S_t}")
                _l.plot_solution(S_b, S_t)
            else: # Homogeneous layer
                if self.method == "Recursive Method":
                    q = LA.solve(_l.SV, _l.Omega_plus@x_plus)
                    _l.plot_solution_recursive(self.plot, q)
                elif self.method == "characteristics":
                    q = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
                    for i, _l in enumerate(self.layers):
                        q = self.interfaces[i].Tau @ q # Transfert through the interface x^+
                        q = _l.Xi@q # Transfert through the layer x^-_{+1}
                        _l.plot_solution_characteristics(self.plot, _l.Omega_minus@q)