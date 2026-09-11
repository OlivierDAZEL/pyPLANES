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
from pyPLANES.pw.multilayer import MultiLayer
from pyPLANES.pw.window import Window

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *

class PwProblem(Calculus, MultiLayer):
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        self.result.Solver = type(self).__name__
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.method = kwargs.get("method", "Global Method")
        if self.method.lower() in ["recursive", "jap", "recursive method", "rm"]:
            self.method = "Recursive Method"
            if self.theta_d == 0:
                self.theta_d = 1e-12
        elif self.method.lower() in ["tmm", "transfer matrix method"]:
            self.method = "TMM"
            if self.theta_d == 0:
                self.theta_d = 1e-12
        elif self.method.lower() in ["h","z", "impedance", "hybrid", "impedance method", "impedances"]:
            self.method = "HMM"
        else: 
            self.method = "Global Method"
        self.method_TM = kwargs.get("method_TM", "diag")
        if self.method_TM in ["cheb_1"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        

        assert "ml" in kwargs
        ml = kwargs.get("ml")
        MultiLayer.__init__(self, ml=ml, method=self.method , method_TM=self.method_TM, material_database=self.material_database)

        # Window 
        self.window = kwargs.get("window", False)
        if self.window is not False:
            window_method = kwargs.get("window_method", "Yu")
            self.window = Window(self.window[0], self.window[1], window_method)

        if self.method_TM in ["cheb_1"]:
            for l in self.layers:
                l.order_chebychev = self.order_chebychev 

        self.termination = kwargs.get("termination", "rigid")
        self.add_excitation_and_termination(self.termination)
        if self.energetic_balance:
            self.create_result_energetic_balance()

    def create_result_energetic_balance(self):
        self.result.input_power = None
        self.result.output_power = None
        self.result.P_viscous = None
        self.result.P_thermal = None
        self.result.P_structural = None
        nb_layers = len(self.layers)
        if self.method == "Global Method":
            nb_layers -= 1
        for _l in self.layers:
            if isinstance(_l, (FluidLayer, PemLayer)):
                self.result.P_viscous = [[] for i in range(nb_layers)]
                self.result.P_thermal = [[] for i in range(nb_layers)]
            if isinstance(_l, (ElasticLayer, PemLayer)):
                self.result.P_structural = [[] for i in range(nb_layers)]

    def update_frequency(self, omega):
        self.k_air = omega/Air.c
        self.kx = self.k_air*np.array([np.sin(self.theta_d*np.pi/180)])
        self.ky = self.k_air*np.array([np.cos(self.theta_d*np.pi/180)])
        MultiLayer.update_frequency(self, omega, self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        if self.method == "Recursive Method":
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
            self.A = np.zeros((self.nb_dofs-1, self.nb_dofs),dtype=complex)
            i_eq = 0
            # Loop on the interfaces
            for _int in self.interfaces:
                i_eq = _int.update_M_global(self.A,i_eq)
            self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
            self.A = np.delete(self.A, 0, axis=1)
        elif self.method == "TMM":
            for _l in self.layers:
                _l.update_TM(omega)
            self.A = np.zeros((self.nb_dofs-1, self.nb_dofs),dtype=complex)
            i_eq = 0 # alpha*(-1+R)-u_y = 0
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            self.A[i_eq,0] = -alpha
            self.A[i_eq,1] = alpha
            self.A[i_eq,2] = -1 # u_y
            i_eq += 1 # 1+R-p = 0
            self.A[i_eq,0] = 1
            self.A[i_eq,1] = 1
            self.A[i_eq,3] = -1 # p 
            i_eq += 1
            for _int in self.interfaces:
                i_eq = _int.update_M_TMM(self.A,i_eq)
            self.F = -self.A[:, 0]# - is for transposition
            self.A = np.delete(self.A, 0, axis=1)
        elif self.method == "HMM":
            self.H = self.interfaces[-1].HMM_update()
            for i, _l in enumerate(self.layers[::-1]):
                _l.H_top = self.H.copy()
                self.H = _l.HMM_update(self.H)
                self.H = self.interfaces[-i-2].HMM_update(self.H)
        else:
            raise NameError("Unknow method")

    def solve_kernel(self):
        
        if self.method in ["Recursive Method"]:
            self.Omega = self.Omega.reshape(2)
            alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
            det = -self.Omega[0]+alpha*self.Omega[1]
            self.R0 = (self.Omega[0]+alpha*self.Omega[1])/det
            self.abs = 1-np.abs(self.R0)**2
            self.X_0_minus = 2*alpha/det
            if self.termination == "transmission":
                Omega_end = (self.back_prop*self.X_0_minus).flatten()
                self.T0 = Omega_end[0]
                self.abs -= np.abs(self.T0)**2
        elif self.method == "Global Method":
            self.X = LA.solve(self.A, self.F)
            self.R0 = self.X[0]
            self.abs = 1-np.abs(self.R0)**2
            if self.termination == "transmission":
                self.T0 = self.X[-1]
                self.abs -= np.abs(self.T0)**2
        elif self.method == "TMM":
            if LA.det(self.A)!=0:
                self.X = LA.solve(self.A, self.F)
                alpha = 1j*(self.ky[0]/self.k_air)/(2*pi*self.f*Air.Z)
                det = -self.X[0]+alpha*self.X[1]
                self.R0 = self.X[0]
                self.abs = 1-np.abs(self.R0)**2
                if self.termination == "transmission":
                    self.T0 = self.X[-1]
                    self.abs -= np.abs(self.T0)**2
            else:
                self.R0 = np.nan
                self.abs = np.nan
                if self.termination == "transmission":
                    self.T0 = np.nan
        elif self.method == "HMM":
            H = self.H[0,0]
            self.R0 = (np.cos(self.theta_d*pi/180)-Air.Z*H)/(np.cos(self.theta_d*pi/180)+Air.Z*H)
            self.abs = 1-np.abs(self.R0)**2
            if self.termination == "transmission":
                self.T0 = np.array([1+self.R0]).reshape((1,1))
                for i, _int in enumerate(self.interfaces[:-1]):
                    self.T0 = _int.I_cal@self.T0
                    self.T0 = self.layers[i].L_cal@self.T0
                self.T0 = self.T0.flatten()[0]
                self.abs -= np.abs(self.T0)**2
        if self.termination == "transmission":
            # Window correction
            if self.window:
                self.window.update_frequency(2*pi*self.f)
                sigma = self.window.sigma_average_Yu(self.k_air*np.sin(self.theta_d*pi/180))
            else:
                sigma = 1/np.cos(self.theta_d*pi/180)
            self.win = np.cos(self.theta_d*pi/180)*sigma
            self.tau = (np.abs(self.T0)**2)*np.cos(self.theta_d*pi/180)*sigma

    def solve(self):
        # print("Frequency: {:.2f} Hz".format(self.f))
        Calculus.solve(self)
        self.solve_kernel()
 
        self.result.R0.append(self.R0)
        self.result.Z_prime.append((self.R0+1)/(1-self.R0)/np.cos(self.theta_d*pi/180))
        self.result.abs.append(self.abs)
        if self.termination == "transmission":
            self.result.T0.append(self.T0)
            self.result.tau.append(self.tau)

    def resolution(self):
        return super().resolution()

    def compute_energetic_balance(self):
        self.compute_amplitude_of_waves()
        # Calculus of the input power
        if self.method == "Global Method":
            v = (1-self.R0)*np.cos(self.theta_d*pi/180)/Air.Z
            self.result.input_power = np.real((1+self.R0)*np.conj(v))/2/self.f # division by f is the intrgration of constant function over one period
            self.absorbed_power = self.result.input_power 
        elif self.method == "Recursive Method":          
            if self.termination == "transmission":
                v_out = self.T0*np.cos(self.theta_d*pi/180)/Air.Z
                self.result.output_power = np.real(self.T0*np.conj(v_out))/2/self.f
                self.absorbed_power -= self.result.output_power
            else:
                self.result.output_power = 0.0
            # print(f"Input power.   : {self.result.input_power}")
        incident_power = (np.real(1*np.conj(1*np.cos(self.theta_d*pi/180)/Air.Z))/2/self.f)


        P_dis = 0.0
        list_layers = self.layers
        
        if self.method == "Global Method":
            list_layers = self.layers[1:]
        self.result.nb_layers = len(list_layers)
        for i, _l in enumerate(list_layers):
            if isinstance(_l, FluidLayer):
                P_viscous, P_thermal = _l.compute_energetic_balance(self.f, incident_power)
                self.result.P_viscous[i].append(np.real(P_viscous))
                self.result.P_thermal[i].append(np.real(P_thermal))
                if self.result.P_structural is not None:
                    self.result.P_structural.append(0.0)
            elif isinstance(_l, ElasticLayer):
                P_structural = _l.compute_energetic_balance(self.f, incident_power)
                self.result.P_structural[i].append(P_structural)
                if self.result.P_viscous is not None:
                    self.result.P_viscous[i].append(0.0)
                if self.result.P_thermal is not None:
                    self.result.P_thermal[i].append(0.0)
            elif isinstance(_l, PemLayer):
                P_viscous, P_thermal,P_structural = _l.compute_energetic_balance(self.f, incident_power)
                self.result.P_viscous[i].append(P_viscous)
                self.result.P_thermal[i].append(P_thermal)
                self.result.P_structural[i].append(P_structural)
            P_dis +=  self.result.P_viscous[i][-1]
            P_dis +=  self.result.P_thermal[i][-1]
            P_dis +=  self.result.P_structural[i][-1]

    def compute_amplitude_of_waves(self):
        # Comput the amplitude of the waves in each layer the refererence is the bottom interface for all of them
        if self.method == "Global Method":
            for _l in self.layers[1:]:
                _l.q = self.X[_l.dofs-1]
                _l.x_refs = None
        elif self.method == "Recursive Method":
            x = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            for i, _l in enumerate(self.layers):
                x = self.interfaces[i].Tau @ x # Transfert through the interface x^+
                _l.q = LA.solve(_l.SV, _l.Omega_plus@x)
                x = _l.Xi@x # Transfert through the layer x^-_{+1}
                _l.x_ref = _l.x[0]
        elif self.method == "HMM":
            p = np.array([1+self.R0]) # pressure (parent variable) at the bottom 
            for i,_l in enumerate(self.layers):
                p = self.interfaces[i].I_cal @ p # Transfert through the interface 
                p = _l.L_cal @ p # Transfert through the layer until the top
                Sigma = (_l.Omega_p + _l.Omega_c@_l.H_top) @ p 
                _l.q = _l.Q@Sigma
                _l.x_ref = _l.x[1]

        else: 
            raise NameError("No method")

    def plot_solution(self):
        self.compute_amplitude_of_waves()
        if self.method == "Global Method":
            for _l in self.layers[1:]:
                _l.plot_solution_global(self.plot,_l.q)  
        elif self.method == "Recursive Method":
            x = np.array([self.X_0_minus]) # Information vector at incident interface  x^-
            for i, _l in enumerate(self.layers):
                x = self.interfaces[i].Tau @ x # Transfert through the interface x^+
                q = LA.solve(_l.SV, _l.Omega_plus@x)
                _l.plot_solution_recursive(self.plot, _l.q)
                x = _l.Xi@x # Transfert through the layer x^-_{+1}
        elif self.method == "TMM":
            for _l in self.layers[1:]:
                _l.plot_solution_TMM(self.plot, self.X[_l.dofs-1])
        elif self.method == "HMM":
            for i, _l in enumerate(self.layers):
                _l.plot_solution_H(self.plot, _l.q, self.f)
        else: 
            raise NameError("No method")
