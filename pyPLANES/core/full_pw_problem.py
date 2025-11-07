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
from scipy import integrate

from pyPLANES.core.calculus import Calculus
from pyPLANES.pw.general.general_multilayer import GeneralMultiLayer
from pyPLANES.pw.studs import Studs
from pyPLANES.pw.pw_solver import PWSolver

from pyPLANES.pw.window import Window


class FullPwProblem(Calculus, GeneralMultiLayer, Studs, PWSolver):
    def __init__(self, **kwargs):
        # General variables
        Calculus.__init__(self, **kwargs)
        self.result.Solver = type(self).__name__
        # Windowing
        self.window = kwargs.get("window", False)
        if self.window is not False:
            self.window = Window(self.window[0], self.window[1])
            self.window.method = kwargs.get("window_method", False)
            if self.window.method is False:
                if self.homogeneous: 
                    self.window.method = "Yu"
                else:
                    self.window.method = "Rhazi"

        # Diffuse field
        PWSolver.__init__(self, **kwargs)

        # multilayer
        assert "ml" in kwargs
        GeneralMultiLayer.__init__(self, ml=kwargs.get("ml"), method=self.method, material_database=self.material_database)
        self.termination = kwargs.get("termination", "rigid")
        self.add_excitation_and_termination(self.termination)

        GeneralMultiLayer.compute_number_of_pw(self)

        # studs 
        studs = kwargs.get("studs", [])
        Studs.__init__(self, studs=studs)



        # self.period = kwargs.get("period", None)
        # if self.studs != []:
        #     self.homogeneous = False
        #     self.method = "Stud"
        #     self.period = kwargs.get("period", .6)
        #     for st in self.studs:
        #         i_eq = 0
        #         for i, _int in enumerate(self.interfaces):
        #             if i == st.layer:
        #                 st.sigma_b = i_eq + np.array(_int.relations_sigma)
        #                 st.sigma_t = i_eq + _int.number_relations+np.array(self.interfaces[i+1].relations_sigma)
        #                 st.B = np.zeros((self.nb_PW-1, 6), dtype=complex)
        #                 st.B[st.sigma_t, :3] = np.eye(3)
        #                 st.B[st.sigma_b, 3:] = -np.eye(3)
        #             i_eq += _int.number_relations
        #         st.nb_PW = self.nb_PW

            
        # Calculus variable (for pylint)
        self.kx, self.ky, self.kz, self.k = None, None, None, None
        self.R, self.T = None, None


    def update_frequency(self, omega, i_w=0):
        # Calculus.update_frequency(self, omega)
        self.k_air = omega/Air.c
        self.kx = self.k_air*np.sin(self.theta_d*np.pi/180)*np.cos(self.phi_d*np.pi/180)
        if self.period is not None:
            self.kx += i_w*2*pi/self.period
        self.kz = self.k_air*np.sin(self.theta_d*np.pi/180)*np.sin(self.phi_d*np.pi/180)
        self.ky = np.sqrt(self.k_air**2-self.kx**2-self.kz**2+0*1j)
        GeneralMultiLayer.update_frequency(self, omega, self.kx, self.kz)
        for st in self.studs:
            st.update_frequency(self.kx)

    def create_linear_system(self, omega):
        Calculus.create_linear_system(self, omega)
        
        if self.method == "Global Method":
            self.A = np.zeros((self.nb_PW-self.nb_waves, self.nb_PW),dtype=complex)
            i_eq = 0
            # Loop on the interfaces
            for _int in self.interfaces:
                i_eq = _int.update_M_global(self.A,i_eq)
            self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
            for i in range(self.nb_waves):
                self.A = np.delete(self.A, 2*(self.nb_waves-i-1), axis=1)
        else:
            raise NameError("Unknow method")

    def solve_at_angles(self, theta, phi):
        if self.method == "Global Method":
            Calculus.solve(self)
            self.X = LA.solve(self.A, self.F)
            self.R0 = self.X[0]
            if self.termination == "transmission":
                self.T0 = self.X[-1]
                self.tau = np.abs(self.T0)**2
                if self.window:
                    sigma = self.window.sigma(self.f, self.theta_d, self.phi_d)
                    self.tau *= sigma*np.cos(self.theta_d*pi/180)

        elif self.method == "Stud":
            omega = 2*pi*self.f
            i_w = 0
            test =True
            S = np.zeros((6,6), dtype=complex)
            S_old = S.copy()
            rho = []
            while test :
                self.update_frequency(omega, i_w)
                self.A = np.zeros((self.nb_PW-1, self.nb_PW),dtype=complex)
                i_eq = 0
                for _int in self.interfaces:
                    i_eq = _int.update_M_global(self.A,i_eq)
                if i_w == 0:
                    self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
                self.A = np.delete(self.A,0, axis=1)
                A_i = LA.inv(self.A)
                for stud in self.studs:
                    U_w = stud.compute_Uw()
                    if i_w == 0:
                        q_0 = A_i@self.F
                        U0q0 = U_w@q_0
                    rho.append(-A_i@stud.B)
                    S += U_w @ rho[-1]
                    
                if LA.norm(S-S_old)< LA.norm(S)*1e-2:
                    test = False
                else:
                    S_old = S.copy()
                    # print(i_w)
                    if i_w > 0: # ok for 0, 1, -1, ...
                        i_w = -i_w
                    else:
                        i_w = -i_w+1

            print(i_w)
            sigma = LA.solve(np.eye(6)-stud.K@S, stud.K@U0q0)
            for i in range(1):
                self.X = q_0 + rho[0]@sigma
            self.R0 = self.X[0]
            if self.termination == "transmission":
                self.T0 = self.X[-1]
                self.tau = np.abs(self.T0)**2
                if self.window:
                    sigma = self.window.sigma(self.f, self.theta_d, self.phi_d)
                    self.tau *= sigma*np.cos(self.theta_d*pi/180)


            
        else:
            raise NameError("Unknow method")

    def solve(self):
        if self.diffuse_field:
            D = 0.5
            def f(theta):
                self.theta_d = theta*180/pi
                self.solve_at_angles(self.theta_d, self.phi_d)
                return np.sin(theta)*np.cos(theta)*self.tau/D
            if self.DF_method == "scipy":
                self.tau, abserror, infodict = integrate.quad(f, 0, pi/2,full_output=1,epsrel=self.epsrel, epsabs=self.epsabs)
                self.result.tau.append(self.tau)
            else:
                raise NotImplementedError(f"Method {self.DF_method} not implemented")
        else: # Resolution at a single angle
            self.solve_at_angles(self.theta_d, self.phi_d)
            self.result.R0.append(self.R0)
            self.result.abs.append(1-np.abs(self.R0)**2)
            if self.termination == "transmission":
                self.result.T0.append(self.T0)
                self.result.tau.append(self.tau)
                self.result.abs[-1] -= np.abs(self.T0)**2
            self.result.Z_prime.append((self.R0+1)/(1-self.R0))

 
