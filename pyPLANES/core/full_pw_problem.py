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
from pyPLANES.pw.window import Window
from pyPLANES.pw.stud import Stud

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
        self.diffuse_field = True if self.theta_d == 90 else False
        self.DF_method = kwargs.get("DF_method", "scipy")
        self.verbose = kwargs.get("verbose", False)
        self.epsrel = kwargs.get("epsrel", 1.49e-1)
        self.epsabs = kwargs.get("epsabs", 1.49e-1)



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
        # if self.homogeneous == False:
        #     self.method = "Stud"
        # Put a non zero angle for some methods
        if self.method in [ "TMM", "Recursive Method"]:
            if self.theta_d == 0:
                self.theta_d = 1e-12


        assert "ml" in kwargs
        GeneralMultiLayer.__init__(self, ml=kwargs.get("ml"), method=self.method, material_database=self.material_database)
        self.termination = kwargs.get("termination", "rigid")
        self.add_excitation_and_termination(self.termination)

        # Studs
        self.homogeneous = True
        studs = kwargs.get("studs", False)
        if studs is not False:
            self.studs = []
            for st in studs:
                st = Stud(st[0],self.layers)
                st.nb_PW = self.nb_PW
                self.studs.append(st)
        
        # self.nb_bloch_waves = kwargs.get("nb_bloch_waves", 0)
        
        
        
        # self.nb_waves = 1+2*self.nb_bloch_waves

        # if (self.studs is not False) or (self.nb_bloch_waves != 0):
            self.homogeneous = False
            self.method = "Stud"
            self.period = kwargs.get("period", 1)
        # Calculus variable (for pylint)
        self.kx, self.ky, self.kz, self.k = None, None, None, None
        self.R, self.T = None, None


    def update_frequency(self, omega, i_w=0):
        Calculus.update_frequency(self, omega)
        self.k_air = omega/Air.c
        # self.kx = self.k_air*np.array([np.sin(self.theta_d*np.pi/180)])*np.array([np.cos(self.phi_d*np.pi/180)])
        
        self.kx = self.k_air*          np.sin(self.theta_d*np.pi/180)            *np.cos(self.phi_d*np.pi/180)
        
        
        # if self.homogeneous is False:
        #     _ = np.zeros(self.nb_waves)
        #     for i in range(self.nb_bloch_waves):
        #         _[1+2*i]= i+1
        #         _[2+2*i]= -(i+1)
        #     self.kx += _*(2*pi/self.period)
        # k_y = np.sqrt(self.k_air**2-self.kx**2+0*1j)
        
        self.kz = self.k_air*          np.sin(self.theta_d*np.pi/180)            *np.sin(self.phi_d*np.pi/180)

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
            self.update_frequency(omega)
            i_w, q_old  = 0, np.zeros(self.nb_PW-1) 
            test =True
            while test :
                self.A = np.zeros((self.nb_PW-1, self.nb_PW),dtype=complex)
                i_eq = 0
                # Loop on the interfaces
                for _int in self.interfaces:
                    i_eq = _int.update_M_global(self.A,i_eq)
                if i_w == 0:
                    self.F = -self.A[:, 0]*np.exp(1j*self.ky[0]*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
                self.A = np.delete(self.A,0, axis=1)
                for stud in self.studs:
                    stud.update(self.A, self.F)
                q = LA.solve(self.A, self.F)
                print(LA.norm(q-q_old))
                print(LA.norm(q))
                if LA.norm(q-q_old)< LA.norm(q):
                    test = True 
                else:
                    q_old = q
                    i_w += 1 
                    
            exit()
                        

            
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

 
