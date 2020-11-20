#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_classes.py
#
# This file is part of pymls, a software distributed under the MIT license.
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

# from pyPLANES.utils.io import initialisation_out_files_plain
from pyPLANES.core.calculus import PwCalculus
from pyPLANES.core.multilayer import MultiLayer

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *


class PwProblem(PwCalculus, MultiLayer):
    """
        Plane Wave Problem 
    """ 
    def __init__(self, **kwargs):
        PwCalculus.__init__(self, **kwargs)
        MultiLayer.__init__(self, **kwargs)
        # Interface associated to the termination 
        if self.termination in ["trans", "transmission","Transmission"]:
            self.interfaces.append(SemiInfinite(self.layers[-1]))
        else: # Case of a rigid backing 
            if self.layers[-1].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                self.interfaces.append(FluidRigidBacking(self.layers[-1]))
            elif self.layers[-1].medium.MEDIUM_TYPE == "pem":
                self.interfaces.append(PemBacking(self.layers[-1]))
            elif self.layers[-1].medium.MEDIUM_TYPE == "elastic":
                self.interfaces.append(ElasticBacking(self.layers[-1]))



        if self.method == "recursive":
            if self.layers[0].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                self.interfaces.insert(0,FluidFluidInterface(None ,self.layers[0]))
            elif self.layers[0].medium.MEDIUM_TYPE == "pem":
                self.interfaces.insert(0,FluidPemInterface(None, self.layers[0]))
            elif self.layers[0].medium.MEDIUM_TYPE == "elastic":
                self.interfaces.insert(0,FluidElasticInterface(None, self.layers[0]))
        elif self.method == "global":
            Air_mat = Air()
            mat = Fluid(c=Air_mat.c,rho=Air_mat.rho)
            self.layers.insert(0,FluidLayer(mat, 1.e-2, -1.e-2))
            if self.layers[1].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                self.interfaces.insert(0,FluidFluidInterface(self.layers[0] ,self.layers[1]))
            elif self.layers[1].medium.MEDIUM_TYPE == "pem":
                self.interfaces.insert(0,FluidPemInterface(self.layers[0], self.layers[1]))
            elif self.layers[1].medium.MEDIUM_TYPE == "elastic":
                self.interfaces.insert(0,FluidElasticInterface(self.layers[0],self.layers[1]))
            # Count of the number of plane waves for the global method
            self.nb_PW = 0
            for _layer in self.layers:
                if _layer.medium.MODEL in ["fluid", "eqf"]:
                    _layer.dofs = self.nb_PW+np.arange(2)
                    self.nb_PW += 2
                elif _layer.medium.MODEL == "pem":
                    _layer.dofs = self.nb_PW+np.arange(6)
                    self.nb_PW += 6
                elif _layer.medium.MODEL == "elastic":
                    _layer.dofs = self.nb_PW+np.arange(4)
                    self.nb_PW += 4
            if isinstance(self.interfaces[-1],SemiInfinite):
                self.nb_PW += 1 

    def update_frequency(self, omega):
        PwCalculus.update_frequency(self, omega)
        MultiLayer.update_frequency(self, omega, self.k, self.kx)
        # if self.method = "recursive":
        #     self.Omega_0 = np.array([self.ky/(1j*self.omega*Air.Z),1], dtype=np.complex)
        #     if self.termination == "Rigid":
        #         self.Omega = np.array([0,1], dtype=np.complex)
        #     else:
        #         self.Omega = np.array([-self.ky/(1j*self.omega*Air.Z),1], dtype=np.complex)

    def create_linear_system(self, omega):
        PwCalculus.create_linear_system(self, omega)
        if self.method == "recursive":
            self.create_linear_system_recursive(omega)
        elif self.method == "global":
            self.create_linear_system_global(omega)

    def create_linear_system_recursive(self, omega):
        if self.termination == "rigid":
            self.Omega = self.interfaces[-1].Omega()
        else:
            self.Omega, self.back_prop = self.interfaces[-1].Omega()
        if self.termination == "transmission":
            for i, _l in enumerate(self.layers[::-1]):
                self.Omega, Xi = _l.transfert(self.Omega)
                self.back_prop = self.back_prop@Xi
                self.Omega, Tau = self.interfaces[i].transfert(self.Omega)
                self.back_prop = self.back_prop@Tau
        else:
            for i, _l in enumerate(self.layers[::-1]):
                self.Omega = _l.transfert(self.Omega)[0]
                self.Omega = self.interfaces[i].transfert(self.Omega)[0]
                

    def create_linear_system_global(self, omega):
        self.A = np.zeros((self.nb_PW-1, self.nb_PW), dtype=complex)
        i_eq = 0
        # Loop on the interfaces
        for _int in self.interfaces:
            if self.method == "global":
                i_eq = _int.update_M_global(self.A, i_eq)
        self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
        self.A = np.delete(self.A, 0, axis=1)

    def solve(self):
        PwCalculus.solve(self)
        if self.method == "recursive":
            self.solve_recursive()
        elif self.method == "global":
            self.solve_global()

    def solve_global(self):
        self.X = LA.solve(self.A, self.F)
        self.R = self.X[0]
        if self.termination == "transmission":
            self.T = self.X[-1]
        else:
            self.T = None

    def solve_recursive(self):
        self.Omega = self.Omega.reshape(2)
        _ = (self.ky/self.k)/(1j*self.omega*Air.Z) 
        detM = -self.Omega[0]+_*self.Omega[1]
        self.R = (self.Omega[0]+_*self.Omega[1])/detM
        if self.termination == "transmission":
            X_0_minus = 2*_/detM
            self.Omega = (self.back_prop*X_0_minus).tolist()
            self.T = self.Omega[0][0]




    def display_sol(self):
         for _layer in self.layers:
            _layer.plot_sol(self.plot, self.X[_layer.dofs-1])  
