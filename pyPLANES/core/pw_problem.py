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
import matplotlib.pyplot as plt


from pyPLANES.utils.io import initialisation_out_files_plain
from pyPLANES.core.calculus import PwCalculus
from pyPLANES.core.multilayer import MultiLayer

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *

Air = Air()

class PwProblem(PwCalculus, MultiLayer):
    """
        Plane Wave Problem 
    """ 
    def __init__(self, **kwargs):
        PwCalculus.__init__(self, **kwargs)
        termination = kwargs.get("termination","rigid")
        self.method = kwargs.get("method","global")

        MultiLayer.__init__(self, **kwargs)

        self.kx, self.ky, self.k = None, None, None
        self.plot = kwargs.get("plot_results", [False]*6)
        self.outfiles_directory = False

        if self.method == "global":
            self.layers.insert(0,FluidLayer(Air,1.e-2,-1.e-2))
            if self.layers[1].medium.MEDIUM_TYPE == "fluid":
                self.interfaces.insert(0,FluidFluidInterface(self.layers[0],self.layers[1]))
            elif self.layers[1].medium.MEDIUM_TYPE == "pem":
                self.interfaces.insert(0,FluidPemInterface(self.layers[0],self.layers[1]))
            elif self.layers[1].medium.MEDIUM_TYPE == "elastic":
                self.interfaces.insert(0,FluidElasticInterface(self.layers[0],self.layers[1]))
            
            self.nb_PW = 0
            for _layer in self.layers:
                if _layer.medium.MODEL == "fluid":
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

    def update_frequency(self, f):
        PwCalculus.update_frequency(self, f)
        MultiLayer.update_frequency(self, f, self.k, self.kx)

    def create_linear_system(self, f):
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
        self.X = LA.solve(self.A, self.F)
        R_pyPLANES_PW = self.X[0]
        print(R_pyPLANES_PW)
        if self.layers[-1] == "transmission":
            T_pyPLANES_PW = self.X[-2]
        else:
            T_pyPLANES_PW = None
        self.layers[0].plot_sol(self.plot, [np.exp(1j*self.ky*self.layers[0].d), self.X[0]]) 
        self.layers.pop(0)

    def display_sol(self):   
        for _layer in self.layers:
            _layer.plot_sol(self.plot, self.X[_layer.dofs-1])  
