#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_interfaces.py
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

class PwInterface():
    """
    Interface for Plane Wave Solver
    """
    def __init__(self, layer1=None, layer2=None):
        self.layers = [layer1, layer2]
    def update_M_global(self, M, i_eq):
        pass

class FluidFluidInterface(PwInterface):
    """
    Fluid-fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Fluid-fluid interface"
        return out

    def update_M_global(self, M, i_eq):
        print(self.layers[0].ky)
        delta_0 = np.exp(-1j*self.layers[0].ky*self.layers[0].d)
        delta_1 = np.exp(-1j*self.layers[1].ky*self.layers[1].d)
        M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[0, 0]*delta_0
        M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[0, 1]
        M[i_eq, self.layers[1].dofs[0]] = -self.layers[1].SV[0, 0]
        M[i_eq, self.layers[1].dofs[1]] = -self.layers[1].SV[0, 1]*delta_1
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[1, 0]*delta_0
        M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[1, 1]
        M[i_eq, self.layers[1].dofs[0]] = -self.layers[1].SV[1, 0]
        M[i_eq, self.layers[1].dofs[1]] = -self.layers[1].SV[1, 1]*delta_1
        i_eq += 1
        return i_eq

class FluidRigidBacking(PwInterface):
    """
    Rigid backing 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def __str__(self):
        out = "\t Rigid backing"
        return out

    def update_M_global(self, M, i_eq):
        M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[0, 0]*np.exp(-1j*self.layers[0].ky*self.layers[0].d)
        M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[0, 1]
        i_eq += 1
        return i_eq