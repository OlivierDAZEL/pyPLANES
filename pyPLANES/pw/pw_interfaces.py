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
from numpy import sqrt
from mediapack import Air, Fluid
from pyPLANES.pw.pw_polarisation import fluid_waves


class PwInterface():
    """
    Interface for Plane Wave Solver
    """
    def __init__(self, layer1=None, layer2=None):
        self.layers = [layer1, layer2]
    def update_M_global(self, M, i_eq):
        pass
    def update_Omega(self, Om):
        pass 
    def update_frequency(self, omega, k, kx):
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

    def update_frequency(self, omega, k, kx):    
        pass

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam[0]*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam[0]*self.layers[1].d)
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

    def transfert(self, Om):
        self.Xi = np.eye(2)
        return Om

    def update_Omega(self, Om):
        return Om 

class FluidPemInterface(PwInterface):
    """
    Fluid-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Fluid-PEM interface"
        return out

    def transfert(self, Om):
        a = -np.array([
            [Om[0,1],Om[0,2]],
            [Om[3,1],Om[3,2]]
        ])
        Tau = np.dot(np.linalg.inv(a), np.array([[Om[0,0]], [Om[3,0]]]))
        Tau_tilde = np.concatenate([np.eye(1),Tau])

        Omega_moins = np.array([[Om[2,0]], [Om[4,0]]]) + np.dot(np.array([[Om[2,1], Om[2,2]], [Om[4,1], Om[4,2]]]), Tau)

        return Omega_moins, Tau_tilde

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam[0]*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV

        M[i_eq, self.layers[0].dofs[0]] = SV_1[0, 0]*delta_0
        M[i_eq, self.layers[0].dofs[1]] = SV_1[0, 1]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[2, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[2, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[2, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[2, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[2, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[2, 5]*delta_1[2]
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = SV_1[1, 0]*delta_0
        M[i_eq, self.layers[0].dofs[1]] = SV_1[1, 1]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[4, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[4, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[4, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[4, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[4, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[4, 5]*delta_1[2]
        i_eq += 1
        M[i_eq, self.layers[1].dofs[0]] = SV_2[0, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[0, 1]
        M[i_eq, self.layers[1].dofs[2]] = SV_2[0, 2]
        M[i_eq, self.layers[1].dofs[3]] = SV_2[0, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = SV_2[0, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = SV_2[0, 5]*delta_1[2]
        i_eq += 1
        M[i_eq, self.layers[1].dofs[0]] = SV_2[3, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[3, 1]
        M[i_eq, self.layers[1].dofs[2]] = SV_2[3, 2]
        M[i_eq, self.layers[1].dofs[3]] = SV_2[3, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = SV_2[3, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = SV_2[3, 5]*delta_1[2]
        i_eq += 1
        return i_eq

class PemFluidInterface(PwInterface):
    """
    PEM-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t PEM-Fluid interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam[0]*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV

        M[i_eq, self.layers[0].dofs[0]] = -SV_1[2, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[2, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[2, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[2, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[2, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[2, 5]
        M[i_eq, self.layers[1].dofs[0]] = SV_2[0, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[0, 1]*delta_1
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[4, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[4, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[4, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[4, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[4, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[4, 5]
        M[i_eq, self.layers[1].dofs[0]] = SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[1, 1]*delta_1
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = SV_1[0, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = SV_1[0, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = SV_1[0, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = SV_1[0, 3]
        M[i_eq, self.layers[0].dofs[4]] = SV_1[0, 4]
        M[i_eq, self.layers[0].dofs[5]] = SV_1[0, 5]
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = SV_1[3, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = SV_1[3, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = SV_1[3, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = SV_1[3, 3]
        M[i_eq, self.layers[0].dofs[4]] = SV_1[3, 4]
        M[i_eq, self.layers[0].dofs[5]] = SV_1[3, 5]
        i_eq += 1
        return i_eq

class FluidElasticInterface(PwInterface):
    """
    Fluid-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Fluid-Elastic interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam[0]*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of u_y
        M[i_eq, self.layers[0].dofs[0]] = SV_1[0, 0]*delta_0
        M[i_eq, self.layers[0].dofs[1]] = SV_1[0, 1]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[1, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[1, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[1, 3]*delta_1[1]
        i_eq += 1
        # sigma_yy = -p
        M[i_eq, self.layers[0].dofs[0]] = SV_1[1, 0]*delta_0
        M[i_eq, self.layers[0].dofs[1]] = SV_1[1, 1]
        M[i_eq, self.layers[1].dofs[0]] = SV_2[2, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[2, 1]
        M[i_eq, self.layers[1].dofs[2]] = SV_2[2, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] = SV_2[2, 3]*delta_1[1]
        i_eq += 1
        # sigma_xy = 0
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[0, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[0, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[0, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[0, 3]*delta_1[1]
        i_eq += 1
        return i_eq

class ElasticFluidInterface(PwInterface):
    """
    Elastic-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Elastic-Fluid interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam[0]*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of u_y
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[1, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[1, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[1, 2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[1, 3]
        M[i_eq, self.layers[1].dofs[0]] = SV_2[0, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[0, 1]*delta_1
        i_eq += 1
        # sigma_yy = -p
        M[i_eq, self.layers[0].dofs[0]] = SV_1[2, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = SV_1[2, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = SV_1[2, 2]
        M[i_eq, self.layers[0].dofs[3]] = SV_1[2, 3]
        M[i_eq, self.layers[1].dofs[0]] = SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] = SV_2[1, 1]*delta_1
        i_eq += 1
        # sigma_xy = 0
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[0, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[0, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[0, 2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[0, 3]
        i_eq += 1
        return i_eq

class ElasticElasticInterface(PwInterface):
    """
    Elastic-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Elastic-Fluid interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of u_y
        for _i in range(4):
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[1, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[1, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[1, 2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[1, 3]
            M[i_eq, self.layers[1].dofs[0]] = SV_2[1, 0]
            M[i_eq, self.layers[1].dofs[1]] = SV_2[1, 1]
            M[i_eq, self.layers[1].dofs[2]] = SV_2[1, 2]*delta_1[0]
            M[i_eq, self.layers[1].dofs[3]] = SV_2[1, 3]*delta_1[1]
            i_eq += 1
        return i_eq

class PemPemInterface(PwInterface):
    """
    PEM-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t PEM-PEM interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of u_y
        for _i in range(6):
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[1, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[1, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[1, 2]*delta_0[2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[1, 3]
            M[i_eq, self.layers[0].dofs[4]] = -SV_1[1, 4]
            M[i_eq, self.layers[0].dofs[5]] = -SV_1[1, 5]

            M[i_eq, self.layers[1].dofs[0]] = SV_2[1, 0]
            M[i_eq, self.layers[1].dofs[1]] = SV_2[1, 1]
            M[i_eq, self.layers[1].dofs[2]] = SV_2[1, 2]
            M[i_eq, self.layers[1].dofs[3]] = SV_2[1, 3]*delta_1[0]
            M[i_eq, self.layers[1].dofs[4]] = SV_2[1, 4]*delta_1[1]
            M[i_eq, self.layers[1].dofs[5]] = SV_2[1, 5]*delta_1[2]
            i_eq += 1

        return i_eq

class ElasticPemInterface(PwInterface):
    """
    Elastic-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Elastic-Fluid interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of simga_xy
        M[i_eq, self.layers[0].dofs[0]] =  SV_1[1, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] =  SV_1[1, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] =  SV_1[1, 2]
        M[i_eq, self.layers[0].dofs[3]] =  SV_1[1, 3]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[2, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[2, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[2, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[2, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[2, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[2, 5]*delta_1[2]
        i_eq += 1
        # Continuity of u_y^s
        M[i_eq, self.layers[0].dofs[0]] =  SV_1[1, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] =  SV_1[1, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] =  SV_1[1, 2]
        M[i_eq, self.layers[0].dofs[3]] =  SV_1[1, 3]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[1, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[1, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[1, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[1, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[1, 5]*delta_1[2]
        i_eq += 1
        # Continuity of u_y^t
        M[i_eq, self.layers[0].dofs[0]] =  SV_1[1, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] =  SV_1[1, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] =  SV_1[1, 2]
        M[i_eq, self.layers[0].dofs[3]] =  SV_1[1, 3]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[2, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[2, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[2, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[2, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[2, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[2, 5]*delta_1[2]
        i_eq += 1
        # Continuity of sigma_yy = \hat{\sigma_yy)-p)
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[1, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[1, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[1, 2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[1, 3]
        M[i_eq, self.layers[1].dofs[0]] =  (SV_2[3, 0]-SV_2[4, 0])
        M[i_eq, self.layers[1].dofs[1]] =  (SV_2[3, 1]-SV_2[4, 1])
        M[i_eq, self.layers[1].dofs[2]] =  (SV_2[3, 2]-SV_2[4, 2])
        M[i_eq, self.layers[1].dofs[3]] =  (SV_2[3, 3]-SV_2[4, 3])*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] =  (SV_2[3, 4]-SV_2[4, 4])*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] =  (SV_2[3, 5]-SV_2[4, 5])*delta_1[2]
        i_eq += 1
        # Continuity of u_x^s
        M[i_eq, self.layers[0].dofs[0]] =  SV_1[3, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] =  SV_1[3, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] =  SV_1[3, 2]
        M[i_eq, self.layers[0].dofs[3]] =  SV_1[3, 3]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[5, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[5, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[5, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[5, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[5, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[5, 5]*delta_1[2]
        i_eq += 1
        return i_eq

class PemElasticInterface(PwInterface):
    """
    PEM-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t PEM-Elastic interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of simga_xy
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[2, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[2, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[2, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[2, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[2, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[2, 5]
        M[i_eq, self.layers[1].dofs[0]] =  SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] =  SV_2[1, 1]
        M[i_eq, self.layers[1].dofs[2]] =  SV_2[1, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] =  SV_2[1, 3]*delta_1[1]

        i_eq += 1
        # Continuity of u_y^s
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[1, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[1, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[1, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[1, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[1, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[1, 5]
        M[i_eq, self.layers[1].dofs[0]] =  SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] =  SV_2[1, 1]
        M[i_eq, self.layers[1].dofs[2]] =  SV_2[1, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] =  SV_2[1, 3]*delta_1[1]
        i_eq += 1
        # Continuity of u_y^t
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[2, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[2, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[2, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[2, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[2, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[2, 5]
        M[i_eq, self.layers[1].dofs[0]] =  SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] =  SV_2[1, 1]
        M[i_eq, self.layers[1].dofs[2]] =  SV_2[1, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] =  SV_2[1, 3]*delta_1[1]
        i_eq += 1
        # Continuity of sigma_yy = \hat{\sigma_yy)-p)
        M[i_eq, self.layers[0].dofs[0]] =  (SV_1[3, 0]-SV_1[4, 0])*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] =  (SV_1[3, 1]-SV_1[4, 1])*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] =  (SV_1[3, 2]-SV_1[4, 2])*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] =  (SV_1[3, 3]-SV_1[4, 3])
        M[i_eq, self.layers[0].dofs[4]] =  (SV_1[3, 4]-SV_1[4, 4])
        M[i_eq, self.layers[0].dofs[5]] =  (SV_1[3, 5]-SV_1[4, 5])
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[1, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[1, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[1, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[1, 3]*delta_1[1]
        i_eq += 1
        # Continuity of u_x^s
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[5, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[5, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[5, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[5, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[5, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[5, 5]
        M[i_eq, self.layers[1].dofs[0]] =  SV_2[3, 0]
        M[i_eq, self.layers[1].dofs[1]] =  SV_2[3, 1]
        M[i_eq, self.layers[1].dofs[2]] =  SV_2[3, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] =  SV_2[3, 3]*delta_1[1]
        i_eq += 1
        return i_eq

class FluidRigidBacking(PwInterface):
    """
    Rigid backing for a fluid layer
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def __str__(self):
        out = "\t Rigid backing"
        return out

    def update_M_global(self, M, i_eq):
        M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[0, 0]*np.exp(self.layers[0].lam[0]*self.layers[0].d)
        M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[0, 1]
        i_eq += 1
        return i_eq

    def Omega(self):
        return np.array([0,1], dtype=np.complex)

class PemBacking(PwInterface):
    """
    Rigid backing for a pem layer
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def __str__(self):
        out = "\t Rigid backing"
        return out

    def update_M_global(self, M, i_eq):
        delta = np.exp(self.layers[0].lam*self.layers[0].d)
        SV = self.layers[0].SV

        M[i_eq, self.layers[0].dofs[0]] = SV[1, 0]*delta[0]
        M[i_eq, self.layers[0].dofs[1]] = SV[1, 1]*delta[1]
        M[i_eq, self.layers[0].dofs[2]] = SV[1, 2]*delta[2]
        M[i_eq, self.layers[0].dofs[3]] = SV[1, 3]
        M[i_eq, self.layers[0].dofs[4]] = SV[1, 4]
        M[i_eq, self.layers[0].dofs[5]] = SV[1, 5]
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = SV[2, 0]*delta[0]
        M[i_eq, self.layers[0].dofs[1]] = SV[2, 1]*delta[1]
        M[i_eq, self.layers[0].dofs[2]] = SV[2, 2]*delta[2]
        M[i_eq, self.layers[0].dofs[3]] = SV[2, 3]
        M[i_eq, self.layers[0].dofs[4]] = SV[2, 4]
        M[i_eq, self.layers[0].dofs[5]] = SV[2, 5]
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = SV[5, 0]*delta[0]
        M[i_eq, self.layers[0].dofs[1]] = SV[5, 1]*delta[1]
        M[i_eq, self.layers[0].dofs[2]] = SV[5, 2]*delta[2]
        M[i_eq, self.layers[0].dofs[3]] = SV[5, 3]
        M[i_eq, self.layers[0].dofs[4]] = SV[5, 4]
        M[i_eq, self.layers[0].dofs[5]] = SV[5, 5]
        i_eq += 1
        return i_eq

    def Omega(self):
        Om = np.zeros((6,3), dtype=np.complex)
        Om[0,1] = 1.
        Om[3,2] = 1.
        Om[4,0] = 1.
        return Om

class ElasticBacking(PwInterface):

    """
    Rigid backing for an elastic layer
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def __str__(self):
        out = "\t Rigid backing"
        return out

    def update_M_global(self, M, i_eq):
        delta = np.exp(self.layers[0].lam*self.layers[0].d)
        SV = self.layers[0].SV
        M[i_eq, self.layers[0].dofs[0]] = SV[1, 0]*delta[0]
        M[i_eq, self.layers[0].dofs[1]] = SV[1, 1]*delta[1]
        M[i_eq, self.layers[0].dofs[2]] = SV[1, 2]
        M[i_eq, self.layers[0].dofs[3]] = SV[1, 3]
        i_eq += 1
        M[i_eq, self.layers[0].dofs[0]] = SV[3, 0]*delta[0]
        M[i_eq, self.layers[0].dofs[1]] = SV[3, 1]*delta[1]
        M[i_eq, self.layers[0].dofs[2]] = SV[3, 2]
        M[i_eq, self.layers[0].dofs[3]] = SV[3, 3]
        i_eq += 1
        return i_eq

class SemiInfinite(PwInterface):
    """
    Semi-infinite boundary
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2) 
        Air_mat = Air()
        self.medium = Fluid(c=Air_mat.c,rho=Air_mat.rho)
        self.SV = None

    def __str__(self):
        out = "\t Semi-infinite transmission medium"
        return out

    def update_frequency(self, omega, k, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = fluid_waves(self.medium, kx)
        self.k = k
        self.kx = kx
        self.omega = omega 

    def Omega(self):
        if self.layers[0].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
            return np.array([-self.lam[0]/(self.medium.rho*self.omega**2), 1], dtype=np.complex), np.eye(1)

        elif self.layers[0].medium.MEDIUM_TYPE == "pem":
            Om = np.zeros((6, 3), dtype=complex)
            Om[1, 1] = 1.
            Om[2, 0] = -self.lam[0]/(self.medium.rho*self.omega**2)
            Om[4, 0] = 1. 
            Om[5, 2] = 1.
        return Om, np.eye(3)

    ''' S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
         


    def update_M_global(self, M, i_eq):
        if self.layers[0].medium.MEDIUM_TYPE == "fluid":
            delta_0 = np.exp(self.layers[0].lam[0]*self.layers[0].d)
            M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[0, 0]*delta_0
            M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[0, 1]
            M[i_eq, -1] = -self.SV[0, 0]
            i_eq += 1
            M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[1, 0]*delta_0
            M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[1, 1]
            M[i_eq, -1] = -self.SV[1, 0]
            i_eq += 1
        elif self.layers[0].medium.MEDIUM_TYPE == "pem":
            delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
            SV_1 = self.layers[0].SV

            M[i_eq, self.layers[0].dofs[0]] = -SV_1[2, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[2, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[2, 2]*delta_0[2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[2, 3]
            M[i_eq, self.layers[0].dofs[4]] = -SV_1[2, 4]
            M[i_eq, self.layers[0].dofs[5]] = -SV_1[2, 5]
            M[i_eq, -1] = self.SV[0, 0]
            i_eq += 1
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[4, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[4, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[4, 2]*delta_0[2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[4, 3]
            M[i_eq, self.layers[0].dofs[4]] = -SV_1[4, 4]
            M[i_eq, self.layers[0].dofs[5]] = -SV_1[4, 5]
            M[i_eq, -1] = self.SV[1, 0]
            i_eq += 1
            M[i_eq, self.layers[0].dofs[0]] = SV_1[0, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = SV_1[0, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = SV_1[0, 2]*delta_0[2]
            M[i_eq, self.layers[0].dofs[3]] = SV_1[0, 3]
            M[i_eq, self.layers[0].dofs[4]] = SV_1[0, 4]
            M[i_eq, self.layers[0].dofs[5]] = SV_1[0, 5]
            i_eq += 1
            M[i_eq, self.layers[0].dofs[0]] = SV_1[3, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = SV_1[3, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = SV_1[3, 2]*delta_0[2]
            M[i_eq, self.layers[0].dofs[3]] = SV_1[3, 3]
            M[i_eq, self.layers[0].dofs[4]] = SV_1[3, 4]
            M[i_eq, self.layers[0].dofs[5]] = SV_1[3, 5]
            i_eq += 1

        elif self.layers[0].medium.MEDIUM_TYPE == "elastic":
            delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
            SV_1 = self.layers[0].SV
            # Continuity of u_y
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[1, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[1, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[1, 2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[1, 3]
            M[i_eq, -1] = self.SV[0, 0]
            i_eq += 1
            # sigma_yy = -p
            M[i_eq, self.layers[0].dofs[0]] = SV_1[2, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = SV_1[2, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = SV_1[2, 2]
            M[i_eq, self.layers[0].dofs[3]] = SV_1[2, 3]
            M[i_eq, -1] = self.SV[1, 0]
            i_eq += 1
            # sigma_xy = 0
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[0, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[0, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[0, 2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[0, 3]
            i_eq += 1


        return i_eq
