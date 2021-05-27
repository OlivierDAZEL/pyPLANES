#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_interfaces.py
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
from numpy import sqrt
from pyPLANES.utils.io import load_material
from pyPLANES.pw.pw_layers import PwLayer
from pyPLANES.pw.periodic_layer import PeriodicLayer
from pyPLANES.pw.pw_layers import PwLayer
from pyPLANES.pw.pw_polarisation import fluid_waves_TMM
from scipy.linalg import block_diag

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
    def update_frequency(self, omega, kx):
        self.nb_waves = len(kx)

class PwInterfaceType0(PwInterface):
    """
    Interface with same number of physical medium on both side
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def transfert(self, Om):
        return Om, np.eye(self.layers[1].nb_waves_in_medium*self.layers[1].nb_waves)

class PwInterfaceType1(PwInterface): 
    """
    Interface with less waves in layer 0 than in layer 1 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def transfert(self, Om):

        n_w = self.nb_waves
        list_null_fields = [2*self.n_1*_w+i for _w in range(n_w) for i in self.null_fields]
        list_kept_fields = [2*self.n_1*_w+i for _w in range(n_w) for i in self.kept_fields]

        list_master = range(0,self.n_0*n_w)
        list_slaves = range(self.n_0*n_w,self.n_1*n_w)

        Tau = -LA.solve(Om[np.ix_(list_null_fields, list_slaves)], Om[np.ix_(list_null_fields, list_master)])
        Om = Om[np.ix_(list_kept_fields, list_master)] + Om[np.ix_(list_kept_fields, list_slaves)]@Tau

        TTau = np.vstack((np.eye(self.n_0*n_w), Tau))

        return Om, TTau

class PwInterfaceType2(PwInterface):
    """
        Interface with more waves in layer 0 than in layer 1
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = None
        self.n_1 = None
        self.D = None
        self.A = None

    def transfert(self, Om_):

        n_w = self.layers[0].nb_waves
        Om = np.hstack((np.kron(np.ones((n_w, n_w)), self.D)@Om_, np.kron(np.eye(n_w), self.A)))
        Tau = np.hstack((np.eye(self.n_1*n_w), np.zeros((self.n_1*n_w,self.n_0-self.n_1))))

        return Om, Tau

class FluidFluidInterface(PwInterfaceType0):
    """
    Fluid-fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Fluid-fluid interface"
        return out

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

    # def transfert(self, Om):
    #     _w = Om.shape[1]
    #     return Om.reshape(2*_w,_w), np.eye(_w)

class FluidPemInterface(PwInterfaceType1):
    """
    Fluid-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

        self.n_0 = 1
        self.n_1 = 3
        self.null_fields = [0,3]
        self.kept_fields = [2,4]


    def __str__(self):
        out = "\t Fluid-PEM interface"
        return out

    def transfert(self, Om_):

        mat_pem = np.eye(6)
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                mat_pem[2, 1] = 1.
                mat_pem[3, 4] = 1.
        Om = np.kron(np.eye(self.nb_waves), mat_pem)@ Om_

        return PwInterfaceType1.transfert(self, Om)

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

class PemFluidInterface(PwInterfaceType2):
    """
    PEM-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                typ = "Biot01"
        else:
            typ = "Biot98"

        if typ == "Biot98":
            self.D = np.zeros((6, 2), dtype=complex)
            self.D[2, 0] = 1
            self.D[4, 1] = 1
            self.A = np.zeros((6, 2), dtype=complex)
            self.A[1, 0] = 1
            self.A[5, 1] = 1
        else: 
            self.D = np.zeros((6, 2), dtype=complex)
            self.D[0, 0] = 1 
            self.D[1, 1] = 1
            self.D[3, 2] = 1
            self.D[5, 3] = 1
            self.A = np.zeros((6, 2), dtype=complex)
            self.A[4, 0] = 1

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

class FluidElasticInterface(PwInterfaceType1):
    """
    Fluid-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 1
        self.n_1 = 2
        self.null_fields = [0]
        self.kept_fields = [1,2]
        self.list_master = [0]
        self.list_slaves = [1]

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

    def transfert(self, Om_):
        Om, Tau = PwInterfaceType1.transfert(self, Om_)
        Om[1,:] *= -1 
        return Om, Tau
    #     n_w = self.nb_waves

    #     list_null_fields = [4*_w for _w in range(n_w)] # sig_x_z
    #     list_kept_fields = [4*_w+i for _w in range(n_w) for i in [1,2]] 
    #     list_master = [2*_w for _w in range(n_w)]
    #     list_slaves = [2*_w+1 for _w in range(n_w)]

    #     if LA.det(Om_[np.ix_(list_null_fields, list_slaves)]) !=0: # Non zero incidence 
    #         Tau = -LA.solve(Om_[np.ix_(list_null_fields, list_slaves)], Om_[np.ix_(list_null_fields, list_master)])
    #     else:
    #         raise NameError("Zero incidence in Fluid-Elastic transfer")
    #         Tau = -LA.solve(Om_[np.ix_(list_null_fields, list_slaves)], Om_[np.ix_(list_null_fields, list_master)])

    #     Om = Om_[np.ix_(list_kept_fields, list_master)] + Om_[np.ix_(list_kept_fields, list_slaves)]@Tau

    #     Mat_sign = (np.diag([(-1)**i for i in range(2*n_w)]))
    #     Om = Mat_sign @ Om

    #     TTau = np.zeros((2*n_w, n_w),dtype=complex)
    #     for _w in range(n_w):
    #         TTau[2*_w, _w] = 1.
    #         TTau[2*_w+1, :] = Tau[_w,:]

    #     return Om, TTau.reshape((2*n_w, n_w))

class ElasticFluidInterface(PwInterfaceType2):
    """
    Elastic-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 2
        self.n_1 = 1
        self.D = np.zeros((4, 2), dtype=complex)
        self.D[1, 0] = 1
        self.D[2, 1] = -1
        self.A = np.zeros((4, 1), dtype=complex)
        self.A[3, 0] = 1



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

class ElasticElasticInterface(PwInterfaceType0):
    """
    Elastic-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Elastic-Elastic interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of u_y
        for _i in range(4):
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[_i, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[_i, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[_i, 2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[_i, 3]
            M[i_eq, self.layers[1].dofs[0]] = SV_2[_i, 0]
            M[i_eq, self.layers[1].dofs[1]] = SV_2[_i, 1]
            M[i_eq, self.layers[1].dofs[2]] = SV_2[_i, 2]*delta_1[0]
            M[i_eq, self.layers[1].dofs[3]] = SV_2[_i, 3]*delta_1[1]
            i_eq += 1
        return i_eq

class PemPemInterface(PwInterfaceType0):
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
            M[i_eq, self.layers[0].dofs[0]] = -SV_1[_i, 0]*delta_0[0]
            M[i_eq, self.layers[0].dofs[1]] = -SV_1[_i, 1]*delta_0[1]
            M[i_eq, self.layers[0].dofs[2]] = -SV_1[_i, 2]*delta_0[2]
            M[i_eq, self.layers[0].dofs[3]] = -SV_1[_i, 3]
            M[i_eq, self.layers[0].dofs[4]] = -SV_1[_i, 4]
            M[i_eq, self.layers[0].dofs[5]] = -SV_1[_i, 5]

            M[i_eq, self.layers[1].dofs[0]] = SV_2[_i, 0]
            M[i_eq, self.layers[1].dofs[1]] = SV_2[_i, 1]
            M[i_eq, self.layers[1].dofs[2]] = SV_2[_i, 2]
            M[i_eq, self.layers[1].dofs[3]] = SV_2[_i, 3]*delta_1[0]
            M[i_eq, self.layers[1].dofs[4]] = SV_2[_i, 4]*delta_1[1]
            M[i_eq, self.layers[1].dofs[5]] = SV_2[_i, 5]*delta_1[2]
            i_eq += 1

        return i_eq

    def transfert(self, Om):

        mat_pem_0, mat_pem_1 = np.eye(6), np.eye(6)

        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[1].typ == "Biot01":
                mat_pem_0[2, 1] = -1.
                mat_pem_0[3, 4] = -1.
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                mat_pem_1[2, 1] = 1.
                mat_pem_1[3, 4] = 1.

        mat_pem_0 = np.kron(np.eye(self.nb_waves), mat_pem_0)
        mat_pem_1 = np.kron(np.eye(self.nb_waves), mat_pem_1)

        Om, Tau = PwInterfaceType0.transfert(self, Om)

        return (mat_pem_0@mat_pem_1)@Om, Tau

class ElasticPemInterface(PwInterfaceType1):
    """
    Elastic-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 2
        self.n_1 = 3
        self.null_fields = [2]
        self.kept_fields = [0, 1, 3, 5]


    def __str__(self):
        out = "\t Elastic-PEM interface"
        return out

    def transfert(self, Om_):

        mat_pem = np.eye(6)
        if isinstance(self.layers[1], PwLayer):
            mat_pem[2, 1] = -1.
            mat_pem[3, 4] = -1.
        elif isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot98":
                mat_pem[2, 1] = 1.
                mat_pem[3, 4] = 1.


        Om = np.kron(np.eye(self.nb_waves), mat_pem)@ Om_

        return PwInterfaceType1.transfert(self, Om)

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        ''' S={0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
        SV_2 = self.layers[1].SV
        ''' S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
        # Continuity of simga_xy
        M[i_eq, self.layers[0].dofs[0]] =  SV_1[0, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] =  SV_1[0, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] =  SV_1[0, 2]
        M[i_eq, self.layers[0].dofs[3]] =  SV_1[0, 3]
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[0, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[0, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[0, 2]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[0, 3]*delta_1[0]
        M[i_eq, self.layers[1].dofs[4]] = -SV_2[0, 4]*delta_1[1]
        M[i_eq, self.layers[1].dofs[5]] = -SV_2[0, 5]*delta_1[2]
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
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[2, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[2, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[2, 2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[2, 3]
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

class PemElasticInterface(PwInterfaceType2):
    """
    PEM-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 3
        self.n_1 = 2
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                typ = "Biot01"
        else:
            typ = "Biot98"

        if typ == "Biot98":
            self.D = np.zeros((6, 4), dtype=complex)
            self.D[0, 0] = 1 
            self.D[1, 1] = 1
            self.D[2, 1] = 1
            self.D[3, 2] = 1
            self.D[5, 3] = 1
            self.A = np.zeros((6, 1), dtype=complex)
            self.A[3, 0] = 1
            self.A[4, 0] = 1
        else: 
            self.D = np.zeros((6, 4), dtype=complex)
            self.D[0, 0] = 1 
            self.D[1, 1] = 1
            self.D[3, 2] = 1
            self.D[5, 3] = 1
            self.A = np.zeros((6, 1), dtype=complex)
            self.A[4, 0] = 1

    def __str__(self):
        out = "\t PEM-Elastic interface"
        return out

    def update_M_global(self, M, i_eq):
        delta_0 = np.exp(self.layers[0].lam*self.layers[0].d)
        delta_1 = np.exp(self.layers[1].lam*self.layers[1].d)
        SV_1 = self.layers[0].SV
        SV_2 = self.layers[1].SV
        # Continuity of simga_xy
        M[i_eq, self.layers[0].dofs[0]] = -SV_1[0, 0]*delta_0[0]
        M[i_eq, self.layers[0].dofs[1]] = -SV_1[0, 1]*delta_0[1]
        M[i_eq, self.layers[0].dofs[2]] = -SV_1[0, 2]*delta_0[2]
        M[i_eq, self.layers[0].dofs[3]] = -SV_1[0, 3]
        M[i_eq, self.layers[0].dofs[4]] = -SV_1[0, 4]
        M[i_eq, self.layers[0].dofs[5]] = -SV_1[0, 5]
        M[i_eq, self.layers[1].dofs[0]] =  SV_2[0, 0]
        M[i_eq, self.layers[1].dofs[1]] =  SV_2[0, 1]
        M[i_eq, self.layers[1].dofs[2]] =  SV_2[0, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] =  SV_2[0, 3]*delta_1[1]
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
        M[i_eq, self.layers[1].dofs[0]] = -SV_2[2, 0]
        M[i_eq, self.layers[1].dofs[1]] = -SV_2[2, 1]
        M[i_eq, self.layers[1].dofs[2]] = -SV_2[2, 2]*delta_1[0]
        M[i_eq, self.layers[1].dofs[3]] = -SV_2[2, 3]*delta_1[1]
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

    def Omega(self, nb_bloch_waves=0):
        out = np.array([0,1]).reshape(2,1)
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=np.complex)

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

    def Omega(self, nb_bloch_waves=1):
        out = np.zeros((6,3), dtype=np.complex)
        out[0,1] = 1.
        out[3,2] = 1.
        out[4,0] = 1.
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=np.complex)

class ElasticBacking(PwInterface):

    """
    Rigid backing for an elastic layer
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

    def __str__(self):
        out = "\t Rigid backing"
        return out

    def Omega(self, nb_bloch_waves=1):
        out = np.zeros((4,2), dtype=np.complex)
        out[0,1] = 1.
        out[2,0] = 1.
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=np.complex)


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
        self.medium = load_material("Air")
        self.SV = None

    def __str__(self):
        out = "\t Semi-infinite transmission medium"
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = fluid_waves_TMM(self.medium, kx)
        self.k = self.medium.k
        self.kx = kx
        self.omega = omega 

    def Omega(self, nb_bloch_waves=1):
        typ =None
        if isinstance(self.layers[0], PwLayer):
            if self.layers[0].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                typ = "fluid"
            elif self.layers[0].medium.MEDIUM_TYPE in ["pem"]:
                typ ="pem"
                formulation = "Biot98"
            elif self.layers[0].medium.MEDIUM_TYPE in ["elastic"]:
                typ ="elastic"
        else:
            if self.layers[0].medium[1].MEDIUM_TYPE in ["fluid", "eqf"]:
                typ ="fluid"
            elif self.layers[0].medium[1].MEDIUM_TYPE in ["pem"]:
                typ ="pem"
                formulation = self.layers[0].pwfem_entities[1].typ
            elif self.layers[0].medium[1].MEDIUM_TYPE in ["elastic"]:
                typ ="elastic"
        # else:
        #     raise NameError("Layer is neither PwLayer nor PeriodicLayer")

        if typ == "fluid":
            self.len_X = 1
            out = np.zeros((2*nb_bloch_waves, nb_bloch_waves), dtype=complex)
            for _w in range(nb_bloch_waves):
                out[0+_w*2, 0+_w] = self.lam[2*_w]/(self.medium.rho*self.omega**2)
                out[1+_w*2, 0+_w] = 1
            return out, np.eye(max([nb_bloch_waves,1]))
        elif typ == "pem":
            self.len_X = 3
            out = np.zeros((6*nb_bloch_waves, 3*nb_bloch_waves), dtype=complex)
            if formulation == "Biot98":
                for _w in range(nb_bloch_waves):
                    # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
                    out[1+_w*6, 1+_w*3] = 1.
                    out[2+_w*6, 0+_w*3] = self.lam[2*_w]/(self.medium.rho*self.omega**2)
                    out[4+_w*6, 0+_w*3] = 1.
                    out[5+_w*6, 2+_w*3] = 1.
            elif formulation == "Biot01":
                for _w in range(nb_bloch_waves):
                    # pem S={0:{\sigma}^t_{xy}, 1:u_y^s, 2:w_y=u_y^t-u_y^s, 3:{\sigma}^t_{yy}, 4:p, 5:u_x^s}'''
                    out[1+_w*6, 1+_w*3] = 1.
                    out[2+_w*6, 0+_w*3] = self.lam[2*_w]/(self.medium.rho*self.omega**2)
                    out[2+_w*6, 1+_w*3] = -1.
                    out[3+_w*6, 0+_w*3] = -1.
                    out[4+_w*6, 0+_w*3] = 1.
                    out[5+_w*6, 2+_w*3] = 1.
            else: 
                raise NameError("Incorrect Biot formulation")
            return out, np.eye(3*max([nb_bloch_waves,1]))
        elif typ  == "elastic":
            self.len_X = 2
            out = np.zeros((4*nb_bloch_waves, 2*nb_bloch_waves), dtype=complex)
            for _w in range(nb_bloch_waves):
                out[1+_w*4, 0+_w*2] = self.lam[2*_w]/(self.medium.rho*self.omega**2)
                out[2+_w*4, 0+_w*2] = -1. # \sigma_{yy} is -p
                out[3+_w*4, 1+_w*2] = 1.

            return out, np.eye(2*max([nb_bloch_waves,1]))

    def update_M_global(self, M, i_eq):
        if self.layers[0].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
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
