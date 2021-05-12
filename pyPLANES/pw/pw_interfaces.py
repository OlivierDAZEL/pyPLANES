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
        _w = Om.shape[1]
        return Om.reshape(2*_w,_w), np.eye(_w)

class FluidPemInterface(PwInterface):
    """
    Fluid-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Fluid-PEM interface"
        return out

    def transfert(self, Om_):
        n_w = self.nb_waves
        mat_pem = np.eye(6)
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                mat_pem[2, 1] = 1.
                mat_pem[3, 4] = 1.
        Om = np.kron(np.eye(n_w), mat_pem)@ Om_


        list_null_fields = [6*_w+i for _w in range(n_w) for i in [0,3]]
        list_kept_fields = [6*_w+i for _w in range(n_w) for i in [2,4]]
        list_master = [3*_w for _w in range(n_w)]
        list_slaves = [3*_w+i for _w in range(n_w) for i in [1,2]]

        # print(list_null_fields)
        # print(list_slaves)
        
        Tau = -LA.solve(Om[np.ix_(list_null_fields, list_slaves)], Om[np.ix_(list_null_fields, list_master)])
        Om__ = Om[np.ix_(list_kept_fields, list_master)] + Om[np.ix_(list_kept_fields, list_slaves)]@Tau

        TTau = np.zeros((3*n_w, n_w),dtype=complex)
        for _w in range(n_w):
            TTau[3*_w, _w] = 1.
            TTau[3*_w+1, :] = Tau[2*_w,:]
            TTau[3*_w+2, :] = Tau[2*_w+1,:]
        # print(np.array([1,0,0,]))
        # print(Tau)
        # print(Om__)
        # print(TTau)
        # Omega_moins, Tau_tilde = [], []

        # for _w in range(n_w):
        #     Om = mat_pem@Om_[6*_w:6*(_w+1), 3*_w:3*(_w+1)]

        #     a = -np.array([
        #         [Om[0,1],Om[0,2]],
        #         [Om[3,1],Om[3,2]]
        #     ])
        #     Tau = np.dot(np.linalg.inv(a), np.array([[Om[0,0]], [Om[3,0]]]))
        #     Tau_tilde.append(np.concatenate([np.eye(1),Tau]))
        #     # print("--")
        #     # print(Tau)
        #     Omega_moins.append(np.array([[Om[2,0]], [Om[4,0]]]) + np.dot(np.array([[Om[2,1], Om[2,2]], [Om[4,1], Om[4,2]]]), Tau).reshape(2,1).tolist())
        #     print(Omega_moins[0])
        # Omega_moins = block_diag(*Omega_moins)
        # Tau_tilde = block_diag(*Tau_tilde)
        # print("Om")
        # print(Om__)
        # return np.block(Omega_moins), np.block(Tau_tilde)
        return Om__, TTau

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

    def transfert(self, Om_):
        Omega_moins, Tau_tilde = [], []
        n_w = self.nb_waves
        for _w in range(n_w):
            Om = Om_[4*_w:4*(_w+1), 2*_w:2*(_w+1)]
            tau = -Om_[0,0]/Om_[0,1]
            Omega_moins.append(np.array([[Om[1,1]], [-Om[2,1]]])*tau + np.array([[Om[1,0]], [-Om[2,0]]]))
            Tau_tilde.append(np.concatenate([np.eye(1,1), np.array([[tau]])]))

        Omega_moins = block_diag(*Omega_moins)
        Tau_tilde = block_diag(*Tau_tilde)
        return np.block(Omega_moins), np.block(Tau_tilde)

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

    def transfert(self, Om_):
        Omega_moins, Tau_tilde = [], []
        n_w = self.nb_waves
        for _w in range(n_w):
            Om = Om_[2*_w:2*(_w+1), 1*_w:1*(_w+1)]
            Om_moins = np.zeros((4,2), dtype=np.complex)
            Om_moins[1,0] = Om[0,0]
            Om_moins[2,0] = -Om[1,0]
            Om_moins[3,1] = 1
            Omega_moins.append(Om_moins)
            T_tilde = np.zeros((1,2), dtype=np.complex)
            T_tilde[0,0] = 1
            Tau_tilde.append(T_tilde)

        Omega_moins = block_diag(*Omega_moins)
        Tau_tilde = block_diag(*Tau_tilde)
        return np.block(Omega_moins), np.block(Tau_tilde)

class ElasticElasticInterface(PwInterface):
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

    def transfert(self, Om):
        return Om, np.eye(2)

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

    def transfert(self, O):
        return (O, np.eye(3*self.nb_waves))

class ElasticPemInterface(PwInterface):
    """
    Elastic-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
    def __str__(self):
        out = "\t Elastic-PEM interface"
        return out

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

    def transfert(self, Om_):
        Omega_moins, Tau_tilde = [], []
        n_w = self.nb_waves
        Dplus = np.array([0, 1, -1, 0, 0, 0])
        Dmoins = np.zeros((4,6), dtype=np.complex)
        Dmoins[0,0] = 1
        Dmoins[1,1] = 1
        Dmoins[2,3] = 1
        Dmoins[2,4] = -1
        Dmoins[3,5] = 1

        mat_pem = np.eye(6)
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                mat_pem[2, 1] = 1.
                mat_pem[3, 4] = 1.


        for _w in range(n_w):
            Om = mat_pem@Om_[6*_w:6*(_w+1), 3*_w:3*(_w+1)]
            Tau = -Dplus.dot(Om[:,2:4])**-1 * np.dot(Dplus, Om[:,0:2])
            Omega_moins.append(Dmoins.dot(Om[:,0:2] + Om[:,2:4]*Tau))
            Tau_tilde.append(np.vstack([np.eye(2), Tau]))

        Omega_moins = block_diag(*Omega_moins)
        Tau_tilde = block_diag(*Tau_tilde)
        return np.block(Omega_moins), np.block(Tau_tilde)

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

    def transfert(self, Om_):
        is_infinite_layer = not any([isinstance(_l, PeriodicLayer) for _l in self.layers])
        Omega_moins, Tau_tilde = [], []
        n_w = self.nb_waves
        for _w in range(n_w):
            if is_infinite_layer:
                Om = Om_[4*_w:4*(_w+1), 2*_w:2*(_w+1)]
                Om_moins = np.zeros((6,3), dtype=np.complex)
                Om_moins[0,0:2] = Om[0,0:2]
                Om_moins[1,0:2] = Om[1,0:2]
                Om_moins[2,0:2] = Om[1,0:2]
                Om_moins[3,0:2] = Om[2,0:2]
                Om_moins[3,2] = 1
                Om_moins[4,2] = 1
                Om_moins[5,0:2] = Om[3,0:2]
                Omega_moins.append(Om_moins)
                T_tilde = np.zeros((2,3), dtype=np.complex)
                T_tilde[0,0] = 1
                T_tilde[1,1] = 1
                Tau_tilde.append(T_tilde)
            else:
                # elastic {0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
                # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
                # print(self.layers[0].entities[-1].formulation98)
                Om = Om_[4*_w:4*(_w+1), 2*_w:2*(_w+1)]
                Om_moins = np.zeros((6,3), dtype=np.complex)
                Om_moins[0,0:2] = Om[0,0:2] #\sigma_{xy}
                Om_moins[1,0:2] = Om[1,0:2] #u_y
                Om_moins[2,0:2] = Om[1,0:2] #2:u_y^t
                Om_moins[3,0:2] = Om[2,0:2] #\hat{\sigma}_{yy} = \sigma_{yy}^e + p
                Om_moins[3,2] = 1
                Om_moins[4,2] = 1
                Om_moins[5,0:2] = Om[3,0:2]
                Omega_moins.append(Om_moins)
                T_tilde = np.zeros((2,3), dtype=np.complex)
                T_tilde[0,0] = 1
                T_tilde[1,1] = 1
                Tau_tilde.append(T_tilde)
        Omega_moins = block_diag(*Omega_moins)
        Tau_tilde = block_diag(*Tau_tilde)



        return np.block(Omega_moins), np.block(Tau_tilde)

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
            elif self.layers[0].medium.MEDIUM_TYPE in ["elastic"]:
                typ ="elastic"
        else:
            if self.layers[0].medium[1].MEDIUM_TYPE in ["fluid", "eqf"]:
                typ ="fluid"
            elif self.layers[0].medium[1].MEDIUM_TYPE in ["pem"]:
                typ ="pem"
            elif self.layers[0].medium[1].MEDIUM_TYPE in ["elastic"]:
                typ ="elastic"
        # else:
        #     raise NameError("Layer is neither PwLayer nor PeriodicLayer")

        if typ == "fluid":
            out = np.zeros((2*nb_bloch_waves, nb_bloch_waves), dtype=complex)
            for _w in range(nb_bloch_waves):
                out[0+_w*2, 0+_w] = self.lam[2*_w]/(self.medium.rho*self.omega**2)
                out[1+_w*2, 0+_w] = 1
            return out, np.eye(max([nb_bloch_waves,1]))
        elif typ == "pem":
            out = np.zeros((6*nb_bloch_waves, 3*nb_bloch_waves), dtype=complex)
            for _w in range(nb_bloch_waves):
                # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
                out[1+_w*6, 1+_w*3] = 1.
                out[2+_w*6, 0+_w*3] = self.lam[2*_w]/(self.medium.rho*self.omega**2)
                out[4+_w*6, 0+_w*3] = 1.
                out[5+_w*6, 2+_w*3] = 1.
            return out, np.eye(3*max([nb_bloch_waves,1]))
        elif typ  == "elastic":
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
