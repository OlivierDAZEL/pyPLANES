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
from pyPLANES.pw.pw_polarisation import fluid_waves_TMM, PEM_waves_TMM, elastic_waves_TMM
from scipy.linalg import block_diag

class PwInterface():
    """
    Interface for Plane Wave Solver
    """
    def __init__(self, layer1=None, layer2=None):
        self.layers = [layer1, layer2]
        self.n_0, self_n_1 = None, None 
        self.number_relations = None
        self.pw_method = None
        self.C_minus, self.C_plus = None, None

    def update_frequency(self, omega, kx):
        self.nb_waves = len(kx)
        if isinstance(self.layers[0],PwLayer):
            self.layers[0].medium.update_frequency(omega)
        if isinstance(self.layers[1],PwLayer):
            self.layers[1].medium.update_frequency(omega)

    def update_M_global(self, M, i_eq):

        SV_0 = self.layers[0].SV
        SV_1 = self.layers[1].SV

        d_0 = [self.layers[0].d]*self.n_0+[0]*self.n_0
        d_1 = [0]*self.n_1 +[-self.layers[1].d]*self.n_1

        delta_0 = np.diag(np.exp(self.layers[0].lam*d_0))
        delta_1 = np.diag(np.exp(self.layers[1].lam*d_1))

        index_rel = slice(i_eq, i_eq+self.number_relations)
        M [index_rel, self.layers[0].dofs] = self.C_minus@(SV_0@delta_0)
        M [index_rel, self.layers[1].dofs] = self.C_plus@(SV_1@delta_1)

        i_eq += self.number_relations
        return i_eq

    def update_Omega(self, Om):

        if isinstance(self.layers[0], PwLayer):
            mat = self.layers[0].medium
        elif isinstance(self.layers[0], PeriodicLayer):
            mat = self.layers[0].medium[1]
        if self.nb_waves == 1:
            SV = self.pw_method(mat, np.zeros(1))[0] 
            P_in = SV[:,:self.n_0].reshape((2*self.n_0,self.n_0))
            P_out = SV[:,self.n_0:].reshape((2*self.n_0,self.n_0))

            M1 = self.C_plus@Om
            M2 = self.C_minus@P_in
            M3 = self.C_minus@P_out

            M = -LA.inv(np.hstack((M1,M2)))@M3

            M_X = M[:self.n_1,:]
            M_S = M[self.n_1:,:]
            Omega = P_in@M_S +P_out 
        else: 
            SV = self.pw_method(mat, np.zeros(1))[0] 
            P_in = SV[:,:self.n_0].reshape((2*self.n_0, self.n_0))
            P_out = SV[:,self.n_0:].reshape((2*self.n_0, self.n_0))

            M1 = np.kron(np.eye(self.nb_waves), self.C_plus)@Om
            M2 = np.kron(np.eye(self.nb_waves), self.C_minus@P_in)
            M3 = np.kron(np.eye(self.nb_waves), self.C_minus@P_out)

            M = -LA.inv(np.hstack((M1,M2)))@M3

            M_X = M[:self.n_1*self.nb_waves,:]
            M_S = M[self.n_1*self.nb_waves:,:]
            Omega = np.kron(np.eye(self.nb_waves), P_in)@M_S +np.kron(np.eye(self.nb_waves), P_out) 
        return Omega, M_X

class FluidFluidInterface(PwInterface):
    """
    Fluid-fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = self.n_1 = 1
        self.number_relations = 2
        self.C_minus = np.eye(self.number_relations)
        self.C_plus = -np.eye(self.number_relations)
        self.pw_method = fluid_waves_TMM

    def __str__(self):
        out = "\t Fluid-fluid interface"
        return out

class FluidPemInterface(PwInterface):
    """
    Fluid-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)

        self.n_0 = 1
        self.n_1 = 3
        self.number_relations = 4
        self.C_minus = np.array([[1,0],[0,1], [0,0], [0, 0]])
        self.C_plus = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                self.C_minus = np.array([[1,0],[0,1], [0,1], [0, 0]])
                self.C_plus = np.array([[0, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
        self.pw_method = fluid_waves_TMM

    def __str__(self):
        out = "\t Fluid-PEM interface"
        return out


class PemFluidInterface(PwInterface):
    """
    PEM-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 3
        self.n_1 = 1
        self.pw_method = PEM_waves_TMM

        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                typ = "Biot01"
            else:
                typ = "Biot98"
        else:
            typ = "Biot98"

        self.number_relations = 4
        self.C_minus = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        self.C_plus = np.array([[1,0],[0,1], [0,0], [0, 0]])
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_minus = np.array([[0, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
                self.C_plus = np.array([[1,0],[0,1], [0,1], [0, 0]])

    def __str__(self):
        out = "\t PEM-Fluid interface"
        return out

class FluidElasticInterface(PwInterface):
    """
    Fluid-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 1
        self.n_1 = 2
        self.number_relations = 3
        self.C_minus = np.array([[0,0],[1,0],[0,1]])
        self.C_plus = np.array([[1, 0, 0, 0], [0, -1., 0, 0 ],[0, 0, 1, 0]])
        self.pw_method = fluid_waves_TMM

    def __str__(self):
        out = "\t Fluid-Elastic interface"
        return out

class ElasticFluidInterface(PwInterface):
    """
    Elastic-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 2
        self.n_1 = 1
        self.number_relations = 3
        self.C_minus = np.array([[1, 0, 0, 0], [0, -1., 0, 0 ],[0, 0, 1, 0]])
        self.C_plus = np.array([[0,0],[1,0],[0,1]])
        self.pw_method = elastic_waves_TMM

    def __str__(self):
        out = "\t Elastic-Fluid interface"
        return out

class ElasticElasticInterface(PwInterface):
    """
    Elastic-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 2
        self.n_1 = 2
        self.number_relations = 4
        self.C_minus = np.eye(self.number_relations)
        self.C_plus = -np.eye(self.number_relations)
        self.pw_method = elastic_waves_TMM

    def __str__(self):
        out = "\t Elastic-Elastic interface"
        return out

class PemPemInterface(PwInterface):
    """
    PEM-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 3
        self.n_1 = 3
        self.number_relations = 6
        self.C_minus = np.eye(self.number_relations)
        self.C_plus = -np.eye(self.number_relations)
        self.pw_method = PEM_waves_TMM

    def __str__(self):
        out = "\t PEM-PEM interface"
        return out

    def transfert(self, Om):

        mat_pem_0, mat_pem_1 = np.eye(6), np.eye(6)
        # if isinstance(self.layers[0], PeriodicLayer):
        #     if self.layers[0].pwfem_entities[1].typ == "Biot01":
        #         mat_pem_0[2, 1] = -1.
        #         mat_pem_0[3, 4] = -1.
        # if isinstance(self.layers[1], PeriodicLayer):
        #     if self.layers[1].pwfem_entities[0].typ == "Biot01":
        #         mat_pem_1[2, 1] = 1.
        #         mat_pem_1[3, 4] = 1.

        mat_pem_0 = np.kron(np.eye(self.nb_waves), mat_pem_0)
        mat_pem_1 = np.kron(np.eye(self.nb_waves), mat_pem_1)

        Om, Tau = PwInterface.transfert(self, Om)

        return (mat_pem_0@mat_pem_1)@Om, Tau

class ElasticPemInterface(PwInterface):
    """
    Elastic-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 2
        self.n_1 = 3
 
        self.number_relations = 5
        self.C_minus = np.zeros((self.number_relations, 2*self.n_0))
        self.C_plus = np.zeros((self.number_relations, 2*self.n_1))
        # \sigma_xz 
        self.C_minus[0, 0], self.C_plus[0, 0] = 1., -1. 
        # u_z =u_z_s  
        self.C_minus[1, 1], self.C_plus[1, 1] = 1., -1.
        # u_z =u_z_t 
        self.C_minus[2, 1], self.C_plus[2, 2] = 1., -1.
        # Case of 2001 formulation 0 =w
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                self.C_minus[2, 1] = 0.
        # sigma_zz = \hat{\sigma_zz} - p
        self.C_minus[3, 2], self.C_plus[3, 3], self.C_plus[3, 4] = 1., -1., 1. 

        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                self.C_plus[3, 4] = 0.
        # u_x = u_x_s 
        self.C_minus[4, 3], self.C_plus[4, 5] = 1., -1.



        self.pw_method = elastic_waves_TMM

    def __str__(self):
        out = "\t Elastic-PEM interface"
        return out

    def transfert(self, Om):
        # Mat_pem@Om returns Om in 2001 format 
        mat_pem = np.eye(6)
        # if isinstance(self.layers[1], PwLayer):
        #     mat_pem[2, 1] = -1.
        #     mat_pem[3, 4] = -1.
        # elif isinstance(self.layers[1], PeriodicLayer):
        #     if self.layers[1].pwfem_entities[0].typ == "Biot98":
        #         mat_pem[2, 1] = -1.
        #         mat_pem[3, 4] = -1.
        Om_ = np.kron(np.eye(self.nb_waves), mat_pem)@ Om

        return PwInterface.transfert(self, Om_)

class PemElasticInterface(PwInterface):
    """
    PEM-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = 3
        self.n_1 = 2
        self.pw_method = PEM_waves_TMM
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                typ = "Biot01"
        else:
            typ = "Biot98"

        self.number_relations = 5

        self.C_minus = np.zeros((self.number_relations, 2*self.n_0))
        self.C_plus = np.zeros((self.number_relations, 2*self.n_1))
        # \sigma_xz 
        self.C_plus[0, 0], self.C_minus[0, 0] = 1., -1. 
        # u_z =u_z_s  
        self.C_plus[1, 1], self.C_minus[1, 1] = 1., -1.
        # u_z =u_z_t 
        self.C_plus[2, 1], self.C_minus[2, 2] = 1., -1.
        # Case of 2001 formulation 0 =w
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_plus[2, 1] = 0.
        # sigma_zz = \hat{\sigma_zz} - p
        self.C_plus[3, 2], self.C_minus[3, 3], self.C_minus[3, 4] = 1., -1., 1. 

        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_minus[3, 4] = 0.
        # u_x = u_x_s 
        self.C_plus[4, 3], self.C_minus[4, 5] = 1., -1.


    def __str__(self):
        out = "\t PEM-Elastic interface"
        return out

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
        out = "\t PEM Rigid backing"
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
        out[4,0] = 1.
        out[0,1] = 1.
        out[3,2] = 1.
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
        out = "\t Elastic Rigid backing"
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
