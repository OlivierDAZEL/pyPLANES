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
from itertools import chain 
import numpy as np
import numpy.linalg as LA
from numpy import sqrt
from mediapack import Air, Fluid
from pyPLANES.utils.io import load_material
from pyPLANES.pw.pw_layers import PwLayer
from pyPLANES.pw.periodic_layer import PeriodicLayer
from pyPLANES.pw.pw_layers import PwLayer, FluidLayer
from pyPLANES.pw.characteristics import Characteristics
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
        self.C_bottom, self.C_top = None, None
        self.C_bottomc, self.C_topc = None, None
        if isinstance(self.layers[0],PwLayer):
            self.carac_bottom = Characteristics(self.layers[0].medium)
        elif isinstance(self.layers[0],PeriodicLayer):
            self.carac_bottom = Characteristics(self.layers[0].medium[1])
            self.carac_bottom.typ = self.layers[0].pwfem_entities[1].typ
        if layer2 != None:
            if isinstance(self.layers[1],PwLayer):
                self.carac_top = Characteristics(self.layers[1].medium)
            elif isinstance(self.layers[1],PeriodicLayer):
                self.carac_top = Characteristics(self.layers[1].medium[0])
                self.carac_top.typ = self.layers[1].pwfem_entities[1].typ
        else:
            self.carac_top = None

    def update_frequency(self, omega, kx=[0]):
        self.nb_waves = len(kx)
        if isinstance(self.layers[0],PwLayer):
            self.layers[0].medium.update_frequency(omega)
        if isinstance(self.layers[1],PwLayer):
            self.layers[1].medium.update_frequency(omega)
        self.carac_bottom.update_frequency(omega)
        if self.carac_top is not None:
            self.carac_top.update_frequency(omega)


    def update_M_global(self, M, i_eq):

        SV_0 = self.layers[0].SV
        SV_1 = self.layers[1].SV

        d_0 = [self.layers[0].d]*self.n_0+[0]*self.n_0
        d_1 = [0]*self.n_1 +[-self.layers[1].d]*self.n_1
        delta_0 = np.diag(np.exp(self.layers[0].lam*d_0))
        delta_1 = np.diag(np.exp(self.layers[1].lam*d_1))

        index_rel = slice(i_eq, i_eq+self.number_relations)
        M [index_rel, self.layers[0].dofs] = self.C_bottom@(SV_0@delta_0)
        M [index_rel, self.layers[1].dofs] = self.C_top@(SV_1@delta_1)

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

            M1 = self.C_top@Om
            M2 = self.C_bottom@P_in
            M3 = self.C_bottom@P_out


            M = -LA.inv(np.hstack((M1,M2)))@M3

            M_X = M[:self.n_1,:]
            M_S = M[self.n_1:,:]
            Omega = P_in@M_S +P_out 
            
        else: 
            SV = self.pw_method(mat, np.zeros(1))[0] 
            P_in = SV[:,:self.n_0].reshape((2*self.n_0, self.n_0))
            P_out = SV[:,self.n_0:].reshape((2*self.n_0, self.n_0))

            M1 = np.kron(np.eye(self.nb_waves), self.C_top)@Om
            M2 = np.kron(np.eye(self.nb_waves), self.C_bottom@P_in)
            M3 = np.kron(np.eye(self.nb_waves), self.C_bottom@P_out)

            M = -LA.inv(np.hstack((M1,M2)))@M3

            M_X = M[:self.n_1*self.nb_waves,:]
            M_S = M[self.n_1*self.nb_waves:,:]
            Omega = np.kron(np.eye(self.nb_waves), P_in)@M_S +np.kron(np.eye(self.nb_waves), P_out) 
        return Omega, M_X

    def update_Omegac(self, Om):
            
        M1 = np.kron(np.eye(self.nb_waves), self.C_topc@self.carac_top.P)@Om
        M2 = np.kron(np.eye(self.nb_waves), self.C_bottomc@self.carac_bottom.P_minus)
        M3 = np.kron(np.eye(self.nb_waves), self.C_bottomc@self.carac_bottom.P_plus)

        M = -LA.inv(np.hstack((M1,M2)))@M3
        M_X = M[:self.n_1*self.nb_waves,:]
        M_qminus = M[self.n_1*self.nb_waves:,:]
        
        Omega = np.zeros((2*self.n_0*self.nb_waves,self.n_0*self.nb_waves), dtype=complex)
        
        index_plus = list(chain.from_iterable([ list(range(2*self.n_0*d,2*self.n_0*d+self.n_0)) for d in range(self.nb_waves)]))
        index_minus = list(chain.from_iterable([ list(range(2*self.n_0*d+self.n_0,2*self.n_0*d+2*self.n_0)) for d in range(self.nb_waves)]))
        Omega[index_plus,:] = np.eye(self.n_0*self.nb_waves)
        Omega[index_minus,:] = M_qminus

        return Omega, M_X

class FluidFluidInterface(PwInterface):
    """
    Fluid-fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_0 = self.n_1 = 1
        self.number_relations = 2
        self.C_bottom = np.eye(self.number_relations)
        self.C_top = -np.eye(self.number_relations)
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top
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
        # 0: u_y-u_y^t 1: p-p=0 2: hat{sigma}_{xy}=0 3 hat{sigma}_{xy}=0
        # 0: u_y-u_y^t 1: p-p=0 2: hat{sigma}_{xy}=0 3 hat{sigma}_{xy}=0
        self.C_bottom = np.array([[1,0],[0,1], [0,0], [0, 0]])
        self.C_top = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                # 0: u_y-u_y^t 1: p-p=0 2: p+{sigma}^t_{yy}=0 3 hat{sigma}_{xy}=0
                self.C_bottom = np.array([[1,0],[0,1], [0,1], [0, 0]])
                self.C_top = np.array([[0, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
                # 0: u_y-u_y^t 1: p-p=0 2: p+{sigma}^t_{yy}=0 3 hat{sigma}_{xy}=0
                self.C_bottomc = np.array([[1,0],[0,1], [0,1], [0, 0]])
                self.C_topc = np.array([[0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
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
        self.C_bottom = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        self.C_top = np.array([[1,0],[0,1], [0,0], [0, 0]])
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_bottom = np.array([[0, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
                self.C_top = np.array([[1,0],[0,1], [0,1], [0, 0]])
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top

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
        self.C_bottom = np.array([[0,0],[1,0],[0,1]])
        self.C_top = np.array([[1, 0, 0, 0], [0, -1., 0, 0 ],[0, 0, 1, 0]])
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top

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
        self.C_bottom = np.array([[1, 0, 0, 0], [0, -1., 0, 0 ],[0, 0, 1, 0]])
        self.C_top = np.array([[0,0],[1,0],[0,1]])
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top

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
        self.C_bottom = np.eye(self.number_relations)
        self.C_top = -np.eye(self.number_relations)
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top

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
        self.C_bottom = np.eye(self.number_relations)
        self.C_top = -np.eye(self.number_relations)
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top

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
        self.C_bottom = np.zeros((self.number_relations, 2*self.n_0))
        self.C_top = np.zeros((self.number_relations, 2*self.n_1))
        # \sigma_xz 
        self.C_bottom[0, 0], self.C_top[0, 0] = 1., -1. 
        # u_z =u_z_s  
        self.C_bottom[1, 1], self.C_top[1, 1] = 1., -1.
        # u_z =u_z_t 
        self.C_bottom[2, 1], self.C_top[2, 2] = 1., -1.
        # Case of 2001 formulation 0 =w
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                self.C_bottom[2, 1] = 0.
        # sigma_zz = \hat{\sigma_zz} - p
        self.C_bottom[3, 2], self.C_top[3, 3], self.C_top[3, 4] = 1., -1., 1. 

        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                self.C_top[3, 4] = 0.
        # u_x = u_x_s 
        self.C_bottom[4, 3], self.C_top[4, 5] = 1., -1.
        
        self.C_bottomc = self.C_bottom
        self.C_topc = self.C_top
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                M_01 = np.zeros((6,6))
                M_01[0,0]=1
                M_01[1,3]=1
                M_01[2,2]=1
                M_01[3,5]=1
                M_01[4,1]=1
                M_01[5,4]=1
                    # ''' S={0: hat{sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:hat{sigma}_{yy}, 4:p, 5:u_x^s}'''
                self.C_topc =  self.C_top@LA.inv(M_01)

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

        self.C_bottom = np.zeros((self.number_relations, 2*self.n_0))
        self.C_top = np.zeros((self.number_relations, 2*self.n_1))
        # \sigma_xz 
        self.C_top[0, 0], self.C_bottom[0, 0] = 1., -1. 
        # u_z =u_z_s  
        self.C_top[1, 1], self.C_bottom[1, 1] = 1., -1.
        # u_z =u_z_t 
        self.C_top[2, 1], self.C_bottom[2, 2] = 1., -1.
        # Case of 2001 formulation 0 =w
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_top[2, 1] = 0.
        # sigma_zz = \hat{\sigma_zz} - p
        self.C_top[3, 2], self.C_bottom[3, 3], self.C_bottom[3, 4] = 1., -1., 1. 

        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_bottom[3, 4] = 0.
        # u_x = u_x_s 
        self.C_top[4, 3], self.C_bottom[4, 5] = 1., -1.
        
        self.C_bottomc, self.C_topc = self.C_bottom, self.C_top
        
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                M_01 = np.zeros((6,6))
                M_01[0,0]=1
                M_01[1,3]=1
                M_01[2,2]=1
                M_01[3,5]=1
                M_01[4,1]=1
                M_01[5,4]=1                
                self.C_bottomc =  self.C_bottom@LA.inv(M_01)

    def __str__(self):
        out = "\t PEM-Elastic interface"
        return out

class FluidRigidBacking(PwInterface):
    """
    Rigid backing for a fluid layer
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        super().__init__(layer1,layer2)
        self.method = method


    def __str__(self):
        out = "\t Rigid backing"
        return out

    def update_M_global(self, M, i_eq):
        M[i_eq, self.layers[0].dofs[0]] = self.layers[0].SV[0, 0]*np.exp(self.layers[0].lam[0]*self.layers[0].d)
        M[i_eq, self.layers[0].dofs[1]] = self.layers[0].SV[0, 1]
        i_eq += 1
        return i_eq

    def Omegac(self, nb_bloch_waves=0):
        out = np.array([1.,1.]).reshape(2,1)
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=complex)


    def Omega(self, nb_bloch_waves=0):
        out = np.array([0,1]).reshape(2,1)
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=complex)

class PemBacking(PwInterface):
    """
    Rigid backing for a pem layer
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        super().__init__(layer1,layer2)
        self.method = method
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

    def Omegac(self, nb_bloch_waves=1):
        # C = np.zeros((3,6), dtype=complex)
        # if self.carac_bottom.typ == None:
        #     C[0,1] = 1.
        #     C[1,2] = 1.
        #     C[2,5] = 1.
        # elif self.carac_bottom.typ in ["Biot01", "Biot98"]:
        #     # P={0: {sigma}_{xy}^t, 1: {sigma}_{yy}^t, 2: w_y=0 3 u_x^s=0  4:u_y^s=0, 5: p}
        #     C[0,2] = 1.
        #     C[1,3] = 1.
        #     C[2,4] = 1.
        # else: 
        #     raise NameError("invalid typ")
        
        out = np.zeros((6, 3), dtype=complex)
        out[:3,:] = np.eye(3)
        out[3:,:] = np.eye(3) #-LA.inv(C@self.carac_bottom.P_minus)@C@self.carac_bottom.P_plus

        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=complex)

    def Omega(self, nb_bloch_waves=1):
        out = np.zeros((6,3), dtype=complex)
        out[4,0] = 1.
        out[0,1] = 1.
        out[3,2] = 1.
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=complex)

class ElasticBacking(PwInterface):

    """
    Rigid backing for an elastic layer
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics" ):
        super().__init__(layer1,layer2)
        self.method = method
        
    def __str__(self):
        out = "\t Elastic Rigid backing"
        return out

    def Omega(self, nb_bloch_waves=1):
        out = np.zeros((4,2), dtype=complex)
        out[0,1] = 1.
        out[2,0] = 1.
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=complex)

    def Omegac(self, nb_bloch_waves=1):
        out = np.zeros((4, 2), dtype=complex)
        out[:2,:] = np.eye(2)
        out[2:,:] = np.eye(2) #-LA.inv(C@self.carac_bottom.P_minus)@C@self.carac_bottom.P_plus
        if nb_bloch_waves !=0:
            out = np.kron(np.eye(nb_bloch_waves), out)
        return np.array(out, dtype=complex)

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
    def __init__(self, layer1=None):
        transmission_layer = FluidLayer(Fluid(c=Air().c,rho=Air().rho), 1.e-2, x_0=-1.e-2)
        self.medium = load_material("Air")
        PwInterface.__init__(self, layer1, transmission_layer)
        self.SV = None
        # Determine the type of the last layer

        self.typ =None
        if isinstance(self.layers[0], PwLayer):
            t = self.layers[0].medium.MEDIUM_TYPE
        elif isinstance(self.layers[0], PeriodicLayer):
            t = self.layers[0].medium[1].MEDIUM_TYPE
        if t in ["fluid", "eqf"]:
            self.typ = "fluid"
            self.n_0 = self.n_1 = 1
            self.number_relations = 2
            self.C_bottom = np.eye(self.number_relations)
            self.C_top = -np.eye(self.number_relations)
            self.C_bottomc, self.C_topc = self.C_bottom, self.C_top

            self.pw_method = fluid_waves_TMM
        elif t in ["pem"]:
            self.typ = "pem"
            formulation = "Biot98"
            self.n_0 = 3
            self.n_1 = 1
            self.pw_method = PEM_waves_TMM

            self.number_relations = 4
            self.C_bottom = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
            self.C_top = np.array([[1,0],[0,1], [0,0], [0, 0]])
            self.C_bottomc, self.C_topc = self.C_bottom, self.C_top
            if isinstance(self.layers[0], PeriodicLayer):
                if self.layers[0].pwfem_entities[0].typ == "Biot01":
                    typ = "Biot01"
                    self.C_bottomc = np.array([[0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
                    self.C_topc = np.array([[1,0],[0,1], [0,1], [0, 0]])
                else:
                    typ = "Biot98"
            else:
                typ = "Biot98"

        elif t in ["elastic"]:
            self.typ ="elastic"
            self.n_0, self.n_1 = 2, 1
            self.number_relations = 3
            self.C_bottom = np.array([[1, 0, 0, 0], [0, -1., 0, 0 ],[0, 0, 1, 0]])
            self.C_top = np.array([[0,0],[1,0],[0,1]])
            self.C_bottomc, self.C_topc = self.C_bottom, self.C_top
            self.pw_method = elastic_waves_TMM
        else:
            raise NameError("Invalid type")

    def __str__(self):
        out = "\t Semi-infinite transmission medium\n\t\t"
        out += f"{self.layers}"
        return out

    def update_frequency(self, omega, kx):
        PwInterface.update_frequency(self, omega, kx)
        self.medium.update_frequency(omega)
        self.SV, self.lam = fluid_waves_TMM(self.medium, kx)
        
        
        
        
        self.k = self.medium.k
        self.kx = kx
        self.k_air = omega/Air.c
        k_y = np.sqrt(self.k_air**2-self.kx**2+0*1j)
        self.ky = np.real(k_y)-1j*np.imag(k_y) # ky is either real or imaginary // - is to impose the good sign
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

    def Omegac(self, nb_bloch_waves=1):
        
        Omega_0 = [np.array([-1j*(self.ky[_w]/self.k_air)/(self.omega*Air.Z),1]).reshape(2,1) for _w in range(self.nb_waves)]
        # Om = [np.array([self.lam[2*_w]/(self.medium.rho*self.omega**2),1]).reshape(2,1) for _w in range(nb_bloch_waves)]
        Om = block_diag(*Omega_0)
        Om = np.kron(np.eye(self.nb_waves),self.carac_top.Q)@Om

        return self.update_Omegac(Om)


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
