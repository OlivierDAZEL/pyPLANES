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
from pyPLANES.pw.general.general_layers import PwLayer_3D
from pyPLANES.pw.periodic_layer import PeriodicLayer
from pyPLANES.pw.general.general_layers import PwLayer_3D, FluidLayer_3D
from pyPLANES.pw.general.characteristics_3D import Characteristics_3D
from pyPLANES.pw.general.polarisation_3D import fluid_waves_3D, PEM_waves_3D, elastic_waves_3D
from scipy.linalg import block_diag

class Interface_3D():
    """
    Interface for Plane Wave Solver
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        self.layers = [layer1, layer2]
        self.n_b, self.n_t = None, None
        self.number_relations = None
        self.pw_method = None
        self.C_b, self.C_t = None, None
        self.C_b, self.C_tc = None, None
        if isinstance(self.layers[0],PwLayer_3D):
            self.carac_b = Characteristics_3D(self.layers[0].medium)
        elif isinstance(self.layers[0],PeriodicLayer):
            self.carac_b = Characteristics_3D(self.layers[0].medium[1])
            self.carac_b.typ = self.layers[0].pwfem_entities[1].typ
        if layer2 != None:
            if isinstance(self.layers[1],PwLayer_3D):
                self.carac_t = Characteristics_3D(self.layers[1].medium)
            elif isinstance(self.layers[1],PeriodicLayer):
                self.carac_t = Characteristics_3D(self.layers[1].medium[0])
                self.carac_t.typ = self.layers[1].pwfem_entities[1].typ
        else:
            self.carac_t = None


    def update_frequency(self, omega, kx=0, kz=0):
        if isinstance(self.layers[0],PwLayer_3D):
            self.layers[0].medium.update_frequency(omega)
        if isinstance(self.layers[1],PwLayer_3D):
            self.layers[1].medium.update_frequency(omega)
        self.carac_b.update_frequency(omega)
        if self.carac_t is not None:
            self.carac_t.update_frequency(omega)

    def update_M_global(self, M, i_eq):
        periodic_layer = [isinstance(l, PeriodicLayer) for l in self.layers]
        if any(periodic_layer):
            raise NameError("Periodic layers are not implemented for General Plane Wave Solver")
            if all(periodic_layer):
                index_rel = slice(i_eq, i_eq+self.number_relations*self.nb_waves)
                M [index_rel, self.layers[0].dofs_top] = np.kron(np.eye(self.nb_waves), self.C_bottom)
                M [index_rel, self.layers[1].dofs_bottom] = np.kron(np.eye(self.nb_waves), self.C_top)
                i_eq += self.number_relations*self.nb_waves
            elif periodic_layer[0]: # Layer 0 is periodic
                SV_1 = self.layers[1].SV
                d_1 = ([0]*self.n_1+[-self.layers[1].d]*self.n_1)*self.nb_waves
                delta_1 = np.diag(np.exp(self.layers[1].lam*d_1))
                index_rel = slice(i_eq, i_eq+self.number_relations*self.nb_waves)
                M [index_rel, self.layers[0].dofs_top] = np.kron(np.eye(self.nb_waves), self.C_bottom)
                M [index_rel, self.layers[1].dofs] = np.kron(np.eye(self.nb_waves), self.C_top)@(SV_1@delta_1)
                i_eq += self.number_relations*self.nb_waves
            else: # Layer 1 is periodic
                SV_0 = self.layers[0].SV
                d_0 = ([self.layers[0].d]*self.n_0+[0]*self.n_0)*self.nb_waves
                delta_0 = np.diag(np.exp(self.layers[0].lam*d_0))
                index_rel = slice(i_eq, i_eq+self.number_relations*self.nb_waves)
                M [index_rel, self.layers[0].dofs] = np.kron(np.eye(self.nb_waves), self.C_bottom)@(SV_0@delta_0)
                M [index_rel, self.layers[1].dofs_bottom] = np.kron(np.eye(self.nb_waves), self.C_top)
                i_eq += self.number_relations*self.nb_waves
        else:
            # Only homogeneous layers
            SV_b = self.layers[0].SV
            SV_t = self.layers[1].SV

            d_b = ([self.layers[0].d]*self.n_b+[0]*self.n_b)
            d_t = ([0]*self.n_t +[-self.layers[1].d]*self.n_t)
            delta_b = np.diag(np.exp(self.layers[0].lam*d_b))
            delta_t = np.diag(np.exp(self.layers[1].lam*d_t))
            index_rel = slice(i_eq, i_eq+self.number_relations)
            M [index_rel, self.layers[0].dofs] = self.C_b@(SV_b@delta_b)
            M [index_rel, self.layers[1].dofs] = self.C_t@(SV_t@delta_t)
            i_eq += self.number_relations
        return i_eq

class FluidFluidInterface_3D(Interface_3D):
    """
    Fluid-fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = self.n_t = 1
        self.number_relations = 2
        self.C_b = np.eye(self.number_relations)
        self.C_t = -np.eye(self.number_relations)
        self.C_bc, self.C_tc = self.C_b, self.C_t
        self.pw_method = fluid_waves_3D

    def __str__(self):
        out = "\t Fluid-fluid interface"
        return out

class FluidPemInterface_3D(Interface_3D):
    """
    Fluid-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 1
        self.n_t = 4
        self.number_relations = 5
        self.pw_method = fluid_waves_3D
        # 0: u_y-u_y^t 1: p-p=0 2: hat{sigma}_{yy}=0 3: hat{sigma}_{yz}=0, 4: hat{sigma}_{yz}=0
        self.C_b= np.zeros((self.number_relations, 2*self.n_b))
        self.C_t= np.zeros((self.number_relations, 2*self.n_t))
        self.C_b[0,0], self.C_t[0,3] = 1,-1 # u_y-u_y^t 
        self.C_b[1,1], self.C_t[1,7] = 1,-1 # p-p
        self.C_t[2,4] = 1 # hat{sigma}_{yy}
        self.C_t[3,5] = 1 # hat{sigma}_{yz}
        self.C_t[4,6] = 1 # hat{sigma}_{xy}
        
        self.C_bc, self.C_tc = self.C_b, self.C_t
        if isinstance(self.layers[1], PeriodicLayer):
            if self.layers[1].pwfem_entities[0].typ == "Biot01":
                pass
            raise NameError("Periodic layers are not implemented for General Plane Wave Solver")



    def __str__(self):
        out = "\t Fluid-PEM interface"
        return out

class PemFluidInterface_3D(Interface_3D):
    """
    PEM-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 4
        self.n_t = 1
        self.number_relations = 5
        self.pw_method = PEM_waves_3D
        if isinstance(self.layers[0], PeriodicLayer):
            raise NameError("Periodic layers are not implemented for General Plane Wave Solver")
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                typ = "Biot01"
            else:
                typ = "Biot98"
        else:
            typ = "Biot98"

        self.C_b= np.zeros((self.number_relations, 2*self.n_b))
        self.C_t= np.zeros((self.number_relations, 2*self.n_t))
        
        self.C_t[0,0], self.C_b[0,3] = 1,-1 # u_y-u_y^t 
        self.C_t[1,1], self.C_b[1,7] = 1,-1 # u_y-u_y^t
        self.C_b[2,4] = 1 # hat{sigma}_{yy}
        self.C_b[3,5] = 1 # hat{sigma}_{yz},
        self.C_b[4,6] = 1 # hat{sigma}_{xy}

        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                pass
        self.C_bc, self.C_tc = self.C_b, self.C_t

    def __str__(self):
        out = "\t PEM-Fluid interface"
        return out

class FluidElasticInterface_3D(Interface_3D):
    """
    Fluid-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 1
        self.n_t = 3
        self.number_relations = 4
        self.C_b= np.zeros((self.number_relations, 2*self.n_b))
        self.C_t= np.zeros((self.number_relations, 2*self.n_t))
        
        self.C_t[0,2], self.C_b[0,0] = 1,-1 # u_y^s-u_y 
        self.C_t[1,3], self.C_b[1,1] = 1, 1 # sigma_{yy}+p
        self.C_t[2,4] = 1 # hat{sigma}_{yy}
        self.C_t[3,5] = 1 # hat{sigma}_{yz},
         
        
        self.C_bc, self.C_tc = self.C_b, self.C_t
        self.pw_method = fluid_waves_3D

    def __str__(self):
        out = "\t Fluid-Elastic interface"
        return out

class ElasticFluidInterface_3D(Interface_3D):
    """
    Elastic-Fluid interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 3
        self.n_t = 1
        self.number_relations = 4
 
        self.C_b= np.zeros((self.number_relations, 2*self.n_b))
        self.C_t= np.zeros((self.number_relations, 2*self.n_t)) 
 
        self.C_b[0,2], self.C_t[0,0] = 1,-1 # u_y-u_y 
        self.C_b[1,3], self.C_t[1,1] = 1, 1 # sigma_{yy}+p
        self.C_b[2,4] = 1 # hat{sigma}_{yz}
        self.C_b[3,5] = 1 # hat{sigma}_{xy},


        self.C_bc, self.C_tc = self.C_b, self.C_t

        self.pw_method = elastic_waves_3D

    def __str__(self):
        out = "\t Elastic-Fluid interface"
        return out

class ElasticElasticInterface_3D(Interface_3D):
    """
    Elastic-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = self.n_t = 3
        self.number_relations = 6
        self.C_b = np.eye(self.number_relations)
        self.C_t = -np.eye(self.number_relations)
        self.C_bc, self.C_tc = self.C_b, self.C_t

        self.pw_method = elastic_waves_3D

    def __str__(self):
        out = "\t Elastic-Elastic interface"
        return out

class PemPemInterface_3D(Interface_3D):
    """
    PEM-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 4
        self.n_t = 4
        self.number_relations = 8
        self.C_b = np.eye(self.number_relations)
        self.C_t = -np.eye(self.number_relations)
        self.C_bc, self.C_tc = self.C_b, self.C_t

        self.pw_method = PEM_waves_3D

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

class ElasticPemInterface_3D(Interface_3D):
    """
    Elastic-PEM interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 3
        self.n_t = 4
        self.number_relations = 7

        self.C_b = np.zeros((self.number_relations, 2*self.n_b))
        self.C_t = np.zeros((self.number_relations, 2*self.n_t))
        self.C_t[0,0], self.C_b[0,0] = 1,-1 # u_x^s-u_x
        self.C_t[1,1], self.C_b[1,1] = 1,-1 # u_z^s-u_z
        self.C_t[2,2], self.C_b[2,2] = 1,-1 # u_y^s-u_y
        self.C_t[3,3], self.C_b[3,2] = 1,-1 # u_y^t-u_y
        self.C_t[4,4], self.C_t[4,7], self.C_b[4,3] = 1,-1, -1 # hat{sigma}_{yy}-p-sigma_{yy}   
        self.C_t[5,5], self.C_b[5,4] = 1,-1 # hat{sigma}_{yz}-sigma_{yz}
        self.C_t[6,6], self.C_b[6,5] = 1,-1 # hat{sigma}_{yz}-sigma_{yz}
        self.C_bc, self.C_tc = self.C_b, self.C_t

        self.pw_method = elastic_waves_3D

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

class PemElasticInterface_3D(Interface_3D):
    """
    PEM-Elastic interface 
    """
    def __init__(self, layer1=None, layer2=None):
        super().__init__(layer1,layer2)
        self.n_b = 4
        self.n_t = 3
        self.number_relations = 7
        self.pw_method = PEM_waves_3D
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                typ = "Biot01"
        else:
            typ = "Biot98"



        self.C_b = np.zeros((self.number_relations, 2*self.n_b))
        self.C_t = np.zeros((self.number_relations, 2*self.n_t))
        self.C_b[0,0], self.C_t[0,0] = 1,-1 # u_x^s-u_x
        self.C_b[1,1], self.C_t[1,1] = 1,-1 # u_z^s-u_z
        self.C_b[2,2], self.C_t[2,2] = 1,-1 # u_y^s-u_y
        self.C_b[3,3], self.C_t[3,2] = 1,-1 # u_y^t-u_y
        self.C_b[4,4], self.C_b[4,7], self.C_t[4,3] = 1,-1, -1 # hat{sigma}_{yy}-p-sigma_{yy}   
        self.C_b[5,5], self.C_t[5,4] = 1,-1 # hat{sigma}_{yz}-sigma_{yz}
        self.C_b[6,6], self.C_t[6,5] = 1,-1 # hat{sigma}_{yz}-sigma_{yz}
        self.C_bc, self.C_tc = self.C_b, self.C_t

       # Case of 2001 formulation 0 =w
        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_top[2, 1] = 0.



        if isinstance(self.layers[0], PeriodicLayer):
            if self.layers[0].pwfem_entities[0].typ == "Biot01":
                self.C_bottom[3, 4] = 0.
         
        
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

class RigidBacking_3D(Interface_3D):
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        super().__init__(layer1,layer2, method)
        self.method = method
        self.C = None
        self.number_relations = None

    def update_M_global(self, M, i_eq):
        if isinstance(self.layers[0], PeriodicLayer):
            index_rel = slice(i_eq, i_eq+self.number_relations*self.nb_waves)
            M [index_rel, self.layers[0].dofs_top] = np.kron(np.eye(self.nb_waves), self.C)
            i_eq += self.number_relations*self.nb_waves
        else:
            d_0 = ([self.layers[0].d]*self.number_relations+[0]*self.number_relations)
            delta_0 = np.diag(np.exp(self.layers[0].lam*d_0))
            lines = slice(i_eq, i_eq+self.number_relations)
            M[lines, self.layers[0].dofs] = self.C@(self.layers[0].SV@delta_0)        
            i_eq += self.number_relations
        return i_eq

    def Omega(self, nb_bloch_waves=0):
        pass

    def Omegac(self, nb_bloch_waves=0):
        pass 
    
class FluidRigidBacking_3D(RigidBacking_3D):
    """
    Rigid backing for a fluid layer
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        super().__init__(layer1,layer2, method)
        self.C = np.array([[1,0]]).reshape(1,2)
        self.number_relations = 1

    def __str__(self):
        out = "\t Rigid backing"
        return out

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

class PemBacking_3D(RigidBacking_3D):
    """
    Rigid backing for a pem layer
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        super().__init__(layer1,layer2, method)
        self.method = method
        
        self.C = np.zeros((4,8))
        self.C[0, 0] = 1.
        self.C[1, 1] = 1.
        self.C[2, 2] = 1.
        self.C[3, 3] = 1.
        self.number_relations = 4
        
    def __str__(self):
        out = "\t PEM Rigid backing"
        return out

    def Omegac(self, nb_bloch_waves=1):
        
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

class ElasticBacking_3D(RigidBacking_3D):

    """
    Rigid backing for an elastic layer
    """
    def __init__(self, layer1=None, layer2=None, method="characteristics"):
        super().__init__(layer1,layer2, method)
        self.method = method

        self.C = np.zeros((3,6))
        self.C[0, 0] = 1.
        self.C[1, 1] = 1.
        self.C[2, 2] = 1.
        self.number_relations = 3
        
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

class SemiInfinite_3D(Interface_3D):
    """
    Semi-infinite boundary
    """
    def __init__(self, layer1=None):
        transmission_layer = FluidLayer_3D(Fluid(c=Air().c,rho=Air().rho), 1.e-2, x_0=-1.e-2)
        self.medium = load_material("Air")
        Interface_3D.__init__(self, layer1, transmission_layer)
        self.SV = None
        self.dofs = None
        # Determine the type of the last layer
        self.determine_type()

    def determine_type(self):
        self.typ =None
        if isinstance(self.layers[0], PwLayer_3D):
            t = self.layers[0].medium.MEDIUM_TYPE
        elif isinstance(self.layers[0], PeriodicLayer):
            t = self.layers[0].medium[1].MEDIUM_TYPE

        if t in ["fluid", "eqf"]:
            self.typ = "fluid"
            self.n_b = self.n_t = 1
            self.number_relations = 2
            self.C_b = np.eye(self.number_relations)
            self.C_t= -np.eye(self.number_relations)
            self.C_bc, self.C_tc = self.C_b, self.C_t
            # self.pw_method = fluid_waves_TMM
        elif t in ["pem"]:
            self.typ = "pem"
            formulation = "Biot98"
            self.n_b, self.n_t = 4, 1
            self.pw_method = PEM_waves_3D
            self.number_relations = 5
            
            self.C_b= np.zeros((self.number_relations, 2*self.n_b))
            self.C_t= np.zeros((self.number_relations, 2*self.n_t))
        
            self.C_t[0,0], self.C_b[0,3] = 1,-1 # u_y-u_y^t 
            self.C_t[1,1], self.C_b[1,7] = 1,-1 # u_y-u_y^t
            self.C_b[2,4] = 1 # hat{sigma}_{yy}
            self.C_b[3,5] = 1 # hat{sigma}_{yz},
            self.C_b[4,6] = 1 # hat{sigma}_{xy}
            
            self.C_bc, self.C_tc = self.C_b, self.C_t
            if isinstance(self.layers[0], PeriodicLayer):
                if self.layers[0].pwfem_entities[0].typ == "Biot01":
                    typ = "Biot01"
                    self.C_bottom = np.array([[0, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
                    self.C_top = np.array([[1,0],[0,1], [0,1], [0, 0]])
                    self.C_bottomc = np.array([[0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
                    self.C_topc = np.array([[1,0],[0,1], [0,1], [0, 0]])
                else:
                    typ = "Biot98"
            else:
                typ = "Biot98"

        elif t in ["elastic"]:
            self.typ ="elastic"
            self.n_b, self.n_t = 3, 1
            self.number_relations = 4
            # \sigma_xy = 0, u_y = u_y^s, \sima_yy = -p
            self.C_b= np.zeros((self.number_relations, 2*self.n_b))
            self.C_t= np.zeros((self.number_relations, 2*self.n_t)) 
    
            self.C_b[0,2], self.C_t[0,0] = 1,-1 # u_y-u_y 
            self.C_b[1,3], self.C_t[1,1] = 1, 1 # sigma_{yy}+p
            self.C_b[2,4] = 1 # hat{sigma}_{yy}
            self.C_b[3,5] = 1 # hat{sigma}_{yz},
            self.C_bc, self.C_tc = self.C_b, self.C_t
            

        else:
            raise NameError("Invalid type")

    def __str__(self):
        out = "\t Semi-infinite transmission medium\n\t\t"
        out += f"{self.layers}"
        return out

    def update_frequency(self, omega, kx, kz):
        Interface_3D.update_frequency(self, omega, kx)
        self.medium.update_frequency(omega)
        self.SV, self.lam = fluid_waves_3D(self.medium, kx, kz)
        
        self.k = self.medium.k
        self.kx = kx
        self.kz = kz
        self.k_air = omega/Air.c
        k_y = np.sqrt(self.k_air**2-self.kx**2-self.kz**2+0*1j)
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
        SV_t = self.SV[:,::2] # Just the outgoing waves
        index_rel = slice(i_eq, i_eq+self.number_relations)
        if isinstance(self.layers[0], PeriodicLayer):
            M [index_rel, self.layers[0].dofs_top] = np.kron(np.eye(self.nb_waves), self.C_bottom)
            M [index_rel, self.dofs] = np.kron(np.eye(self.nb_waves), self.C_top)@(SV_1)
        else:
            SV_b = self.layers[0].SV
            d_b = ([self.layers[0].d]*self.n_b+[0]*self.n_b)
            delta_b = np.diag(np.exp(self.layers[0].lam*d_b))     
            index_rel = slice(i_eq, i_eq+self.number_relations)
            M [index_rel, self.layers[0].dofs] = self.C_b@(SV_b@delta_b)
            M [index_rel, self.dofs] = self.C_t@(SV_t)
        i_eq += self.number_relations
        return i_eq
