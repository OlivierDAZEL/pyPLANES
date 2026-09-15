#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
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

import platform
import time, timeit
import os

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, linalg as sla
from scipy.sparse.linalg import spsolve


from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *

from pyPLANES.pw.periodic_layer_base import PeriodicLayerBase
from pyPLANES.gmsh.mesh import GmshMesh
import numpy.linalg as LA


class PeriodicLayer_HMM(PeriodicLayerBase, GmshMesh):
    def __init__(self, **kwargs):
        self.condensation = kwargs.get("condensation", True)
        GmshMesh.__init__(self, **kwargs)
        PeriodicLayerBase.__init__(self, **kwargs)
        # if self.pwfem_entities[0].pi == self.pwfem_entities[1].pi:
        #     self.pi = self.pwfem_entities[0].pi
        # else:
        #     raise ValueError("The pi of the bottom and top interfaces must be the same")
        # if self.pwfem_entities[0].delta == self.pwfem_entities[1].delta:
        #     self.delta = self.pwfem_entities[0].delta
        # else:
        #     raise ValueError("The delta of the bottom and top interfaces must be the same")
        if self.pwfem_entities[0].nb_dof_per_node == self.pwfem_entities[1].nb_dof_per_node:
            self.nb_dof_per_node = self.pwfem_entities[0].nb_dof_per_node
        else:
            raise ValueError("The nb_dof_per_node of the bottom and top interfaces must be the same")

        
        
    def create_HMM_matrices(self):
        self.create_bulk_matrices()
        self.linear_system_2_numpy()

        D_XX = csr_matrix((self.A_v, (self.A_i, self.A_j)), shape=(self.n_dof+1, self.n_dof+1))[1:,1:]
        D_XX = self.P_periodicity_H@D_XX@self.P_periodicity
        self.cond = np.linalg.cond(D_XX.toarray())
        self.A_i, self.A_j, self.A_v = [], [], []
        DD_iX = [] # Initialisation of the list of the R will be [D_bX D_tX]
        DD_Xi = [] # Initialisation of the list of the R will be [D_Xb D_Xt]
        
        nb_w_b = self.pwfem_entities[0].nb_dof_per_node*self.nb_waves
        nb_w_t = self.pwfem_entities[1].nb_dof_per_node*self.nb_waves
        if nb_w_b != nb_w_t:
            raise ValueError("The number of waves at the bottom and top interfaces must be the same")
        else:
            nb_w = nb_w_b
        
        for _ent in self.pwfem_entities:
            dof_FEM, row_D_tX, column_D_Xt, D_val = [], [], [], []
            for _w, kx in enumerate(self.kx):
                for _elem in _ent.elements:
                    M_elem = imposed_pw_elementary_vector(_elem, kx)
                    if _ent.typ == "fluid":
                        # Columns of D_xi
                        dof_p, orient_p, __ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        row_D_tX.extend(len(dof_p)*[_w])
                        # Rows of D_ix
                        column_D_Xt.extend(len(dof_p)*[2*_ent.nb_dof_per_node*_w])
                        # Values for D_ix and D_xi (will be conjugated below)                      
                        D_val.extend(list(orient_p@M_elem))
                    elif _ent.typ in ["Biot98", "Biot01"]:
                        # u_x
                        dof_ux, orient_ux = dof_ux_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_ux])
                        column_D_Xt.extend(len(dof_ux)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        row_D_tX.extend(len(dof_ux)*[0+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_ux@M_elem))
                        # u_y
                        dof_uy, orient_uy = dof_uy_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_uy])
                        column_D_Xt.extend(len(dof_uy)*[_ent.dual[1]+2*_ent.nb_dof_per_node*_w])
                        row_D_tX.extend(len(dof_uy)*[1+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_uy@M_elem))
                        #  p 
                        dof_p, orient_p, _ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        column_D_Xt.extend(len(dof_p)*[_ent.dual[2]+2*_ent.nb_dof_per_node*_w])
                        row_D_tX.extend(len(dof_p)*[2+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_p@M_elem))
                    elif _ent.typ == "elastic":
                        # u_x                        
                        dof_ux, orient_ux = dof_ux_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_ux])
                        column_D_Xt.extend(len(dof_ux)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        row_D_tX.extend(len(dof_ux)*[0+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_ux@M_elem))
                        # D_xx[0+_ent.nb_dof_per_node*_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        # u_y
                        dof_uy, orient_uy = dof_uy_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_uy])
                        column_D_Xt.extend(len(dof_uy)*[_ent.dual[1]+2*_ent.nb_dof_per_node*_w])
                        row_D_tX.extend(len(dof_uy)*[1+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_uy@M_elem))
                        # D_xx[1+ent.nb_dof_per_node*_w, _ent.primal[1]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    else:
                        raise NameError("_ent.typ has no valid type")

            DD_iX.append(coo_matrix((np.conj(D_val), (row_D_tX, dof_FEM)), shape=(_ent.nb_dof_per_node*self.nb_waves, self.n_dof))@self.P_periodicity)
            # Creation of the D_Xi, minus sign <- transposition +normal 
            DD_Xi.append(coo_matrix((-_ent.ny*np.array(D_val), (dof_FEM, column_D_Xt)), shape=(self.n_dof, 2*_ent.nb_dof_per_node*self.nb_waves)))
        
        D_iX = np.hstack([self.P_periodicity_H@D_i.todense() for D_i in DD_Xi])

        RR = -spsolve(D_XX, D_iX).reshape((self.n_dof-len(self.dof_left), 2*2*_ent.nb_dof_per_node*self.nb_waves))

        primal = [x + 2*i*_ent.nb_dof_per_node for i in range(self.nb_waves) for x in _ent.primal]
        dual = [x + 2*i*_ent.nb_dof_per_node for i in range(self.nb_waves) for x in _ent.dual]

        R_b = RR[:,:2*nb_w]
        R_t = RR[:,2*nb_w:]
  
        self.R_b = R_b[:,dual]
        self.R_t = R_t[:,dual]
        
        self.D_bb = -self.period*np.eye(nb_w)
        self.D_tt = -self.period*np.eye(nb_w)     
        self.D_bX = DD_iX[0]
        self.D_tX = DD_iX[1]
        
    def HMM_update(self, H):
        # print(H)
        self.create_HMM_matrices()
        nb_physical_waves = self.Omega_c.shape[1]
        Omega_p = np.kron(np.eye(self.nb_waves),self.Omega_p)
        Omega_c = np.kron(np.eye(self.nb_waves),self.Omega_c)
        # print(f"Omega_p=\n{Omega_p}")
        # print(f"Omega_c=\n{Omega_c}")
        P_cal = np.kron(np.eye(self.nb_waves), self.P_cal)
        # print(f"P_cal=\n{P_cal}")
        C_cal = np.kron(np.eye(self.nb_waves), self.C_cal)
        # print(f"C_cal=\n{C_cal}")
        NMEfromPQ = np.kron(np.eye(self.nb_waves), self.NMEfromPQ)
        # print(f"NMEfromPQ=\n{NMEfromPQ}")
        PQfromNME = np.kron(np.eye(self.nb_waves), self.PQfromNME)
        # print(f"PQfromNME=\n{PQfromNME}")
        # pi    = [x + 2*i*self.nb_dof_per_node for i in range(self.nb_waves) for x in self.pi]
        # delta = [x + 2*i*self.nb_dof_per_node for i in range(self.nb_waves) for x in self.delta]

        mat_u     = np.kron(np.eye(self.nb_waves), np.diag([1]*nb_physical_waves+[0]*nb_physical_waves))
        mat_sigma = np.kron(np.eye(self.nb_waves), np.diag([0]*nb_physical_waves+[1]*nb_physical_waves))
        u2v = ((1/(1j*self.omega))*mat_u+mat_sigma)
        # print(f"u2v=\n{u2v}")
        v2u = (1j*self.omega)*mat_u+mat_sigma
        P = permutation_bloch_waves_amplitudes(self.nb_dof_per_node, self.nb_waves)
        # print(f"P=\n{P}")
        
        Omega = Omega_p+Omega_c@H # in HMM formalism
        Omega= u2v@Omega # replace velocities by displacements
        Omega = P@NMEfromPQ@Omega # Go from PQ variables to HMM_periodic variables

        Omega_hat_u = np.hstack([  self.D_tt, self.D_tX@self.R_t])@Omega
        Omega_hat_b = np.hstack([0*self.D_bb, self.D_bX@self.R_t])@Omega
        
   
        U, Sigma, Vh = LA.svd(self.D_tX@self.R_b)
        Uh, V = U.conj().T, Vh.conj().T
        Q = -LA.inv(Uh@Omega_hat_u)@np.diag(Sigma)
        Omega = np.vstack([-(-1/self.period)*(self.D_bX@self.R_b@V+Omega_hat_b@Q), V])
        Omega = PQfromNME@P.T@Omega # Go from IJNME paper variables to HMM variables 
        Omega= v2u@Omega # replace displacements by velocities
        HH = C_cal@Omega@LA.inv(P_cal@Omega)
        self.L_cal = Q@LA.inv(P_cal@Omega)

        
        # print(self.kx)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.abs(HH))
        # plt.colorbar(label="Module")
        # plt.show()
        return HH


def permutation_bloch_waves_amplitudes(nb_type_of_wave, nb_Bw):
    if nb_type_of_wave == 1:
        P = [2*i for i in range(nb_Bw)]
        P += [2*i+1 for i in range(nb_Bw)]
        return np.eye(2*nb_Bw)[P]
    elif nb_type_of_wave == 2:
        P = [2*i for i in range(nb_Bw)]
        P += [2*i+1 for i in range(nb_Bw)]
        P += [2*i+2*nb_Bw for i in range(nb_Bw)]
        P += [2*i+1+2*nb_Bw for i in range(nb_Bw)]
        return np.eye(4*nb_Bw)[P]
    else:
        raise (f"Unknown number of waves {nb_type_of_wave}")
    
