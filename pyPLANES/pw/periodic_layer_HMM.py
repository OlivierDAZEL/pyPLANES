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


    def create_HMM_matrices(self):
        self.create_bulk_matrices()
        self.linear_system_2_numpy()

        D_XX = csr_matrix((self.A_v, (self.A_i, self.A_j)), shape=(self.n_dof+1, self.n_dof+1))[1:,1:]
        D_XX = self.P_periodicity_H@D_XX@self.P_periodicity
        self.cond = np.linalg.cond(D_XX.toarray())
        self.A_i, self.A_j, self.A_v = [], [], []
        DD = [] # Initialisation of the list of the R will be [D_bb D_tt]
        DD_Xi = [] # Initialisation of the list of the R will be [D_bX D_tX]
        DD_iX = [] # Initialisation of the list of the R will be [D_Xb D_Xt]

        for _ent in self.pwfem_entities:
            dof_FEM, dof_S_primal, dof_S_dual, D_val = [], [], [], []
            D_xx = np.zeros((_ent.nb_dof_per_node*self.nb_waves, 2*_ent.nb_dof_per_node*self.nb_waves))
            for _w, kx in enumerate(self.kx):
                for _elem in _ent.elements:
                    M_elem = imposed_pw_elementary_vector(_elem, kx)
                    if _ent.typ == "fluid":
                        # Columns of D_xi
                        dof_p, orient_p, __ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        # Rows of D_ix
                        dof_S_dual.extend(len(dof_p)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        # Rows of D_xx and D_xi
                        dof_S_primal.extend(len(dof_p)*[_w])
                        # Values for D_ix and D_xi (will be conjugated below)                      
                        D_val.extend(list(orient_p@M_elem))
                        # Values for D_xx
                        D_xx[_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    elif _ent.typ in ["Biot98", "Biot01"]:
                        # u_x
                        dof_ux, orient_ux = dof_ux_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_ux])
                        dof_S_dual.extend(len(dof_ux)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_ux)*[0+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_ux@M_elem))
                        D_xx[0+_ent.nb_dof_per_node*_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        # u_y
                        dof_uy, orient_uy = dof_uy_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_uy])
                        dof_S_dual.extend(len(dof_uy)*[_ent.dual[1]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_uy)*[1+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_uy@M_elem))
                        D_xx[1+_ent.nb_dof_per_node*_w, _ent.primal[1]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        #  p 
                        dof_p, orient_p, _ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        dof_S_dual.extend(len(dof_p)*[_ent.dual[2]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_p)*[2+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_p@M_elem))
                        D_xx[2+_ent.nb_dof_per_node*_w, _ent.primal[2]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    elif _ent.typ == "elastic":
                        # u_x                        
                        dof_ux, orient_ux = dof_ux_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_ux])
                        dof_S_dual.extend(len(dof_ux)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_ux)*[0+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_ux@M_elem))
                        D_xx[0+_ent.nb_dof_per_node*_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        # u_y
                        dof_uy, orient_uy = dof_uy_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_uy])
                        dof_S_dual.extend(len(dof_uy)*[_ent.dual[1]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_uy)*[1+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_uy@M_elem))
                        D_xx[1+_ent.nb_dof_per_node*_w, _ent.primal[1]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    else:
                        raise NameError("_ent.typ has no valid type")
            # DD.append(D_xx[:,_ent.primal[0]])
            DD.append(D_xx)
            DD_Xi.append(coo_matrix((np.conj(D_val), (dof_S_primal, dof_FEM)), shape=(_ent.nb_dof_per_node*self.nb_waves, self.n_dof))@self.P_periodicity)
            # Creation of the D_ix, minus sign <- transposition +normal 
            DD_iX.append(coo_matrix((-_ent.ny*np.array(D_val), (dof_FEM, dof_S_dual)), shape=(self.n_dof, 2*_ent.nb_dof_per_node*self.nb_waves)))
        
        D_iX = np.hstack([self.P_periodicity_H@D_i.todense() for D_i in DD_iX])
        RR = -spsolve(D_XX, D_iX).reshape((self.n_dof-len(self.dof_left), 2*2*_ent.nb_dof_per_node*self.nb_waves))


        R_b = RR[:,:2*_ent.nb_dof_per_node*self.nb_waves]
        R_t = RR[:,2*_ent.nb_dof_per_node*self.nb_waves:]


        self.R_b = R_b[:,_ent.dual]
        self.R_t = R_t[:,_ent.dual]


        D_bb_NME =  DD[0]
        D_tt_NME = DD[1]
        D_bX_NME = DD_Xi[0]
        D_tX_NME = DD_Xi[1]
        
        self.D_bb = DD[0][:,_ent.primal[0]].reshape((self.nb_waves, self.nb_waves))     
        self.D_tt = DD[1][:,_ent.primal[0]].reshape((self.nb_waves, self.nb_waves))
        self.D_bX = DD_Xi[0]
        self.D_tX = DD_Xi[1]
        
        _s = _ent.nb_dof_per_node*self.nb_waves
    
        self.M_b = np.zeros((2*_s, 2*_s), dtype=complex)
        self.M_t = np.zeros((2*_s, 2*_s), dtype=complex)
        
        self.M_b[:_s,:] = D_tX_NME@R_b# [D_ti][R_b]
        self.M_b[_s:,:] = D_bb_NME+D_bX_NME@R_b# [D_bb]+[D_bi][R_b]
        self.M_t[:_s,:] = D_tt_NME+D_tX_NME@R_t# [D_tt]+[D_ti][R_t]
        self.M_t[_s:,:] = D_bX_NME@R_t# [D_bi][R_t]
        
        self.TM = np.linalg.inv(self.M_b)@self.M_t

    def HMM_update(self, H):
        
        

        self.create_HMM_matrices()

        Omega = self.Omega_p+self.Omega_c@H # in HMM formalism
        Omega[:self.nb_waves,:] /= 1j*self.omega # replace velocities by displacements
        Omega = self.FfW@Omega # Go from HMM paper variables to HMM_peridic variables
        
        Omega_hat_u = np.hstack([  self.D_tt, self.D_tX@self.R_t])@Omega
        Omega_hat_b = np.hstack([0*self.D_bb, self.D_bX@self.R_t])@Omega
        
        U, Sigma, Vh = LA.svd(self.D_tX@self.R_b)
        Q = -LA.inv(U.conj().T@Omega_hat_u)@np.diag(Sigma)
        Omega = np.vstack([-LA.inv(self.D_bb)@(self.D_bX@self.R_b@Vh.conj().T+Omega_hat_b@Q), Vh.conj().T])
        
        
        Omega = self.WfF@Omega # Go from IJNME paper variables to HMM variables 
        Omega[:self.nb_waves, :] *= 1j*self.omega # replace displacements by velocities
        HH = self.C_cal@Omega@LA.inv(self.P_cal@Omega)
        self.L_cal = Q@LA.inv(self.P_cal@Omega)
        return HH
