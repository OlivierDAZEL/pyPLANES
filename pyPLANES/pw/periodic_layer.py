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

from pyPLANES.core.mesh import Mesh
from pyPLANES.core.calculus import Calculus

from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *
# from pyPLANES.fem.fem_entities_pw import IncidentPwFem, TransmissionPwFem

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from pyPLANES.fem.fem_preprocess import fem_preprocess
from pyPLANES.utils.io import plot_fem_solution, export_paraview
from pyPLANES.fem.dofs import periodic_dofs_identification
from pyPLANES.fem.fem_entities_pw import PwFem

class PeriodicLayer(Mesh):
    def __init__(self, **kwargs):
        Mesh.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.order = kwargs.get("order", 2)
        self.verbose = kwargs.get("verbose", False)
        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        self.medium = [None, None]
        self.start_time = time.time()
        self.info_file = open("findanothername.txt", "w")
        self.TM = None
        fem_preprocess(self)
        for _ent in self.pwfem_entities:
            _ent.theta_d = self.theta_d
        # The bottom interface is the first of self.pwfem_entities
        if self.pwfem_entities[0].ny ==1:
            self.pwfem_entities.reverse()
        periodic_dofs_identification(self)


    def update_frequency(self, omega, kx):
        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        for _ent in self.fem_entities:
            _ent.update_frequency(omega)
        for _ent in self.pwfem_entities:
            _ent.update_frequency(omega)
        # Wave numbers and periodic shift
        self.kx = kx
        self.nb_waves = len(kx)
        self.delta_periodicity = np.exp(-1j*self.kx[0]*self.period)
        # self.nb_dofs = self.nb_dof_FEM
        for _ent in self.pwfem_entities:
            _ent.dofs = np.arange(_ent.nb_dof_per_node*len(self.kx))
            _ent.nb_dofs = len(_ent.dofs)
        self.create_TM(omega)



    def create_TM(self, omega):
        # Initialisation of the lists
        # self.F = csr_matrix((self.nb_dof_master, 1), dtype=complex)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        # Creation of the D_ii matrix
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(omega))
        # Application of periodicity 
        for i_left, dof_left in enumerate(self.dof_left):
            # Corresponding dof
            dof_right = self.dof_right[i_left]
            # Summation of the columns for the Matrix
            index = [i for i, value in enumerate(self.A_j) if value == dof_right]
            for _i in index:
                self.A_j[_i] = dof_left
                self.A_v[_i] *= self.delta_periodicity
            # Summation of the rows for the Matrix
            index = np.where(self.A_i == dof_right)
            index = [i for i, value in enumerate(self.A_i) if value == dof_right]

            for _i in index:
                self.A_i[_i] = dof_left
                self.A_v[_i] /= self.delta_periodicity
            # Summation of the rows for the Matrix
            self.A_i.append(dof_right)
            self.A_j.append(dof_left)
            self.A_v.append(self.delta_periodicity)
            self.A_i.append(dof_right)
            self.A_j.append(dof_right)
            self.A_v.append(-1)

        self.linear_system_2_numpy()
        index_A = np.where(((self.A_i*self.A_j) != 0) )
        D_ii = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_master-1, self.nb_dof_master-1)).tocsr()

        # R = [] # Initialisation of the list of the R will be [R_b R_t]
        # D = [] # Initialisation of the list of the R will be [D_bb D_tt]
        # D_i = [] # Initialisation of the list of the R will be [D_bi D_ti]


        RR = [] # Initialisation of the list of the R will be [R_b R_t]
        DD = [] # Initialisation of the list of the R will be [D_bb D_tt]
        DD_xi = [] # Initialisation of the list of the R will be [D_bi D_ti]
        # for _ent in self.pwfem_entities:
        #     M_global = np.zeros((self.nb_dof_master-1), dtype=complex)
        #     for _elem in _ent.elements:
        #         _l = 0
        #         M_elem = imposed_pw_elementary_vector(_elem, self.kx[_l])
        #         if _ent.typ == "fluid":
        #             dof_p, orient_p, _ = dof_p_element(_elem)
        #             dof_p = [d-1 for d in dof_p]
        #             _ = orient_p@M_elem
        #             M_global[dof_p] += _
        #             # D_ib += coo_matrix((_, (dof_p, dof_1)), shape=(self.nb_dof_master-1, _ent.nb_dofs))
        #         # Matrices D_bb and D_tt
        #         D_ =np.zeros((_ent.nb_dof_per_node, 2*_ent.nb_dof_per_node))
        #         D_[:_ent.nb_dof_per_node, _ent.primal[0]] = -_ent.period
        #         D.append(D_)
        #         # Matrices D_bi and D_ti

        #         D_ = np.zeros((_ent.nb_dofs, self.nb_dof_master-1), dtype=complex)
        #         D_[_ent.dual[0], :] = np.conj(M_global)
        #         D_i.append(D_)

        #         D_ = np.zeros((self.nb_dof_master-1, 2*_ent.nb_dofs), dtype=complex)
        #         D_[:, _ent.dual[0]] = M_global
        #         for i_left, dof_left in enumerate(self.dof_left):
        #             D_[dof_left-1, :] += D_[self.dof_right[i_left]-1, :]/self.delta_periodicity
        #             D_[self.dof_right[i_left]-1, :] = 0
        #         R.append(-_ent.ny*linsolve.spsolve(D_ii, D_))

        
        for _ent in self.pwfem_entities:
            dof_FEM, dof_S_primal, dof_S_dual, dof_S, D_val = [], [], [], [], []
            D_period = np.zeros((_ent.nb_dof_per_node*self.nb_waves, 2*_ent.nb_dof_per_node*self.nb_waves))


            for _w, kx in enumerate(self.kx):
                for _elem in _ent.elements:
                    M_elem = imposed_pw_elementary_vector(_elem, kx)
                    if _ent.typ == "fluid":
                        dof_p, orient_p, _ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        D_period[_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        dof_S_dual.extend(len(dof_p)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        dof_S.extend(len(dof_p)*[_w])
                        D_val.extend(list(orient_p@M_elem))
            # print(_ent.nb_dof_per_node*self.nb_waves)
            # print( self.nb_dof_master-1)
            # print(dof_S_primal)
            # print(dof_FEM)
            

            DD.append(D_period)
            DD_xi.append(coo_matrix((np.conj(D_val), (dof_S, dof_FEM)), shape=(_ent.nb_dof_per_node*self.nb_waves, self.nb_dof_master-1)))
            for i_left, _dof_left in enumerate(self.dof_left):
                # Corresponding dof
                _dof_right = self.dof_right[i_left]-1
                index = [i for i,d in enumerate(dof_FEM) if d==_dof_right]
                for _i in index:
                    dof_FEM[_i] = _dof_left-1
                    D_val[_i] /= self.delta_periodicity
            D_ix = coo_matrix((D_val, (dof_FEM, dof_S_dual)), shape=(self.nb_dof_master-1, 2*_ent.nb_dof_per_node*self.nb_waves))
            RR.append(-_ent.ny*linsolve.spsolve(D_ii, D_ix.todense()).reshape((self.nb_dof_master-1, 2*_ent.nb_dof_per_node*self.nb_waves)))

        # print("---")
        _s = _ent.nb_dof_per_node*self.nb_waves
        M_1 = np.zeros((2*_s, 2*_s), dtype=complex)
        M_2 = np.zeros((2*_s, 2*_s), dtype=complex)
        # print(DD_xi[1]@RR[0])
        # M_1[0,:] = (DD_xi[1]@R[0])
        # M_1[1,:] = (D[0]+DD_xi[0]@R[0]) 
        # M_2 = np.zeros((2,2), dtype=complex)
        # M_2[0,:] = D[1]+DD_xi[1]@RR[1] 
        # M_2[1,:] = D_i[0]@RR[1]

        # print(DD_xi[1]@RR[0])

        M_1[:_s,:] = (DD_xi[1]@RR[0])#.todense()
        M_1[_s:,:] = DD[0]+DD_xi[0]@RR[0] 
        
        M_2[:_s,:] = DD[1]+DD_xi[1]@RR[1] 
        # print(DD_xi[0])
        # print("---")
        # print(RR[1].shape)
        # print("------")
        # print((DD_xi[0]@RR[1]).shape)
        M_2[_s:,:] = (DD_xi[0]@RR[1])#.todense()

        self.TM = -LA.solve(M_1, M_2)



    def transfert(self, Om):
        # Creation of the Transfer Matrix 
        if self.verbose: 
            print("Creation of the Transfer Matrix of the FEM layer")
        return self.TM@Om, np.eye(Om.shape[1])


    def update_system(self, _A_i, _A_j, _A_v, _T_i, _T_j, _T_v, _F_i, _F_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.T_i.extend(_T_i)
        self.T_j.extend(_T_j)
        self.T_v.extend(_T_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def linear_system_2_numpy(self):
        self.F_i = np.array(self.F_i)
        self.F_v = np.array(self.F_v, dtype=complex)
        self.A_i = np.array(self.A_i)
        self.A_j = np.array(self.A_j)
        self.A_v = np.array(self.A_v, dtype=complex)
        self.T_i = np.array(self.T_i)-self.nb_dof_master
        self.T_j = np.array(self.T_j)
        self.T_v = np.array(self.T_v, dtype=complex)

    def plot_solution(self):
        plot_fem_solution(self)


    def solve(self):
        self.nb_dof_condensed = self.nb_dof_FEM - self.nb_dof_master
        start = timeit.default_timer()
        self.linear_system_2_numpy()

        index_A = np.where(((self.A_i*self.A_j) != 0) )
        A = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_master-1, self.nb_dof_master-1)).tocsr()
        F = np.zeros(self.nb_dof_master-1, dtype=complex)
        for _i, f_i in enumerate(self.F_i):
            F[f_i-1] += self.F_v[_i]

        # Resolution of the sparse linear system
        if self.verbose:
            print("Resolution of the linear system")
        X = linsolve.spsolve(A, F)
        # Concatenation of the first (zero) dof at the begining of the vector
        X = np.insert(X, 0, 0)
        # Concatenation of the slave dofs at the end of the vector
        T = coo_matrix((self.T_v, (self.T_i, self.T_j)), shape=(self.nb_dof_condensed, self.nb_dof_master)).tocsr()
        X = np.insert(T@X, 0, X)
        stop = timeit.default_timer()
        if self.verbose:
            print("Elapsed time for linsolve = {} ms".format((stop-start)*1e3))

        for _vr in self.vertices[1:]:
            for i_dim in range(4):
                _vr.sol[i_dim] = X[_vr.dofs[i_dim]]
        for _ed in self.edges:
            for i_dim in range(4):
                _ed.sol[i_dim] = X[_ed.dofs[i_dim]]
        for _fc in self.faces:
            for i_dim in range(4):
                _fc.sol[i_dim] = X[_fc.dofs[i_dim]]
        for _bb in self.bubbles:
            for i_dim in range(4):
                _bb.sol[i_dim] = X[_bb.dofs[i_dim]]       
        if self.export_paraview is not False:
            export_paraview(self)
        return X

    def resolution(self):
        Calculus.resolution(self)
        if self.name_server == "il-calc1":
            mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
            os.system(mail)

