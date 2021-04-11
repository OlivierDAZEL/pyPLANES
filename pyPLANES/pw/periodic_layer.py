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
from pyPLANES.fem.fem_entities_pw import IncidentPwFem, TransmissionPwFem

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
        self.verbose = kwargs.get("verbose", True)
        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        self.medium = [None, None]
        self.start_time = time.time()
        self.info_file = open("findanothername.txt", "w")

        fem_preprocess(self)
        for _ent in self.pwfem_entities:
            if isinstance(_ent, IncidentPwFem):
                _ent.theta_d = self.theta_d
                _ent.nb_R = 1
                _ent.typ = "fluid"
            if isinstance(_ent, TransmissionPwFem):
                _ent.theta_d = self.theta_d
                _ent.nb_R = 1
                _ent.typ = "fluid"
        periodic_dofs_identification(self)

    def create_linear_system(self, omega):
        # Initialisation of the lists
        # self.F = csr_matrix((self.nb_dof_master, 1), dtype=complex)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        # Creation of the D_ii matrix
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(omega))
        # Creation of Rt and Ri 
        self.linear_system_2_numpy()
        index_A = np.where(((self.A_i*self.A_j) != 0) )
        D_ii = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_master-1, self.nb_dof_master-1)).todense()

        for _ent in self.pwfem_entities:
            if isinstance(_ent, IncidentPwFem):
                M_global = np.zeros((self.nb_dof_master-1), dtype=complex)
                # D_ib = coo_matrix((self.nb_dof_master-1, 2*_ent.nb_dofs), dtype=complex)                
                for _elem in _ent.elements:
                    _l = 0
                    M_elem = imposed_pw_elementary_vector(_elem, self.kx)
                    if _ent.typ == "fluid":
                        dof_p, orient_p, _ = dof_p_element(_elem)
                        dof_p = [d-1 for d in dof_p]
                        dof_1 = [_ent.dofs[_ent.nb_R*_l]]*len(dof_p)
                        _ = orient_p@M_elem
                        M_global[dof_p] += _
                        # D_ib += coo_matrix((_, (dof_p, dof_1)), shape=(self.nb_dof_master-1, _ent.nb_dofs))

                D_bb =np.zeros((1, 2))
                D_bb[0, 1] = -_ent.period

                D_ib = np.zeros((self.nb_dof_master-1, 2*_ent.nb_dofs), dtype=complex)
                D_ib[:, 0] = M_global

                D_bi = np.zeros((_ent.nb_dofs, self.nb_dof_master-1), dtype=complex)
                D_bi[0, :] = np.conj(M_global)

                R_b = -LA.solve(D_ii, D_ib)
                R_b *= -1.


            if isinstance(_ent, TransmissionPwFem):
                M_global = np.zeros((self.nb_dof_master-1), dtype=complex)
                # D_ib = coo_matrix((self.nb_dof_master-1, 2*_ent.nb_dofs), dtype=complex)                
                for _elem in _ent.elements:
                    _l = 0
                    M_elem = imposed_pw_elementary_vector(_elem, self.kx)
                    if _ent.typ == "fluid":
                        dof_p, orient_p, _ = dof_p_element(_elem)
                        dof_p = [d-1 for d in dof_p]
                        dof_1 = [_ent.dofs[_ent.nb_R*_l]]*len(dof_p)
                        _ = orient_p@M_elem
                        M_global[dof_p] += _

                D_tt =np.zeros((1, 2))
                D_tt[0, 1] = -_ent.period

                D_it = np.zeros((self.nb_dof_master-1, 2*_ent.nb_dofs), dtype=complex)
                D_it[:, 0] = M_global

                D_ti = np.zeros((_ent.nb_dofs, self.nb_dof_master-1), dtype=complex)
                D_ti[0, :] = np.conj(M_global)

                R_t = -LA.solve(D_ii, D_it)




        # print("D_it")
        # print(D_it)
        # print("D_ib")
        # print(D_ib)
        # print("R_t")
        # print(R_t)
        # print("R_b")
        # print(R_b)

    
        # D_ti = (D_it.H).reshape((_ent.nb_dofs, self.nb_dof_master-1))
        # D_bi = (D_ib.H).reshape((_ent.nb_dofs, self.nb_dof_master-1))

        M_1 = np.zeros((2,2), dtype=complex)
        # print(D_ti.shape)
        # print(R_b.shape)
        M_1[0,:] = D_ti@R_b 
        M_1[1,:] = D_bb+D_bi@R_b 
        M_2 = np.zeros((2,2), dtype=complex)
        M_2[0,:] = D_tt+D_ti@R_t 
        M_2[1,:] = D_bi@R_t
        # print("M_1")
        # print(M_1)
        self.M = -LA.solve(M_1, M_2)
        print(self.M) 
        fdsdfsfdsfds

        # print(self.pwfem_entities)



        # dssqsdsqddsqdsqsqdsqd
            


    def update_frequency(self, omega, kx):
            self.F_i, self.F_v = [], []
            self.A_i, self.A_j, self.A_v = [], [], []
            self.T_i, self.T_j, self.T_v = [], [], []
            for _ent in self.fem_entities:
                _ent.update_frequency(omega)
            for _ent in self.pwfem_entities:
                _ent.update_frequency(omega)
            # Wave numbers and periodic shift
            self.kx = (omega/Air.c)*np.sin(self.theta_d*np.pi/180)
            self.ky = (omega/Air.c)*np.cos(self.theta_d*np.pi/180)
            self.delta_periodicity = np.exp(-1j*self.kx*self.period)

            self.nb_dofs = self.nb_dof_FEM
            self.create_linear_system(omega)
            

    def transfert(self, Om):
        # Creation of the Transfer Matrix 
        if self.verbose: 
            print("Creation of the Transfer Matrix of the FEM layer")
        




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

