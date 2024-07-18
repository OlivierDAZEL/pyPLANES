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
import itertools

import numpy as np


from pyPLANES.core.calculus import Calculus

from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from pyPLANES.fem.fem_preprocess import fem_preprocess
from pyPLANES.utils.io import plot_fem_solution, export_paraview

from pyPLANES.core.mesh import Mesh
from pyPLANES.gmsh.mesh import GmshMesh

from mediapack import Air

class FemBase(Mesh, Calculus):
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)

        self.result.Solver = type(self).__name__
        self.condensation = kwargs.get("condensation", True)
        self.order = kwargs.get("order", 2)
        Mesh.__init__(self, **kwargs)

        self.interface_zone = kwargs.get("interface_zone", 0.01)
        self.interface_ml = kwargs.get("interface_ml", False)

        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        if self.condensation:
            self.T_i, self.T_j, self.T_v = None, None, None

        self.result.R0 = []
        self.result.order = self.order
        self.result.n_dof = []
        self.result.D_lambda = []


    def create_linear_system(self, omega):
        
        # Initialisation of the lists
        Calculus.create_linear_system(self, omega)
        self.F = csr_matrix((self.nb_dof_master, 1), dtype=complex)
        self.A_i, self.A_j, self.A_v = [], [], []
        if self.condensation:
            self.T_i, self.T_j, self.T_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(omega))
            # if _ent.dim == 2:
            #     self.update_Q(*_ent.update_Q())

    def update_frequency(self, omega):
            Calculus.update_frequency(self, omega)
            self.F_i, self.F_v = [], []
            self.A_i, self.A_j, self.A_v = [], [], []
            if self.condensation:
                self.T_i, self.T_j, self.T_v = [], [], []
            for _ent in self.fem_entities:
                _ent.update_frequency(omega)

    def update_Q(self, _A_i, _A_j, _A_v):
        self.Q_i.extend(_A_i)
        self.Q_j.extend(_A_j)
        self.Q_v.extend(_A_v)

    def update_system(self, _A_i, _A_j, _A_v, _F_i, _F_v, _T_i=None, _T_j=None, _T_v=None):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)
        if _T_i is not None:
            self.T_i.extend(_T_i)
            self.T_j.extend(_T_j)
            self.T_v.extend(_T_v)

    def linear_system_2_numpy(self):
        self.F_i = np.array(self.F_i)
        self.F_v = np.array(self.F_v, dtype=complex)
        self.A_i = np.array(self.A_i)
        self.A_j = np.array(self.A_j)
        self.A_v = np.array(self.A_v, dtype=complex)
        if self.condensation:
            self.T_i = np.array(self.T_i)-self.nb_dof_master
            self.T_j = np.array(self.T_j)
            self.T_v = np.array(self.T_v, dtype=complex)

    def plot_solution(self):
        plot_fem_solution(self)

    def solve(self):
        Calculus.solve(self)


        # self.nb_dof_condensed = self.nb_dof_FEM - self.nb_dof_master
        start = timeit.default_timer()
        self.linear_system_2_numpy()

        index_A = np.where(((self.A_i*self.A_j) != 0) )
        if self.condensation:
            A = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_master-1, self.nb_dof_master-1)).tocsr()
        else:
            A = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_FEM-1, self.nb_dof_FEM-1)).tocsr()
        F = np.zeros(A.shape[0], dtype=complex)


        for _i, f_i in enumerate(self.F_i):
            F[f_i-1] += self.F_v[_i]

        self.Q_i = np.array(self.Q_i)
        self.Q_j = np.array(self.Q_j)
        self.Q_v = np.array(self.Q_v, dtype=complex)


        index_Q = np.where(((self.Q_i*self.Q_j) != 0) )
        Q = coo_matrix((self.Q_v, (self.Q_i, self.Q_j)), shape=(self.nb_dof_FEM, self.nb_dof_FEM)).tocsr()

        # Resolution of the sparse linear system
        if self.verbose:
            print("Resolution of the linear system")
        X = linsolve.spsolve(A, F)
        X = np.insert(X, 0, 0)
        if self.condensation:
           # Concatenation of the slave dofs at the end of the vector
            T = coo_matrix((self.T_v, (self.T_i, self.T_j)), shape=(self.nb_dof_FEM-self.nb_dof_master, self.nb_dof_master)).tocsr()
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
        # if self.export_paraview is not False:
        #     export_paraview(self)
        # if self.list_vr is not False:
        #     l_x = [v.coord[0] for v in self.list_vr]
        #     d = max(l_x)- min(l_x)


            # pr = X[1:self.nb_dof_master].dot(F)/d/(1j*ome/(Air.Z*ome**2))
        for _ent in self.fem_entities:
            if isinstance(_ent, (ImposedPwFem, ImposedDisplacementFem)):
                ome = 2*np.pi*self.f
                Z = _ent.impedance_on_entity(X, ome)
                R= (Z-Air.Z)/(Z+Air.Z)

                # kl
                # k = ome/Air.c
                # L2 = X.T@(Q@X)
                self.result.R0.append(R)

        self.result.n_dof.append(self.nb_dof_master)
        return X

    def resolution(self):
        Calculus.resolution(self)
        
        # if self.name_server == "helmholtz":
        #     mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
        #     os.system(mail)


class FemProblem(FemBase, GmshMesh):
    def __init__(self, **kwargs):
        self.condensation = kwargs.get("condensation", True)
        GmshMesh.__init__(self, **kwargs)
        FemBase.__init__(self, **kwargs)
        fem_preprocess(self)



