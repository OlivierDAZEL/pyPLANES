#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
#
# This file is part of pymls, a software distributed under the MIT license.
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

import numpy as np

from pyPLANES.core.mesh import Mesh
from pyPLANES.core.calculus import Calculus

from pyPLANES.fem.entities_surfacic import *
from pyPLANES.fem.entities_volumic import *

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from pyPLANES.fem.fem_preprocess import fem_preprocess
from pyPLANES.utils.io import plot_fem_solution

class FemProblem(Mesh, Calculus):
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        Mesh.__init__(self, **kwargs)
        self.interface_zone = kwargs.get("interface_zone", 0.01)
        self.incident_ml = kwargs.get("incident_ml", False)
        self.interface_ml = kwargs.get("interface_ml", False)

        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        
        self.out_file_name = self.name_project + ".FEM.txt"
        self.info_file_name = self.name_project + ".info.FEM.txt"

    def preprocess(self):
        Calculus.preprocess(self)
        fem_preprocess(self)

    def create_linear_system(self, omega):
        # Initialisation of the lists
        self.F = csr_matrix((self.nb_dof_master, 1), dtype=complex)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(omega))


    def update_frequency(self, omega):
            Calculus.update_frequency(self, omega)
            self.F_i, self.F_v = [], []
            self.A_i, self.A_j, self.A_v = [], [], []
            self.T_i, self.T_j, self.T_v = [], [], []
            for _ent in self.fem_entities:
                _ent.update_frequency(omega)

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
        return X

    def resolution(self):
        Calculus.resolution(self)
        if self.name_server == "il-calc1":
            mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
            os.system(mail)

