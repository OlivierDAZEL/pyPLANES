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
import scipy.sparse.linalg as sla
import numpy as np

from pyPLANES.core.mesh import Mesh
from pyPLANES.core.calculus import Calculus

from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *

from pyPLANES.gmsh.gmsh import GmshMesh

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from pyPLANES.fem.fem_preprocess import fem_preprocess
from pyPLANES.utils.io import plot_fem_solution, export_paraview

from mediapack import Air

class EigenProblemBase(Mesh, Calculus):
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        Mesh.__init__(self, **kwargs)
        self.order = kwargs.get("order", 2)
        self.interface_zone = kwargs.get("interface_zone", 0.01)
        self.interface_ml = kwargs.get("interface_ml", False)

        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        

        self.out_file_method = "FEM"

        fem_preprocess(self)
        
        self.Results["order"] = self.order
        self.Results["n_elem"] = []
        self.Results["eigen"] = []

    def create_linear_system(self, omega):
        # Initialisation of the lists
        self.F = csr_matrix((self.nb_dof_master, 1), dtype=complex)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        self.Q_i, self.Q_j, self.Q_v = [], [], []
        self.H_i, self.H_j, self.H_v = [], [], []
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(omega))
            if _ent.dim == 2:
                self.update_Q(*_ent.update_Q())
                self.update_H(*_ent.update_H())

    def update_frequency(self, omega):
            Calculus.update_frequency(self, omega)
            self.F_i, self.F_v = [], []
            self.A_i, self.A_j, self.A_v = [], [], []
            self.T_i, self.T_j, self.T_v = [], [], []
            for _ent in self.fem_entities:
                _ent.update_frequency(omega)

    def update_Q(self, _A_i, _A_j, _A_v):
        self.Q_i.extend(_A_i)
        self.Q_j.extend(_A_j)
        self.Q_v.extend(_A_v)

    def update_H(self, _A_i, _A_j, _A_v):
        self.H_i.extend(_A_i)
        self.H_j.extend(_A_j)
        self.H_v.extend(_A_v)

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
        self.Q_i, self.Q_j, self.Q_v = [], [], []
        self.H_i, self.H_j, self.H_v = [], [], []
        for _ent in self.fem_entities:
            if _ent.dim == 2:
                self.update_Q(*_ent.update_Q())
                self.update_H(*_ent.update_H())

        self.H_i = np.array(self.H_i)
        self.H_j = np.array(self.H_j)
        self.H_v = np.array(self.H_v, dtype=complex)

        self.Q_i = np.array(self.Q_i)
        self.Q_j = np.array(self.Q_j)
        self.Q_v = np.array(self.Q_v, dtype=complex)


        index_Q = np.where(((self.Q_i*self.Q_j) != 0) )
        Q = coo_matrix((self.Q_v, (self.Q_i-1, self.Q_j-1)), shape=(self.nb_dof_FEM-1, self.nb_dof_FEM-1)).tocsr()
        index_H = np.where(((self.H_i*self.H_j) != 0) )
        H = coo_matrix((self.H_v, (self.H_i-1, self.H_j-1)), shape=(self.nb_dof_FEM-1, self.nb_dof_FEM-1)).tocsr()

        D, V = sla.eigs(H, 5, Q, which='SM')
        D = np.sqrt(np.sort(np.real(D))[1:])

        for _e in self.entities:
            if _e.dim ==2:
                self.Results["n_elem"].append(len(_e.elements))
        self.Results["eigen"] = (D-np.arange(1,5)).tolist()






    def resolution(self):
        self.solve()
        self.close_info_file()
        if self.save_format == "json":
            self.save_json()

        if self.name_server == "il-calc1":
            mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
            os.system(mail)


class EigenProblem(EigenProblemBase, GmshMesh):
    def __init__(self, **kwargs):
        GmshMesh.__init__(self, **kwargs)
        EigenProblemBase.__init__(self, **kwargs)
