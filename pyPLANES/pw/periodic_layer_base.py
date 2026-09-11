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

from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix, linalg as sla
from scipy.sparse.linalg import svds

from pyPLANES.fem.fem_preprocess import fem_preprocess
from pyPLANES.utils.io import plot_fem_solution, export_paraview
from pyPLANES.fem.dofs import periodic_dofs_identification
from pyPLANES.fem.fem_entities_pw import PwFem
from pyPLANES.fem.utils_fem import dof_p_element, dof_up_element

class PeriodicLayerBase(Mesh):
    def __init__(self, **kwargs):
        self.condensation = kwargs.get("condensation", True)
        Mesh.__init__(self, **kwargs)
        _x = kwargs.get("_x", 0)
        for _v in self.vertices[1:]:
            _v.coord[1] += _x
        for _e in self.elements[1:]:
            _e.coord[1,:] += _x
             
             
        y = [_v.coord[1] for _v in self.vertices[1:]]
        self.d = np.max(y)-np.min(y)

        self.theta_d = kwargs.get("theta_d", 0.0)
        self.order = kwargs.get("order", 2)
        self.verbose = kwargs.get("verbose", False)
        self.plot = kwargs.get("plot", [False]*6)

        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        self.medium = [None, None]
        self.start_time = time.time()
        self.TM = None


        fem_preprocess(self)
        
        if self.condensation:
            self.n_dof = self.nb_dof_master-1
        
        for _ent in self.pwfem_entities:
            _ent.theta_d = self.theta_d
            if _ent.typ == "fluid":
                _ent.method_dof = dof_p_element
            elif _ent.typ in ["Biot98", "Biot01"]:
                _ent.method_dof = dof_up_element  
                
        # The bottom interface is the first of self.pwfem_entities
        if self.pwfem_entities[0].physicalTags["condition"] == "top":
            self.pwfem_entities.reverse()
        # Its normal is -1
        self.pwfem_entities[0].ny =-1.
        if len(self.pwfem_entities) !=0:
            periodic_dofs_identification(self)

            # determination of the internal dofs // We cancel the +1
            dof_periodic = [i-1 for i in self.dof_left]+[i-1 for i in self.dof_right] # We cancel the +1
            self.dof_internal = [i for i in range(self.n_dof) if i not in dof_periodic]
            self.n_dof_minus_periodicity = self.n_dof - len(self.dof_left)
            n = self.n_dof
            m = self.n_dof_minus_periodicity
            
            # creation of P_periodicity_master 
            # first left dofs then internal dofs
            rows = [d-1 for d in self.dof_left]+ [d for d in self.dof_internal]
            self.orientation_periodic_dofs += [1. for d in self.dof_internal]
            columns = [i for i in range(m)]

            
            self.P_periodicity_master = csr_matrix((self.orientation_periodic_dofs, (rows, columns)), shape=(n, m))
            # creation of P_periodicity_delta
            rows = [d-1 for d in self.dof_right]
            columns = [d for d in range(len(self.dof_left))]
            
            self.P_periodicity_delta = csr_matrix((np.ones(len(self.dof_left),dtype=complex), (rows, columns)), shape=(n, m))

        self.characteristics = [None, None] # Will be completed in PeriodicPwProblem.__init__()

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
        self.P_periodicity = self.P_periodicity_master+self.delta_periodicity*self.P_periodicity_delta
        self.P_periodicity_H = np.conj(self.P_periodicity.transpose())

        self.omega = omega # Needs to be stored
        for _ent in self.pwfem_entities:
            _ent.dofs = np.arange(_ent.nb_dof_per_node*len(self.kx))
            _ent.nb_dofs = len(_ent.dofs)

    def apply_periodicity_on_Dii(self):
        # Application of periodicity on Dii
        for i_left, dof_left in enumerate(self.dof_left):
            # Corresponding dof
            dof_right = self.dof_right[i_left]
            index = np.where(np.array(self.A_j) == dof_right)[0]
            for _i in index:
                self.A_j[_i] = dof_left
                self.A_v[_i] *= self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            # Summation of the rows for the Matrix
            index = np.where(np.array(self.A_i) == dof_right)[0]
            for _i in index:
                self.A_i[_i] = dof_left
                self.A_v[_i] /= self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            # Periodicity of the physical dofs
            self.A_i.append(dof_right)
            self.A_j.append(dof_left)
            self.A_v.append(self.delta_periodicity)
            self.A_i.append(dof_right)
            self.A_j.append(dof_right)
            self.A_v.append(-self.orientation_periodic_dofs[i_left])

    def create_bulk_matrices(self):
        # Initialisation of the lists
        self.A_i, self.A_j, self.A_v = [], [], []
        # Number of dof of the D_ii marix
        if self.condensation:
            self.T_i, self.T_j, self.T_v = [], [], []
        else:
            self.n_dof = self.nb_dof_FEM-1

        # Creation of the D_ii matrix (volumic term of the weak form) 
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(self.omega))



        
    def update_TM(self, omega=None):
        self.create_global_method_matrices()
        self.TM = -LA.solve(self.M_b, self.M_t)

    def update_system(self, _A_i, _A_j, _A_v, _F_i, _F_v, _T_i=None, _T_j=None, _T_v=None):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)
        if self.condensation:
            self.T_i.extend(_T_i)
            self.T_j.extend(_T_j)
            self.T_v.extend(_T_v)

    def linear_system_2_numpy(self):
        if self.condensation:
            self.T_i = np.array(self.T_i)-self.nb_dof_master
            self.T_j = np.array(self.T_j)
            self.T_v = np.array(self.T_v, dtype=complex)

    def plot_solution(self, S_b, S_t):
        X = self.R_b@S_b +self.R_t@S_t
        X = self.P_periodicity@X
        X = np.insert(X, 0, 0)
        # Concatenation of the slave dofs at the end of the vector
        self.nb_dof_condensed = self.nb_dof_FEM - self.nb_dof_master
        if self.condensation:
            T = coo_matrix((self.T_v, ([t-self.nb_dof_master for t_i in  self.T_i], self.T_j)), shape=(self.nb_dof_FEM-self.nb_dof_master, self.nb_dof_master)).tocsr()
            X = np.insert(T@X, 0, X)
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
        plot_fem_solution(self, self.kx)

