#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# model.py
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
import os
import timeit

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

import numpy as np
import numpy.linalg as LA

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mediapack import Air

from pyPLANES.classes.entity_classes import *
from pyPLANES.utils.utils_io import display_sol
from pyPLANES.classes.model import Model




class FemModel(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        self.incident_ml = kwargs.get("incident_ml", False)


        self.reference_elements = dict() # dictionary of reference_elements
        self.edges = []
        self.faces = []
        self.bubbles = []
        self.nb_edges = self.nb_faces = self.nb_bubbles = 0

    def initialisation_out_files(self):
        Model.initialisation_out_files(self)

    def write_out_files(self):
        self.out_file.write("{:.12e}\t".format(self.current_frequency))
        if any([isinstance(_ent, PwFem) for _ent in self.model_entities]):
            self.out_file.write("{:.12e}\t".format(self.abs))
        if any([isinstance(_ent, (IncidentPwFem)) for _ent in self.model_entities]):
            self.out_file.write("{:.12e}\t".format(self.modulus_reflex))
        if any([isinstance(_ent, (TransmissionPwFem)) for _ent in self.model_entities]):
            self.out_file.write("{:.12e}\t".format(self.modulus_trans))
        self.out_file.write("\n")

    def __str__(self):
        out = "TBD"
        return out

    def resolution(self):
        Model.resolution(self)
        if self.name_server == "il-calc1":
            mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
            os.system(mail)


    def linear_system_2_numpy(self):
        Model.linear_system_2_numpy(self)
        for _ent in self.model_entities:
            if isinstance(_ent, PwFem):
                _ent.phi_i = np.array(_ent.phi_i)
                _ent.phi_j = np.array(_ent.phi_j)
                _ent.phi_v = np.array(_ent.phi_v, dtype=complex)

    def create_linear_system(self, f):
        Model.create_linear_system(self, f)
        omega = 2*np.pi*f
        # Initialisation of the lists
        self.F = csr_matrix((self.nb_dof_master, 1), dtype=complex)
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        for _ent in self.model_entities:
            if isinstance(_ent, (PwFem)):
                self.extend_A_F_from_coo(_ent.create_dynamical_matrices(omega, self.nb_dof_master))
                # print("F_create={}".format(self.F))
            else:
                _A_i, _A_j, _A_v, _T_i, _T_j, _T_v = _ent.append_linear_system(omega)
                self.extend_AT(_A_i, _A_j, _A_v, _T_i, _T_j, _T_v)
        self.linear_system_2_numpy()
        self.apply_periodicity()

    def apply_periodicity(self):
        A_i, A_j, A_v = [], [], []
        # print("apply periodicity")
        # print("self.F_i={}".format(self.F_i))
        if self.dof_left != []:
            for i_left, dof_left in enumerate(self.dof_left):
                # Corresponding dof
                dof_right = self.dof_right[i_left]
                # Summation of the columns for the Matrix
                index = np.where(self.A_j_c == dof_right)
                self.A_j[index] = dof_left
                for _i in index:
                    self.A_v[_i] *= self.delta_periodicity
                # Summation of the rows for the Matrix
                index = np.where(self.A_i == dof_right)
                self.A_i[index] = dof_left
                for _i in index:
                    self.A_v[_i] /= self.delta_periodicity
                # Summation of the rows for the Matrix
                A_i.append(dof_right)
                A_j.append(dof_left)
                A_v.append(self.delta_periodicity)
                A_i.append(dof_right)
                A_j.append(dof_right)
                A_v.append(-1)
                index = np.where(self.F_i == dof_right)
                self.F_i[index] = dof_left
                for _i in index:
                    self.F_v[_i] /= self.delta_periodicity
        self.A_i = np.append(self.A_i, A_i)
        self.A_j = np.append(self.A_j, A_j)
        self.A_v = np.append(self.A_v, A_v)

        for _ent in self.model_entities:
            if isinstance(_ent, PwFem):
                _ent.apply_periodicity(self.nb_dof_master, self.dof_left, self.dof_right, self.delta_periodicity)


    def solve(self):
        out = dict()
        self.nb_dof_condensed = self.nb_dof_FEM - self.nb_dof_master

        start = timeit.default_timer()

        index_A = np.where(((self.A_i*self.A_j) != 0) )
        A = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_master-1, self.nb_dof_master-1)).tocsr()
        F = np.zeros(self.nb_dof_master-1, dtype=complex)
        for _i, f_i in enumerate(self.F_i):
            F[f_i-1] += self.F_v[_i]
        # print(F)

        self.A_i, self.A_j, self.F_v, self.F_i, self.F_v = None, None, None, None, None
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
        # self.abs has been sent to 1 in the __init__ () of the model class
        for _ent in self.entities[1:]:
            if isinstance(_ent, IncidentPwFem):
                _ent.sol = _ent.phi.H@(X[:self.nb_dof_master])/_ent.period
                _ent.sol[:_ent.nb_R] -= _ent.Omega_0_orth
                _ent.sol = _ent.eta_TM@_ent.sol
                self.modulus_reflex = np.sqrt(np.sum(np.real(_ent.ky)*np.abs(_ent.sol[::_ent.nb_R]**2)/np.real(self.ky)))
                out["R"] = _ent.sol[0]
                # print("R pyPLANES_FEM   = {}".format((_ent.sol[0])))
                self.abs -= np.abs(self.modulus_reflex)**2
            elif isinstance(_ent, TransmissionPwFem):
                _ent.sol = _ent.phi.H@(X[:self.nb_dof_master])/_ent.period
                _ent.sol = _ent.eta_TM@_ent.sol
                # print("T pyPLANES_FEM   = {}".format((_ent.sol[0])))
                out["T"] = _ent.sol[0]
                self.modulus_trans = np.sqrt(np.sum(np.real(_ent.ky)*np.abs(_ent.sol[::_ent.nb_R])**2/np.real(self.ky)))
                self.abs -= self.modulus_trans**2
        # print("abs pyPLANES_FEM   = {}".format(self.abs))

        return out
