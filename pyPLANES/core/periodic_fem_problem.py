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


# from scipy.sparse.linalg.dsolve import linsolve
# from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.fem.entities_pw import *


class PeriodicFemProblem(FemProblem):
    def __init__(self, **kwargs):
        FemProblem.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        # Incident angle is affected to all pwfem entities
        for _ent in self.pwfem_entities:
            if isinstance(_ent, PwFem):
                _ent.theta_d = self.theta_d
        self.modulus_reflex, self.modulus_trans, self.abs = None, None, None


    def update_frequency(self, omega):
        FemProblem.update_frequency(self, omega)
        # Wave numbers and periodic shift
        self.kx = (omega/Air.c)*np.sin(self.theta_d*np.pi/180)
        self.ky = (omega/Air.c)*np.cos(self.theta_d*np.pi/180)
        self.delta_periodicity = np.exp(-1j*self.kx*self.period)

        self.nb_dofs = self.nb_dof_FEM
        for _ent in self.pwfem_entities:
            _ent.update_frequency(omega)
        self.modulus_reflex, self.modulus_trans, self.abs = 0, 0, 1

    def create_linear_system(self, omega):
        FemProblem.create_linear_system(self, omega)
        for _ent in self.pwfem_entities:
            self.update_system(*_ent.update_system(omega, self.nb_dof_master))      
        # Application of the periodicity
        if self.dof_left != []:
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
                # index = np.where(self.F_i == dof_right)
                index = [i for i, value in enumerate(self.F_i) if value == dof_right]
                for _i in index:
                    self.F_i[_i] = dof_left
                    self.F_v[_i] /= self.delta_periodicity
        for _ent in self.pwfem_entities: # For the postprocess and the determination of reflexion coefficients 
            _ent.apply_periodicity(self.nb_dof_master, self.dof_left, self.dof_right, self.delta_periodicity)
        # End of Application of the periodicity

    def solve(self):
        out = dict()
        X = FemProblem.solve(self)
        # self.abs has been sent to 1 in the __init__ () of the model class
        for _ent in self.entities[1:]:
            if isinstance(_ent, IncidentPwFem):
                _ent.sol = _ent.phi.H@(X[:self.nb_dof_master])/_ent.period
                _ent.sol[:_ent.nb_R] -= _ent.Omega_0_orth
                _ent.sol = _ent.eta_TM@_ent.sol
                self.modulus_reflex = np.sqrt(np.sum(np.real(_ent.ky)*np.abs(_ent.sol[::_ent.nb_R]**2)/np.real(self.ky)))
                out["R"] = _ent.sol[0]
                self.abs -= np.abs(self.modulus_reflex)**2
            elif isinstance(_ent, TransmissionPwFem):
                _ent.sol = _ent.phi.H@(X[:self.nb_dof_master])/_ent.period
                _ent.sol = _ent.eta_TM@_ent.sol
                # print("T pyPLANES_FEM   = {}".format((_ent.sol[0])))
                out["T"] = _ent.sol[0]
                self.modulus_trans = np.sqrt(np.sum(np.real(_ent.ky)*np.abs(_ent.sol[::_ent.nb_R])**2/np.real(self.ky)))
                self.abs -= self.modulus_trans**2
        print("abs pyPLANES_FEM   = {}".format(self.abs))
        return out