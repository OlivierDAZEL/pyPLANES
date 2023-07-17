#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# model.py
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
import os
import time

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

import numpy as np
import numpy.linalg as LA

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mediapack import Air

from pyPLANES.classes.entity_classes import *
from pyPLANES.utils.utils_io import initialisation_out_files_plain, close_out_files, display_sol

class Model():
    def __init__(self, **kwargs):
        self.plot = kwargs.get("plot_results", [False]*6)
        pass

    def extend_AF(self, _A_i, _A_j, _A_v, _F_i, _F_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def extend_F(self, _F_i, _F_v):
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def extend_A(self, _A_i, _A_j, _A_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)

    def extend_A_F_from_coo(self, AF):
        self.A_i.extend(list(AF[0].row))
        self.A_j.extend(list(AF[0].col))
        self.A_v.extend(list(AF[0].data))
        # print("extend_A_F_from_coo")
        # print(AF[1])
        # print(AF[1].row)
        self.F_i.extend(list(AF[1].row))
        self.F_v.extend(list(AF[1].data))
        # print("self.F_i={}".format(self.F_i))

    def extend_AT(self, _A_i, _A_j, _A_v, _T_i, _T_j, _T_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.T_i.extend(_T_i)
        self.T_j.extend(_T_j)
        self.T_v.extend(_T_v)

    def linear_system_2_numpy(self):
        self.F_i = np.array(self.F_i)
        self.F_v = np.array(self.F_v, dtype=complex)
        self.A_i = np.array(self.A_i)
        self.A_j = np.array(self.A_j)
        self.A_v = np.array(self.A_v, dtype=complex)
        self.T_i = np.array(self.T_i)-self.nb_dof_master
        self.T_j = np.array(self.T_j)
        self.T_v = np.array(self.T_v, dtype=complex)

    def write_out_files(self):
        pass

    def create_linear_system(self, f):
        if self.verbose:
            print("Creation of the linear system for f={}".format(f))
        self.update_frequency(f)

    def initialisation_out_files(self):
        initialisation_out_files_plain(self)
        self.start_time = time.time()

    def close_out_files(self):
        close_out_files(self)

    def resolution(self):
        if self.verbose:
            print("%%%%%%%%%%%%% Resolution of PLANES %%%%%%%%%%%%%%%%%")
        for f in self.frequencies:
            self.create_linear_system(f)
            out = self.solve()
            self.write_out_files()
            # if self.verbose:
                # print("|R pyPLANES_FEM|  = {}".format(self.modulus_reflex))
                # print("|abs pyPLANES_FEM| = {}".format(self.abs))
            if any(self.plot):
                display_sol(self)
        self.close_out_files()

        return out




