#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# characteristics.py
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

import numpy as np
import numpy.linalg as LA
from numpy import sqrt
from pyPLANES.pw.pw_polarisation import fluid_waves_TMM, elastic_waves_TMM, PEM_waves_TMM


class Characteristics():
    """
    Interface for Plane Wave Solver
    """
    def __init__(self, medium):
        self.medium = medium
        self.P = None
        self.Q = None
        self.typ = None


    def update_frequency(self, omega):
        self.medium.update_frequency(omega)
        if self.medium.MEDIUM_TYPE in ['eqf', 'fluid']:
            self.n_w = 1
            if self.medium.MEDIUM_TYPE == "eqf":
                K = self.medium.K_eq_til
            else:
                K = self.medium.K
            _ = 1j/(K*self.medium.k)
            self.P = np.array([[-_, _],[1,1]])
            self.lam = np.array([-1j*self.medium.k, 1j*self.medium.k])
            self.Q = np.array([[-1, _],[1,_]])/(2*_)
            
            
        elif self.medium.MEDIUM_TYPE == "pem":
            self.n_w = 3
            # {0: hat{sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:hat{sigma}_{yy}, 4:p, 5:u_x^s}
            self.P, self.lam = PEM_waves_TMM(self.medium, np.array([0]))
            self.Q = LA.inv(self.P)
            if self.typ == "Biot98":
                # TO CHECK
                # {0: {sigma}_{xy}, 1: {sigma}_{yy}, 2: u_y^f 3 u_x^s  4:u_y^s, 5: p}
                M_98 = np.array([[1, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1],[0,1,0,0,0,0],[0,0,0,0,1,0]])
                self.P, self.Q = M_98@self.P, self.Q@LA.inv(M_98)
            elif self.typ == "Biot01":
                # {0: {sigma}_{xy}^t, 1: {sigma}_{yy}^t, 2: w_y 3 u_x^s  4:u_y^s, 5: p}
                M_01 = np.array([[1, 0, 0, 0, 0, 0],[0, 0, 0, 1, -1, 0], [0, -1, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1],[0,1,0,0,0,0],[0,0,0,0,1,0]])
                self.P, self.Q = M_01@self.P, self.Q@LA.inv(M_01)


            
        elif self.medium.MEDIUM_TYPE == "elastic":
            self.n_w = 2
            self.P, self.lam = elastic_waves_TMM(self.medium, np.array([0]))
            self.Q = LA.inv(self.P)
        else:
            pass
        self.P_plus, self.P_minus = self.P[:,:self.n_w], self.P[:,self.n_w:]
        self.Q_plus, self.Q_minus = self.Q[:self.n_w,:], self.Q[self.n_w:,:]
        self.lam_plus, self.lam_minus = self.lam[:self.n_w], self.lam[self.n_w:]
