#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# elastic.py
#
# This file is part of pymls, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2017
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

from .medium import Medium
from numpy import sqrt


class Elastic(Medium):

    MEDIUM_TYPE = 'elastic'
    MODEL = MEDIUM_TYPE
    EXPECTED_PARAMS = [
        ('E', float),  # Young's modulus
        ('nu', float),  # Poisson ratio
        ('rho', float),  # Density
        ('eta', float),  # loss factor
    ]
    OPT_PARAMS = [
        ('lambda_', complex),
        ('mu', complex),
        ('law', str)
    ]

    def __init__(self, **params):
        self.E = None
        self.rho = None
        self.nu = None
        self.eta = None
        self.lambda_ = None
        self.mu = None
        self.delta_p = None
        self.delta_s = None
        self.law = None

        super().__init__(**params)

    def _compute_missing(self):
        if self.law is None:
            self.law = "structural"
            self.E *= (1+1j*self.eta)
        if self.lambda_ is None:
            self.lambda_ = (self.E*self.nu)/((1+self.nu)*(1-2*self.nu))
        if self.mu is None:
            self.mu = (self.E)/(2*(1+self.nu))

    def update_frequency(self, omega):
        self.omega = omega
        if self.law == "rubber":
            self.E = (self.E.real)+1j*omega*self.eta
            self.lambda_ = (self.E*self.nu)/((1+self.nu)*(1-2*self.nu))
            self.mu = (self.E)/(2*(1+self.nu))
        P_mat = self.lambda_ + 2*self.mu
        self.delta_p = omega*sqrt(self.rho/P_mat)
        self.delta_s = omega*sqrt(self.rho/self.mu)
