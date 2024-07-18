#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# quadrature_1d.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@univ-lemans.fr>

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
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.interpolate import lagrange
from scipy import integrate
from .quadrature_schemes import gauss_kronrod


class ReferenceScheme():
    def __init__(self, typ, **kwargs):

        self.typ = typ
        if self.typ == "CC":
            self.order = kwargs.get("order",2)
            self.xi_c = np.cos([np.pi*k/(self.order) for k in range(self.order+1)])[::-1]
            self.xi_r = np.cos([np.pi*k/(2*self.order) for k in range(2*self.order+1)])[::-1]
            self.w_r = self.w_c = None
            xi_c_interval =  self.xi_c
            self.indices_c = np.array([i for i in range(0, 2*self.order+1, 2)])

        elif self.typ == "GK":
            self.order = kwargs.get("order",2)
            sc = gauss_kronrod[f"{order}"]
            # Coarse and refined nodes and weights
            self.xi_c, self.w_c = np.array(sc["xi_c"]), np.array(sc["w_c"])
            self.xi_r, self.w_r = np.array(sc["xi_r"]), np.array(sc["w_r"])
            xi_c_interval = np.hstack([self.a, self.xi_c, self.b])
            self.indices_c = np.array([i for i in range(0, 2*self.order+1, 2)])

        elif self.typ == "PV":
            self.x = kwargs.get("x")
            self.f = kwargs.get("f")
            a, b = self.x[0], self.x[-1]
            self.xi_r = -1 + 2*(self.x-a)/(b-a)
            self.xi_c = self.xi_r[::2]
            xi_c_interval =  self.xi_c
            self.indices_c = np.array([i for i in range(0, len(self.xi_r), 2)])
        else:
            print(f"Unknown scheme {typ}")
            raise NotImplementedError()

        self.n_c, self.n_r = len(self.xi_c), len(self.xi_r)

        # Powers (for polynomial evaluation) at nodes 
        self.xi_powers = [np.array([xi**i for i in range(self.n_r+1)]) for xi in xi_c_interval]
        
        self.xia_powers = self.xi_powers[0]
        self.xib_powers = self.xi_powers[-1]


        self.matrix_poly_c = np.linalg.inv(np.vander(self.xi_c,increasing=True))
        self.matrix_poly_r = np.linalg.inv(np.vander(self.xi_r,increasing=True))

        ## Coarse scheme
        # Power integration diagonal matrix 
        d_c = np.vstack([np.zeros((1,self.n_c)), np.diag([1/i for i in range(1,self.n_c+1)])])
        # Coefficients of the antiderivative polynomial 
        self.matrix_Poly_c = np.dot(d_c, self.matrix_poly_c)

        ## Refined scheme
        # Power integration diagonal matrix 
        d_r = np.vstack([np.zeros((1,self.n_r)), np.diag([1/i for i in range(1, self.n_r+1)])])
        # Coefficients of the antiderivative polynomial 
        self.matrix_Poly_r = np.dot(d_r, self.matrix_poly_r)
        
        
def ReferenceSchemes(typ):
    if typ == "CC":
        d =dict()
        for i in range(1, 10):
            d[f"{i}"] = ReferenceScheme(typ, order=i)
    else:
        print(f"Unknown scheme {typ}")
        raise NotImplementedError()
    return d

class PreviousValues(ReferenceScheme):
    def __init__(self, x, f):
        self.x = x
        self.f = f
        self.typ = "PV"