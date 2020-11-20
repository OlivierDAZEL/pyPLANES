#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# multilayer.py
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

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from mediapack import from_yaml
from mediapack import Air, PEM, EqFluidJCA, Fluid
import mediapack as mediapack 

# from pyPLANES.utils.io import initialisation_out_files_plain
from pyPLANES.core.calculus import PwCalculus

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *


class MultiLayer():
    """
    Multilayer structure
    """
    def __init__(self, **kwargs):
        ml = kwargs.get("ml")
        # Creation of the list of layers
        self.layers = []
        _x = 0   
        for _l in ml:
            if _l[0] == "Air":
                Air_mat = Air()
                mat = Fluid(c=Air_mat.c,rho=Air_mat.rho)
            else:
                mat = from_yaml(_l[0]+".yaml")
            d = _l[1]
            if mat.MEDIUM_TYPE in ["fluid", "eqf"]:
                self.layers.append(FluidLayer(mat, d, _x))
            if mat.MEDIUM_TYPE == "pem":
                self.layers.append(PemLayer(mat, d, _x))
            if mat.MEDIUM_TYPE == "elastic":
                self.layers.append(ElasticLayer(mat, d, _x))
            _x += d
        # Creation of the list of interfaces     
        self.interfaces = []
        for i_l, _layer in enumerate(self.layers[:-1]):
            if _layer.medium.MEDIUM_TYPE == "fluid":
                if self.layers[i_l+1].medium.MEDIUM_TYPE == "fluid":
                    self.interfaces.append(FluidFluidInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(FluidPemInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(FluidElasticInterface(_layer,self.layers[i_l+1]))

    def __str__(self):
        out = "Interface #0\n"
        out += self.interfaces[0].__str__()+"\n"
        for i_l, _layer in enumerate(self.layers):
            out += "Layer #{}".format(i_l)+"\n"
            out += _layer.__str__()+"\n"
            out +="Interface #{}".format(i_l+1)+"\n"
            out += self.interfaces[i_l+1].__str__()+"\n"
        return out 

    def update_frequency(self, omega, k, kx):
        for _l in self.layers:
            _l.update_frequency(omega, k, kx)
        for _i in self.interfaces:
            _i.update_frequency(omega, k, kx)
