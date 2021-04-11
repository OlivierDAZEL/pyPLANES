#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# multilayer.py
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
import matplotlib.pyplot as plt
import os.path
from mediapack import Air, Fluid

from pyPLANES.utils.io import load_material
from pyPLANES.pw.periodic_layer import PeriodicLayer
# from pyPLANES.core.calculus import PwCalculus

from pyPLANES.pw.pw_layers import *
from pyPLANES.pw.pw_interfaces import *

class PeriodicMultiLayer():
    """
    Periodic Multilayer structure
    """
    def __init__(self, ml, **kwargs):
        # Creation of the list of layers
        self.layers = []
        self.interfaces = []
        self.verbose = kwargs.get("dim", True)
        _x = 0
        for _l in ml:
            mat = load_material(_l[0])
            print(_l[0])
            if mat != None:
                d = _l[1]
                if mat.MEDIUM_TYPE in ["fluid", "eqf"]:
                    self.layers.append(FluidLayer(mat, d, _x))
                if mat.MEDIUM_TYPE == "pem":
                    self.layers.append(PemLayer(mat, d, _x))
                if mat.MEDIUM_TYPE == "elastic":
                    self.layers.append(ElasticLayer(mat, d, _x))
                _x += d
            
            elif os.path.isfile(_l[0] + ".msh"):
                self.layers.append(PeriodicLayer(name_mesh=_l[0], verbose=self.verbose))

        # Creation of the list of interfaces
        for i_l, _layer in enumerate(self.layers[:-1]):
            if _layer.medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                if self.layers[i_l+1].medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                    self.interfaces.append(FluidFluidInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(FluidPemInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(FluidElasticInterface(_layer,self.layers[i_l+1]))
            elif _layer.medium.MEDIUM_TYPE in  ["pem"]:
                if self.layers[i_l+1].medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                    self.interfaces.append(PemFluidInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(PemPemInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(PemElasticInterface(_layer,self.layers[i_l+1]))
            elif _layer.medium.MEDIUM_TYPE in  ["elastic"]:
                if self.layers[i_l+1].medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                    self.interfaces.append(ElasticFluidInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(ElasticPemInterface(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(ElasticElasticInterface(_layer,self.layers[i_l+1]))

    def __str__(self):
        out = "Interface #0\n"
        out += self.interfaces[0].__str__()+"\n"
        for i_l, _layer in enumerate(self.layers):
            out += "Layer #{}".format(i_l)+"\n"
            out += _layer.__str__()+"\n"
            out +="Interface #{}".format(i_l+1)+"\n"
            out += self.interfaces[i_l+1].__str__()+"\n"
        return out 

    def add_excitation_and_termination(self, method, termination):
        # Interface associated to the termination 
        if termination in ["trans", "transmission","Transmission"]:
            self.interfaces.append(SemiInfinite(self.layers[-1]))
        else: # Case of a rigid backing 
            if isinstance(self.layers[-1].medium, list):
                self.interfaces.append(FluidRigidBacking(self.layers[-1]))
            else:
                if self.layers[-1].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                    self.interfaces.append(FluidRigidBacking(self.layers[-1]))
                elif self.layers[-1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(PemBacking(self.layers[-1]))
                elif self.layers[-1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(ElasticBacking(self.layers[-1]))
        if method == "Recursive Method":
            if isinstance(self.layers[-1].medium, list):
                self.interfaces.insert(0,FluidFluidInterface(None ,self.layers[0]))
            else:
                if self.layers[0].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                    self.interfaces.insert(0,FluidFluidInterface(None ,self.layers[0]))
                elif self.layers[0].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.insert(0,FluidPemInterface(None, self.layers[0]))
                elif self.layers[0].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.insert(0,FluidElasticInterface(None, self.layers[0]))
        else: # Global method
            Air_mat = Air()
            mat = Fluid(c=Air_mat.c,rho=Air_mat.rho)
            self.layers.insert(0,FluidLayer(mat, 1.e-2, -1.e-2))
            if self.layers[1].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                self.interfaces.insert(0,FluidFluidInterface(self.layers[0] ,self.layers[1]))
            elif self.layers[1].medium.MEDIUM_TYPE == "pem":
                self.interfaces.insert(0,FluidPemInterface(self.layers[0], self.layers[1]))
            elif self.layers[1].medium.MEDIUM_TYPE == "elastic":
                self.interfaces.insert(0,FluidElasticInterface(self.layers[0],self.layers[1]))
            # Count of the number of plane waves for the global method
            self.nb_PW = 0
            for _layer in self.layers:
                if _layer.medium.MODEL in ["fluid", "eqf"]:
                    _layer.dofs = self.nb_PW+np.arange(2)
                    self.nb_PW += 2
                elif _layer.medium.MODEL == "pem":
                    _layer.dofs = self.nb_PW+np.arange(6)
                    self.nb_PW += 6
                elif _layer.medium.MODEL == "elastic":
                    _layer.dofs = self.nb_PW+np.arange(4)
                    self.nb_PW += 4
            if isinstance(self.interfaces[-1], SemiInfinite):
                self.nb_PW += 1 

    def update_frequency(self, omega, kx):
        for _l in self.layers:
            _l.update_frequency(omega, kx)
        for _i in self.interfaces:
            _i.update_frequency(omega, kx)
