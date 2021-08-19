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
from pyPLANES.pw.pw_layers import PwLayer

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
        self.verbose = kwargs.get("verbose", False)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.order = kwargs.get("order", 2)
        self.plot = kwargs.get("plot", [False*6])
        self.condensation = kwargs.get("condensation", True)
        self.period = False # If period is false: homogeneous layer
        _x = 0
        for _l in ml:
            mat = load_material(_l[0])
            # print(_l[0])
            if mat is not None:
                d = _l[1]
                if mat.MEDIUM_TYPE in ["fluid", "eqf"]:
                    self.layers.append(FluidLayer(mat, d, _x))
                if mat.MEDIUM_TYPE == "pem":
                    self.layers.append(PemLayer(mat, d, _x))
                if mat.MEDIUM_TYPE == "elastic":
                    self.layers.append(ElasticLayer(mat, d, _x))
                _x += d
            elif os.path.isfile(_l[0] + ".msh"):
                self.layers.append(PeriodicLayer(name_mesh=_l[0], _x=_x, theta_d= self.theta_d, verbose=self.verbose, order=self.order, plot=self.plot, condensation=self.condensation))
                self.Results["n_dof"] = self.layers[-1].nb_dof_master
                self.period = self.layers[-1].period
                _x += self.layers[-1].d


        # Creation of the list of interfaces
        for i_l, _layer in enumerate(self.layers[:-1]):
            if isinstance(_layer, PwLayer):
                _medium_type_bottom = _layer.medium.MEDIUM_TYPE
            else:
                _medium_type_bottom = _layer.medium[1].MEDIUM_TYPE
            if isinstance(self.layers[i_l+1], PwLayer):
                _medium_type_top = self.layers[i_l+1].medium.MEDIUM_TYPE
            else:
                _medium_type_top = self.layers[i_l+1].medium[1].MEDIUM_TYPE

            if _medium_type_bottom in  ["fluid", "eqf"]:
                if _medium_type_top in  ["fluid", "eqf"]:
                    self.interfaces.append(FluidFluidInterface(_layer, self.layers[i_l+1]))
                elif _medium_type_top == "pem":
                    self.interfaces.append(FluidPemInterface(_layer, self.layers[i_l+1]))
                elif _medium_type_top == "elastic":
                    self.interfaces.append(FluidElasticInterface(_layer, self.layers[i_l+1]))
            elif _medium_type_bottom in  ["pem"]:
                if _medium_type_top in  ["fluid", "eqf"]:
                    self.interfaces.append(PemFluidInterface(_layer,  self.layers[i_l+1]))
                elif _medium_type_top == "pem":
                    self.interfaces.append(PemPemInterface(_layer, self.layers[i_l+1]))
                elif _medium_type_top == "elastic":
                    self.interfaces.append(PemElasticInterface(_layer, self.layers[i_l+1]))
            elif _medium_type_bottom in  ["elastic"]:
                if _medium_type_top in  ["fluid", "eqf"]:
                    self.interfaces.append(ElasticFluidInterface(_layer, self.layers[i_l+1]))
                elif _medium_type_top == "pem":
                    self.interfaces.append(ElasticPemInterface(_layer, self.layers[i_l+1]))
                elif _medium_type_top == "elastic":
                    self.interfaces.append(ElasticElasticInterface(_layer, self.layers[i_l+1]))

    def __str__(self):
        out = "Interface #0\n"
        out += self.interfaces[0].__str__()+"\n"
        for i_l, _layer in enumerate(self.layers):
            out += "Layer #{}".format(i_l)+"\n"
            out += _layer.__str__()+"\n"
            out +="Interface #{}".format(i_l+1)+"\n"
            out += self.interfaces[i_l+1].__str__()+"\n"
        return out 

    def add_excitation_and_termination(self, termination):
        # Interface associated to the termination 
        if termination in ["trans", "transmission","Transmission"]:
            self.interfaces.append(SemiInfinite(self.layers[-1]))
        else: # Case of a rigid backing 
            if isinstance(self.layers[-1].medium, list):
                medium_type = self.layers[-1].medium[1].MEDIUM_TYPE
            else: 
                medium_type = self.layers[-1].medium.MEDIUM_TYPE
            if medium_type in ["fluid", "eqf"]:
                self.interfaces.append(FluidRigidBacking(self.layers[-1]))
            elif medium_type == "pem":
                self.interfaces.append(PemBacking(self.layers[-1]))
            elif medium_type == "elastic":
                self.interfaces.append(ElasticBacking(self.layers[-1]))
        if isinstance(self.layers[0].medium, list):
            medium_type = self.layers[0].medium[0].MEDIUM_TYPE
        else: 
            medium_type = self.layers[0].medium.MEDIUM_TYPE
        if medium_type in ["fluid", "eqf"]:
            self.interfaces.insert(0,FluidFluidInterface(None ,self.layers[0]))
        elif medium_type == "pem":
            self.interfaces.insert(0,FluidPemInterface(None, self.layers[0]))
        elif medium_type == "elastic":
            self.interfaces.insert(0,FluidElasticInterface(None, self.layers[0]))

    def update_frequency(self, omega, kx):
        for _l in self.layers:
            _l.update_frequency(omega, kx)
        for _i in self.interfaces:
            _i.update_frequency(omega, kx)
