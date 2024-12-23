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

from mediapack import Air, Fluid
from mediapack.medium import Medium
from pyPLANES.utils.io import load_material

from pyPLANES.pw.general.general_layers import *
from pyPLANES.pw.general.general_interfaces import *

class GeneralMultiLayer():
    """
    Multilayer structure
    """
    def __init__(self, **kwargs):
        ml = kwargs.get("ml", None) 
        self.method = kwargs.get("method","Global Method")
        self.method_TM = kwargs.get("method_TM","diag")
        self.material_database = kwargs.get("material_database", None)
        # Creation of the list of layers
        self.layers = []
        self.nb_waves = kwargs.get("nb_waves", 1)
        self.kx, self.kz = None, None
        _x = 0 
        for _l in ml:
            if isinstance(_l,(list,tuple)):
                mat,d = _l
                assert isinstance(mat,str) & np.isscalar(d)
                if self.material_database is None:
                    load_mat = load_material(mat)
                else:
                    load_mat = load_material(self.material_database, mat)
                if isinstance(load_mat, Medium):
                    mat = load_mat
                    if mat.MODEL in ["fluid", "eqf"]:
                        self.layers.append(FluidLayer_3D(mat, d, x_0=_x, method_TM=self.method_TM))
                    elif mat.MODEL == "pem":
                        self.layers.append(PemLayer_3D(mat, d, x_0=_x, method_TM=self.method_TM))
                    elif mat.MODEL == "elastic":
                        self.layers.append(ElasticLayer_3D(mat, d, x_0=_x, method_TM=self.method_TM))
            else:
                raise NameError("Invalid Material")
            _x += d
        # Creation of the list of interfaces
        self.interfaces = []
        for i_l, _layer in enumerate(self.layers[:-1]):
            if _layer.medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                if self.layers[i_l+1].medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                    self.interfaces.append(FluidFluidInterface_3D(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(FluidPemInterface_3D(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(FluidElasticInterface_3D(_layer,self.layers[i_l+1]))
            elif _layer.medium.MEDIUM_TYPE in  ["pem"]:
                if self.layers[i_l+1].medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                    self.interfaces.append(PemFluidInterface_3D(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(PemPemInterface_3D(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(PemElasticInterface_3D(_layer,self.layers[i_l+1]))
            elif _layer.medium.MEDIUM_TYPE in  ["elastic"]:
                if self.layers[i_l+1].medium.MEDIUM_TYPE in  ["fluid", "eqf"]:
                    self.interfaces.append(ElasticFluidInterface_3D(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "pem":
                    self.interfaces.append(ElasticPemInterface_3D(_layer,self.layers[i_l+1]))
                elif self.layers[i_l+1].medium.MEDIUM_TYPE == "elastic":
                    self.interfaces.append(ElasticElasticInterface_3D(_layer,self.layers[i_l+1]))

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
            self.interfaces.append(SemiInfinite_3D(self.layers[-1]))
        else: # Case of a rigid backing 
            if self.layers[-1].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
                self.interfaces.append(FluidRigidBacking_3D(self.layers[-1], None, self.method))
            elif self.layers[-1].medium.MEDIUM_TYPE == "pem":
                self.interfaces.append(PemBacking_3D(self.layers[-1]))
            elif self.layers[-1].medium.MEDIUM_TYPE == "elastic":
                self.interfaces.append(ElasticBacking_3D(self.layers[-1]))
        
        incident_layer = FluidLayer_3D(Fluid(c=Air().c,rho=Air().rho), 1.e-2, x_0=-1.e-2)
        if self.layers[0].medium.MEDIUM_TYPE in ["fluid", "eqf"]:
            self.interfaces.insert(0,FluidFluidInterface_3D(incident_layer ,self.layers[0]))
        elif self.layers[0].medium.MEDIUM_TYPE == "pem":
            self.interfaces.insert(0,FluidPemInterface_3D(incident_layer, self.layers[0]))
        elif self.layers[0].medium.MEDIUM_TYPE == "elastic":
            self.interfaces.insert(0,FluidElasticInterface_3D(incident_layer, self.layers[0]))
            # # Addition of a fictious Air-Layer for the interface.
            # self.interfaces[0].layers[0] = incident_layer
        if self.method == "Global Method":
            self.layers.insert(0, incident_layer)
            self.nb_PW = 0
            for _layer in self.layers:
                _layer.dofs = self.nb_PW+np.arange(2*_layer.nb_waves_in_medium*self.nb_waves)
                self.nb_PW += 2*_layer.nb_waves_in_medium*self.nb_waves
            if isinstance(self.interfaces[-1], SemiInfinite_3D):
                self.interfaces[-1].dofs = slice(self.nb_PW, self.nb_PW+self.nb_waves)
                self.nb_PW += self.nb_waves

    def update_frequency(self, omega, kx, kz):
        self.kx = kx
        self.kz = kz
        for _l in self.layers:
            _l.update_frequency(omega, self.kx, self.kz)
        for _i in self.interfaces:
            _i.update_frequency(omega, self.kx, self.kz)
        
