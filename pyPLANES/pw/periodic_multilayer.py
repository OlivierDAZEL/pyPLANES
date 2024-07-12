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
from mediapack.medium import Medium
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
    def __init__(self, ml, method_TM="JAP", **kwargs):
        # Creation of the list of layers
        
        self.method = kwargs.get("method","Global Method")
        self.method_TM = method_TM
        self.layers = []
        self.interfaces = []
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.k_x = kwargs.get("k_x", None)
        self.order = kwargs.get("order", 2)
        self.plot = kwargs.get("plot", [False*6])
        self.condensation = kwargs.get("condensation", True)
        self.period = False # If period is false: homogeneous layer
        self.nb_waves = None
        _x = 0
        periodic_layer = False
        
        for _l in ml:
            if isinstance(_l, PwGeneric):
                if _l.method_TM != self.method_TM:
                    _l.method_TM = self.method_TM
                _l.x_0 = _x
                d = _l.d
                self.layers.append(_l)
            elif isinstance(_l,(list,tuple)):
                mat,d = _l
                assert isinstance(mat,str)
                load_mat = load_material(mat)
                if isinstance(load_mat, Medium):
                    mat = load_mat
                    if mat.MEDIUM_TYPE in ["fluid", "eqf"]:
                        self.layers.append(FluidLayer(mat, d, x_0=_x, method_TM=self.method_TM))
                    if mat.MEDIUM_TYPE == "pem":
                        self.layers.append(PemLayer(mat, d, x_0=_x, method_TM=self.method_TM))
                    if mat.MEDIUM_TYPE == "elastic":
                        self.layers.append(ElasticLayer(mat, d, x_0=_x, method_TM=self.method_TM))
                    _x += d
                elif os.path.isfile("msh/" + _l[0] + ".msh"):
                    if periodic_layer:
                        raise NameError ("Only a periodic layer is supported"
                                         )
                    else: 
                        self.layers.append(PeriodicLayer(name_mesh=_l[0], _x=_x, theta_d= self.theta_d, verbose=self.verbose, order=self.order, plot=self.plot, condensation=self.condensation))
                        self.result.n_dof = self.layers[-1].nb_dof_master
                        self.period = self.layers[-1].period
                        _x += self.layers[-1].d
                        periodic_layer = True
            else:
                raise NameError ("layer {} is neither a mediapack material nor a msh file ".format(_l[0]))
        if self.period == False:
            self.period = 1.

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

        # Addition of a fictious Air-Layer for the incident interface.
        incident_layer=FluidLayer(Fluid(c=Air().c,rho=Air().rho), 1.e-2, x_0=-1.e-2)
        if medium_type in ["fluid", "eqf"]:
            self.interfaces.insert(0,FluidFluidInterface(incident_layer ,self.layers[0]))
        elif medium_type == "pem":
            self.interfaces.insert(0,FluidPemInterface(incident_layer, self.layers[0]))
        elif medium_type == "elastic":
            self.interfaces.insert(0,FluidElasticInterface(incident_layer, self.layers[0]))
        if self.method == "Global Method":
            self.layers.insert(0, incident_layer)
            self.nb_waves = 1+2*self.nb_bloch_waves
            self.nb_PW =0
            for _layer in self.layers:
                if isinstance(_layer, PeriodicLayer):
                    _layer.dofs_bottom = self.nb_PW+np.arange(2*_layer.nb_waves_in_medium*self.nb_waves)
                    self.nb_PW += 2*_layer.nb_waves_in_medium*self.nb_waves
                    _layer.dofs_top = self.nb_PW+np.arange(2*_layer.nb_waves_in_medium*self.nb_waves)
                    self.nb_PW += 2*_layer.nb_waves_in_medium*self.nb_waves
                else:
                    _layer.dofs = self.nb_PW+np.arange(2*_layer.nb_waves_in_medium*self.nb_waves)
                    self.nb_PW += 2*_layer.nb_waves_in_medium*self.nb_waves
            if isinstance(self.interfaces[-1], SemiInfinite):
                self.interfaces[-1].dofs = slice(self.nb_PW, self.nb_PW+self.nb_waves)
                self.nb_PW += self.nb_waves
                

        # for i, _l in enumerate(self.layers):
        #     if isinstance(_l, PeriodicLayer):
        #         print(f"dof({i})={_l.dofs_bottom} // {_l.dofs_top}")
        #     else:
        #         print(f"dof({i})={_l.dofs}")

    def update_frequency(self, omega, kx):
        for _l in self.layers:
            _l.update_frequency(omega, kx)
        for _i in self.interfaces:
            _i.update_frequency(omega, kx)
