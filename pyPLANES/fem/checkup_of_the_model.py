#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# preprocess.py
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

import itertools
import time, timeit

import numpy as np
import numpy.linalg as LA

from mediapack import Air, from_yaml
from pymls import Layer
from pyPLANES.fem.utils_fem import normal_to_element
from pyPLANES.fem.elements_fem import FemEdge, FemFace
from pyPLANES.core.mesh import NeighbourElement
from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *
from pyPLANES.fem.fem_entities_pw import *

from pyPLANES.utils.geometry import getOverlap, local_abscissa
# from pyPLANES.core.multilayer import MultiLayer

def checkup_of_the_model(self):
    ''' This function checks if the model is correct and adapt it if not '''
    check_neighbours_1D_entity(self)
    check_IFS(self)
    check_pem_formulation(self)
    check_interfaces(self)
    if len(self.pwfem_entities) !=0:
        check_pwfem(self)

    entities = [isinstance(e, (ImposedDisplacementFem, RobinAirFem)) for e in self.entities]
    if any(entities):
        _ent= self.entities[entities.index(True)]
        self.list_vr = set(itertools.chain(*[_el.vertices for _el in _ent.elements]))
    else:
        self.list_vr = False


def check_pwfem(self):
    for _e in self.pwfem_entities:
        if isinstance(_e, PwFem):
            for s in _e.up:
                if isinstance(s, (Air, FluidFem)):
                    _e.nb_dof_per_node = 1
                    _e.medium = s.mat
                    _e.primal = [1]
                    _e.dual = [0]
                elif (isinstance(s, ElasticFem)):
                    _e.nb_dof_per_node = 2
                    _e.medium = s.mat
                    _e.typ = "elastic"
                    _e.primal = [3, 1]
                    _e.dual = [0, 2]
                elif isinstance(s, PemFem):
                    if s.formulation98:
                        _e.nb_dof_per_node = 3
                        _e.medium = s.mat
                        _e.typ = "Biot98"
                        _e.primal = [5, 1, 4]
                        _e.dual = [0, 3, 2]
                    else:
                        _e.nb_dof_per_node = 3
                        _e.medium = s.mat
                        _e.typ = "Biot01"
                        _e.primal = [5, 1, 4]
                        _e.dual = [0, 3, 2]
    

    self.medium[0] = self.pwfem_entities[0].medium
    self.medium[1] = self.pwfem_entities[1].medium
    if self.medium[0] == self.medium[1]:
        if self.medium[0].MEDIUM_TYPE in ["fluid", "eqf"]:
            self.nb_waves_in_medium = 1 
        elif self.medium[0].MEDIUM_TYPE == "pem":
            self.nb_waves_in_medium = 3
        elif self.medium[0].MEDIUM_TYPE == "elastic":
            self.nb_waves_in_medium = 2
    else: 
        raise NameError("self.medium[0] != self.medium[1]")

def check_neighbours_1D_entity(self):
    pass
    # for _e in self.entities:
    #     if _e.dim == 1: # 1D entities
    #         if len(_e.neighbouring_surfaces) > 1:
    #             if isinstance(_e, (PwFem, RigidWallFem, PeriodicityFem)):
    #                 raise ValueError("Error in checkup_of_the_model: 1D entity is linked to more than one surface")    

def check_IFS(self):
    pass
    # unfinished = True
    # while unfinished:
    #     unfinished = False
    #     for _e in self.entities:
    #         if _e.dim == 1: # 1D entities
    #             if len(_e.neighbouring_surfaces) > 1:
    #                 if isinstance(_e, FluidStructureFem):
    #                     if len(_e.neighbouring_surfaces) != 2:
    #                         raise NameError("For FluidStructureFem, the number of neighbours should be 2")
    #                     else:
    #                         if isinstance(_e.neighbouring_surfaces[0], FluidFem) and isinstance(_e.neighbouring_surfaces[1], ElasticFem):
    #                             _e.fluid_neighbour, _e.struc_neighbour = _e.neighbouring_surfaces[0], _e.neighbouring_surfaces[1]
    #                         elif isinstance(_e.neighbouring_surfaces[0], ElasticFem) and isinstance(_e.neighbouring_surfaces[1], FluidFem):
    #                             _e.fluid_neighbour, _e.struc_neighbour = _e.neighbouring_surfaces[1], _e.neighbouring_surfaces[0]
    #                         elif isinstance(_e.neighbouring_surfaces[0], FluidFem) and isinstance(_e.neighbouring_surfaces[1], PemFem):
    #                             if _e.neighbouring_surfaces[1].formulation98 == True:
    #                                 raise NameError("IFS Fluid 98 Error")
    #                             else:
    #                                 _e.fluid_neighbour, _e.struc_neighbour = _e.neighbouring_surfaces[0], _e.neighbouring_surfaces[1]   
    #                         elif isinstance(_e.neighbouring_surfaces[0], PemFem) and isinstance(_e.neighbouring_surfaces[1], FluidFem):
    #                             if _e.neighbouring_surfaces[0].formulation98 == True:
    #                                 raise NameError("IFS Fluid 98 Error")
    #                             else:
    #                                 _e.fluid_neighbour, _e.struc_neighbour = _e.neighbouring_surfaces[1], _e.neighbouring_surfaces[0]
    #                         else:
    #                             raise NameError("FluidStructureFem does not relate a fluid and elastic struture")

    #                         for _elem in _e.elements:
    #                             vert = [_elem.vertices[0].tag, _elem.vertices[1].tag] # Vertices of the FSI element
    #                             # Determination of the neighbouring element in neighbouring_surfaces[0]
    #                             _iter = iter(_e.fluid_neighbour.elements)
    #                             while True:
    #                                 _el = next(_iter)
    #                                 vert_2D = [_el.vertices[0].tag, _el.vertices[1].tag, _el.vertices[2].tag]
    #                                 _ = len(set(vert).intersection(vert_2D)) # Number of common vertices
    #                                 if _ == 2: # Case of two common vertices
    #                                         _elem.elem2d = _el
    #                                         _elem.normal_fluid = normal_to_element(_elem, _el)
    #                                         _elem.normal_struc = -_elem.normal_fluid
    #                                         break

def check_pem_formulation(self):
    for _e in self.entities:
        if isinstance(_e, PemFem):
            _e.formulation98 = True
            _e.formulation98 = False


def check_interfaces(self):
    # Check that the number of interfaces go by pairs
    list_interfaces = [_ent for _ent in self.fem_entities if isinstance(_ent, InterfaceFem)]
    name_interfaces = [_ent.ml for _ent in list_interfaces]
    n_interface = len(list_interfaces)
    if  n_interface%2 == 1:
        raise ValueError("Error in check model: Number of interfaces is odd")
    else:
        while n_interface != 0:
            # Current minus interface
            _int_minus = list_interfaces[0]
            # Determination of the corresponding plus interface
            _index = name_interfaces[1:].index(_int_minus.ml)+1
            _int_plus = list_interfaces[_index]
            # Updates of list and name of interfaces not to consider these two
            del list_interfaces[_index]
            del list_interfaces[0]
            del name_interfaces[_index]
            del name_interfaces[0]
            # For bilateral communication between interfaces
            _int_minus.neighbour = _int_plus
            _int_plus.neighbour = _int_minus
            n_interface -= 2
            # Check that the side chars are correctly
            if _int_minus.side == "+":
                if _int_plus.side == "-":
                    _int_minus, _int_plus = _int_plus, _int_minus
                else:
                    raise ValueError("_int_minus.side = + and _int_plus.side != - ")
            elif _int_minus.side == "-":
                if _int_plus.side != "+":
                    raise ValueError("_int_minus.side = - and _int_plus.side != + ")
            else:
                raise ValueError("_int_minus.side is neither + or - ")
            # Affectation of the bounding nodes to the two interfaces
            _int_minus.nodes = _int_minus.bounding_points.copy()
            _int_plus.nodes = _int_plus.bounding_points.copy()

            # Test if the nodes of minus and plus interface coincide
            if LA.norm(_int_minus.nodes[0].coord-_int_plus.nodes[0].coord) > LA.norm(_int_minus.nodes[1].coord-_int_plus.nodes[0].coord):
               # _int_minus.nodes[1] is closer of _int_plus.nodes[0] than _int_minus.nodes[0]
               _int_plus.nodes.reverse()

            _int_minus.delta = _int_plus.nodes[0].coord - _int_minus.nodes[0].coord
            _int_plus.delta = _int_minus.nodes[1].coord - _int_plus.nodes[1].coord
            if not(np.allclose(_int_minus.delta, -_int_plus.delta)):
                raise ValueError(" Error on delta ")
            #Determination for each element of the local abscissa in the interface coordinate system
            for _elem in _int_minus.elements:
                _elem.neighbours = [] # To be filled later
                _elem.delta = _int_minus.delta
                s_node_0 = local_abscissa(_int_minus.nodes[0].coord, _int_minus.nodes[1].coord, _elem.vertices[0].coord)
                s_node_1 = local_abscissa(_int_minus.nodes[0].coord, _int_minus.nodes[1].coord, _elem.vertices[1].coord)
                _elem.s_interface = [s_node_0, s_node_1]
            for _elem in _int_plus.elements:
                _elem.neighbours = [] # To be filled later
                _elem.delta = _int_plus.delta
                s_node_0 = local_abscissa(_int_plus.nodes[0].coord, _int_plus.nodes[1].coord, _elem.vertices[0].coord)
                s_node_1 = local_abscissa(_int_plus.nodes[0].coord, _int_plus.nodes[1].coord, _elem.vertices[1].coord)
                _elem.s_interface = [s_node_0, s_node_1]
            # Determination of neighbours
            for _elem_minus in _int_minus.elements:
                for _elem_plus in _int_plus.elements:
                    if getOverlap(_elem_minus.s_interface, _elem_plus.s_interface) != 0:
                       # Coordinates
                        s_0_minus = local_abscissa(np.array(_elem_minus.vertices[0].coord), np.array(_elem_minus.vertices[1].coord), np.array(_elem_plus.vertices[0].coord)-_int_minus.delta)
                        s_1_minus = local_abscissa(np.array(_elem_minus.vertices[0].coord), np.array(_elem_minus.vertices[1].coord), np.array(_elem_plus.vertices[1].coord)-_int_minus.delta)

                        s_0_minus = min(max(s_0_minus, 0.), 1.)
                        s_1_minus = min(max(s_1_minus, 0.), 1.)

                        s_0_plus = local_abscissa(np.array(_elem_plus.vertices[0].coord), np.array(_elem_plus.vertices[1].coord), np.array(_elem_minus.vertices[0].coord)+_int_minus.delta)
                        s_1_plus = local_abscissa(np.array(_elem_plus.vertices[0].coord), np.array(_elem_plus.vertices[1].coord), np.array(_elem_minus.vertices[1].coord)+_int_minus.delta)

                        s_0_plus = min(max(s_0_plus, 0.), 1.)
                        s_1_plus = min(max(s_1_plus, 0.), 1.)

                        direction = (np.array(_elem_minus.vertices[1].coord)-np.array(_elem_minus.vertices[0].coord)).dot(np.array(_elem_plus.vertices[1].coord)-np.array(_elem_plus.vertices[0].coord))
                        if direction > 0:
                            _elem_minus.neighbours.append(NeighbourElement(_elem_plus, s_0_minus, s_1_minus, s_0_plus, s_1_plus))
                            _elem_plus.neighbours.append(NeighbourElement(_elem_minus, s_0_plus, s_1_plus, s_0_minus, s_1_minus))
                        else:
                            _elem_minus.neighbours.append(NeighbourElement(_elem_plus, s_0_minus, s_1_minus, s_1_plus, s_0_plus))
                            _elem_plus.neighbours.append(NeighbourElement(_elem_minus, s_0_plus, s_1_plus, s_1_minus, s_0_minus))
            # Afffectation of the ml to the - interface
            _int_minus.ml = MultiLayer(ml=self.interface_ml[_int_minus.ml_name],incident=_int_minus.neighbouring_surfaces[0].mat,termination=_int_plus.neighbouring_surfaces[0].mat)
            # Deletion of the + interface from the model_entities
            self.model_entities.remove(_int_plus)