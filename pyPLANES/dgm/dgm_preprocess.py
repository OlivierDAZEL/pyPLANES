#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# preprocess.py
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
from termcolor import colored
import itertools
import time, timeit

import numpy as np
import numpy.linalg as LA

from mediapack import Air, from_yaml
from pymls import Layer

# from pyPLANES.fem.elements_reference import Ka, KaPw, Kt
# from pyPLANES.fem.utils_fem import normal_to_element
# from pyPLANES.fem.elements_fem import FemEdge, FemFace
# from pyPLANES.core.mesh import NeighbourElement
# from pyPLANES.fem.fem_entities_surfacic import *
# from pyPLANES.fem.fem_entities_volumic import *
# from pyPLANES.fem.fem_entities_pw import *

from pyPLANES.utils.geometry import getOverlap, local_abscissa
from pyPLANES.core.multilayer import MultiLayer

from pyPLANES.fem.dofs import activate_dofs, affect_dofs_to_elements
from pyPLANES.fem.checkup_of_the_model import checkup_of_the_model

from pyPLANES.dgm.dgm_entities_surfacic import ImposedDisplacementDgm, RigidWallDgm
from pyPLANES.dgm.dgm_entities_volumic import FluidDgm

from pyPLANES.dgm.dgm_edges import ImposedDisplacementDgmEdge, RigidWallDgmEdge
from pyPLANES.dgm.dgm_edges import InternalFluidDgmEdge

def dgm_preprocess(self):
    if self.verbose:
        print("%%%%%%%%%%%% DGM Preprocess of PLANES  %%%%%%%%%%%%%%%%%")


    # Creation of edges and faces
    create_dgm_edges_lists(self)
    # Identification of active dofs and their numbering
    if self.verbose:
        print("Activation of dofs based on physical media" + "\t"*4 + "["+ colored("OK", "green")  +"]")
    activate_dofs(self)
    if self.verbose:
        print("Affectation of dofs to the elements" + "\t"*5 + "["+ colored("OK", "green")  +"]")
    affect_dofs_to_elements(self)
    if self.verbose:
        print("Identification of periodic dofs" + "\t"*6 + "["+ colored("OK", "green")  +"]")
    # periodic_dofs_identification(self)
    if self.verbose:
        print("Checkup of the model")
    checkup_of_the_model(self)
    self.duration_importation = time.time() - self.start_time
    self.info_file.write("Duration of importation ={} s\n".format(self.duration_importation))
    if self.verbose:
        print("Creation of elementary matrices in the elements")
    for _ent in self.fem_entities:
        for _el in _ent.elements:
            _ent.elementary_matrices(_el)
    self.duration_assembly = time.time() - self.start_time - self.duration_importation
    self.info_file.write("Duration of assembly ={} s\n".format(self.duration_assembly))   


def create_dgm_edges_lists(self):
    ''' Create the list of edges, faces and bubbles of the Model '''
    existing_edges = [] # List of element vertices for redundancy check
    for _ent in self.dgm_entities:
        if _ent.dim ==1:
            if isinstance(_ent, ImposedDisplacementDgm):
                typ_edge = ImposedDisplacementDgmEdge
            elif isinstance (_ent, RigidWallDgm):
                typ_edge = RigidWallDgmEdge
            for _el in _ent.elements:
                element_vertices = [_el.vertices[0], _el.vertices[1]]
                update_dgm_edges(self, existing_edges, element_vertices, typ_edge)
        elif _ent.dim ==2:
            if _el.typ == 1:
                update_dgm_edges(self, _el, existing_edges, element_vertices, _ent)
            elif _el.typ == 2:
                # loop on the the edges of the triangle
                for ii in range(3):
                    element_vertices = [_el.vertices[ii], _el.vertices[(ii+1)%3]]
                    update_dgm_edges(self, _el, existing_edges, element_vertices)



def update_dgm_edges(self, existing_edges, element_vertices, _el=None):
    element_vertices_tag = [element_vertices[0].tag, element_vertices[1].tag]
    element_vertices_tag_sorted = sorted(element_vertices_tag)
    if element_vertices_tag_sorted in existing_edges: # If the edge already exists
        # What is its index ?
        index_edge = existing_edges.index(element_vertices_tag_sorted)
        # add the element to the edge's edge list
        self.edges[index_edge].elements.append(_el)
        # add the edge to the element's edge list
        _el.edges.append(self.edges[index_edge])
    else:  # The edge does not exist already exists
        # Insertion the new edge in the existing_edges list
        existing_edges.append(element_vertices_tag_sorted)
        # Creation of the new edge
        new_edge = DgmEdge(self.nb_edges, element_vertices, _el)
        self.edges.append(new_edge)
        _el.edges.append(new_edge)
        self.nb_edges +=1

    for _e in self.edges:
        print(_e)
    eza

