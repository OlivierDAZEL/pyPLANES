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
from termcolor import colored
import itertools
import time, timeit

import numpy as np
import numpy.linalg as LA

from mediapack import Air, from_yaml
from pymls import Layer

from pyPLANES.fem.elements_reference import Ka, KaPw, Kt
from pyPLANES.fem.utils_fem import normal_to_element
from pyPLANES.fem.elements_fem import FemEdge, FemFace
from pyPLANES.core.mesh import NeighbourElement
from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *
from pyPLANES.fem.fem_entities_pw import *

from pyPLANES.utils.geometry import getOverlap, local_abscissa
from pyPLANES.core.multilayer import MultiLayer

from pyPLANES.fem.dofs import activate_dofs, affect_dofs_to_elements
from pyPLANES.fem.checkup_of_the_model import checkup_of_the_model


def fem_preprocess(self):
    if self.verbose:
        print("%%%%%%%%%%%% FEM Preprocess of PLANES  %%%%%%%%%%%%%%%%%")
    # Assign reference elements and order to elements 
    assign_reference_element(self)

    # Creation of edges and faces
    create_vertices_edges_faces_bubbles_lists(self)
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


# def assign_order_to_elements

def assign_reference_element(self):
    for _ent in self.fem_entities:
        for _el in _ent.elements:
            # For FEM entities reference element key is the type
            reference_element_key = _el.typ
            if reference_element_key not in self.reference_elements.keys():
                self.reference_elements[reference_element_key] = reference_element(reference_element_key, self.order)
            _el.reference_element = self.reference_elements[reference_element_key] 
    for _ent in self.pwfem_entities:
        for _el in _ent.elements:
            # For PWFEM entities reference element key is a tuple with type and PW string
            reference_element_key = (_el.typ, "PW")
            if reference_element_key not in self.reference_elements.keys():
                self.reference_elements[reference_element_key] = reference_element(reference_element_key, self.order)
            _el.reference_element = self.reference_elements[reference_element_key]

def reference_element(key, order):
    if isinstance(key, int):
        if key == 2:
            out = Kt(order, 2*order)
        elif key == 1:
            out = Ka(order, 2*order)
    else:
        if key[0] == 2:
            out = Kt(order, 2*order)
        elif key[0] == 1:
            out = KaPw(order, 3*order)

    return out

 


def create_vertices_edges_faces_bubbles_lists(self):
    ''' Create the list of edges, faces and bubbles of the Model '''
    existing_edges = [] # List of element vertices for redundancy check
    for __, _el in enumerate(self.elements[1:]):
        if _el.typ == 1:
            element_vertices = [_el.vertices[0], _el.vertices[1]]
            update_edges(self, _el, existing_edges, element_vertices)
        if _el.typ == 2:
            # Edges
            for ii in range(3):
                element_vertices = [_el.vertices[ii], _el.vertices[(ii+1)%3]]
                update_edges(self, _el, existing_edges, element_vertices)
            # Faces
            element_vertices = [_el.vertices[0], _el.vertices[1], _el.vertices[2]]
            new_face = FemFace(self.nb_faces, element_vertices, _el, self.order)
            self.faces.append(new_face)
            _el.faces.append(new_face)
            _el.faces_orientation.append(1)
            self.nb_faces +=1



def update_edges(self, _el, existing_edges, element_vertices):
    element_vertices_tag = [element_vertices[0].tag, element_vertices[1].tag]
    element_vertices_tag_sorted = sorted(element_vertices_tag)
    if element_vertices_tag_sorted in existing_edges: # If the edge already exists
        # What is its index ?
        index_edge = existing_edges.index(element_vertices_tag_sorted)
        # add the element to the edge's edge list
        self.edges[index_edge].elements.append(_el)
        # add the edge to the element's edge list
        _el.edges.append(self.edges[index_edge])
        # Determination of the edge orientation for the element
        if element_vertices_tag == element_vertices_tag_sorted:
            _el.edges_orientation.append(1)
        else:
            _el.edges_orientation.append(-1)
    else:  # The edge does not exist already exists
        # Insertion the new edge in the existing_edges list
        existing_edges.append(element_vertices_tag_sorted)
        # Creation of the new edge
        if element_vertices_tag == element_vertices_tag_sorted:
            new_edge = FemEdge(self.nb_edges, element_vertices, _el, self.order)
            self.edges.append(new_edge)
            _el.edges.append(new_edge)
            _el.edges_orientation.append(1)
        else:
            new_edge = FemEdge(self.nb_edges, element_vertices[::-1], _el, self.order)
            self.edges.append(new_edge)
            _el.edges.append(new_edge)
            _el.edges_orientation.append(-1)
        self.nb_edges +=1

