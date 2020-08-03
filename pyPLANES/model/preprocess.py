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

import itertools
import numpy as np

from mediapack import Air, from_yaml
from pymls import Layer
from pyPLANES.utils.utils_fem import normal_to_element
from pyPLANES.classes.fem_classes import Edge, Face
from pyPLANES.classes.entity_classes import PwFem, FluidFem, RigidWallFem, PemFem, ElasticFem, PeriodicityFem, IncidentPwFem, TransmissionPwFem, FluidStructureFem, InterfaceFem



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
            new_edge = Edge(self.nb_edges, element_vertices, _el, self.order)
            self.edges.append(new_edge)
            _el.edges.append(new_edge)
            _el.edges_orientation.append(1)
        else:
            new_edge = Edge(self.nb_edges, element_vertices[::-1], _el, self.order)
            self.edges.append(new_edge)
            _el.edges.append(new_edge)
            _el.edges_orientation.append(-1)
        self.nb_edges +=1

def create_lists(self):
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
            new_face = Face(self.nb_faces, element_vertices, _el, self.order)
            self.faces.append(new_face)
            _el.faces.append(new_face)
            _el.faces_orientation.append(1)
            self.nb_faces +=1

def activate_dofs(self):
    for _en in self.entities[1:]:
        if _en.dim == 2:
            if isinstance(_en, FluidFem):
                for _el in _en.elements:
                    for _v in _el.vertices:
                        _v.dofs[3] = 1
                    for _e in _el.edges:
                        _e.dofs[3] = [1]*len(_e.dofs[3])
                    for _f in _el.faces:
                        _f.dofs[3] = [1]*len(_f.dofs[3])
            elif isinstance(_en, PemFem):
                for _el in _en.elements:
                    for _v in _el.vertices:
                        _v.dofs[0:4] = [1, 1, 1, 1]
                    for _e in _el.edges:
                        _e.dofs[0] = [1]*len(_e.dofs[0])
                        _e.dofs[1] = [1]*len(_e.dofs[1])
                        _e.dofs[2] = [1]*len(_e.dofs[2])
                        _e.dofs[3] = [1]*len(_e.dofs[3])
                    for _f in _el.faces:
                        _f.dofs[0] = [1]*len(_f.dofs[0])
                        _f.dofs[1] = [1]*len(_f.dofs[1])
                        _f.dofs[2] = [1]*len(_f.dofs[2])
                        _f.dofs[3] = [1]*len(_f.dofs[3])
            elif isinstance(_en, ElasticFem):
                for _el in _en.elements:
                    for _v in _el.vertices:
                        _v.dofs[0:3] = [1, 1, 1]
                    for _e in _el.edges:
                        _e.dofs[0] = [1]*len(_e.dofs[0])
                        _e.dofs[1] = [1]*len(_e.dofs[1])
                        _e.dofs[2] = [1]*len(_e.dofs[2])
                    for _f in _el.faces:
                        _f.dofs[0] = [1]*len(_f.dofs[0])
                        _f.dofs[1] = [1]*len(_f.dofs[1])
                        _f.dofs[2] = [1]*len(_f.dofs[2])

def desactivate_dofs_dimension(self):
    if self.dim < 3:
        for _ in self.vertices[1:]: _.dofs[2] = 0
        for _ in self.edges: _.dofs[2] = [0]*len(_.dofs[2])
        for _ in self.faces: _.dofs[2] = [0]*len(_.dofs[2])
        for _ in self.bubbles: _.dofs[2] = [0]*len(_.dofs[2])
    if self.dim < 2:
        for _ in self.vertices[1:]: _.dofs[1] = 0
        for _ in self.edges: _.dofs[1] = [0]*len(_.dofs[1])
        for _ in self.faces: _.dofs[1] = [0]*len(_.dofs[1])
        for _ in self.bubbles: _.dofs[1] = [0]*len(_.dofs[1])

def desactivate_dofs_BC(self):
    for _en in self.entities[1:]:
        if isinstance(_en, RigidWallFem):
            for _el in _en.elements:
                for _v in _el.vertices:
                    _v.dofs[0:3] = [0,0,0]
                for _e in _el.edges:
                    _e.dofs[0] = [0]*len(_e.dofs[0])
                    _e.dofs[1] = [0]*len(_e.dofs[1])
                    _e.dofs[2] = [0]*len(_e.dofs[2])

def renumber_dof(_list,start=0):
    ''' renumber a list from start '''
    index = [idx for idx, val in enumerate(_list) if val != 0]
    _start = start
    for _i in index:
        _list[_i] = _start
        _start += 1
    return start+len(index)

def renumber_dofs(self):
    self.nb_dofs = 1
    for _nd in self.vertices[1:]:
        self.nb_dofs = renumber_dof(_nd.dofs, self.nb_dofs)
    self.nb_dof_nodes = self.nb_dofs
    for _ed in self.edges:
        for i_dim in range(4):
            self.nb_dofs = renumber_dof(_ed.dofs[i_dim], self.nb_dofs)
    self.nb_dof_edges = self.nb_dofs
    self.nb_dof_master = self.nb_dofs
    self.nb_master_dofs = self.nb_dofs
    for _fc in self.faces:
        for i_dim in range(4):
            self.nb_dofs = renumber_dof(_fc.dofs[i_dim], self.nb_dofs)
    self.nb_dof_faces = self.nb_dofs

    for _bb in self.bubbles:
        for i_dim in range(4):
            self.nb_dofs = renumber_dof(_bb.dofs[i_dim], self.nb_dofs)
    self.nb_dof_FEM = self.nb_dofs
    self.nb_dofs_to_condense = self.nb_dofs - self.nb_dof_master

def affect_dofs_to_elements(self):
    for _el in self.elements[1:]:
        if _el.typ ==1:
            for i_dim in range(4):
                _el.dofs[i_dim] += [_v.dofs[i_dim] for _v in _el.vertices]
                _el.dofs[i_dim] += [_e.dofs[i_dim] for _e in _el.edges]
        if _el.typ ==2:
            for i_dim in range(4):
                _el.dofs[i_dim] += [_v.dofs[i_dim] for _v in _el.vertices]
                _el.dofs[i_dim] += [_e.dofs[i_dim] for _e in _el.edges]
                _el.dofs[i_dim] += [_f.dofs[i_dim] for _f in _el.faces]

def periodicity_initialisation(self):
    edges_left, edges_right = [], []
    for _en in self.model_entities:
        if _en.dim == 1:
            for _el in _en.elements:
                _vertices_tag = [_v.tag for _v in _el.edges[0].vertices]
                if len(set(_vertices_tag).intersection(self.vertices_right)) == 2:
                    edges_right.append(_el.edges[0])
                elif len(set(_vertices_tag).intersection(self.vertices_left)) == 2:
                    edges_left.append(_el.edges[0])

    # Determination of the correspondance between edges (we did not divide by two for the average of the position)
    y_left =  [(_e.vertices[0].coord[1]+_e.vertices[1].coord[1]) for _e in edges_left]
    y_right = [(_e.vertices[0].coord[1]+_e.vertices[1].coord[1]) for _e in edges_right]
    # corr_edges = [y_right.index(_y) for _y in y_left]

    corr_edges = [ next(i for i, _ in enumerate(y_right) if np.isclose(_, _yl, 1e-8)) for _yl in y_left]

    # dof_left, dof_right, orient = [],[],[]
    dof_left, dof_right = [],[]
    for _il, _vl in enumerate(self.vertices_left):
        dof_left.extend(self.vertices[_vl].dofs)
        _vr = self.vertices_right[_il]
        dof_right.extend(self.vertices[_vr].dofs)
    #     # orient += [1]*4

    for _il, _ed in enumerate(edges_left):
        dof_left += list(itertools.chain(*_ed.dofs))
        dof_right += list(itertools.chain(*edges_right[corr_edges[_il]].dofs))
    #  Suppression of zeros dofs
    _ =np.sum(np.array([dof_left, dof_right]), axis=0)
    _nz = np.where(_!=0)[0].tolist()
    self.dof_left = [dof_left[ii] for ii in _nz]
    self.dof_right = [dof_right[ii] for ii in _nz]

def elementary_matrices(self):
    '''Creation of elementary matrices in the elements'''
    print("Creation of elementary matrices in the elements")
    for _ent in self.model_entities:
        for _el in _ent.elements:
            _ent.elementary_matrices(_el)

def check_model(self):
    ''' This function checks if the model is correct and adapt it if not '''
    unfinished = True
    while unfinished:
        unfinished = False
        for _e in self.entities:
            if _e.dim == 1: # 1D entities
                if len(_e.neighbouring_surfaces) > 1:
                    if isinstance(_e, (PwFem, RigidWallFem, PeriodicityFem)):
                        raise ValueError("Error in check model: 1D entity is linked to more than one surface")
                    if isinstance(_e, FluidStructureFem):
                        if len(_e.neighbouring_surfaces) != 2:
                            raise NameError("For FluidStructureFem, the number of neighbours should be 2")
                        else:
                            if isinstance(_e.neighbouring_surfaces[0], FluidFem) and isinstance(_e.neighbouring_surfaces[1], ElasticFem):
                                _e.fluid_neighbour, _e.struc_neighbour = _e.neighbouring_surfaces[0], _e.neighbouring_surfaces[1]
                            elif isinstance(_e.neighbouring_surfaces[0], ElasticFem) and isinstance(_e.neighbouring_surfaces[1], FluidFem):
                                _e.fluid_neighbour, _e.struc_neighbour = _e.neighbouring_surfaces[1], _e.neighbouring_surfaces[0]
                            else:
                                raise NameError("FluidStructureFem doest not relate a fluid and elastic struture")
                            for _elem in _e.elements:
                                vert = [_elem.vertices[0].tag, _elem.vertices[1].tag] # Vertices of the FSI element
                                # Determination of the neighbouring element in neighbouring_surfaces[0]
                                _iter = iter(_e.fluid_neighbour.elements)
                                while True:
                                    _el = next(_iter)
                                    vert_2D = [_el.vertices[0].tag, _el.vertices[1].tag, _el.vertices[2].tag]
                                    _ = len(set(vert).intersection(vert_2D)) # Number of common vertices
                                    if _ == 2: # Case of two common vertices
                                        _elem.normal_fluid = normal_to_element(_elem, _el)
                                        _elem.normal_struc = -_elem.normal_fluid
                                        break
            if _e.dim == 2:
                if isinstance(_e, PemFem):
                    # _e.formulation98 = True
                    _e.formulation98 = False
    # Check that the number of interfaces go by pairs
    list_interfaces = [_ent for _ent in self.model_entities if isinstance(_ent, InterfaceFem)]
    name_interfaces = [_ent.ml for _ent in list_interfaces]
    n_interface = len(list_interfaces)
    if  n_interface%2 == 1:
        raise ValueError("Error in check model: Number of interfaces is odd")
    else:
        while n_interface != 0:
            _int_minus = list_interfaces[0]
            _index = name_interfaces[1:].index(_int_minus.ml)+1
            _int_plus = list_interfaces[_index]

            _int_minus.neighbour = _int_plus
            _int_plus.neighbour = _int_minus

            del list_interfaces[_index]
            del list_interfaces[0]
            del name_interfaces[_index]
            del name_interfaces[0]

            if _int_minus.side == "+":
                if _int_plus.side == "-":
                    _int_minus, _int_plus =_int_plus, _int_minus
                else:
                    raise ValueError("_int_minus.side = + and _int_plus.side != + ")
            elif _int_minus.side == "-":
                if _int_plus.side != "+":
                    raise ValueError("_int_minus.side = - and _int_plus.side != + ")
            else:
                raise ValueError("_int_minus.side ins neither + or - ")

            for _el_minus in _int_minus.elements:
                print(_el_minus)
                center_minus = _el_minus.get_center()
                for _el_plus in _int_plus.elements:
                    center_plus = _el_plus.get_center()
                    if LA.norm(center_minus-center_plus) < self.interface_zone:
            n_interface -= 2

    for _e in self.model_entities:
        if isinstance(_e, PwFem):
            for s in _e.neighbouring_surfaces:
                if isinstance(s, (Air, FluidFem)):
                    _e.nb_R = 1
                    _e.typ = "fluid"
                elif (isinstance(s, ElasticFem)):
                    _e.nb_R = 2
                    _e.typ = "elastic"
                elif isinstance(s, PemFem):
                    if s.formulation98:
                        _e.nb_R = 3
                        _e.typ = "Biot98"
                    else:
                        _e.nb_R = 3
                        _e.typ = "Biot01"
            if isinstance(_e, IncidentPwFem):
                if self.incident_ml:
                    _e.ml = []
                    for _l in self.incident_ml:
                        mat = from_yaml(_l[0]+".yaml")
                        d = _l[1]
                        _e.ml.append(Layer(mat,d))
            if isinstance(_e, TransmissionPwFem):
                if self.transmission_ml:
                    _e.ml = []
                    for _l in self.transmission_ml:
                        mat = from_yaml(_l[0]+".yaml")
                        d = -_l[1]
                        # Thickness of transmission layers is set negative
                        _e.ml.append(Layer(mat,d))
                    for _l in _e.ml:
                        _l.thickness *= -1

def preprocess(self, p):
    if self.verbose:
        print("%%%%%%%%%%%% Preprocess of PLANES  %%%%%%%%%%%%%%%%%")

    # self.frequencies = init_vec_frequencies(p.frequencies)
    # Creation of edges and faces

    create_lists(self, p)
    if p.verbose:
        print("Activation of dofs based on physical media")
    activate_dofs(self)
    if p.verbose:
        print("Desactivation of dofs (dimension & BC)")
    desactivate_dofs_dimension(self)
    desactivate_dofs_BC(self)
    if p.verbose:
        print("Renumbering of dofs")
    renumber_dofs(self)
    if p.verbose:
        print("Affectation of dofs to the elements")
    affect_dofs_to_elements(self)
    if p.verbose:
        print("Identification of periodic dofs")
    periodicity_initialisation(self)
    if p.verbose:
        print("Creation of voids shape global matrices")
    check_model(self, p)
    elementary_matrices(self)
