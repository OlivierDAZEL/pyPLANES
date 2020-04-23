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


import numpy as np
import itertools

from pyPLANES.classes.fem_classes import Edge, Face
from pyPLANES.classes.entity_classes import IncidentPwFem, TransmissionPwFem, AirFem, EquivalentFluidFem, RigidWallFem, Pem98Fem, Pem01Fem
# import sys
# import itertools

def update_edges(self, _el, existing_edges, element_vertices, p):
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
            new_edge = Edge(self.nb_edges, element_vertices, _el, p.order)
            self.edges.append(new_edge)
            _el.edges.append(new_edge)
            _el.edges_orientation.append(1)
        else:
            new_edge = Edge(self.nb_edges, element_vertices[::-1], _el, p.order)
            self.edges.append(new_edge)
            _el.edges.append(new_edge)
            _el.edges_orientation.append(-1)
        self.nb_edges +=1

def create_lists(self, p):
    ''' Create the list of edges, faces and bubbles of the Model '''
    existing_edges = [] # List of element vertices for redundancy check
    for ie, _el in enumerate(self.elements[1:]):
        if _el.typ == 1:
            element_vertices = [_el.vertices[0], _el.vertices[1]]
            update_edges(self, _el, existing_edges, element_vertices, p)
        if _el.typ == 2:
            # Edges
            for ii in range(3):
                element_vertices = [_el.vertices[ii], _el.vertices[(ii+1)%3]]
                update_edges(self, _el, existing_edges, element_vertices, p)
            # Faces
            element_vertices = [_el.vertices[0], _el.vertices[1], _el.vertices[2]]
            new_face = Face(self.nb_faces, element_vertices, _el, p.order)
            self.faces.append(new_face)
            _el.faces.append(new_face)
            _el.faces_orientation.append(1)
            self.nb_faces +=1

def activate_dofs(self):
    for _en in self.entities[1:]:
        if _en.dim == 2:
            if isinstance(_en, (EquivalentFluidFem, AirFem)):
                for _el in _en.elements:
                    for _v in _el.vertices:
                        _v.dofs[3] = 1
                    for _e in _el.edges:
                        _e.dofs[3] = [1]*len(_e.dofs[3])
                    for _f in _el.faces:
                        _f.dofs[3] = [1]*len(_f.dofs[3])
            elif isinstance(_en, (Pem98Fem, Pem01Fem)):
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
        elif _en.dim ==1:
            if isinstance(_en, IncidentPwFem):
                _en.dofs = [1]
            elif isinstance(_en, TransmissionPwFem):
                _en.dofs = [1]

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
    self.nb_master_dofs = self.nb_dofs
    for _fc in self.faces:
        for i_dim in range(4):
            self.nb_dofs = renumber_dof(_fc.dofs[i_dim], self.nb_dofs)
    self.nb_dof_faces = self.nb_dofs
    for _bb in self.bubbles:
        for i_dim in range(4):
            self.nb_dofs = renumber_dof(_bb.dofs[i_dim], self.nb_dofs)
    self.nb_dof_FEM = self.nb_dofs
    for _en in self.entities[1:]:
        if isinstance(_en, IncidentPwFem):
            self.nb_dofs = renumber_dof(_en.dofs, self.nb_dofs)
        elif isinstance(_en, TransmissionPwFem):
            self.nb_dofs = renumber_dof(_en.dofs, self.nb_dofs)


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


def init_vec_frequencies(frequency):
    if frequency[2] > 0:
            frequencies = np.linspace(frequency[0],frequency[1],frequency[2])
    elif frequency[2]<0:
        frequencies = np.logspace(np.log10(frequency[0]),np.log10(frequency[1]),abs(frequency[2]))
    # else % Case of complex frequency
    #     temp_1=linspace(frequency.min,frequency.max,frequency.nb(1));
    #     temp_2=linspace(frequency.min_imag,frequency.max_imag,frequency.nb(2));
    #     frequency.vec=[];
    #     for ii=1:frequency.nb(2)
    #         frequency.vec=[frequency.vec temp_1+1j*temp_2(ii)];
    #     end
    #     frequency.nb=frequency.nb(1)*frequency.nb(2);
    return frequencies

def preprocess(self, p):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%% Preprocess of PLANES  %%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    self.frequencies = init_vec_frequencies(p.frequencies)
    # Creation of edges and faces
    create_lists(self, p)

    print("Activation of dofs")
    activate_dofs(self)
    print("Desactivation of orders")
    desactivate_dofs_dimension(self)
    desactivate_dofs_BC(self)
    print("Renumbering of dofs")
    renumber_dofs(self)
    print("Affectation of dofs to the elements")
    affect_dofs_to_elements(self)
    periodicity_initialisation(self)
    # print("Creation of voids shape global matrices")