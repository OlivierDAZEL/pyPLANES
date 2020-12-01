#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# fem_classes.py
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

class PlainVertex:
    '''Vertex Finite-Element'''
    def __init__(self, coord, tag):
        self.coord = coord
        self.tag = tag
    def __str__(self):
        out = "Vertex #{}\n".format(self.tag)
        out  += "(x,y,z)=({},{},{})\n".format(*self.coord)
        return out

class PlainEdge:
    ''' TODO '''
    def __init__(self, tag, vertices, element):
        self.tag = tag
        self.vertices = vertices
        self.elements = [element]
    def center(self):
        x = (self.vertices[0].x + self.vertices[1].x)/2.
        y = (self.vertices[0].y + self.vertices[1].y)/2.
        z = (self.vertices[0].z + self.vertices[1].z)/2.
        return np.array([x, y, z])

    def __str__(self):
        out = "Edge #{}\n".format(self.tag)
        out  += "Vertices=[{},{}], ".format(self.vertices[0].tag, self.vertices[1].tag)
        related_elements = [_el.tag for _el in self.elements]
        out  += "related elements={}\n".format(related_elements)
        return out

class PlainElement:
    ''' Element of pyPLANES

    Parameters:
    -----------
    typ : int
        GMSH type of the element

    coorde : numpy array
        Array of nodes coordinates (dim = 3x nb vertices )

    Ref_Elem : Reference Element


    Attributes :
    ------------------------

    edges : List of edge instances associated to the element

    faces : List of face instances associated to the element (optional)

    bubbles : List of bubble instances associated to the element (optional)

    '''
    def __init__(self, typ, tag, vertices):
        self.typ = typ
        self.tag = tag
        self.vertices = vertices
        # Rules for the dofs indices vector

    def __str__(self):
        out = "Element #{} / typ={} / reference element ={}\n".format(self.tag, self.typ, self.reference_element)
        if self.typ == 1:
            out += "Vertices = [{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag)
            print(self.edges)
            out += "edge={} /orientation ={}\n".format(self.edges[0].tag, self.edges_orientation)
        elif self.typ == 2:
            out += "Vertices = [{},{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag, self.vertices[2].tag)
            out += "edges =[{},{},{}]\n".format(self.edges[0].tag, self.edges[1].tag, self.edges[2].tag)
            out += "edge orientation ={}\n".format(self.edges_orientation)
        # out += "dofs={}".format(self.dofs)
        return out

    def get_coordinates(self):
        ''' Method that gives the geometrical coordinates of the element'''
        if self.typ == 1:
            coorde = np.zeros((3, 2))
        elif self.typ == 2:
            # Coordinates of the element
            coorde = np.zeros((3, 3))
        for i, _v in enumerate(self.vertices):
            coorde[0:3, i] = _v.coord[0:3]
        return coorde

    def get_center(self):
        """ 
        Returns
        -------
        numpy array
            coordinates of the center of the element
        """
        coorde = self.get_coordinates()
        return np.mean(coorde, axis=1)



    ''' Element of pyPLANES

    Parameters:
    -----------
    typ : int
        GMSH type of the element

    coorde : numpy array
        Array of nodes coordinates (dim = 3x nb vertices )

    Ref_Elem : Reference Element


    Attributes :
    ------------------------

    edges : List of edge instances associated to the element

    faces : List of face instances associated to the element (optional)

    bubbles : List of bubble instances associated to the element (optional)

    '''
    def __init__(self, typ, tag, vertices):
        PlainElement.__init__(self, typ, tag, vertices)
        self.reference_element = None
        self.dofs = [[], [], [], []]
        # Rules for the dofs indices vector
        if typ == 1:
            self.edges = []
            self.edges_orientation = []
            self.nb_edges = 0
        elif typ == 2:
            self.edges, self.faces = [], []
            self.edges_orientation, self.faces_orientation = [], []

    def __str__(self):
        out = "Element #{} / typ={} / reference element ={}\n".format(self.tag, self.typ, self.reference_element)
        if self.typ == 1:
            out += "Vertices = [{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag)
            print(self.edges)
            out += "edge={} /orientation ={}\n".format(self.edges[0].tag, self.edges_orientation)
        elif self.typ == 2:
            out += "Vertices = [{},{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag, self.vertices[2].tag)
            out += "edges =[{},{},{}]\n".format(self.edges[0].tag, self.edges[1].tag, self.edges[2].tag)
            out += "edge orientation ={}\n".format(self.edges_orientation)
        # out += "dofs={}".format(self.dofs)
        return out

    def display_sol(self, field):
        """
        Returns the values of the coordinates 

        Parameters
        ----------
        field : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        order = self.reference_element.order
        if not isinstance(field, list):
            field = [field]
        f_elem = np.zeros((self.reference_element.nb_SF, len(field)), dtype=complex)
        coorde = self.get_coordinates()
        if self.typ == 2:
            for i_f, _field in enumerate(field):
                f_elem[0, i_f] = self.vertices[0].sol[_field]
                f_elem[1, i_f] = self.vertices[1].sol[_field]
                f_elem[2, i_f] = self.vertices[2].sol[_field]
                for ie, _edge in enumerate(self.edges):
                    f_elem[3+ie*(order-1):3+(ie+1)*(order-1), i_f] = _edge.sol[_field]*(self.edges_orientation[ie]**np.arange((order-1)))
                f_elem[3+3*(order-1):, i_f] = self.faces[0].sol[_field]

            x = np.dot(coorde[0, :].T, self.reference_element.phi_plot[:3, :])
            y = np.dot(coorde[1, :].T, self.reference_element.phi_plot[:3, :])
            f = np.dot(self.reference_element.phi_plot.T, f_elem)
        return x, y, f

    def elementary_matrices(self):
        pass