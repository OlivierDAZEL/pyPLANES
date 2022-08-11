#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# fem_classes.py
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
from pyPLANES.generic.elements_generic import GenericVertex, GenericEdge, GenericElement

class FemVertex(GenericVertex):
    '''Vertex Finite-Element'''
    def __init__(self, coord, tag):
        GenericVertex.__init__(self, coord, tag)
        self.dofs = [0] * 4
        self.sol = np.zeros(4,dtype=complex)
    def __str__(self):
        out = GenericVertex.__str__(self)
        out += "dofs=" + format(self.dofs)+"\n"
        return out

class FemEdge(GenericEdge):
    ''' TODO '''
    def __init__(self, tag, vertices, element, order):
        GenericEdge.__init__(self, tag, vertices, element)
        self.tag = tag
        self.vertices = vertices
        self.elements = [element]
        self.order = [order, order, order, order]
        self.dofs = [[0]*(order-1)]*4
        self.sol = [np.zeros((order-1))]*4

    def __str__(self):
        out = GenericEdge.__str__(self)
        out += "dofs=" + format(self.dofs)+"\n"
        return out

class FemFace:
    """ Finite-Element Face

    Parameters
    ----------
    tag : int
        GMSH tag
    vertices : list
        vertices linked to the element
    element : [type]
        [description]
    order : [type]
        [description]
    """

    def __init__(self, tag, vertices, element, order):
        self.tag = tag
        self.vertices = vertices
        self.elements = [element]
        self.order = [order, order, order, order]
        _ = [0]*int((order-1)*(order-2)/2)
        self.dofs = [_, _, _, _]
        self.sol = [np.zeros(int((order-1)*(order-2)/2))]*4
    def __str__(self):
        out = "Face #{}\n".format(self.tag)
        out  += "Vertices=[{},{},{}], ".format(self.vertices[0].tag, self.vertices[1].tag, self.vertices[2].tag)
        related_elements = [_el.tag for _el in self.elements]
        out  += "related elements={}\n".format(related_elements)
        out += "dofs=" + format(self.dofs)+"\n"
        return out

class FemBubble:
    ''' Class Bubble '''
    """
    [summary]
    """
    def __init__(self,nodes,element,order,geo):
        """
        [summary]

        Parameters
        ----------
        nodes : [type]
            [description]
        element : [type]
            [description]
        order : [type]
            [description]
        geo : [type]
            [description]
        """
        self.geo = geo
        self.nodes = nodes
        if hasattr(self, 'elements'):
            self.elements.append(elements)
        else:
            self.elements=[element]
        self.order = [order, order, order, order]
        self.dofs = [[],[],[],[]]
        self.sol = [np.zeros(int((order-1)*(order-2)*(order-3)/2))]*4

    def __str__(self):
        out = "geo/Nodes/Elements/order/dofs = " + str(self.geo)+"/" + str(self.nodes)+"/"+ format(self.elements)
        out += "/".format(self.order)
        out += format(self.dofs)+"\n"
        return out

class FemElement(GenericElement):
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
        GenericElement.__init__(self, typ, tag, vertices)
        self.reference_element = None
        self.dofs = [[], [], [], []]
        # Rules for the dofs indices vector
        if typ in [1,8]: # Line Elements
            self.edges = []
            self.edges_orientation = []
            self.nb_edges = 0
        elif typ in [2, 9]: # triangles
            self.edges, self.faces = [], []
            self.edges_orientation, self.faces_orientation = [], []
        else: 
            raise NameError("typ {} is not a supported element".format(typ))

    def get_jacobian_matrix(self, xi=0, eta=0):
        
        if self.typ == 2: # TR3
            nb_v = 3 
            coorde = np.zeros((3, nb_v))
            for i, _v in enumerate(self.vertices):
                coorde[0:3, i] = _v.coord[0:3]
            JJ = np.array([[-1./2,1./2,0.],[-1./2,0.,1./2]])

        elif self.typ == 9: # TR6
            nb_v = 3 
            coorde = np.zeros((3, nb_v))
            for i, _v in enumerate(self.vertices[:3]):
                coorde[0:3, i] = _v.coord[0:3]
            JJ = np.array([[-1./2,1./2,0.],[-1./2,0.,1./2]])

        else: 
            raise NameError("Unknown type of element in fem/elements_fem/FemElement/get_jacobian_matrix")
        
        return JJ.dot(coorde[:2,:].T)


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