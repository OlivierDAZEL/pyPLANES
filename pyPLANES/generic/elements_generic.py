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

class GenericVertex:
    '''Vertex Finite-Element'''
    def __init__(self, coord, tag):
        self.coord = coord
        self.tag = tag
    def __str__(self):
        out = "Vertex #{}\n".format(self.tag)
        out  += "(x,y,z)=({},{},{})\n".format(*self.coord)
        return out

class GenericEdge:
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

class GenericElement:
    ''' Generic Element of pyPLANES

    Parameters:
    -----------
    typ : int
        GMSH type of the element

    coorde : numpy array
        Array of nodes coordinates (dim = 3x nb vertices )

    '''
    def __init__(self, typ, tag, vertices):
        self.typ = typ
        self.tag = tag

        self.coord = np.zeros((2, len(vertices)))
        for i, _v in enumerate(vertices):
            self.coord[:, i] = _v.coord[0:2]
        if self.typ in [1, 8]: # Line
            self.vertices = vertices[:2]
        elif self.typ in [2, 9]: # TR
            self.vertices = vertices[:3]

    def __str__(self):
        out = "Element #{} / typ={} / reference element ={}\n".format(self.tag, self.typ, self.reference_element)
        if self.typ == 1:
            out += "Vertices = [{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag)
            out += "edge={} /orientation ={}\n".format(self.edges[0].tag, self.edges_orientation)
        elif self.typ == 2:
            out += "Vertices = [{},{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag, self.vertices[2].tag)
            out += "edges =[{},{},{}]\n".format(self.edges[0].tag, self.edges[1].tag, self.edges[2].tag)
        return out




    def get_center(self):
        """ 
        Returns
        -------
        numpy array
            coordinates of the center of the element
        """
        return np.mean(self.coord, axis=1)


