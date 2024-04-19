#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# mesh.py
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


import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class Mesh():
    def __init__(self, dim=2, name_mesh=None, material_directory="", verbose=False, **kwargs):
        self.dim = dim
        self.name_mesh = name_mesh
        self.materials_directory = material_directory
        self.verbose = verbose

        self.materials = kwargs.get("materials", dict())

        self.entities = [] # List of all GMSH Entities
        self.fem_entities = [] # List of FEM Entities
        self.pwfem_entities = [] # List of Plane Wave FEM Entities
        self.dgm_entities = [] # List of DGM entities
        self.vertices = [] # List of vertices 
        self.elements = [] # List of elements 
        self.edges = []
        self.faces = []
        self.bubbles = []
        self.nb_edges = self.nb_faces = self.nb_bubbles = 0
        self.reference_elements = dict() # dictionary of reference_elements

        if self.name_mesh is not None:
            if not self.name_mesh.endswith('.msh'):
                self.msh_file = "msh/" + self.name_mesh + ".msh"
            self.load_msh_file()

    def load_msh_file(self):
        raise NotImplementedError("This method should be implemented in a child class")

    def display_mesh(self):
        for _el in self.elements[1:]:
            if _el.typ == 2:
                x_vertices =[_v.coord[0] for _v in _el.vertices]
                y_vertices =[_v.coord[1] for _v in _el.vertices]
                plt.plot(x_vertices, y_vertices, 'k-', lw=0.5)

class NeighbourElement():
    def __init__(self, _elem=None, s_0_minus=None, s_1_minus=None, s_0_plus=None, s_1_plus=None):
        self._elem = _elem
        self.s = [s_0_minus, s_1_minus]
        self.S = [s_0_plus, s_1_plus]

