#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
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


from pyPLANES.gmsh.load_msh_file import load_msh_file

class Mesh():
    def __init__(self, **kwargs):
        self.entities = [] # List of all GMSH Entities
        self.model_entities = [] # List of Entities used in the Model
        self.vertices = []
        self.elements = []
        self.materials_directory = kwargs.get("materials_directory", "")
        self.reference_elements = dict() # dictionary of reference_elements
        load_msh_file(self, **kwargs)


class FemMesh(Mesh):
    def __init__(self, **kwargs):
        Mesh.__init__(self, **kwargs)