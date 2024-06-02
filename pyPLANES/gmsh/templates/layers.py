#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# layers.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@@univ-lemans.fr>
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

from pyPLANES.gmsh.tools.write_geo_file import Gmsh as Gmsh
from pyPLANES.gmsh.write_gmsh import GmshModelpyPLANES

def one_layer( **kwargs):
    name_mesh = kwargs.get("name_mesh", "unnamed_project")
    L = kwargs.get("L", 1.0)
    d = kwargs.get("d", 1.0)
    lcar = kwargs.get("lcar", 0.5)
    order_geometry = kwargs.get("order_geometry", 1)
    mat = kwargs.get("mat", "Air")
    method = kwargs.get("method", "FEM")

    vertices = {}
    vertices["A"] = (0, 0)
    vertices["B"] = (L, 0)
    vertices["C"] = (L, d)
    vertices["D"] = (0, d)
    
    GS = GmshModelpyPLANES(vertices, lcar)
    GS.addSurface(mat, "ABCD", mat)
    GS.addCondition("BA", "condition=bottom")
    GS.addCondition("CD", "condition=top")
    GS.addPeriodicity("BC", "DA")
    GS.create_msh_file(name_mesh)

