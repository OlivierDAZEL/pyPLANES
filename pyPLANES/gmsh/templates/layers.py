#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# one_inclusion.py
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

from pyPLANES.gmsh.tools.write_geo_file import Gmsh as Gmsh

def one_layer( **kwargs):
    name_mesh = kwargs.get("name_mesh", "unnamed_project")
    L = kwargs.get("L", 1.0)
    d = kwargs.get("d", 1.0)
    lcar = kwargs.get("lcar", 0.5)
    order_geometry = kwargs.get("order_geometry", 1)
    mat = kwargs.get("mat", "Air")
    method = kwargs.get("method", "FEM")
    BC = kwargs.get("BC", ["bottom", "Periodicity", "top", "Periodicity"])



    G = Gmsh(name_mesh, order_geometry)

    p_0 = G.new_point(0, 0, lcar)
    p_1 = G.new_point(L, 0,lcar)
    p_2 = G.new_point(L, d, lcar)
    p_3 = G.new_point(0, d, lcar)

    lines = [None]*4
    lines[0] = G.new_line(p_0, p_1)
    lines[1] = G.new_line(p_1, p_2)
    lines[2] = G.new_line(p_2, p_3)
    lines[3] = G.new_line(p_3, p_0)
    boundary_domain = G.new_line_loop(lines)

    matrice = G.new_surface([boundary_domain.tag])
    for bc in set(BC):
        # Determine the lines 
        list_lines = [lines[i] for i, _bc in enumerate(BC) if _bc == bc]
        G.new_physical(list_lines, "condition="+ bc)
    G.new_physical(lines, "typ=1D")
    G.new_physical(matrice, "mat="+mat)
    G.new_physical([matrice], "typ=2D")
    G.new_physical(lines+[matrice], "method="+method)
    if (BC[1] == "Periodicity") and (BC[3] == "Periodicity"):
        G.new_periodicity(lines[1], lines[3], (L, 0, 0))
    option = "-2 -v 0 "
    G.run_gmsh(option)
