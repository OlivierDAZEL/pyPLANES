#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# one_inclusion.py
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

from pyPLANES.gmsh.tools.write_geo_file import Gmsh as Gmsh


def one_layer(name_mesh,L=2e-2,d=2e-2, lcar=1e-2, mat="pem_benchmark_1", termination="Rigid Wall"):
    G = Gmsh(name_mesh)

    p_0 = G.new_point(0, 0, lcar)
    p_1 = G.new_point(L, 0,lcar)
    p_2 = G.new_point(L, d, lcar)
    p_3 = G.new_point(0, d, lcar)
    l_0 = G.new_line(p_0, p_1)
    l_1 = G.new_line(p_1, p_2)
    l_2 = G.new_line(p_2, p_3)
    l_3 = G.new_line(p_3, p_0)
    ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
    matrice = G.new_surface([ll_0.tag])
    # G.new_physical(l_2, "condition=Transmission")
    G.new_physical(l_2, "condition="+termination)
    G.new_physical([l_1, l_3], "condition=Periodicity")
    G.new_physical(l_0, "condition=Incident_PW")
    G.new_physical(matrice, "mat="+mat)
    G.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
    G.new_physical([matrice], "model=FEM2D")
    G.new_periodicity(l_1, l_3, (L, 0, 0))
    option = "-2 -v 0 "
    G.run_gmsh(option)