#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# inclusions.py
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
from pyPLANES.utils.io import load_material

def one_inclusion_rigid(name_mesh, L=2e-2, d=2e-2, a=0.008, lcar=1e-2, mat="pem_benchmark_1"):

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
    c_0 = G.new_circle(L/2, d/2, a, lcar/2)

    matrice = G.new_surface([ll_0.tag, -c_0.tag])
    

    G.new_physical(l_2, "condition=top")
    G.new_physical([l_1, l_3], "condition=Periodicity")
    G.new_physical(l_0, "condition=bottom")
    G.new_physical([matrice], "mat="+mat)
    G.new_physical([l_0, l_1, l_3, l_2], "typ=1D")
    G.new_physical([matrice], "typ=2D")
    G.new_physical([l_0, l_1, l_3, l_2, matrice], "method=FEM")
    G.new_periodicity(l_1, l_3, (L, 0, 0))

    option = "-2 -v 0 "
    G.run_gmsh(option)

def one_inclusion(name_mesh, L=2e-2, d=2e-2, a=0.008, lcar=1e-2, mat_core="pem_benchmark_1", mat_inclusion="pem_benchmark_1"):

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
    c_0 = G.new_circle(L/2, d/2, a, lcar/2)

    matrice = G.new_surface([ll_0.tag, -c_0.tag])
    inclusion = G.new_surface([c_0.tag])

    G.new_physical(l_2, "condition=top")
    G.new_physical([l_1, l_3], "condition=Periodicity")
    G.new_physical(l_0, "condition=bottom")
    if mat_core == mat_inclusion:
        G.new_physical([matrice, inclusion], "mat="+mat_core)
    else:
        G.new_physical(matrice, "mat="+mat_core)
        G.new_physical(inclusion, "mat="+mat_inclusion)
    G.new_physical([l_0, l_1, l_3, l_2], "typ=1D")
    G.new_physical([matrice, inclusion], "typ=2D")
    G.new_physical([l_0, l_1, l_3, l_2, matrice, inclusion], "method=FEM")
    G.new_periodicity(l_1, l_3, (L, 0, 0))

    option = "-2 -v 0 "
    G.run_gmsh(option)



def one_inclusion_bicomposite(name_mesh, L=2e-2, d=2e-2, a=0.008, r_i=0.0078, lcar=1e-2, mat_core="pem_benchmark_1", mat_ring="pem_benchmark_1", mat_internal="pem_benchmark_1"):

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
    c_0 = G.new_circle(L/2, d/2, a, lcar/2)
    c_1 = G.new_circle(L/2, d/2, r_i, lcar/2)

    matrice = G.new_surface([ll_0.tag, -c_0.tag])
    ring = G.new_surface([c_0.tag, -c_1.tag])
    internal_inclusion = G.new_surface([c_1.tag])

    G.new_physical(l_2, "condition=top")
    G.new_physical([l_1, l_3], "condition=Periodicity")
    G.new_physical(l_0, "condition=bottom")
    G.new_physical(matrice, "mat="+mat_core)
    G.new_physical(ring, "mat="+mat_ring)
    G.new_physical(internal_inclusion, "mat="+mat_internal)

    list_FEM_1D = [l_0.tag, l_1.tag, l_3.tag, l_2.tag]
    list_FSI =[]
    material_core = load_material(mat_core)
    material_ring = load_material(mat_ring)
    material_internal = load_material(mat_internal)

    if set([material_ring.MODEL, material_core.MODEL]) == set(["elastic", "fluid"]):
        list_FEM_1D.extend(c_0.tag_arcs)
        list_FSI.extend(c_0.tag_arcs)
    if set([material_ring.MODEL, material_internal.MODEL]) == set(["elastic", "fluid"]):
        list_FEM_1D.extend(c_1.tag_arcs)
        list_FSI.extend(c_1.tag_arcs)


    if len(list_FSI)>0:
        G.new_physical_curve(list_FSI, "condition=Fluid_Structure")
    G.new_physical_curve(list_FEM_1D, "typ=1D")
    G.new_physical([matrice, ring, internal_inclusion], "typ=2D")
    G.new_physical([matrice, ring, internal_inclusion], "method=FEM")
    G.new_physical_curve(list_FEM_1D + [matrice.tag, ring.tag, internal_inclusion.tag],  "method=FEM")
    
    
    G.new_periodicity(l_1, l_3, (L, 0, 0))

    option = "-2 -v 0 "
    G.run_gmsh(option)