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
import gmsh
import numpy as np


def one_inclusion_rigid(name_mesh="one_inclusion_rigid", L=2e-2, d=2e-2, a=0.008, lcar=1e-2, mat="Air"):

    order_geometry = 2
    G = Gmsh(name_mesh, order_geometry)

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


def one_inclusion(name_mesh, L=2e-2, d=2e-2, a=0.008, lcar=1, mat_core="pem_benchmark_1", mat_inclusion="pem_benchmark_1"):

    gmsh.initialize()
    vertice_A = gmsh.model.geo.addPoint(0, 0, 0, lcar)
    vertice_B = gmsh.model.geo.addPoint(L, 0, 0, lcar) 
    vertice_C = gmsh.model.geo.addPoint(L, d, 0, lcar)
    vertice_D = gmsh.model.geo.addPoint(0, d, 0, lcar)
    line_AB = gmsh.model.geo.addLine(vertice_A, vertice_B)
    line_BC = gmsh.model.geo.addLine(vertice_B, vertice_C)
    line_CD = gmsh.model.geo.addLine(vertice_C, vertice_D)
    line_DA = gmsh.model.geo.addLine(vertice_D, vertice_A)
    line_loop = gmsh.model.geo.addCurveLoop([line_AB, line_BC, line_CD, line_DA])
        
    center = gmsh.model.geo.addPoint(L/2, d/2, 0,  lcar)
    north = gmsh.model.geo.addPoint(L/2, d/2+a, 0, lcar)
    south = gmsh.model.geo.addPoint(L/2, d/2-a, 0, lcar)
    east = gmsh.model.geo.addPoint(L/2+a, d/2, 0,  lcar)
    west = gmsh.model.geo.addPoint(L/2-a, d/2, 0,  lcar)
    circle_north_west = gmsh.model.geo.addCircleArc(north, center, west)
    circle_west_south = gmsh.model.geo.addCircleArc(west, center, south)
    circle_south_east = gmsh.model.geo.addCircleArc(south, center, east)
    circle_east_north = gmsh.model.geo.addCircleArc(east, center, north)
    
    circle_loop = gmsh.model.geo.addCurveLoop([circle_north_west, circle_west_south, circle_south_east, circle_east_north])
    
    core = gmsh.model.geo.addPlaneSurface([line_loop, -circle_loop])
    inclusion = gmsh.model.geo.addPlaneSurface([circle_loop])


    gmsh.model.addPhysicalGroup(1, [line_AB], name="condition=bottom")
    gmsh.model.addPhysicalGroup(1, [line_CD], name="condition=top") 
    gmsh.model.addPhysicalGroup(1, [line_BC, line_DA], name="condition=Periodicity")
    gmsh.model.addPhysicalGroup(2,[core], name="mat="+mat_core)
    gmsh.model.addPhysicalGroup(2, [inclusion], name="mat="+mat_inclusion)
    gmsh.model.addPhysicalGroup(1, [line_AB, line_BC, line_DA, line_CD], name="typ=1D")
    gmsh.model.addPhysicalGroup(2, [core, inclusion], name="typ=2D")
    gmsh.model.addPhysicalGroup(1, [line_AB, line_BC, line_DA, line_CD], name="method=FEM")
    gmsh.model.addPhysicalGroup(2, [core, inclusion], name="method=FEM")
    gmsh.model.geo.synchronize()
     # Generate mesh:
    gmsh.model.mesh.generate()
    gmsh.model.mesh.setOrder(2)
    affine_transform = np.eye(4)
    affine_transform[0,3] = d # taken on the first elements because they are all equal
    affine_transform = list(affine_transform.flatten())
    gmsh.model.mesh.setPeriodic(1,[line_BC],[line_DA],affine_transform)
    gmsh.write(f"msh/{name_mesh}.geo_unrolled")
    gmsh.write(f"msh/{name_mesh}.msh")
    gmsh.finalize()

    
def one_inclusion_bicomposite(name_mesh, L=2e-2, d=2e-2, a=0.008, r_i=0.0078, lcar=1e-2, mat_core="pem_benchmark_1", mat_ring="pem_benchmark_1", mat_internal="pem_benchmark_1"):

    
    gmsh.initialize()
    vertice_A = gmsh.model.geo.addPoint(0, 0, 0, lcar)
    vertice_B = gmsh.model.geo.addPoint(L, 0, 0, lcar) 
    vertice_C = gmsh.model.geo.addPoint(L, d, 0, lcar)
    vertice_D = gmsh.model.geo.addPoint(0, d, 0, lcar)
    line_AB = gmsh.model.geo.addLine(vertice_A, vertice_B)
    line_BC = gmsh.model.geo.addLine(vertice_B, vertice_C)
    line_CD = gmsh.model.geo.addLine(vertice_C, vertice_D)
    line_DA = gmsh.model.geo.addLine(vertice_D, vertice_A)
    line_loop = gmsh.model.geo.addCurveLoop([line_AB, line_BC, line_CD, line_DA])
    
    center = gmsh.model.geo.addPoint(L/2, d/2, 0, lcar)
    
    outer_north = gmsh.model.geo.addPoint(L/2, d/2+a, 0, lcar)
    outer_south = gmsh.model.geo.addPoint(L/2, d/2-a, 0, lcar)
    outer_east = gmsh.model.geo.addPoint(L/2+a, d/2, 0, lcar)
    outer_west = gmsh.model.geo.addPoint(L/2-a, d/2, 0, lcar)
    outer_circle_north_west = gmsh.model.geo.addCircleArc(outer_north,center, outer_west)
    outer_circle_west_south = gmsh.model.geo.addCircleArc(outer_west, center, outer_south)
    outer_circle_south_east = gmsh.model.geo.addCircleArc(outer_south,center, outer_east)
    outer_circle_east_north = gmsh.model.geo.addCircleArc(outer_east, center, outer_north)

    inner_north = gmsh.model.geo.addPoint(L/2, d/2+r_i, 0, lcar)
    inner_south = gmsh.model.geo.addPoint(L/2, d/2-r_i, 0, lcar)
    inner_east = gmsh.model.geo.addPoint(L/2+r_i, d/2, 0, lcar)
    inner_west = gmsh.model.geo.addPoint(L/2-r_i, d/2, 0, lcar)
    inner_circle_north_west = gmsh.model.geo.addCircleArc(inner_north, center, inner_west)
    inner_circle_west_south = gmsh.model.geo.addCircleArc(inner_west,  center, inner_south)
    inner_circle_south_east = gmsh.model.geo.addCircleArc(inner_south, center, inner_east)
    inner_circle_east_north = gmsh.model.geo.addCircleArc(inner_east,  center, inner_north)


    inner_circle_lines = [inner_circle_north_west, inner_circle_west_south, inner_circle_south_east, inner_circle_east_north]
    inner_circle_loop = gmsh.model.geo.addCurveLoop(inner_circle_lines)

    outer_circle_lines =[outer_circle_north_west, outer_circle_west_south, outer_circle_south_east, outer_circle_east_north]
    outer_circle_loop = gmsh.model.geo.addCurveLoop(outer_circle_lines)
     
     
    core = gmsh.model.geo.addPlaneSurface([line_loop, -outer_circle_loop])
    ring = gmsh.model.geo.addPlaneSurface([outer_circle_loop, -inner_circle_loop])
    inclusion = gmsh.model.geo.addPlaneSurface([inner_circle_loop])
    
    
    list_FEM_1D = [line_AB, line_BC, line_DA, line_CD]
    list_FSI =[]
    
    material_core = load_material(mat_core)
    material_ring = load_material(mat_ring)
    material_internal = load_material(mat_internal)

    if set([material_ring.MODEL, material_core.MODEL]) == set(["elastic", "fluid"]):
        list_FSI.extend(outer_circle_lines)
    if set([material_ring.MODEL, material_internal.MODEL]) == set(["elastic", "fluid"]):
        list_FSI.extend(inner_circle_lines)
    list_FEM_1D.extend(list_FSI)

    if len(list_FSI)>0:
        gmsh.model.addPhysicalGroup(1, list_FSI, name="condition=Fluid_Structure")
    
    gmsh.model.addPhysicalGroup(1, [line_AB], name="condition=bottom")
    gmsh.model.addPhysicalGroup(1, [line_CD], name="condition=top") 
    gmsh.model.addPhysicalGroup(1, [line_BC, line_DA], name="condition=Periodicity")
    gmsh.model.addPhysicalGroup(2,[core], name="mat="+mat_core)
    gmsh.model.addPhysicalGroup(2, [inclusion], name="mat="+mat_internal)
    gmsh.model.addPhysicalGroup(2,[ring], name="mat="+mat_ring)
    gmsh.model.addPhysicalGroup(1, list_FEM_1D, name="typ=1D")
    gmsh.model.addPhysicalGroup(2, [core, ring, inclusion], name="typ=2D")
    gmsh.model.addPhysicalGroup(1, list_FEM_1D, name="method=FEM")
    

    gmsh.model.geo.synchronize()
     # Generate mesh:
    gmsh.model.mesh.generate()
    gmsh.model.mesh.setOrder(2)
    affine_transform = np.eye(4)
    affine_transform[0,3] = d # taken on the first elements because they are all equal
    affine_transform = list(affine_transform.flatten())
    gmsh.model.mesh.setPeriodic(1,[line_BC],[line_DA],affine_transform)
    gmsh.write(f"msh/{name_mesh}.geo_unrolled")
    gmsh.write(f"msh/{name_mesh}.msh")
    gmsh.finalize()
