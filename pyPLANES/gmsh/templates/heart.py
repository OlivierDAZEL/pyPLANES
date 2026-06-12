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

import numpy as np
import gmsh


def heart( **kwargs):
    name_mesh = kwargs.get("name_mesh", "heart")
    lcar = kwargs.get("lcar", 0.001)
    mat_core = kwargs.get("mat_core", "pem_benchmark_1")
    mat_inclusion = kwargs.get("mat_inclusion", "Steel")

    # Dimensions in meters
    d = 0.03   # 3 cm = 0.03 m
    h = 0.02  # assumed height = 2 cm = 0.02 m

    # Heart placement and scale (meters)
    cx = d / 2
    cy = h / 2 + 0.0005
    scale = 0.00055

    NPTS = 30

    gmsh.initialize()
    vertice_A = gmsh.model.geo.addPoint(0, 0, 0, lcar)
    vertice_B = gmsh.model.geo.addPoint(d, 0, 0, lcar) 
    vertice_C = gmsh.model.geo.addPoint(d, h, 0, lcar)
    vertice_D = gmsh.model.geo.addPoint(0, h, 0, lcar)
    line_AB = gmsh.model.geo.addLine(vertice_A, vertice_B)
    line_BC = gmsh.model.geo.addLine(vertice_B, vertice_C)
    line_CD = gmsh.model.geo.addLine(vertice_C, vertice_D)
    line_DA = gmsh.model.geo.addLine(vertice_D, vertice_A)
    line_loop = gmsh.model.geo.addCurveLoop([line_AB, line_BC, line_CD, line_DA])

    heart_vertices = []
    for i in range(NPTS):
        t = 2 * np.pi * i / NPTS
        x = scale * 16 * (np.sin(t) ** 3)
        y = scale * (
            13 * np.cos(t)
            - 5 * np.cos(2 * t)
            - 2 * np.cos(3 * t)
            - np.cos(4 * t)
        )
        v = gmsh.model.geo.addPoint(cx + x, cy + y, 0, lcar)
        heart_vertices.append(v)

    # creation des lignes entre les points consécutifs
    line_ids = []
    for i in range(len(heart_vertices)-1):
        l = gmsh.model.geo.addLine(heart_vertices[i], heart_vertices[i+1])
        line_ids.append(l)
    l = gmsh.model.geo.addLine(heart_vertices[-1], heart_vertices[0])
    line_ids.append(l)

    heart_loop = gmsh.model.geo.addCurveLoop(line_ids)

    core = gmsh.model.geo.addPlaneSurface([line_loop, heart_loop])
    inclusion = gmsh.model.geo.addPlaneSurface([-heart_loop])

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
