#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# ZOD.py
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


def ZOD(name_mesh, L=1., d_1=1., d_2=1., lcb=1., lct=1., shift=0.05, mat="Air", termination="Rigid Wall"):
    G = Gmsh(name_mesh)

    pb_0 = G.new_point(0, 0, lcb)
    pb_1 = G.new_point(L, 0, lcb)
    pb_2 = G.new_point(L, d_1, lcb)
    pb_3 = G.new_point(0, d_1, lcb)

    pt_0 = G.new_point(0, d_1+shift, lct)
    pt_1 = G.new_point(L, d_1+shift, lct)
    pt_2 = G.new_point(L, d_1+d_2+shift, lct)
    pt_3 = G.new_point(0, d_1+d_2+shift, lct)


    lb_0 = G.new_line(pb_0, pb_1)
    lb_1 = G.new_line(pb_1, pb_2)
    lb_2 = G.new_line(pb_2, pb_3)
    lb_3 = G.new_line(pb_3, pb_0)
    llb_0 = G.new_line_loop([lb_0, lb_1, lb_2, lb_3])

    lt_0 = G.new_line(pt_0, pt_1)
    lt_1 = G.new_line(pt_1, pt_2)
    lt_2 = G.new_line(pt_2, pt_3)
    lt_3 = G.new_line(pt_3, pt_0)
    llt_0 = G.new_line_loop([lt_0, lt_1, lt_2, lt_3])


    bottom = G.new_surface([llb_0.tag])
    top = G.new_surface([llt_0.tag])

    G.new_physical(lt_2, "condition="+termination)
    G.new_physical([lb_1, lb_3, lt_1, lt_3], "condition=Periodicity")
    G.new_physical(lb_0, "condition=Incident_PW")
    G.new_physical(bottom, "mat="+mat)
    G.new_physical(top, "mat="+mat + " 1")
    G.new_physical([lb_2], "condition=Interface/ml/-")
    G.new_physical([lt_0], "condition=Interface/ml/+")

    # G.new_physical([lb_2], "condition=Rigid Wall")
    # G.new_physical([lt_0], "condition=Rigid Wall")
    # G.new_physical([lb_0, lb_1, lb_3, lt_1, lt_2, lt_3], "model=FEM1D")


    G.new_physical([lb_0, lb_1, lb_3, lb_2, lt_0, lt_1, lt_2, lt_3], "model=FEM1D")
    G.new_physical([bottom, top], "model=FEM2D")
    G.new_periodicity(lb_1, lb_3, (L, 0, 0))
    G.new_periodicity(lt_1, lt_3, (L, 0, 0))
    option = "-2 -v 0 "

    G.run_gmsh(option)


def ZOD_curved(name_mesh, L=1., d_1=1., d_2=1., lcb=1., lct=1., shift=0.05, d_c=0.1, mat="Air", termination="Rigid Wall"):
    G = Gmsh(name_mesh)

    pb_0 = G.new_point(0, 0, lcb)
    pb_1 = G.new_point(L, 0, lcb)
    pb_2 = G.new_point(L, d_1, lcb)
    pb_2p5 = G.new_point(L/2, d_1+d_c, lcb)
    pb_3 = G.new_point(0, d_1, lcb)

    pt_0 = G.new_point(0, d_1+shift, lct)
    pt_0p5 = G.new_point(L/2, d_1+shift+d_c, lcb)
    pt_1 = G.new_point(L, d_1+shift, lct)
    pt_2 = G.new_point(L, d_1+d_2+shift, lct)
    
    pt_3 = G.new_point(0, d_1+d_2+shift, lct)


    lb_0 = G.new_line(pb_0, pb_1)
    lb_1 = G.new_line(pb_1, pb_2)
    lb_2 = G.new_spline([pb_2, pb_2p5, pb_3])
    lb_3 = G.new_line(pb_3, pb_0)
    llb_0 = G.new_line_loop([lb_0, lb_1, lb_2, lb_3])

    lt_0 = G.new_spline([pt_0, pt_0p5, pt_1])
    lt_1 = G.new_line(pt_1, pt_2)
    lt_2 = G.new_line(pt_2, pt_3)
    lt_3 = G.new_line(pt_3, pt_0)
    llt_0 = G.new_line_loop([lt_0, lt_1, lt_2, lt_3])


    bottom = G.new_surface([llb_0.tag])
    top = G.new_surface([llt_0.tag])

    G.new_physical(lt_2, "condition="+termination)
    G.new_physical([lb_1, lb_3, lt_1, lt_3], "condition=Periodicity")
    G.new_physical(lb_0, "condition=Incident_PW")
    G.new_physical(bottom, "mat="+mat)
    G.new_physical(top, "mat="+mat + " 1")
    G.new_physical([lb_2], "condition=Interface/ml/-")
    G.new_physical([lt_0], "condition=Interface/ml/+")

    # G.new_physical([lb_2], "condition=Rigid Wall")
    # G.new_physical([lt_0], "condition=Rigid Wall")
    # G.new_physical([lb_0, lb_1, lb_3, lt_1, lt_2, lt_3], "model=FEM1D")


    G.new_physical([lb_0, lb_1, lb_3, lb_2, lt_0, lt_1, lt_2, lt_3], "model=FEM1D")
    G.new_physical([bottom, top], "model=FEM2D")
    G.new_periodicity(lb_1, lb_3, (L, 0, 0))
    G.new_periodicity(lt_1, lt_3, (L, 0, 0))
    option = "-2 -v 0 "

    G.run_gmsh(option)