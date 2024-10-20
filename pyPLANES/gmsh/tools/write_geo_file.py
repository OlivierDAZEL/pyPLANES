#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# write_geo_file.py
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

import os
import sys
import numpy as np
from copy import deepcopy


class Gmsh():
    def __init__(self, file="noname", order=1):
        if not os.path.exists("msh"):
            os.mkdir("msh")
        self.geo_file = "msh/" + file + ".geo"
        self.f= open(self.geo_file, "w")
        self.f.write("// This code was created by pyPLANES\n")
        if order ==2: 
            # self.f.write("Mesh.ElementOrder = 2;\n")
            # self.f.write("Mesh.SecondOrderLinear = 0;\n")
            self.f.write("Mesh.Algorithm = 6;\n")
            self.f.write("Mesh.Smoothing = 30;\n")
            self.f.write("Mesh.ElementOrder = 2;\n")
        self.nb_tags = 0
        self.list_points = []
        self.list_lines = []
        self.list_lineloops = []

    class Point():
        def __init__(self, f, tag, x, y, lc):
            self.x = x
            self.y = y
            self.lc = lc
            self.tag = tag
            self.typ = "Point"
            f.write("Point({})= {{{}, {}, {},{}}};\n".format(tag, x, y, 0.0,self.lc))

    class Line():
        def __init__(self, f, tag, _1, _2):
            self.p_1 = _1
            self.p_2 = _2
            self.tag = tag
            self.typ = "Line"
            f.write("Line({})= {{{}, {}}};\n".format(tag, _1.tag, _2.tag))
        def __str__(self):
            out ="Line entity from points {} to point {}. Tag={}".format(self.p_1.tag, self.p_2.tag, self.tag)
            return out
        def inverted(self):
            out = deepcopy(self)
            out.tag = -self.tag
            return out

    class Spline():
        def __init__(self, f, tag, _list_points):
            self.list_points = _list_points
            self.tag = tag
            self.typ = "Spline"
            f.write("Spline({})= {{{}".format(tag, _list_points[0].tag))
            for _p in _list_points[1:]:
                f.write(", {}".format(_p.tag))
            f.write("};\n")

        def __str__(self):
            out ="Line entity from points {} to point {}. Tag={}".format(self.p_1.tag, self.p_2.tag, self.tag)
            return out
        def inverted(self):
            out = deepcopy(self)
            out.tag = -self.tag
            return out

    class Bezier():
        def __init__(self, f, tag, _1, _2, _3, _4):
            self.p_1 = _1
            self.p_2 = _2
            self.p_3 = _3
            self.p_3 = _4
            self.tag = tag
            self.typ = "Bezier"
            f.write("Bezier({})= {{{}, {}, {}, {}}};\n".format(tag, _1.tag, _2.tag, _3.tag, _4.tag))
        def __str__(self):
            out ="Bezier entity from points {} to point {}. Tag={}".format(self.p_1.tag, self.p_4.tag, self.tag)
            return out
        def inverted(self):
            out = deepcopy(self)
            out.tag = -self.tag
            return out

    class LineLoop():
        def __init__(self, f, tag, _list):
            self.lines = [_l.tag for _l in _list]
            self.tag = tag
            self.typ = "Line loop"
            f.write("Line Loop({})= {{{}".format(tag, _list[0].tag))
            for _l in _list[1:]:
                f.write(",{} ".format(_l.tag))
            f.write("};\n")

    class Circle():
        def __init__(self, f, start_tag, points):
            self.typ = "Circle"
            _tag = start_tag

            center, east, north, west, south = points
            f.write("Circle({})= {{{}, {}, {}}};\n".format(_tag, east.tag, center.tag, north.tag)); _tag +=1
            f.write("Circle({})= {{{}, {}, {}}};\n".format(_tag, north.tag, center.tag, west.tag)); _tag +=1
            f.write("Circle({})= {{{}, {}, {}}};\n".format(_tag, west.tag, center.tag, south.tag)); _tag +=1
            f.write("Circle({})= {{{}, {}, {}}};\n".format(_tag, south.tag, center.tag, east.tag)); _tag +=1
            f.write("Curve Loop({})= {{{}, {}, {}, {}}};\n".format(_tag, _tag-4, _tag-3, _tag-2, _tag-1))
            self.tag_arcs = [_tag-4, _tag-3, _tag-2, _tag-1]
            self.tag = _tag

    class Surface():
        def __init__(self, f, tag, ll):
            if isinstance(ll, list):
                self.ll = ll
            else:
                self.ll = [ll]
            self.tag = tag
            self.typ = "Surface"
            f.write("Plane Surface({})= {{{}".format(tag, ll[0]))
            for _l in self.ll[1:]:
                f.write(", {}".format(_l))
            f.write("};\n")

    def add(self,txt):
        self.f.write(txt+"\n")

    def new_point(self, x, y, lc):
        self.nb_tags += 1
        p = self.Point(self.f, self.nb_tags, x, y, lc)
        # self.file.write("Point({})= {{{}, {}, {}, {}}};\n".format(self.nb_tags, x, y, 0.0, lc))
        self.list_points.append(p)
        return p

    def duplicate_point(self, point, x, y):
        self.nb_tags += 1
        p = self.Point(self.f, self.nb_tags, point.x+x, point.y+y, point.lc)
        # self.file.write("Point({})= {{{}, {}, {}, {}}};\n".format(self.nb_tags, x, y, 0.0, lc))
        self.list_points.append(p)
        return p


    def new_line(self, _1, _2):
        self.nb_tags += 1
        l = self.Line(self.f, self.nb_tags, _1, _2)
        # self.file.write("Line({})= {{{}, {}}};\n".format(self.nb_tags, _1, _2))
        self.list_lines.append(l)
        return l

    def new_bezier(self, _1, _2, _3, _4):
        self.nb_tags += 1
        b = self.Bezier(self.f, self.nb_tags, _1, _2, _3, _4)
        # self.file.write("Line({})= {{{}, {}}};\n".format(self.nb_tags, _1, _2))
        self.list_lines.append(b)
        return b


    def new_spline(self, _list_points):
        self.nb_tags += 1
        l = self.Spline(self.f, self.nb_tags, _list_points)
        # self.file.write("Line({})= {{{}, {}}};\n".format(self.nb_tags, _1, _2))
        return l


    def new_line_loop(self, _list):
        self.nb_tags += 1
        ll = self.LineLoop(self.f, self.nb_tags, _list)
        self.list_lineloops.append(ll)
        return ll

    def new_surface(self, ll):
        self.nb_tags += 1
        s = self.Surface(self.f, self.nb_tags, ll)
        return s

    def new_circle(self, x_0, y_0, r, lc, lccenter=-1):
        if lccenter == -1:
            lccenter = lc
        self.nb_tags += 1
        center = self.Point(self.f, self.nb_tags, x_0, y_0, lccenter)
        self.nb_tags +=1
        east = self.Point(self.f, self.nb_tags, x_0+r, y_0, lc)
        self.nb_tags +=1
        north = self.Point(self.f, self.nb_tags, x_0, y_0+r, lc)
        self.nb_tags +=1
        west = self.Point(self.f, self.nb_tags, x_0-r, y_0, lc)
        self.nb_tags+=1
        south = self.Point(self.f, self.nb_tags, x_0, y_0-r, lc)
        self.nb_tags +=1
        points = [center, east, north, west, south]
        c = self.Circle(self.f, self.nb_tags, points)
        self.nb_tags += 4 # 4 additional entities created in Circle
        return c

    def new_physical(self, obj, label):
        if isinstance(obj, list):
            objects = obj
        else:
            objects = [obj]
        line_objects = [_obj.tag for _obj in objects if _obj.typ == "Line"]
        if len(line_objects) != 0:
            self.f.write("Physical Line(\"" + label + "\")={"+ str(line_objects[0]))
            for _obj in line_objects[1:]:
                self.f.write(", {}".format(_obj))
            self.f.write("};\n")

        surf_objects = [_obj.tag for _obj in objects if _obj.typ == "Surface"]
        if len(surf_objects) != 0:
            self.f.write("Physical Surface(\"" + label + "\")={"+ str(surf_objects[0]))
            for _obj in surf_objects[1:]:
                self.f.write(", {}".format(_obj))
            self.f.write("};\n")

    def new_physical_curve(self, list, label):
        self.f.write("Physical Curve(\"" + label + "\")={"+ str(list[0]))
        for _l in list[1:]:
            self.f.write(", {}".format(_l))
        self.f.write("};\n")

    def new_periodicity(self, obj1, obj2, Delta):
        if obj1.typ != obj2.typ:
            print("Error in GEO file in periodicity : obj1.typ != obj2.typ")
            sys.exit()
        else:
            self.f.write("Periodic " + obj1.typ + " {{{}}} = {{{}}} Translate {{{},{},{}}};\n".format(obj1.tag, obj2.tag, Delta[0],Delta[1],Delta[2]))
            
            
    def generate_points_from_dict(self, d,lcar):
        dp = {}
        for key in d.keys():
            self.new_point(d[key][0],d[key][1],lcar)

    # def new_circle(self,)

    def run_gmsh(self, option=""):
        self.f.close()
        if sys.platform == "darwin":
            os.system("/Applications/Gmsh.app/Contents/MacOS/gmsh " + option + self.geo_file)
        else:
            
            os.system("gmsh " + option + self.geo_file)

# def one_layer(p):
#     g = Gmsh(p.gmsh_file)
#     p_0 = g.new_point(0,0,p.lcar)
#     p_1 = g.new_point(p.L,0,p.lcar)
#     p_2 = g.new_point(p.L,p.d,p.lcar)
#     p_3 = g.new_point(0,p.d,p.lcar)
#     l_0 = g.new_line(p_0, p_1)
#     l_1 = g.new_line(p_1, p_2)
#     l_2 = g.new_line(p_2, p_3)
#     l_3 = g.new_line(p_3, p_0)
#     ll_0 = g.new_line_loop([l_0, l_1, l_2, l_3])

#     matrice = g.new_surface([ll_0.tag])

#     g.new_physical(l_2,"condition=Transmission")
#     g.new_physical([l_1, l_3], "condition=Periodicity")
#     g.new_physical(l_0, "condition=Incident_PW")
#     g.new_physical(matrice, "mat="+p.pem1)
#     g.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
#     g.new_physical([matrice], "model=FEM2D")
#     g.new_periodicity(l_1,l_3,(p.L, 0, 0))

#     option = "-2 "
#     g.run_gmsh(option)








# def one_layer(p):
#     geom = pg.built_in.Geometry()

#     L = p.lx
#     l_1 = p.l1

#     lcar = p.lcar
#     P1 = geom.add_point([0., 0., 0.], lcar)
#     P2 = geom.add_point([L, 0., 0.0], lcar)
#     P3 = geom.add_point([L, l_1, 0.0], lcar)
#     P4 = geom.add_point([0., l_1, 0.0], lcar)

#     l1 = geom.add_line(P1, P2)
#     l2 = geom.add_line(P2, P3)
#     l3 = geom.add_line(P3, P4)
#     l4 = geom.add_line(P4, P1)


#     geom.add_physical([l3], label=p.transmission)
#     geom.add_physical([l2], label="Periodicity right")
#     geom.add_physical([l4], label="Periodicity left")
#     geom.add_physical(l1, label=p.excitation)

#     # Layer 1
#     ll1 = geom.add_line_loop([l1, l2, l3, l4])
#     s1 = geom.add_plane_surface(ll1)
#     geom.add_physical(s1, label=p.mat1)
#     # For periodicity
#     # geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(l2.id, l4.id))

#     mesh = pg.generate_mesh(geom, geo_filename='GMSH/toto.geo', msh_filename='GMSH/toto.msh',verbose=False)

#     return mesh

# def two_layers(p):
#     geom = pg.built_in.Geometry()

#     L = p.lx
#     l_1 = p.l1
#     l_2 = p.l2

#     lcar = p.lcar
#     P1 = geom.add_point([0., 0., 0.], lcar)
#     P2 = geom.add_point([L, 0., 0.0], lcar)
#     P3 = geom.add_point([L, l_1, 0.0], lcar)
#     P4 = geom.add_point([0., l_1, 0.0], lcar)
#     P5 = geom.add_point([0., l_1+l_2, 0.0], lcar)
#     P6 = geom.add_point([L, l_1+l_2, 0.0], lcar)

#     l1 = geom.add_line(P1, P2)
#     l2 = geom.add_line(P2, P3)
#     l3 = geom.add_line(P3, P4)
#     l4 = geom.add_line(P4, P1)
#     l5 = geom.add_line(P3, P6)
#     l6 = geom.add_line(P6, P5)
#     l7 = geom.add_line(P5, P4)


#     geom.add_physical([l6], label=p.transmission)
#     geom.add_physical([l2, l5], label="Periodicity right")
#     geom.add_physical([l4, l7], label="Periodicity left")
#     geom.add_physical(l1, label=p.excitation)

#     # Layer 1
#     ll1 = geom.add_line_loop([l1, l2, l3, l4])
#     s1 = geom.add_plane_surface(ll1)
#     geom.add_physical(s1, label=p.mat1 + " 1")
#     # Layer 1
#     ll2 = geom.add_line_loop([l5, l6, l7, -l3])
#     s2 = geom.add_plane_surface(ll2)
#     geom.add_physical(s2, label=p.mat2 + " 2")
#     # For periodicity
#     geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(l2.id, l4.id))
#     geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(l5.id, l7.id))

#     mesh = pg.generate_mesh(geom, verbose=False)

#     return mesh

# def three_layers(p):
#     geom = pg.built_in.Geometry()

#     L = p.lx
#     l_1 = p.l1
#     l_2 = p.l2
#     l_3 = p.l3

#     lcar = p.lcar
#     P_00 = geom.add_point([0., 0., 0.], lcar)
#     P_01 = geom.add_point([0.,l_1, 0.], lcar)
#     P_02 = geom.add_point([0.,l_1+l_2, 0.], lcar)
#     P_03 = geom.add_point([0.,l_1+l_2+l_3, 0.], lcar)

#     P_L0 = geom.add_point([L, 0., 0.], lcar)
#     P_L1 = geom.add_point([L,l_1, 0.], lcar)
#     P_L2 = geom.add_point([L,l_1+l_2, 0.], lcar)
#     P_L3 = geom.add_point([L,l_1+l_2+l_3, 0.], lcar)


#     lh_0 = geom.add_line(P_00, P_L0)
#     lh_1 = geom.add_line(P_01, P_L1)
#     lh_2 = geom.add_line(P_02, P_L2)
#     lh_3 = geom.add_line(P_03, P_L3)

#     lv_00 = geom.add_line(P_00, P_01)
#     lv_01 = geom.add_line(P_01, P_02)
#     lv_02 = geom.add_line(P_02, P_03)

#     lv_L0 = geom.add_line(P_L0, P_L1)
#     lv_L1 = geom.add_line(P_L1, P_L2)
#     lv_L2 = geom.add_line(P_L2, P_L3)


#     geom.add_physical(lh_3, label=p.transmission)
#     geom.add_physical([lv_L0, lv_L1, lv_L2], label="Periodicity right")
#     geom.add_physical([lv_00, lv_01, lv_02], label="Periodicity left")
#     geom.add_physical(lh_0, label=p.excitation)

#     # Lineloops for a layer
#     ll1 = geom.add_line_loop([lh_0, lv_L0, -lh_1, -lv_00])
#     ll2 = geom.add_line_loop([lh_1, lv_L1, -lh_2, -lv_01])
#     ll3 = geom.add_line_loop([lh_2, lv_L2, -lh_3, -lv_02])


#     s1 = geom.add_plane_surface(ll1)
#     s2 = geom.add_plane_surface(ll2)
#     s3 = geom.add_plane_surface(ll3)

#     geom.add_physical(s1, label=p.mat1 + " 1")
#     geom.add_physical(s2, label=p.mat2 + " 2")
#     geom.add_physical(s3, label=p.mat3 + " 3")

#     # For periodicity
#     geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(lv_00.id, lv_L0.id))
#     geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(lv_01.id, lv_L1.id))
#     geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(lv_02.id, lv_L2.id))

#     mesh = pg.generate_mesh(geom,verbose=False)

#     return mesh


# def five_layers(p):
#     geom = pg.built_in.Geometry()

#     L = p.lx
#     l_1 = p.l1
#     l_2 = p.l2
#     l_3 = p.l3
#     l_4 = p.l4
#     l_5 = p.l5


#     lcar = p.lcar
#     P_00 = geom.add_point([0., 0., 0.], lcar)
#     P_01 = geom.add_point([0., l_1, 0.], lcar)
#     P_02 = geom.add_point([0., l_1+l_2, 0.], lcar)
#     P_03 = geom.add_point([0., l_1+l_2+l_3, 0.], lcar)
#     P_04 = geom.add_point([0., l_1+l_2+l_3+l_4, 0.], lcar)
#     P_05 = geom.add_point([0., l_1+l_2+l_3+l_4+l_5, 0.], lcar)

#     P_L0 = geom.add_point([L, 0., 0.], lcar)
#     P_L1 = geom.add_point([L, l_1, 0.], lcar)
#     P_L2 = geom.add_point([L, l_1+l_2, 0.], lcar)
#     P_L3 = geom.add_point([L, l_1+l_2+l_3, 0.], lcar)
#     P_L4 = geom.add_point([L, l_1+l_2+l_3+l_4, 0.], lcar)
#     P_L5 = geom.add_point([L, l_1+l_2+l_3+l_4+l_5, 0.], lcar)


#     lh_0 = geom.add_line(P_00, P_L0)
#     lh_1 = geom.add_line(P_01, P_L1)
#     lh_2 = geom.add_line(P_02, P_L2)
#     lh_3 = geom.add_line(P_03, P_L3)
#     lh_4 = geom.add_line(P_04, P_L4)
#     lh_5 = geom.add_line(P_05, P_L5)


#     lv_00 = geom.add_line(P_00, P_01)
#     lv_01 = geom.add_line(P_01, P_02)
#     lv_02 = geom.add_line(P_02, P_03)
#     lv_03 = geom.add_line(P_03, P_04)
#     lv_04 = geom.add_line(P_04, P_05)


#     lv_L0 = geom.add_line(P_L0, P_L1)
#     lv_L1 = geom.add_line(P_L1, P_L2)
#     lv_L2 = geom.add_line(P_L2, P_L3)
#     lv_L3 = geom.add_line(P_L3, P_L4)
#     lv_L4 = geom.add_line(P_L4, P_L5)


#     geom.add_physical(lh_5, label=p.transmission)
#     geom.add_physical([lv_L0, lv_L1, lv_L2, lv_L3, lv_L4], label="Periodicity right")
#     geom.add_physical([lv_00, lv_01, lv_02, lv_03, lv_04], label="Periodicity left")
#     geom.add_physical(lh_0, label=p.excitation)

#     # Lineloops for a layer
#     ll1 = geom.add_line_loop([lh_0, lv_L0, -lh_1, -lv_00])
#     ll2 = geom.add_line_loop([lh_1, lv_L1, -lh_2, -lv_01])
#     ll3 = geom.add_line_loop([lh_2, lv_L2, -lh_3, -lv_02])
#     ll4 = geom.add_line_loop([lh_3, lv_L3, -lh_4, -lv_03])
#     ll5 = geom.add_line_loop([lh_4, lv_L4, -lh_5, -lv_04])

#     s1 = geom.add_plane_surface(ll1)
#     s2 = geom.add_plane_surface(ll2)
#     s3 = geom.add_plane_surface(ll3)
#     s4 = geom.add_plane_surface(ll4)
#     s5 = geom.add_plane_surface(ll5)

#     geom.add_physical(s1, label=p.mat1 + " 1")
#     geom.add_physical(s2, label=p.mat2 + " 2")
#     geom.add_physical(s3, label=p.mat3 + " 3")
#     geom.add_physical(s4, label=p.mat4 + " 4")
#     geom.add_physical(s5, label=p.mat5 + " 5")

#     # mesh = pg.generate_mesh(geom,verbose=False)
#     mesh = pg.generate_mesh(geom, geo_filename='GMSH/toto.geo', msh_filename='GMSH/toto.msh',verbose=False)

#     return mesh

# def metaporous_benchmark_7(p):
#     geom = pg.built_in.Geometry()
#     # circle = geom.add_circle(
#     #     x0=[0.5, 0.5, 0.0], radius=0.25, lcar=0.1, num_sections=4, make_surface=False
#     # )

#     # geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, lcar=0.1, holes=[circle.line_loop])

#     L = p.L
#     d = p.d
#     a = p.a

#     lcar = p.lcar
#     P1 = geom.add_point([0., 0., 0.], lcar)
#     P2 = geom.add_point([d, 0., 0.0], lcar)
#     P3 = geom.add_point([d, L, 0.0], lcar)
#     P4 = geom.add_point([0., L, 0.0], lcar)

#     l1 = geom.add_line(P1, P2)
#     l2 = geom.add_line(P2, P3)
#     l3 = geom.add_line(P3, P4)
#     l4 = geom.add_line(P4, P1)

#     inclusion =  ll_circle_plus(geom,a+a/2,d/2,a/4,0.01)
#     ll_inclusion = geom.add_line_loop(inclusion)

#     geom.add_physical([l3], label=p.transmission)
#     geom.add_physical([l2], label="Periodicity right")
#     geom.add_physical([l4], label="Periodicity left")
#     geom.add_physical(l1, label=p.excitation)

#     # # Layer 1
#     ll1 = geom.add_line_loop([l1, l2, l3, l4]+inclusion)
#     geom.add_physical(inclusion, label="Interface Inclusion -")
#     s1 = geom.add_plane_surface(ll1)
#     geom.add_physical(s1, label=p.pem1)

#     inclusion_minus = ll_circle_minus(geom, a+a/2, d/2, a/4, 0.01)
#     ll2 = geom.add_line_loop(inclusion_minus)
#     geom.add_physical(inclusion_minus, label="Interface Inclusion +")
#     s2 = geom.add_plane_surface(ll2)
#     geom.add_physical(s2, label=p.pem2)

#     # # For periodicity
#     # # geom.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(l2.id, l4.id))

#     mesh = pg.generate_mesh(geom, geo_filename='GMSH/pygmsh.geo', msh_filename='GMSH/pygmsh.msh',verbose=False)

#     return mesh

# def ll_circle_plus(g,x,y,r,lc):
#     n = 5
#     P_list =[g.add_point([x+r,y,0.],lc)]
#     ll_list = []
#     for i_p in range(1,n):
#         theta = i_p*2*np.pi/n
#         P_list.append(g.add_point([x+r*np.cos(theta),y+r*np.sin(theta),0.],lc))
#         ll_list.append(g.add_line(P_list[i_p-1], P_list[i_p]))
#     ll_list.append(g.add_line(P_list[-1], P_list[0]))
#     return ll_list

# def ll_circle_minus(g,x,y,r,lc):
#     n = 5
#     P_list =[g.add_point([x+r,y,0.],lc)]
#     ll_list = []
#     for i_p in range(1,n):
#         theta = i_p*2*np.pi/n
#         P_list.append(g.add_point([x+r*np.cos(theta),y-r*np.sin(theta),0.],lc))
#         ll_list.append(g.add_line(P_list[i_p-1], P_list[i_p]))
#     ll_list.append(g.add_line(P_list[-1], P_list[0]))
#     return ll_list