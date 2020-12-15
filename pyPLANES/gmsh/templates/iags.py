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

from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh
from pyPLANES.gmsh.tools.write_geo_file import Gmsh as Gmsh

def find_points(string):
    _ind = string.find("+")
    x = string[:_ind]
    # print(x)
    string = string[_ind+1:]
    # print(string)
    ind_L = string.find("L")
    ind_C = string.find("C")
    ind_plus = string.find("+")
    # print(ind_L)
    # print(ind_C)
    # print(ind_plus)
    if (ind_L!=-1) or (ind_C!=-1) or (ind_plus!=-1):
        if ind_plus == -1: ind_plus = len(string)
        if ind_L == -1: ind_L = len(string)
        if ind_C == -1: ind_C = len(string)
        _ind = min([ind_L, ind_C, ind_plus]) 
        # print("_ind={}".format(_ind))
        y = string[:_ind]
        # print(y)
        string = string[_ind:]
        if string[0] == "+" : string = string[1:]
        # print(string)
    else:
        y = string
        string = ""
    # print(y)
    # print("x=" + x +"; y="+y)
    return (float(x),-float(y),string)

def import_line_loop(_p, G, lcar):
    G.list_lines = []
    x, y, _p = find_points(_p)
    p = G.new_point(x, y, lcar)
    while len(_p) >0:
        if _p[0] == "L":
            _p = _p[1:]
            x, y, _p = find_points(_p)
            if _p != "":
                G.new_point(x, y, lcar)
                G.new_line(G.list_points[-2], G.list_points[-1])
            else: 
                G.new_line(G.list_points[-1], p)
        elif _p[0] == "C":
            _p = _p[1:]
            # print(_p)
            x, y, _p = find_points(_p)
            G.new_point(x, y, lcar)
            x, y, _p = find_points(_p)
            G.new_point(x, y, lcar)
            x, y, _p = find_points(_p)
            if _p != "":
                G.new_point(x, y, lcar)
                G.new_bezier(G.list_points[-4], G.list_points[-3], G.list_points[-2], G.list_points[-1])
            else: 
                G.new_bezier(G.list_points[-3], G.list_points[-2], G.list_points[-1], p)
        else:
            raise NameError("First caracter is neither L nor C")

    ll = G.new_line_loop(G.list_lines)
    return ll


def import_surface(_p, G, _list, lcar):
    ll = import_line_loop(_p, G, lcar)
    _list.append(G.new_surface([ll.tag]))
    return ll 

def import_double_surface(_p_1, _p_2, G, _list_air, _list_porous, lcar):
    ll_interior = import_line_loop(_p_1, G, lcar)
    _list_air.append(G.new_surface([ll_interior.tag]))
    ll = import_line_loop(_p_2, G, lcar)
    _list_porous.append(G.new_surface([ll.tag, -ll_interior.tag]))
    return ll 


def iags( **kwargs):
    name_mesh = kwargs.get("name_mesh", "iags-2021")
    lcar = kwargs.get("lcar", 100)
    BC = kwargs.get("BC", ["Incident_PW", "Periodicity", "Rigid Wall", "Periodicity"])

    permeable_letters = kwargs.get("permeable_letters", False)
    permeable_letters = True

    G = Gmsh(name_mesh)

    L = 840.
    d = 350.

    list_interfaces = []

    svg_file = open("Logo_Iags_Blanc_Od.svg", "r")

    paths = svg_file.readlines()
    paths =paths[0].split("ZM")
    paths_iterator = iter(paths)

    list_2D_air = []
    list_2D_porous = []

    # Upper tower
    import_surface(next(paths_iterator), G, list_2D_air, lcar/10)
    # Lower tower
    import_surface(next(paths_iterator), G, list_2D_air, lcar/10)
    # Blason
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/30))
    # Vertical Wall 
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/30))
    # L 
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))
    # e 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # M 
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))    
    # a 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # ns Univ
    for ii in range(7): 
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))  
    # e 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # rsit + accent
    for ii in range(6): 
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))  
    # e 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # Institut 
    for ii in range(9): 
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10)) 
    # d 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar/10))
    # Apostrophe
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))
    # A
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar/10))
    # c
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))
    # o
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # usti
    for ii in range(5): 
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10)) 
    # q
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # u
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar))
    # e 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # Gr
    for ii in range(2):
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10)) 
    # ad
    for ii in range(2):
        list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # u 
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10))
    # a 
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # t 
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10))
    # e
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))
    # Sch 
    for ii in range(3):
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10))
    # oo 
    for ii in range(2):
        list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))     
    # l
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10))
    # 2
    list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10))
    # 0
    list_interfaces.append(import_double_surface(next(paths_iterator), next(paths_iterator), G, list_2D_air, list_2D_porous, lcar))  
    # 21
    for ii in range(2):
        list_interfaces.append(import_surface(next(paths_iterator), G, list_2D_porous,lcar/10))


    list_negative = [-ll.tag for ll in list_interfaces]

    p_0 = G.new_point(0, 0, lcar/10)
    p_1 = G.new_point(L, 0,lcar/10)
    p_2 = G.new_point(L, -d, lcar/10)
    p_3 = G.new_point(0, -d, lcar/10)
    lines = [None]*4
    lines[0] = G.new_line(p_0, p_1)
    lines[1] = G.new_line(p_1, p_2)
    lines[2] = G.new_line(p_2, p_3)
    lines[3] = G.new_line(p_3, p_0)
    boundary_domain = G.new_line_loop(lines)

    cavite = G.new_surface([boundary_domain.tag] +list_negative)
    list_2D_air.append(cavite)

    G.new_physical(lines, "method=FEM")
    G.new_physical(lines, "typ=1D")
    for bc in set(BC):
        # Determine the lines 
        list_lines = [lines[i] for i, _bc in enumerate(BC) if _bc == bc]
        G.new_physical(list_lines, "condition="+ bc)

    if permeable_letters:
        G.new_physical(list_2D_air + list_2D_porous, "method=FEM")
        G.new_physical(list_2D_air + list_2D_porous, "typ=2D")
        G.new_physical(list_2D_air, "mat=Air")
        G.new_physical(list_2D_porous, "mat=melamine_eqf_sigma")
    else:
        G.new_physical(list_2D_air, "method=FEM")
        G.new_physical(list_2D_air, "mat=Air")




    option = "-2 -v 0 "
    G.run_gmsh(option)