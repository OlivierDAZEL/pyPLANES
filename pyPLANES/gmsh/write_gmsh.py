#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# gmsh.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
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
import gmsh
import numpy as np

class GmshSurface():
    def __init__(self, name, list_vertices, mat):
        self.name = name 
        self.list_vertices = list_vertices
        self.material = mat
        # list of the lines, each line is a tuple ("v_1","v_2") with v_1 and v_2 being the vertices
        self.lines= []
        for i, vertice in enumerate(self.list_vertices[:-1]):
            self.lines.append((vertice, self.list_vertices[i+1]))
        self.lines.append((self.list_vertices[-1], self.list_vertices[0]))
        # line_loop 
        self.line_loop = self.list_vertices
        self.string_lineloop = "["
        for i, v  in enumerate(self.list_vertices[:-1]):
            self.string_lineloop  += "line_" + v + self.list_vertices[i+1] + ","
        self.string_lineloop  += "line_" + self.list_vertices[i+1] + self.list_vertices[0] +"]"

class GmshPhysicalCondition():
    def __init__(self, curve, cond):
        if isinstance(curve, str):
            self.list_curves = [curve]
        else:
            self.list_curves = curve
        self.cond = cond
        self.list_index_curves = None # will be define by add_condition
    def __str__(self):
        return f"list: {self.list_curves}\ncond: {self.cond}"

class GmshPeriodicitycondition():
    def __init__(self, s_1, s_2):
        self.s_1 = s_1
        self.s_2 = s_2
        self.period = None
        
    def __str__(self):
        out = "Periodicity condition\n"
        out += f"to segments {self.s_1}/{self.index_1} and {self.s_2}/{self.index_2} with period={self.period}"
        return out
        
class GmshModelpyPLANES():
    def __init__(self, dic_vertices, lcar=1):
        self.dic_vertices = dic_vertices
        self.lcar = lcar
        self.list_curves = [] # list of curves, each one of them is a tuple with the vertices
        self.list_surfaces = [] # list of curves, each one of them is a GmshSurface instance
        self.list_conditions = [] # list of curves, each one of them is a GmshPhysicalcondition instance
        self.list_periodicity = [] # list of curves, each one of them is a GmshPeriodicitycondition instance
        # self.list_1d_entities_with_conditions = []
        
    def addSurface(self, name, list_vertices, mat):
        # test if vertices exists
        for v in list_vertices:
            if v not in self.dic_vertices.keys():
                raise NameError(f"{v} is not an existing vertex")
        s = GmshSurface(name, list_vertices, mat)
        self.list_curves.extend(s.lines)
        self.list_surfaces.append(s)





    def addCondition(self, curve, cond):
        list_existing_conditions = [c.cond for c in self.list_conditions]
        if cond in list_existing_conditions: # 
            # print("it exists")
            # Corresponding condition
            c = self.list_conditions[list_existing_conditions.index(cond)]
            list_index_curves = self.index_of_curves(curve)
            # if isinstance(curve, list):
            c.list_curves.extend(curve)
            c.list_index_curves.extend(list_index_curves)
            # else:
            #     c.list_curves.append(curve)
            #     c.list_index_curves.append(list_index_curves)
        else: # creation of a new condition
            # print("it does not exists")
            c =GmshPhysicalCondition(curve, cond)
            c.list_index_curves = self.index_of_curves(c.list_curves)
            # self.list_1d_entities_with_conditions.extend(c.list_index_curves)
            self.list_conditions.append(c)

    def addPeriodicity(self, s_1, s_2):
        
        self.addCondition([s_1, s_2], "condition=Periodicity")
        c = GmshPeriodicitycondition(s_1, s_2)
        c.i_1, c.i_2 = self.index_of_curves([s_1, s_2])
        # Determination of the two lines 
        model_list_curves = [set(l) for l in self.list_curves]
        set_1, set_2 = set(s_1), set(s_2)
        x_1= self.dic_vertices[s_1[0]][0]
        x_2= self.dic_vertices[s_2[0]][0]
        # check if the period match with the previous ones
        for p in self.list_periodicity:
            if p.period != x_2-x_1:
                raise NameError("Error in Addperiodicity condition different periods")

        c.period = x_2-x_1
        self.list_periodicity.append(c)

    def index_of_curves(self, list_curves):
        """ Indentify the gmsh indices of the curves given as parameter 
        Input list_curves 
        Output list of indices
        """
        list_index_curves = [] # output list 
        if isinstance(list_curves, str):
            list_curves = [list_curves]
        list_curves = [set(l) for l in list_curves] # list of model curve as sets (to skip order)
        model_list_curves = [set(l) for l in self.list_curves]
        for c in list_curves:
            if isinstance(c, set) and len(c)==2:
                if c in model_list_curves:
                    list_index_curves.append(model_list_curves.index(c)+1)
                else:
                    raise NameError(f"condition on a line {c} which does not exist")
            else:
                raise NameError(f"Invalid entity name {c}")
        return list_index_curves

    def create_msh_file(self, name_file):
        # self.checkup()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        # Creation of the vertices
        for name_point, coord in self.dic_vertices.items():
            exec( f"vertice_{name_point} = gmsh.model.geo.addPoint({coord[0]},{coord[1]},0,{self.lcar})")
        # Creation of the curves
        for line in self.list_curves:
            exec(f"line_{''.join(line)} = gmsh.model.geo.addLine(vertice_{line[0]},vertice_{line[1]})")
        # Creation of the surfaces
        for s in self.list_surfaces:
            exec(f"line_loop_{s.line_loop} = gmsh.model.geo.addCurveLoop({s.string_lineloop})")
            exec(f"surface_{s.name} = gmsh.model.geo.addPlaneSurface([line_loop_{s.line_loop}])")
        # Remove duplicate materials and assign them to the right surfaces
        list_materials = [s.material for s in self.list_surfaces]
        list_unique_materials = list(set(list_materials))
        list_surfaces_by_material = []
        for m in list_unique_materials:
            list_surfaces_by_material.append([i+1 for i, x in enumerate(list_materials) if x == m])
        for i_m, m in enumerate(list_unique_materials):
            exec(f"gmsh.model.geo.addPhysicalGroup(2,{str(list_surfaces_by_material[i_m])},name=\"mat={m}\")")
        for c in self.list_conditions:
            exec( f"gmsh.model.geo.addPhysicalGroup(1,{str(c.list_index_curves)},name=\"{c.cond}\")")
        gmsh.model.geo.synchronize()
        # Generate mesh:
        gmsh.model.mesh.generate()
        # Periodic conditions for the mesh
        if len(self.list_periodicity) !=0:
            affine_transform = np.eye(4)
            affine_transform[0,3] = self.list_periodicity[0].period # taken on the first elements because they are all equal
            affine_transform = str(list(affine_transform.flatten()))
            i_2 = [p.i_2 for p in self.list_periodicity]
            i_1 = [p.i_1 for p in self.list_periodicity]

            exec( f"gmsh.model.mesh.setPeriodic(1,{str(i_2)},{str(i_1)},{affine_transform})") 

        # Write mesh data:
        gmsh.write(f"msh/{name_file}.geo_unrolled")
        gmsh.write(f"msh/{name_file}.msh")
        # gmsh.write(f"msh/{name_file}.pdf")
        gmsh.finalize()

