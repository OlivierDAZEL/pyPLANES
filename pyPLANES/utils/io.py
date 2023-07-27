#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_io.py
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


import socket
import datetime
import time
import os.path
import numpy as np

import pyvtk 

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *

from pyPLANES.core.result import Result

# def initialisation_out_files_plain(self):
#     pass

from pymls import from_yaml, Solver, Layer, backing
from mediapack import Air, Fluid
import importlib

plot_color = ["r", "b", "m", "k", "g", "y"]

def load_material(mat):
    if mat == "Air":
        Air_mat = Air()
        return Fluid(c=Air_mat.c,rho=Air_mat.rho)
    elif os.path.isfile("materials/" + mat + ".yaml") : # mediapack case
        return from_yaml("materials/" + mat + ".yaml")
    elif os.path.isfile("materials/" + mat + ".py") : # python dedicated file
        module = importlib.import_module("materials." + mat ) # Import the py 
        return module.mat()
    elif os.path.isfile("msh/" + mat + ".msh"): # Case of a metaporous
        return None
    else:
        raise NameError("Invalid Material {}".format(mat))

def run_pymls(**kwargs):
    name_project = kwargs.get("name_project", "unnamed_project")
    ml = kwargs.get("ml", False) 
    termination = kwargs.get("termination", "rigid")
    theta_d = kwargs.get("theta_d", 45) 
    freq = kwargs.get("frequencies", np.array([440])) 
    plot_RT = kwargs.get("plot_RT", False)
    result = Result()
    solver = Solver()
    for _l in ml:
        mat = load_material(_l[0])
        solver.layers.append(Layer(mat, _l[1]))
        
    R = []
    if termination in ["rigid", "Rigid", "Rigid Wall", "Wall"]:
        solver.backing = backing.rigid
    else: 
        solver.backing = backing.transmission
    for _f in freq:
        _ = solver.solve(_f, theta_d)
        result.f.append(_f)
        result.R0.append(_["R"][0])
        if termination == "transmission" :
            result.T0.append(_["T"][0])
    result.save("out/" + name_project + "_pymls","w")

def close_out_files(self):
    duration = time.time()-self.start_time
    self.info_file.write("Calculus ended at %s.\n"%(datetime.datetime.now()))
    self.info_file.write("Total duration = {} s\n".format(duration))
    self.info_file.write("duration / freq (averaged) = {} s\n".format(duration/len(self.frequencies)))
    self.out_file.close()
    self.info_file.close()

def print_entities(self):
    for _ in self.entities:
        print(_)

def print_elements(self):
    for _ in self.elements[1:]:
        print(_)

def print_vertices(self):
    for _ in self.vertices[1:]:
        print(_)

def print_edges(self):
    for _ in self.edges:
        print(_)

def print_faces(self):
    for _ in self.faces:
        print(_)

def print_model_entities(self):
    for _ in self.model_entities:
        print(_)

def print_reference_elements(self):
    print(self.reference_elements)

def plot_fem_solution(self, kx=0.):
    if self.plot[5]: # Determination of the maximum value of the pressure
        p_max = 0
        p_min = 1e308
        for _en in self.entities:
            if isinstance(_en, (FluidFem, PemFem)):
                for _elem in _en.elements:
                    _, __, p_elem = _elem.display_sol(3)
                    _max = np.amax(np.abs(p_elem))
                    _min = np.amin(np.abs(p_elem))
                    if _max >p_max: p_max = _max
                    if _min <p_min: p_min = _min

    if any(self.plot[3:]):
        x, y, u_x, u_y, pr = [], [], [], [], []
    for _en in self.entities:
        if isinstance(_en, FluidFem):
            if any(self.plot[2::3]): # Plot of pressure  == True
                for ie, _elem in enumerate(_en.elements):
                    x_elem, y_elem, p_elem = _elem.display_sol(3)
                    p_elem = p_elem[:, 0]
                    if kx !=0:
                        p_elem *= np.exp(1j*kx[0]*x_elem)
                    if self.plot[2]:
                        plt.figure("Pressure")
                        plt.plot(y_elem, np.abs(p_elem), 'r+')
                        plt.plot(y_elem, np.imag(p_elem), 'm+')
                    if self.plot[5]:
                        triang = mtri.Triangulation(x_elem, y_elem)
                        plt.figure("Pressure map")
                        plt.tricontourf(triang, np.abs(p_elem), cmap=cm.jet, levels=np.linspace(p_min, p_max,40))
                        # x.extend(list(x_elem))
                        # y.extend(list(y_elem))
                        # pr.extend(list(p_elem))
        elif isinstance(_en, PemFem):
            if any(self.plot): # Plot of pressure  == True
                for _elem in _en.elements:
                    x_elem, y_elem, f_elem = _elem.display_sol([0, 1, 3])
                    if type(kx)== float:
                        kx = [kx]
                    ux_elem = f_elem[:, 0]*np.exp(1j*kx[0]*x_elem)
                    uy_elem = f_elem[:, 1]*np.exp(1j*kx[0]*x_elem)
                    p_elem = f_elem[:, 2]*np.exp(1j*kx[0]*x_elem)
                    if self.plot[0]:
                        plt.figure("Solid displacement along x")
                        plt.plot(y_elem, np.abs(ux_elem), 'r+')
                        plt.plot(y_elem, np.imag(ux_elem), 'm+')
                    if self.plot[1]:
                        plt.figure("Solid displacement along y")
                        plt.plot(y_elem, np.abs(uy_elem), 'r+')
                        plt.plot(y_elem, np.imag(uy_elem), 'm+')
                    if self.plot[2]:
                        plt.figure("Pressure")
                        plt.plot(y_elem, np.abs(p_elem), 'r+')
                        plt.plot(y_elem, np.imag(p_elem), 'm+')
                    if self.plot[5]:
                        triang = mtri.Triangulation(x_elem, y_elem)
                        plt.figure("Pressure map")
                        plt.tricontourf(triang, np.abs(p_elem), cmap=cm.jet, levels=np.linspace(p_min, p_max,40))

        elif isinstance(_en, ElasticFem):
            if any(self.plot): # Plot of pressure  == True
                for _elem in _en.elements:
                    x_elem, y_elem, f_elem = _elem.display_sol([0, 1, 3])
                    ux_elem = f_elem[:, 0]
                    uy_elem = f_elem[:, 1]
                    if kx !=0:
                        ux_elem *= np.exp(1j*kx[0]*x_elem)
                        uy_elem *= np.exp(1j*kx[0]*x_elem)
                    if self.plot[0]:
                        plt.figure("Solid displacement along x")
                        plt.plot(y_elem, np.abs(ux_elem), 'r+')
                        plt.plot(y_elem, np.imag(ux_elem), 'm.')
                    if self.plot[1]:
                        plt.figure("Solid displacement along y")
                        plt.plot(y_elem, np.abs(uy_elem), 'r+')
                        plt.plot(y_elem, np.imag(uy_elem), 'm.')

    if any(self.plot[3:]):
        # triang = mtri.Triangulation(x, y)
        # if self.plot[5]:
        #     plt.figure("Pressure map")
        #     plt.tricontourf(triang, np.abs(pr), 40, cmap=cm.jet)
        self.display_mesh()
        plt.colorbar()
        plt.axis("off")
        plt.axis('equal')

    # plt.show()

def export_paraview(self):
    if self.export_paraview == 0: 
        self.vtk_points = [_v.coord for _v in self.vertices[1:]]
        self.vtk_triangle = [[_e.vertices[0].tag-1, _e.vertices[1].tag-1, _e.vertices[2].tag-1] for _e in self.elements[1:] if _e.typ==2]
    pressure = [np.abs(_v.sol[3]) for _v in self.vertices[1:]]

    # Bidouille pour que la tour et les lettres clignotent dans IAGS 2021
    pressure_max = max(pressure)
    light_on = self.export_paraview%4
    if light_on<2:
        tower_on = 0
    else:
        tower_on = 1

    for _ent in self.fem_entities:
        if _ent.dim ==2:
            if _ent.mat.MEDIUM_TYPE == "eqf":
                if _ent.mat.name == "tower":
                    for _elem in _ent.elements:
                        for _v in _elem.vertices:
                            _v.sol[3] = (1+(-1)**tower_on)*(pressure_max/2.)
                if _ent.mat.name == "letter":
                    for _elem in _ent.elements:
                        for _v in _elem.vertices:
                            _v.sol[3] = (1+(-1)**(tower_on+1))*(pressure_max/2.)

    pressure = [np.abs(_v.sol[3]) for _v in self.vertices[1:]]
    vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(self.vtk_points,triangle=self.vtk_triangle), pyvtk.PointData(pyvtk.Scalars(pressure,name='Pressure')))
    vtk.tofile("vtk/"+self.name_project + "-{}".format(self.export_paraview))
    self.export_paraview +=1
