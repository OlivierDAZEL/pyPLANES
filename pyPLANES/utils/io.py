#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_io.py
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


import socket
import datetime
import time

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from pyPLANES.fem.entities_surfacic import *
from pyPLANES.fem.entities_volumic import *

# def initialisation_out_files_plain(self):
#     pass

from pymls import from_yaml, Solver, Layer, backing
from mediapack import Air, Fluid



def load_material(mat):
    if mat == "Air":
        Air_mat = Air()
        return Fluid(c=Air_mat.c,rho=Air_mat.rho)
    else:
        return from_yaml("materials/" + mat + ".yaml")

def result_pymls(**kwargs):
    name_project = kwargs.get("name_project", "unnamed_project")
    ml = kwargs.get("ml", False) 
    termination = kwargs.get("termination", "rigid")
    theta_d = kwargs.get("theta_d", 45) 
    freq = kwargs.get("frequencies", np.array([440])) 
    plot_RT = kwargs.get("plot_RT", False)
    solver = Solver()
    for _l in ml:
        mat = load_material(_l[0])
        solver.layers.append(Layer(mat, _l[1]))
        
    R = []
    if termination in ["rigid", "Rigid", "Rigid Wall", "Wall"]:
        solver.backing = backing.rigid
        T = False 
    else: 
        T = []
        solver.backing = backing.transmission
    for _f in freq:
        _ = solver.solve(_f, theta_d)
        R.append(_["R"][0])
        if termination == "transmission":
            T.append(_["T"][0])
    if plot_RT:
        plt.figure(name_project + "/ Reflection coefficient")
        plt.plot(freq, [_.real for _ in R], 'r',label="Re(R) pymls")
        plt.plot(freq, [_.imag for _ in R], 'b',label="Im(R) pymls")
        plt.legend()
        if T is not False:
            plt.figure(name_project + "/ Transmission coefficient")
            plt.plot(freq, [_.real for _ in T], 'r',label="Re(T) pymls")
            plt.plot(freq, [_.imag for _ in T], 'b',label="Im(T) pymls")
            plt.legend()
    return freq, R, T

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

def plot_fem_solution(self):
    if any(self.plot[3:]):
        x, y, u_x, u_y, pr = [], [], [], [], []
    for _en in self.entities:
        if isinstance(_en, FluidFem):
            if any(self.plot[2::3]): # Plot of pressure  == True
                for _elem in _en.elements:
                    x_elem, y_elem, p_elem = _elem.display_sol(3)
                    p_elem = p_elem[:, 0]
                    p_elem *= np.exp(1j*self.kx*x_elem)
                    if self.plot[2]:
                        plt.figure("Pressure")
                        plt.plot(y_elem, np.abs(p_elem), 'r+')
                        plt.plot(y_elem, np.imag(p_elem), 'm.')
                    if self.plot[5]:
                        triang = mtri.Triangulation(x_elem, y_elem)
                        plt.figure(5)
                        plt.tricontourf(triang, np.abs(p_elem), 40, cmap=cm.jet)
                        # x.extend(list(x_elem))
                        # y.extend(list(y_elem))
                        # pr.extend(list(p_elem))
        elif isinstance(_en, PemFem):
            if any(self.plot): # Plot of pressure  == True
                for _elem in _en.elements:
                    x_elem, y_elem, f_elem = _elem.display_sol([0, 1, 3])
                    ux_elem = f_elem[:, 0]*np.exp(1j*self.kx*x_elem)
                    uy_elem = f_elem[:, 1]*np.exp(1j*self.kx*x_elem)
                    p_elem = f_elem[:, 2]*np.exp(1j*self.kx*x_elem)
                    if self.plot[0]:
                        plt.figure("Solid displacement along x")
                        plt.plot(y_elem, np.abs(ux_elem), 'r+')
                        plt.plot(y_elem, np.imag(ux_elem), 'm.')
                    if self.plot[1]:
                        plt.figure("Solid displacement along y")
                        plt.plot(y_elem, np.abs(uy_elem), 'r+')
                        plt.plot(y_elem, np.imag(uy_elem), 'm.')
                    if self.plot[2]:
                        plt.figure("Pressure")
                        plt.plot(y_elem, np.abs(p_elem), 'r+')
                        plt.plot(y_elem, np.imag(p_elem), 'm.')
                    if self.plot[5]:
                        x.extend(list(x_elem))
                        y.extend(list(y_elem))
                        pr.extend(list(p_elem))
        elif isinstance(_en, ElasticFem):
            if any(self.plot): # Plot of pressure  == True
                for _elem in _en.elements:
                    x_elem, y_elem, f_elem = _elem.display_sol([0, 1, 3])
                    ux_elem = f_elem[:, 0]*np.exp(1j*self.kx*x_elem)
                    uy_elem = f_elem[:, 1]*np.exp(1j*self.kx*x_elem)
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
        if self.plot[5]:
            plt.figure(5)
            # plt.tricontourf(triang, np.abs(pr), 40, cmap=cm.jet)
        self.display_mesh()
        plt.colorbar()
        plt.axis('equal')
    # plt.show()







