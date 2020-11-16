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

from os import path, mkdir
import socket
import datetime
import time

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from pyPLANES.fem.entities_surfacic import *
from pyPLANES.fem.entities_volumic import *

def initialisation_out_files_plain(self):
    pass

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

def display_sol(self):
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
                        plt.figure(2)
                        plt.plot(y_elem, np.abs(p_elem), 'r+')
                        plt.plot(y_elem, np.imag(p_elem), 'm.')
                        plt.title("Pressure")
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
                        plt.figure(0)
                        plt.plot(y_elem, np.abs(ux_elem), 'r+')
                        plt.plot(y_elem, np.imag(ux_elem), 'm.')
                        plt.title("Solid displacement along x")
                    if self.plot[1]:
                        plt.figure(1)
                        plt.plot(y_elem, np.abs(uy_elem), 'r+')
                        plt.plot(y_elem, np.imag(uy_elem), 'm.')
                        plt.title("Solid displacement along y")
                    if self.plot[2]:
                        plt.figure(2)
                        plt.plot(y_elem, np.abs(p_elem), 'r+')
                        plt.plot(y_elem, np.imag(p_elem), 'm.')
                        plt.title("Pressure")
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
                        plt.figure(0)
                        plt.plot(y_elem, np.abs(ux_elem), 'r+')
                        plt.plot(y_elem, np.imag(ux_elem), 'm.')
                        plt.title("Solid displacement along x")
                    if self.plot[1]:
                        plt.figure(1)
                        plt.plot(y_elem, np.abs(uy_elem), 'r+')
                        plt.plot(y_elem, np.imag(uy_elem), 'm.')
                        plt.title("Solid displacement along y")
    if any(self.plot[3:]):
        # triang = mtri.Triangulation(x, y)
        if self.plot[5]:
            plt.figure(5)
            # plt.tricontourf(triang, np.abs(pr), 40, cmap=cm.jet)
        self.display_mesh()
        plt.colorbar()
        plt.axis('equal')
    # plt.show()







