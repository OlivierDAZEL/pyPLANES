#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# model.py
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
import os
import platform
import timeit

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, linalg as sla

import numpy as np
import numpy.linalg as LA

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mediapack import Air

from pyPLANES.classes.entity_classes import PwFem, IncidentPwFem, FluidFem, PemFem, TransmissionPwFem
from pyPLANES.gmsh.load_msh_file import load_msh_file
from pyPLANES.model.preprocess import preprocess, renumber_dof
from pyPLANES.utils.utils_outfiles import initialisation_out_files, write_out_files



class Model():
    def __init__(self, p):
        self.verbose = p.verbose
        self.name_project = p.name_project
        self.dim = 2
        self.theta_d = p.theta_d
        self.entities = [] # List of all GMSH Entities
        self.model_entities = [] # List of Entities used in the Model
        self.reference_elements = dict() # dictionary of reference_elements
        self.vertices = []
        self.elements = []
        self.edges = []
        self.faces = []
        self.bubbles =[]
        self.nb_edges, self.nb_faces, self.nb_bubbles = 0, 0, 0
        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.A_i_c, self.A_j_c, self.A_v_c = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        self.outfile = None
        self.modulus_reflex, self.modulus_trans, self.abs = 0, 0, 1
        if hasattr(p, "materials_directory"):
            self.materials_directory = p.materials_directory
        else:
            self.materials_directory = ""
        load_msh_file(self, p)
        initialisation_out_files(self, p)
        preprocess(self, p)

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

    def update_frequency(self, f):
        self.current_frequency = f
        omega = 2*np.pi*f
        self.kx = (omega/Air.c)*np.sin(self.theta_d*np.pi/180)
        self.ky = (omega/Air.c)*np.cos(self.theta_d*np.pi/180)
        self.delta_periodicity = np.exp(-1j*self.kx*self.period)
        self.nb_dofs = self.nb_dof_FEM
        for _ent in self.model_entities:
            _ent.update_frequency(omega)
        self.modulus_reflex, self.modulus_trans, self.abs = 0, 0, 1

    def __str__(self):
        out = "TBD"
        return out

    def extend_AF(self, _A_i, _A_j, _A_v, _F_i, _F_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def extend_F(self, _F_i, _F_v):
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def extend_A(self, _A_i, _A_j, _A_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)

    def extend_AT(self, _A_i, _A_j, _A_v, _T_i, _T_j, _T_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.T_i.extend(_T_i)
        self.T_j.extend(_T_j)
        self.T_v.extend(_T_v)

    def linear_system_2_numpy(self):
        self.F_i = np.array(self.F_i)
        self.F_v = np.array(self.F_v, dtype=complex)
        self.T_i = np.array(self.T_i)-self.nb_dof_master
        self.T_j = np.array(self.T_j)
        self.T_v = np.array(self.T_v, dtype=complex)
        self.A_i = np.array(self.A_i)
        self.A_j = np.array(self.A_j)
        self.A_v = np.array(self.A_v, dtype=complex)
        for _ent in self.model_entities:
            if isinstance(_ent, PwFem):
                _ent.rho_i = np.array(_ent.rho_i)
                _ent.rho_j = np.array(_ent.rho_j)
                _ent.rho_v = np.array(_ent.rho_v, dtype=complex)

    def create_linear_system(self, f):
        print("Creation of the linear system for f={}".format(f))
        self.update_frequency(f)
        omega = 2*np.pi*f
        # Initialisation of the lists
        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        for _ent in self.model_entities:
            if isinstance(_ent, (PwFem)):
                _ent.create_dynamical_matrices(omega)
            else: # only the matrix Matrix + RHS
                _A_i, _A_j, _A_v, _T_i, _T_j, _T_v = _ent.append_linear_system(omega)
                self.extend_AT(_A_i, _A_j, _A_v, _T_i, _T_j, _T_v)
        self.linear_system_2_numpy()
        self.apply_periodicity()


    def apply_periodicity(self):
        A_i, A_j, A_v = [], [], []
        if self.dof_left != []:
            for i_left, dof_left in enumerate(self.dof_left):
                # Corresponding dof
                dof_right = self.dof_right[i_left]
                # Summation of the columns for the Matrix
                index = np.where(self.A_j_c == dof_right)
                self.A_j[index] = dof_left
                for _i in index:
                    self.A_v[_i] *= self.delta_periodicity
                # Summation of the rows for the Matrix
                index = np.where(self.A_i == dof_right)
                self.A_i[index] = dof_left
                for _i in index:
                    self.A_v[_i] /= self.delta_periodicity
                # Summation of the rows for the Matrix
                A_i.append(dof_right)
                A_j.append(dof_left)
                A_v.append(self.delta_periodicity)
                A_i.append(dof_right)
                A_j.append(dof_right)
                A_v.append(-1)
        self.A_i = np.append(self.A_i, A_i)
        self.A_j = np.append(self.A_j, A_j)
        self.A_v = np.append(self.A_v, A_v)

        for _ent in self.model_entities:
            if isinstance(_ent, PwFem):
                for i_left, dof_left in enumerate(self.dof_left):
                    # Corresponding dof
                    dof_right = self.dof_right[i_left]
                    # Summation of the rows for the rho matrix
                    index = np.where(_ent.rho_i == dof_right-1)
                    _ent.rho_i[index] = dof_left-1
                    for _i in index:
                        _ent.rho_v[_i] /= self.delta_periodicity
                    _ent.rho = coo_matrix((_ent.rho_v, (_ent.rho_i, _ent.rho_j)), shape=(self.nb_dof_master-1, _ent.nb_dofs)).tocsr()

    def resolution(self, p):
        if p.verbose:
            print("%%%%%%%%%%%%% Resolution of PLANES %%%%%%%%%%%%%%%%%")
        for f in self.frequencies:
            self.create_linear_system(f)
            self.solve()
            write_out_files(self)
            # if self.verbose:
                # print("|R pyPLANES_FEM|  = {}".format(self.modulus_reflex))
                # print("|abs pyPLANES_FEM| = {}".format(self.abs))
            if any(p.plot):
                self .display_sol(p)
        self.outfile.close()
        self.resfile.close()
        name_server = platform.node()
        mail = " mailx -s \"Calculation of pyPLANES over on \"" + name_server + " olivier.dazel@univ-lemans.fr < " + self.resfile_name
        if name_server == "il-calc1":
            os.system(mail)

    def solve(self):
        self.nb_dof_condensed = self.nb_dof_FEM - self.nb_dof_master

        start = timeit.default_timer()

        index_A = np.where(((self.A_i*self.A_j) != 0) )
        A = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(self.nb_dof_master-1, self.nb_dof_master-1)).tocsr()
        rhs = np.zeros(self.nb_dof_master-1, dtype=complex)

        T = coo_matrix((self.T_v, (self.T_i, self.T_j)), shape=(self.nb_dof_condensed, self.nb_dof_master)).tocsr()

        for _ent in self.model_entities:
            if isinstance(_ent, PwFem):
                rho = _ent.rho[:self.nb_dof_master-1, :]
                A += rho.dot(_ent.Omega).dot(rho.H)/_ent.period
                if isinstance(_ent, IncidentPwFem):
                    rho_0 = _ent.rho[:self.nb_dof_master-1, _ent.dof_spec].toarray().reshape(self.nb_dof_master-1)
                    rhs += 2*rho_0*_ent.Omega[_ent.dof_spec, _ent.dof_spec]
        # Resolution of the sparse linear system
        X = linsolve.spsolve(A, rhs)
        # Concatenation of the first (zero) dof at the begining of the vector
        X = np.insert(X, 0, 0)
        # Concatenation of the slave dofs at the end of the vector
        X = np.insert(T@X, 0, X)
        stop = timeit.default_timer()
        if self.verbose:
            print("Elapsed time for linsolve = {} ms".format((stop-start)*1e3))

        for _vr in self.vertices[1:]:
            for i_dim in range(4):
                _vr.sol[i_dim] = X[_vr.dofs[i_dim]]
        for _ed in self.edges:
            for i_dim in range(4):
                _ed.sol[i_dim] = X[_ed.dofs[i_dim]]
        for _fc in self.faces:
            for i_dim in range(4):
                _fc.sol[i_dim] = X[_fc.dofs[i_dim]]
        for _bb in self.bubbles:
            for i_dim in range(4):
                _bb.sol[i_dim] = X[_bb.dofs[i_dim]]
        # self.abs has been sent to 1 in the __init__ () of the model class
        for _ent in self.entities[1:]:
            if isinstance(_ent, IncidentPwFem):
                _ent.sol = _ent.rho.H .dot(X[1:self.nb_dof_master])/_ent.period
                _ent.sol[_ent.dof_spec] -= 1
                self.modulus_reflex = np.sqrt(np.sum(np.real(_ent.ky)*np.abs(_ent.sol**2)/np.real(self.ky)))
                print("R pyPLANES_FEM   = {}".format((_ent.sol[_ent.dof_spec])))
                self.abs -= np.abs(self.modulus_reflex)**2
            elif isinstance(_ent, TransmissionPwFem):
                _ent.sol = _ent.rho.H .dot(X[1:])/_ent.period
                self.modulus_trans = np.sqrt(np.sum(np.real(_ent.ky)*np.abs(_ent.sol)**2/np.real(self.ky)))
                self.abs -= self.modulus_trans**2
        # print("abs pyPLANES_FEM   = {}".format(self.abs))

    def display_sol(self, p):
        if any(p.plot[3:]):
            x, y, u_x, u_y, pr = [], [], [], [], []
        for _en in self.entities:
            if isinstance(_en, FluidFem):
                if any(p.plot[2::3]): # Plot of pressure  == True
                    for _elem in _en.elements:
                        x_elem, y_elem, p_elem = _elem.display_sol(3)
                        p_elem = p_elem[:, 0]
                        p_elem *= np.exp(1j*self.kx*x_elem)
                        if p.plot[2]:
                            plt.figure(2)
                            plt.plot(y_elem, np.abs(p_elem), 'r+')
                            plt.plot(y_elem, np.imag(p_elem), 'm.')
                        if p.plot[5]:
                            x.extend(list(x_elem))
                            y.extend(list(y_elem))
                            pr.extend(list(p_elem))
            elif isinstance(_en, PemFem):
                if any(p.plot): # Plot of pressure  == True
                    for _elem in _en.elements:
                        x_elem, y_elem, f_elem = _elem.display_sol([0, 1, 3])
                        ux_elem = f_elem[:, 0]*np.exp(1j*self.kx*x_elem)
                        uy_elem = f_elem[:, 1]*np.exp(1j*self.kx*x_elem)
                        p_elem = f_elem[:, 2]*np.exp(1j*self.kx*x_elem)
                        if p.plot[0]:
                            plt.figure(0)
                            plt.plot(y_elem, np.abs(ux_elem), 'r+')
                            plt.plot(y_elem, np.imag(ux_elem), 'm.')
                        if p.plot[1]:
                            plt.figure(1)
                            plt.plot(y_elem, np.abs(uy_elem), 'r+')
                            plt.plot(y_elem, np.imag(uy_elem), 'm.')
                        if p.plot[2]:
                            plt.figure(2)
                            plt.plot(y_elem, np.abs(p_elem), 'r+')
                            plt.plot(y_elem, np.imag(p_elem), 'm.')
                        if p.plot[5]:
                            x.extend(list(x_elem))
                            y.extend(list(y_elem))
                            pr.extend(list(p_elem))
        if any(p.plot[3:]):
            triang = mtri.Triangulation(x, y)
            if p.plot[5]:
                plt.figure(5)
                plt.tricontourf(triang, np.abs(pr), 40, cmap=cm.jet)
            self.display_mesh()
            plt.colorbar()
            plt.axis('equal')
        # plt.show()

    def display_mesh(self):
        x_vertices =[_nd.coord[0] for _nd in self.vertices[1:]]
        y_vertices =[_nd.coord[1] for _nd in self.vertices[1:]]
        tri_vertices = mtri.Triangulation(x_vertices, y_vertices)
        plt.triplot(tri_vertices, 'ko-', lw=0.5, ms=2)





