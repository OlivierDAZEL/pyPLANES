#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_classes.py
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

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from mediapack import from_yaml
from mediapack import Air, PEM, EqFluidJCA

from pyPLANES.utils.io import initialisation_out_files_plain
from pyPLANES.core.calculus import PwCalculus
from pyPLANES.core.multilayer import MultiLayer

from pyPLANES.pw.pw_layers import FluidLayer
from pyPLANES.pw.pw_interfaces import FluidFluidInterface, RigidBacking

Air = Air()

# def initialise_PW_solver(L, b):
#     nb_PW = 0
#     dofs = []
#     for _layer in L:
#         if _layer.medium.MODEL == "fluid":
#             dofs.append(nb_PW+np.arange(2))
#             nb_PW += 2
#         elif _layer.medium.MODEL == "pem":
#             dofs.append(nb_PW+np.arange(6))
#             nb_PW += 6
#         elif _layer.medium.MODEL == "elastic":
#             dofs.append(nb_PW+np.arange(4))
#             nb_PW += 4
#     interface = []
#     for i_l, _layer in enumerate(L[:-1]):
#         interface.append((L[i_l].medium.MODEL, L[i_l+1].medium.MODEL))
#     return nb_PW, interface, dofs


class PwProblem(PwCalculus, MultiLayer):
    """
        Plane Wave Problem 
    """ 
    def __init__(self, **kwargs):
        PwCalculus.__init__(self, **kwargs)
        termination = kwargs.get("termination","rigid")
        self.method = kwargs.get("termination","global")

        MultiLayer.__init__(self, **kwargs)

        self.kx, self.ky, self.k = None, None, None
        self.shift_plot = kwargs.get("shift_pw", 0.)
        self.plot = kwargs.get("plot_results", [False]*6)
        self.result = {}
        self.outfiles_directory = False

        if self.method == "global":
            self.layers.insert(0,FluidLayer(Air,1.e-2))
            if self.layers[1].medium.MEDIUM_TYPE == "fluid":
                self.interfaces.append(FluidFluidInterface(self.layers[0],self.layers[1]))
            self.nb_PW = 0
            for _layer in self.layers:
                if _layer.medium.MODEL == "fluid":
                    _layer.dofs = self.nb_PW+np.arange(2)
                    self.nb_PW += 2
                elif _layer.medium.MODEL == "pem":
                    _layer.dofs = self.nb_PW+np.arange(6)
                    self.nb_PW += 6
                elif _layer.medium.MODEL == "elastic":
                    _layer.dofs = self.nb_PW+np.arange(4)
                    self.nb_PW += 4


    def update_frequency(self, f):
        PwCalculus.update_frequency(self, f)
        MultiLayer.update_frequency(self, f, self.k, self.kx)


    def create_linear_system(self, f):
        self.A = np.zeros((self.nb_PW-1, self.nb_PW), dtype=complex)
        i_eq = 0
        # Loop on the interfaces
        for _int in self.interfaces:
            if self.method == "global":
                i_eq = _int.update_M_global(self.A, i_eq)
        



        # for i_inter, _inter in enumerate(self.interfaces):
        #     if _inter[0] == "fluid":
        #         if _inter[1] == "fluid":
        #             i_eq = self.interface_fluid_fluid(i_eq, i_inter, Layers, dofs, M)
        #         if _inter[1] == "pem":
        #             i_eq = self.interface_fluid_pem(i_eq, i_inter, Layers, dofs, M)
        #         if _inter[1] == "elastic":
        #             i_eq = self.interface_fluid_elastic(i_eq, i_inter, Layers, dofs, M)
        #     elif _inter[0] == "pem":
        #         if _inter[1] == "fluid":
        #             i_eq = self.interface_pem_fluid(i_eq, i_inter, Layers, dofs, M)
        #         if _inter[1] == "pem":
        #             i_eq = self.interface_pem_pem(i_eq, i_inter, Layers, dofs, M)
        #         if _inter[1] == "elastic":
        #             i_eq = self.interface_pem_elastic(i_eq, i_inter, Layers, dofs, M)
        #     elif _inter[0] == "elastic":
        #         if _inter[1] == "fluid":
        #             i_eq = self.interface_elastic_fluid(i_eq, i_inter, Layers, dofs, M)
        #         if _inter[1] == "pem":
        #             i_eq = self.interface_elastic_pem(i_eq, i_inter, Layers, dofs, M)
        #         if _inter[1] == "elastic":
        #             i_eq = self.interface_elastic_elastic(i_eq, i_inter, Layers, dofs, M)
        # if self.backing == backing.rigid:
        #     if Layers[-1].medium.MODEL == "fluid":
        #         i_eq = self.interface_fluid_rigid(M, i_eq, Layers[-1], dofs[-1] )
        #     elif Layers[-1].medium.MODEL == "pem":
        #         i_eq = self.interface_pem_rigid(M, i_eq, Layers[-1], dofs[-1])
        #     elif Layers[-1].medium.MODEL == "elastic":
        #         i_eq = self.interface_elastic_rigid(M, i_eq, Layers[-1], dofs[-1])
        # elif self.backing == "transmission":
        #     i_eq = self.semi_infinite_medium(M, i_eq, Layers[-1], dofs[-1] )

        self.F = -self.A[:, 0]*np.exp(1j*self.ky*self.layers[0].d) # - is for transposition, exponential term is for the phase shift
        self.A = np.delete(self.A, 0, axis=1)
        # print(self.A)
        X = LA.solve(self.A, self.F)
        # print(X)
        # R_pyPLANES_PW = X[0]
        # if self.backing == "transmission":
        #     T_pyPLANES_PW = X[-2]
        # else:
        #     T_pyPLANES_PW = 0.
        # X = np.delete(X, 0)
        # del(dofs[0])
        # for i, _ld in enumerate(dofs):
        #     dofs[i] -= 2
        # if self.plot:
        #     self.plot_sol_PW(X, dofs)
        # out["R"] = R_pyPLANES_PW
        # out["T"] = T_pyPLANES_PW
        # return out





# class Solver_PW(PwCalculus):
#     def __init__(self, **kwargs):
#         PwCalculus.__init__(self, **kwargs)
#         ml = kwargs.get("ml")
#         termination = kwargs.get("termination")
#         self.layers = []
#         for _l in ml:
#             if _l[0] == "Air":
#                 mat = Air
#             else:
#                 mat = from_yaml(_l[0]+".yaml")
#             d = _l[1]
#             self.layers.append(Layer(mat,d))
#         if termination in ["trans", "transmission","Transmission"]:
#             self.backing = "Transmission"
#         else:
#             self.backing = backing.rigid

#         self.kx, self.ky, self.k = None, None, None
#         self.shift_plot = kwargs.get("shift_pw", 0.)
#         self.plot = kwargs.get("plot_results", [False]*6)
#         self.result = {}
#         self.outfiles_directory = False
#         initialisation_out_files_plain(self)

#     def write_out_files(self, out):

#         self.out_file.write("{:.12e}\t".format(self.current_frequency))
#         abs = 1-np.abs(out["R"])**2
#         self.out_file.write("{:.12e}\t".format(abs))
#         self.out_file.write("\n")




#     def interface_fluid_fluid(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = fluid_SV(self.kx, self.k, L[iinter].medium.K)
#         SV_2, k_y_2 = fluid_SV(self.kx, self.k, L[iinter+1].medium.K)
#         M[ieq, d[iinter+0][0]] = SV_1[0, 0]*np.exp(-1j*k_y_1*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[0, 1]
#         M[ieq, d[iinter+1][0]] = -SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[0, 1]*np.exp(-1j*k_y_2*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[1, 0]*np.exp(-1j*k_y_1*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[1, 1]
#         M[ieq, d[iinter+1][0]] = -SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[1, 1]*np.exp(-1j*k_y_2*L[iinter+1].thickness)
#         ieq += 1
#         return ieq

#     def interface_fluid_rigid(self, M, ieq, L, d):
#         SV, k_y = fluid_SV(self.kx, self.k, L.medium.K)
#         M[ieq, d[0]] = SV[0, 0]*np.exp(-1j*k_y*L.thickness)
#         M[ieq, d[1]] = SV[0, 1]
#         ieq += 1
#         return ieq

#     def semi_infinite_medium(self, M, ieq, L, d):
#         M[ieq, d[1]] = 1.
#         ieq += 1
#         return ieq

#     def interface_pem_pem(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = PEM_SV(L[iinter].medium, self.kx)
#         SV_2, k_y_2 = PEM_SV(L[iinter+1].medium, self.kx)
#         for _i in range(6):
#             M[ieq, d[iinter+0][0]] =  SV_1[_i, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#             M[ieq, d[iinter+0][1]] =  SV_1[_i, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#             M[ieq, d[iinter+0][2]] =  SV_1[_i, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#             M[ieq, d[iinter+0][3]] =  SV_1[_i, 3]
#             M[ieq, d[iinter+0][4]] =  SV_1[_i, 4]
#             M[ieq, d[iinter+0][5]] =  SV_1[_i, 5]
#             M[ieq, d[iinter+1][0]] = -SV_2[_i, 0]
#             M[ieq, d[iinter+1][1]] = -SV_2[_i, 1]
#             M[ieq, d[iinter+1][2]] = -SV_2[_i, 2]
#             M[ieq, d[iinter+1][3]] = -SV_2[_i, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#             M[ieq, d[iinter+1][4]] = -SV_2[_i, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#             M[ieq, d[iinter+1][5]] = -SV_2[_i, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#             ieq += 1
#         return ieq

#     def interface_fluid_pem(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = fluid_SV(self.kx, self.k, L[iinter].medium.K)
#         SV_2, k_y_2 = PEM_SV(L[iinter+1].medium,self.kx)
#         # print(k_y_2)
#         M[ieq, d[iinter+0][0]] = SV_1[0, 0]*np.exp(-1j*k_y_1*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[0, 1]
#         M[ieq, d[iinter+1][0]] = -SV_2[2, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[2, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[2, 2]
#         M[ieq, d[iinter+1][3]] = -SV_2[2, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = -SV_2[2, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = -SV_2[2, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[1, 0]*np.exp(-1j*k_y_1*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[1, 1]
#         M[ieq, d[iinter+1][0]] = -SV_2[4, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[4, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[4, 2]
#         M[ieq, d[iinter+1][3]] = -SV_2[4, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = -SV_2[4, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = -SV_2[4, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+1][0]] = SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[0, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[0, 2]
#         M[ieq, d[iinter+1][3]] = SV_2[0, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = SV_2[0, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = SV_2[0, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+1][0]] = SV_2[3, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[3, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[3, 2]
#         M[ieq, d[iinter+1][3]] = SV_2[3, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = SV_2[3, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = SV_2[3, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         return ieq

#     def interface_elastic_pem(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = elastic_SV(L[iinter].medium,self.kx, self.omega)
#         SV_2, k_y_2 = PEM_SV(L[iinter+1].medium,self.kx)
#         # print(k_y_2)
#         M[ieq, d[iinter+0][0]] = -SV_1[0, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[0, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[0, 2]
#         M[ieq, d[iinter+0][3]] = -SV_1[0, 3]
#         M[ieq, d[iinter+1][0]] = SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[0, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[0, 2]
#         M[ieq, d[iinter+1][3]] = SV_2[0, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = SV_2[0, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = SV_2[0, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = -SV_1[1, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[1, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[1, 2]
#         M[ieq, d[iinter+0][3]] = -SV_1[1, 3]
#         M[ieq, d[iinter+1][0]] = SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[1, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[1, 2]
#         M[ieq, d[iinter+1][3]] = SV_2[1, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = SV_2[1, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = SV_2[1, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = -SV_1[1, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[1, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[1, 2]
#         M[ieq, d[iinter+0][3]] = -SV_1[1, 3]
#         M[ieq, d[iinter+1][0]] = SV_2[2, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[2, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[2, 2]
#         M[ieq, d[iinter+1][3]] = SV_2[2, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = SV_2[2, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = SV_2[2, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = -SV_1[2, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[2, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[2, 2]
#         M[ieq, d[iinter+0][3]] = -SV_1[2, 3]
#         M[ieq, d[iinter+1][0]] = (SV_2[3, 0]-SV_2[4, 0])
#         M[ieq, d[iinter+1][1]] = (SV_2[3, 1]-SV_2[4, 1])
#         M[ieq, d[iinter+1][2]] = (SV_2[3, 2]-SV_2[4, 2])
#         M[ieq, d[iinter+1][3]] = (SV_2[3, 3]-SV_2[4, 3])*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = (SV_2[3, 4]-SV_2[4, 4])*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = (SV_2[3, 5]-SV_2[4, 5])*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = -SV_1[3, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[3, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[3, 2]
#         M[ieq, d[iinter+0][3]] = -SV_1[3, 3]
#         M[ieq, d[iinter+1][0]] = SV_2[5, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[5, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[5, 2]
#         M[ieq, d[iinter+1][3]] = SV_2[5, 3]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][4]] = SV_2[5, 4]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][5]] = SV_2[5, 5]*np.exp(-1j*k_y_2[2]*L[iinter+1].thickness)
#         ieq += 1
#         return ieq

#     def interface_pem_elastic(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = PEM_SV(L[iinter].medium,self.kx)
#         SV_2, k_y_2 = elastic_SV(L[iinter+1].medium,self.kx, self.omega)
#         # print(k_y_2)
#         M[ieq, d[iinter+0][0]] = SV_1[0, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[0, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[0, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = SV_1[0, 3]
#         M[ieq, d[iinter+0][4]] = SV_1[0, 4]
#         M[ieq, d[iinter+0][5]] = SV_1[0, 5]
#         M[ieq, d[iinter+1][0]] = -SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[0, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[0, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = -SV_2[0, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)

#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[1, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[1, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[1, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = SV_1[1, 3]
#         M[ieq, d[iinter+0][4]] = SV_1[1, 4]
#         M[ieq, d[iinter+0][5]] = SV_1[1, 5]
#         M[ieq, d[iinter+1][0]] = -SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[1, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[1, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = -SV_2[1, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)

#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[2, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[2, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[2, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = SV_1[2, 3]
#         M[ieq, d[iinter+0][4]] = SV_1[2, 4]
#         M[ieq, d[iinter+0][5]] = SV_1[2, 5]
#         M[ieq, d[iinter+1][0]] = -SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[1, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[1, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = -SV_2[1, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)

#         ieq += 1
#         M[ieq, d[iinter+0][0]] = (SV_1[3, 0]-SV_1[4, 0])*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = (SV_1[3, 1]-SV_1[4, 1])*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = (SV_1[3, 2]-SV_1[4, 2])*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = (SV_1[3, 3]-SV_1[4, 3])
#         M[ieq, d[iinter+0][4]] = (SV_1[3, 4]-SV_1[4, 4])
#         M[ieq, d[iinter+0][5]] = (SV_1[3, 5]-SV_1[4, 5])
#         M[ieq, d[iinter+1][0]] = -SV_2[2, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[2, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[2, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = -SV_2[2, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)

#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[5, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[5, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[5, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = SV_1[5, 3]
#         M[ieq, d[iinter+0][4]] = SV_1[5, 4]
#         M[ieq, d[iinter+0][5]] = SV_1[5, 5]
#         M[ieq, d[iinter+1][0]] = -SV_2[3, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[3, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[3, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = -SV_2[3, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         ieq += 1
#         return ieq

#     def interface_elastic_elastic(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = elastic_SV(L[iinter].medium,self.kx, self.omega)
#         SV_2, k_y_2 = elastic_SV(L[iinter+1].medium,self.kx, self.omega)
#         for _i in range(4):
#             M[ieq, d[iinter+0][0]] =  SV_1[_i, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#             M[ieq, d[iinter+0][1]] =  SV_1[_i, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#             M[ieq, d[iinter+0][2]] =  SV_1[_i, 2]
#             M[ieq, d[iinter+0][3]] =  SV_1[_i, 3]
#             M[ieq, d[iinter+1][0]] = -SV_2[_i, 0]
#             M[ieq, d[iinter+1][1]] = -SV_2[_i, 1]
#             M[ieq, d[iinter+1][2]] = -SV_2[_i, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#             M[ieq, d[iinter+1][3]] = -SV_2[_i, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#             ieq += 1
#         return ieq

#     def interface_fluid_elastic(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = fluid_SV(self.kx, self.k, L[iinter].medium.K)
#         SV_2, k_y_2 = elastic_SV(L[iinter+1].medium, self.kx, self.omega)
#         # Continuity of u_y
#         M[ieq, d[iinter+0][0]] =  SV_1[0, 0]*np.exp(-1j*k_y_1*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] =  SV_1[0, 1]
#         M[ieq, d[iinter+1][0]] = -SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = -SV_2[1, 1]
#         M[ieq, d[iinter+1][2]] = -SV_2[1, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = -SV_2[1, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         ieq += 1
#         # sigma_yy = -p
#         M[ieq, d[iinter+0][0]] = SV_1[1, 0]*np.exp(-1j*k_y_1*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[1, 1]
#         M[ieq, d[iinter+1][0]] = SV_2[2, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[2, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[2, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = SV_2[2, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         ieq += 1
#         # sigma_xy = 0
#         M[ieq, d[iinter+1][0]] = SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[0, 1]
#         M[ieq, d[iinter+1][2]] = SV_2[0, 2]*np.exp(-1j*k_y_2[0]*L[iinter+1].thickness)
#         M[ieq, d[iinter+1][3]] = SV_2[0, 3]*np.exp(-1j*k_y_2[1]*L[iinter+1].thickness)
#         ieq += 1
#         return ieq

#     def interface_pem_fluid(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = PEM_SV(L[iinter].medium, self.kx)
#         SV_2, k_y_2 = fluid_SV(self.kx, self.k, L[iinter+1].medium.K)
#         # print(k_y_2)
#         M[ieq, d[iinter+0][0]] = -SV_1[2, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[2, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[2, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = -SV_1[2, 3]
#         M[ieq, d[iinter+0][4]] = -SV_1[2, 4]
#         M[ieq, d[iinter+0][5]] = -SV_1[2, 5]
#         M[ieq, d[iinter+1][0]] = SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[0, 1]*np.exp(-1j*k_y_2*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = -SV_1[4, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[4, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[4, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = -SV_1[4, 3]
#         M[ieq, d[iinter+0][4]] = -SV_1[4, 4]
#         M[ieq, d[iinter+0][5]] = -SV_1[4, 5]
#         M[ieq, d[iinter+1][0]] = SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[1, 1]*np.exp(-1j*k_y_2*L[iinter+1].thickness)
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[0, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[0, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[0, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = SV_1[0, 3]
#         M[ieq, d[iinter+0][4]] = SV_1[0, 4]
#         M[ieq, d[iinter+0][5]] = SV_1[0, 5]
#         ieq += 1
#         M[ieq, d[iinter+0][0]] = SV_1[3, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[3, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[3, 2]*np.exp(-1j*k_y_1[2]*L[iinter].thickness)
#         M[ieq, d[iinter+0][3]] = SV_1[3, 3]
#         M[ieq, d[iinter+0][4]] = SV_1[3, 4]
#         M[ieq, d[iinter+0][5]] = SV_1[3, 5]
#         ieq += 1
#         return ieq

#     def interface_elastic_fluid(self, ieq, iinter, L, d, M):
#         SV_1, k_y_1 = elastic_SV(L[iinter].medium, self.kx, self.omega)
#         SV_2, k_y_2 = fluid_SV(self.kx, self.k, L[iinter+1].medium.K)
#         # Continuity of u_y
#         M[ieq, d[iinter+0][0]] = -SV_1[1, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = -SV_1[1, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = -SV_1[1, 2]
#         M[ieq, d[iinter+0][3]] = -SV_1[1, 3]
#         M[ieq, d[iinter+1][0]] = SV_2[0, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[0, 1]*np.exp(-1j*k_y_2*L[iinter+1].thickness)
#         ieq += 1
#         # sigma_yy = -p
#         M[ieq, d[iinter+0][0]] = SV_1[2, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[2, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[2, 2]
#         M[ieq, d[iinter+0][3]] = SV_1[2, 3]
#         M[ieq, d[iinter+1][0]] = SV_2[1, 0]
#         M[ieq, d[iinter+1][1]] = SV_2[1, 1]*np.exp(-1j*k_y_2*L[iinter+1].thickness)
#         ieq += 1
#         # sigma_xy = 0
#         M[ieq, d[iinter+0][0]] = SV_1[0, 0]*np.exp(-1j*k_y_1[0]*L[iinter].thickness)
#         M[ieq, d[iinter+0][1]] = SV_1[0, 1]*np.exp(-1j*k_y_1[1]*L[iinter].thickness)
#         M[ieq, d[iinter+0][2]] = SV_1[0, 2]
#         M[ieq, d[iinter+0][3]] = SV_1[0, 3]
#         ieq += 1
#         return ieq

#     def interface_elastic_rigid(self, M, ieq, L, d):
#         SV, k_y = elastic_SV(L.medium,self.kx, self.omega)
#         M[ieq, d[0]] = SV[1, 0]*np.exp(-1j*k_y[0]*L.thickness)
#         M[ieq, d[1]] = SV[1, 1]*np.exp(-1j*k_y[1]*L.thickness)
#         M[ieq, d[2]] = SV[1, 2]
#         M[ieq, d[3]] = SV[1, 3]
#         ieq += 1
#         M[ieq, d[0]] = SV[3, 0]*np.exp(-1j*k_y[0]*L.thickness)
#         M[ieq, d[1]] = SV[3, 1]*np.exp(-1j*k_y[1]*L.thickness)
#         M[ieq, d[2]] = SV[3, 2]
#         M[ieq, d[3]] = SV[3, 3]
#         ieq += 1
#         return ieq

#     def interface_pem_rigid(self, M, ieq, L, d):
#         SV, k_y = PEM_SV(L.medium, self.kx)
#         M[ieq, d[0]] = SV[1, 0]*np.exp(-1j*k_y[0]*L.thickness)
#         M[ieq, d[1]] = SV[1, 1]*np.exp(-1j*k_y[1]*L.thickness)
#         M[ieq, d[2]] = SV[1, 2]*np.exp(-1j*k_y[2]*L.thickness)
#         M[ieq, d[3]] = SV[1, 3]
#         M[ieq, d[4]] = SV[1, 4]
#         M[ieq, d[5]] = SV[1, 5]
#         ieq += 1
#         M[ieq, d[0]] = SV[2, 0]*np.exp(-1j*k_y[0]*L.thickness)
#         M[ieq, d[1]] = SV[2, 1]*np.exp(-1j*k_y[1]*L.thickness)
#         M[ieq, d[2]] = SV[2, 2]*np.exp(-1j*k_y[2]*L.thickness)
#         M[ieq, d[3]] = SV[2, 3]
#         M[ieq, d[4]] = SV[2, 4]
#         M[ieq, d[5]] = SV[2, 5]
#         ieq += 1
#         M[ieq, d[0]] = SV[5, 0]*np.exp(-1j*k_y[0]*L.thickness)
#         M[ieq, d[1]] = SV[5, 1]*np.exp(-1j*k_y[1]*L.thickness)
#         M[ieq, d[2]] = SV[5, 2]*np.exp(-1j*k_y[2]*L.thickness)
#         M[ieq, d[3]] = SV[5, 3]
#         M[ieq, d[4]] = SV[5, 4]
#         M[ieq, d[5]] = SV[5, 5]
#         ieq += 1
#         return ieq

#     def plot_sol_PW(self, X, dofs):
#         x_start = self.shift_plot
#         for _l, _layer in enumerate(self.layers):
#             x_f = np.linspace(0, _layer.thickness,200)
#             x_b = x_f-_layer.thickness
#             if _layer.medium.MODEL == "fluid":
#                 SV, k_y = fluid_SV(self.kx, self.k, _layer.medium.K)
#                 pr = SV[1, 0]*np.exp(-1j*k_y*x_f)*X[dofs[_l][0]]
#                 pr += SV[1, 1]*np.exp( 1j*k_y*x_b)*X[dofs[_l][1]]
#                 ut = SV[0, 0]*np.exp(-1j*k_y*x_f)*X[dofs[_l][0]]
#                 ut += SV[0, 1]*np.exp( 1j*k_y*x_b)*X[dofs[_l][1]]
#                 if self.plot[2]:
#                     plt.figure(2)
#                     plt.plot(x_start+x_f, np.abs(pr), 'r')
#                     plt.plot(x_start+x_f, np.imag(pr), 'm')
#                     plt.title("Pressure")
#                 # plt.figure(5)
#                 # plt.plot(x_start+x_f,np.abs(ut),'b')
#                 # plt.plot(x_start+x_f,np.imag(ut),'k')
#             if _layer.medium.MODEL == "pem":
#                 SV, k_y = PEM_SV(_layer.medium, self.kx)
#                 ux, uy, pr, ut = 0*1j*x_f, 0*1j*x_f, 0*1j*x_f, 0*1j*x_f
#                 for i_dim in range(3):
#                     ux += SV[1, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     ux += SV[1, i_dim+3]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+3]]
#                     uy += SV[5, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     uy += SV[5, i_dim+3]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+3]]
#                     pr += SV[4, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     pr += SV[4, i_dim+3]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+3]]
#                     ut += SV[2, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     ut += SV[2, i_dim+3]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+3]]
#                 if self.plot[0]:
#                     plt.figure(0)
#                     plt.plot(x_start+x_f, np.abs(uy), 'r')
#                     plt.plot(x_start+x_f, np.imag(uy), 'm')
#                     plt.title("Solid displacement along x")
#                 if self.plot[1]:
#                     plt.figure(1)
#                     plt.plot(x_start+x_f, np.abs(ux), 'r')
#                     plt.plot(x_start+x_f, np.imag(ux), 'm')
#                     plt.title("Solid displacement along y")
#                 if self.plot[2]:
#                     plt.figure(2)
#                     plt.plot(x_start+x_f, np.abs(pr), 'r')
#                     plt.plot(x_start+x_f, np.imag(pr), 'm')
#                     plt.title("Pressure")
#             if _layer.medium.MODEL == "elastic":
#                 SV, k_y = elastic_SV(_layer.medium, self.kx, self.omega)
#                 ux, uy, pr, sig = 0*1j*x_f, 0*1j*x_f, 0*1j*x_f, 0*1j*x_f
#                 for i_dim in range(2):
#                     ux += SV[1, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     ux += SV[1, i_dim+2]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+2]]
#                     uy += SV[3, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     uy += SV[3, i_dim+2]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+2]]
#                     pr -= SV[2, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     pr -= SV[2, i_dim+2]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+2]]
#                     sig -= SV[0, i_dim  ]*np.exp(-1j*k_y[i_dim]*x_f)*X[dofs[_l][i_dim]]
#                     sig -= SV[0, i_dim+2]*np.exp( 1j*k_y[i_dim]*x_b)*X[dofs[_l][i_dim+2]]
#                 if self.plot[0]:
#                     plt.figure(0)
#                     plt.plot(x_start+x_f, np.abs(uy), 'r')
#                     plt.plot(x_start+x_f, np.imag(uy), 'm')
#                     plt.title("Solid displacement along x")
#                 if self.plot[1]:
#                     plt.figure(1)
#                     plt.plot(x_start+x_f, np.abs(ux), 'r')
#                     plt.plot(x_start+x_f, np.imag(ux), 'm')
#                     plt.title("Solid displacement along y")
#                 # if self.plot[2]:
#                 #     plt.figure(2)
#                 #     plt.plot(x_start+x_f, np.abs(pr), 'r')
#                 #     plt.plot(x_start+x_f, np.imag(pr), 'm')
#                 #     plt.title("Sigma_yy")
#                 # if self.plot[2]:
#                 #     plt.figure(3)
#                 #     plt.plot(x_start+x_f, np.abs(sig), 'r')
#                 #     plt.plot(x_start+x_f, np.imag(sig), 'm')
#                 #     plt.title("Sigma_xy")




#             x_start += _layer.thickness

# def PEM_SV(mat,ky):
#     ''' S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
#     kx_1 = np.sqrt(mat.delta_1**2-ky**2)
#     kx_2 = np.sqrt(mat.delta_2**2-ky**2)
#     kx_3 = np.sqrt(mat.delta_3**2-ky**2)

#     kx = np.array([kx_1, kx_2, kx_3])
#     delta = np.array([mat.delta_1, mat.delta_2, mat.delta_3])

#     alpha_1 = -1j*mat.A_hat*mat.delta_1**2-1j*2*mat.N*kx[0]**2
#     alpha_2 = -1j*mat.A_hat*mat.delta_2**2-1j*2*mat.N*kx[1]**2
#     alpha_3 = -2*1j*mat.N*kx[2]*ky

#     SV = np.zeros((6,6), dtype=complex)
#     SV[0:6, 0] = np.array([-2*1j*mat.N*kx[0]*ky, kx[0], mat.mu_1*kx[0], alpha_1, 1j*delta[0]**2*mat.K_eq_til*mat.mu_1, ky])
#     SV[0:6, 3] = np.array([ 2*1j*mat.N*kx[0]*ky,-kx[0],-mat.mu_1*kx[0], alpha_1, 1j*delta[0]**2*mat.K_eq_til*mat.mu_1, ky])

#     SV[0:6, 1] = np.array([-2*1j*mat.N*kx[1]*ky, kx[1], mat.mu_2*kx[1],alpha_2, 1j*delta[1]**2*mat.K_eq_til*mat.mu_2, ky])
#     SV[0:6, 4] = np.array([ 2*1j*mat.N*kx[1]*ky,-kx[1],-mat.mu_2*kx[1],alpha_2, 1j*delta[1]**2*mat.K_eq_til*mat.mu_2, ky])

#     SV[0:6, 2] = np.array([1j*mat.N*(kx[2]**2-ky**2), ky, mat.mu_3*ky, alpha_3, 0., -kx[2]])
#     SV[0:6, 5] = np.array([1j*mat.N*(kx[2]**2-ky**2), ky, mat.mu_3*ky, -alpha_3, 0., kx[2]])
#     return SV, kx

# def elastic_SV(mat,ky, omega):
#     ''' S={0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''

#     P_mat = mat.lambda_ + 2.*mat.mu
#     delta_p = omega*np.sqrt(mat.rho/P_mat)
#     delta_s = omega*np.sqrt(mat.rho/mat.mu)

#     kx_p = np.sqrt(delta_p**2-ky**2)
#     kx_s = np.sqrt(delta_s**2-ky**2)

#     kx = np.array([kx_p, kx_s])

#     alpha_p = -1j*mat.lambda_*delta_p**2 - 2j*mat.mu*kx[0]**2
#     alpha_s = 2j*mat.mu*kx[1]*ky

#     SV = np.zeros((4, 4), dtype=np.complex)
#     SV[0:4, 0] = np.array([-2.*1j*mat.mu*kx[0]*ky,  kx[0], alpha_p, ky])
#     SV[0:4, 2] = np.array([ 2.*1j*mat.mu*kx[0]*ky, -kx[0], alpha_p, ky])

#     SV[0:4, 1] = np.array([1j*mat.mu*(kx[1]**2-ky**2), ky,-alpha_s, -kx[1]])
#     SV[0:4, 3] = np.array([1j*mat.mu*(kx[1]**2-ky**2), ky, alpha_s, kx[1]])

#     return SV, kx

# def fluid_SV(kx, k, K):
#     ''' S={0:u_y , 1:p}'''
#     ky = np.sqrt(k**2-kx**2)
#     SV = np.zeros((2, 2), dtype=complex)
#     SV[0, 0:2] = np.array([ky/(1j*K*k**2), -ky/(1j*K*k**2)])
#     SV[1, 0:2] = np.array([1, 1])
#     return SV, ky

# def resolution_PW_imposed_displacement(S, p):
#     # print("k={}".format(p.k))
#     Layers = S.layers.copy()
#     n, interfaces, dofs = initialise_PW_solver(Layers, S.backing)
#     M = np.zeros((n, n), dtype=complex)
#     i_eq = 0
#     # Loop on the layers
#     for i_inter, _inter in enumerate(interfaces):
#         if _inter[0] == "fluid":
#             if _inter[1] == "fluid":
#                 i_eq = interface_fluid_fluid(i_eq, i_inter, Layers, dofs, M, p)
#             if _inter[1] == "pem":
#                 i_eq = interface_fluid_pem(i_eq, i_inter, Layers, dofs, M, p)
#         elif _inter[0] == "pem":
#             if _inter[1] == "fluid":
#                 i_eq = interface_pem_fluid(i_eq, i_inter, Layers, dofs, M, p)
#             if _inter[1] == "pem":
#                 i_eq = interface_pem_pem(i_eq, i_inter, Layers, dofs, M, p)
#     if S.backing == backing.rigid:
#         if Layers[-1].medium.MODEL == "fluid":
#             i_eq = interface_fluid_rigid(M, i_eq, Layers[-1], dofs[-1], p)
#         elif Layers[-1].medium.MODEL == "pem":
#             i_eq = interface_pem_rigid(M, i_eq, Layers[-1], dofs[-1], p)

#     if Layers[0].medium.MODEL == "fluid":
#         F = np.zeros(n, dtype=complex)
#         SV, k_y = fluid_SV(p.kx, p.k, Layers[0].medium.K)
#         M[i_eq, dofs[0][0]] = SV[0, 0]
#         M[i_eq, dofs[0][1]] = SV[0, 1]*np.exp(-1j*k_y*Layers[0].thickness)
#         F[i_eq] = 1.
#     elif Layers[0].medium.MODEL == "pem":
#         SV, k_y = PEM_SV(Layers[0].medium, p.kx)
#         M[i_eq, dofs[0][0]] = SV[2, 0]
#         M[i_eq, dofs[0][1]] = SV[2, 1]
#         M[i_eq, dofs[0][2]] = SV[2, 2]
#         M[i_eq, dofs[0][3]] = SV[2, 3]*np.exp(-1j*k_y[0]*Layers[0].thickness)
#         M[i_eq, dofs[0][4]] = SV[2, 4]*np.exp(-1j*k_y[1]*Layers[0].thickness)
#         M[i_eq, dofs[0][5]] = SV[2, 5]*np.exp(-1j*k_y[2]*Layers[0].thickness)
#         F = np.zeros(n, dtype=complex)
#         F[i_eq] = 1.
#         i_eq +=1
#         M[i_eq, dofs[0][0]] = SV[0, 0]
#         M[i_eq, dofs[0][1]] = SV[0, 1]
#         M[i_eq, dofs[0][2]] = SV[0, 2]
#         M[i_eq, dofs[0][3]] = SV[0, 3]*np.exp(-1j*k_y[0]*Layers[0].thickness)
#         M[i_eq, dofs[0][4]] = SV[0, 4]*np.exp(-1j*k_y[1]*Layers[0].thickness)
#         M[i_eq, dofs[0][5]] = SV[0, 5]*np.exp(-1j*k_y[2]*Layers[0].thickness)
#         i_eq += 1
#         M[i_eq, dofs[0][0]] = SV[3, 0]
#         M[i_eq, dofs[0][1]] = SV[3, 1]
#         M[i_eq, dofs[0][2]] = SV[3, 2]
#         M[i_eq, dofs[0][3]] = SV[3, 3]*np.exp(-1j*k_y[0]*Layers[0].thickness)
#         M[i_eq, dofs[0][4]] = SV[3, 4]*np.exp(-1j*k_y[1]*Layers[0].thickness)
#         M[i_eq, dofs[0][5]] = SV[3, 5]*np.exp(-1j*k_y[2]*Layers[0].thickness)


#     X = LA.solve(M, F)
#     # print("|R pyPLANES_PW| = {}".format(np.abs(X[0])))
#     print("R pyPLANES_PW           = {}".format(X[0]))

#     plot_sol_PW(S, X, dofs, p)

