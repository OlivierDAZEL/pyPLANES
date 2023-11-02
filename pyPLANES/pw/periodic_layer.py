#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
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

import platform
import time, timeit
import os

import numpy as np

from pyPLANES.core.mesh import Mesh
from pyPLANES.core.calculus import Calculus

from pyPLANES.fem.fem_entities_surfacic import *
from pyPLANES.fem.fem_entities_volumic import *
# from pyPLANES.fem.fem_entities_pw import IncidentPwFem, TransmissionPwFem

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from pyPLANES.fem.fem_preprocess import fem_preprocess
from pyPLANES.utils.io import plot_fem_solution, export_paraview
from pyPLANES.fem.dofs import periodic_dofs_identification
from pyPLANES.fem.fem_entities_pw import PwFem
from pyPLANES.fem.utils_fem import dof_p_element, dof_up_element

class PeriodicLayer(Mesh):
    def __init__(self, **kwargs):
        self.condensation = kwargs.get("condensation", True)
        Mesh.__init__(self, **kwargs)
        _x = kwargs.get("_x", 0)
        for _v in self.vertices[1:]:
            _v.coord[1] += _x
        for _e in self.elements[1:]:
             _e.coord[1,:] += _x
             
             
        y = [_v.coord[1] for _v in self.vertices[1:]]
        self.d = np.max(y)-np.min(y)

        self.theta_d = kwargs.get("theta_d", 0.0)
        self.order = kwargs.get("order", 2)
        self.verbose = kwargs.get("verbose", False)
        self.plot = kwargs.get("plot", [False]*6)

        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        self.medium = [None, None]
        self.start_time = time.time()
        self.TM = None


        fem_preprocess(self)
        for _ent in self.pwfem_entities:
            _ent.theta_d = self.theta_d
            if _ent.typ == "fluid":
                _ent.method_dof = dof_p_element
            elif _ent.typ in ["Biot98", "Biot01"]:
                _ent.method_dof = dof_up_element  
        # The bottom interface is the first of self.pwfem_entities
        if self.pwfem_entities[0].ny == 1:
            self.pwfem_entities.reverse()

        periodic_dofs_identification(self)

        self.characteristics = [None, None] # Will be completed in PeriodicPwProblem.__init__()
        

    def update_frequency(self, omega, kx):
        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        for _ent in self.fem_entities:
            _ent.update_frequency(omega)
        for _ent in self.pwfem_entities:
            _ent.update_frequency(omega)
        # Wave numbers and periodic shift
        self.kx = kx
        self.nb_waves = len(kx)
        self.delta_periodicity = np.exp(-1j*self.kx[0]*self.period)
        self.omega = omega # Needs to be stored
        for _ent in self.pwfem_entities:
            _ent.dofs = np.arange(_ent.nb_dof_per_node*len(self.kx))
            _ent.nb_dofs = len(_ent.dofs)

    def create_transfert_matrix(self):
        # Initialisation of the lists
        self.A_i, self.A_j, self.A_v = [], [], []
        # Number of dof of the D_ii marix
        if self.condensation:
            n_dof = self.nb_dof_master-1
            self.T_i, self.T_j, self.T_v = [], [], []
        else:
            n_dof = self.nb_dof_FEM-1

        # Creation of the D_ii matrix (volumic term of the weak form) 
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(self.omega))
        # Application of periodicity on Dii
        for i_left, dof_left in enumerate(self.dof_left):
            # Corresponding dof
            dof_right = self.dof_right[i_left]
            index = np.where(np.array(self.A_j) == dof_right)[0]
            for _i in index:
                self.A_j[_i] = dof_left
                self.A_v[_i] *= self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            # Summation of the rows for the Matrix
            index = np.where(np.array(self.A_i) == dof_right)[0]
            for _i in index:
                self.A_i[_i] = dof_left
                self.A_v[_i] /= self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            # Periodicity of the physical dofs
            self.A_i.append(dof_right)
            self.A_j.append(dof_left)
            self.A_v.append(self.delta_periodicity)
            self.A_i.append(dof_right)
            self.A_j.append(dof_right)
            self.A_v.append(-self.orientation_periodic_dofs[i_left])

        self.linear_system_2_numpy()
        index_A = np.where(((self.A_i*self.A_j) != 0) )

        D_ii = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(n_dof, n_dof)).tocsr()        

        self.A_i, self.A_j, self.A_v = [], [], []
        RR = [] # Initialisation of the list of the R will be [R_b R_t]
        DD = [] # Initialisation of the list of the R will be [D_bb D_tt]
        DD_xi = [] # Initialisation of the list of the R will be [D_bi D_ti]
        for _ent in self.pwfem_entities:
            dof_FEM, dof_S_primal, dof_S_dual, D_val = [], [], [], []
            D_xx = np.zeros((_ent.nb_dof_per_node*self.nb_waves, 2*_ent.nb_dof_per_node*self.nb_waves))
            for _w, kx in enumerate(self.kx):
                for _elem in _ent.elements:
                    M_elem = imposed_pw_elementary_vector(_elem, kx)
                    if _ent.typ == "fluid":
                        # Columns of D_xi
                        dof_p, orient_p, __ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        # Rows of D_ix
                        dof_S_dual.extend(len(dof_p)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        # Rows of D_xx and D_xi
                        dof_S_primal.extend(len(dof_p)*[_w])
                        # Values for D_ix and D_xi (will be conjugated below)                      
                        D_val.extend(list(orient_p@M_elem))
                        # Values for D_xx
                        D_xx[_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    elif _ent.typ in ["Biot98", "Biot01"]:
                        # u_x
                        dof_ux, orient_ux = dof_ux_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_ux])
                        dof_S_dual.extend(len(dof_ux)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_ux)*[0+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_ux@M_elem))
                        D_xx[0+_ent.nb_dof_per_node*_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        # u_y
                        dof_uy, orient_uy = dof_uy_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_uy])
                        dof_S_dual.extend(len(dof_uy)*[_ent.dual[1]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_uy)*[1+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_uy@M_elem))
                        D_xx[1+_ent.nb_dof_per_node*_w, _ent.primal[1]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        #  p 
                        dof_p, orient_p, _ = dof_p_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_p])
                        dof_S_dual.extend(len(dof_p)*[_ent.dual[2]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_p)*[2+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_p@M_elem))
                        D_xx[2+_ent.nb_dof_per_node*_w, _ent.primal[2]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    elif _ent.typ == "elastic":
                        # u_x                        
                        dof_ux, orient_ux = dof_ux_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_ux])
                        dof_S_dual.extend(len(dof_ux)*[_ent.dual[0]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_ux)*[0+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_ux@M_elem))
                        D_xx[0+_ent.nb_dof_per_node*_w, _ent.primal[0]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                        # u_y
                        dof_uy, orient_uy = dof_uy_element(_elem)
                        dof_FEM.extend([d-1 for d in dof_uy])
                        dof_S_dual.extend(len(dof_uy)*[_ent.dual[1]+2*_ent.nb_dof_per_node*_w])
                        dof_S_primal.extend(len(dof_uy)*[1+_ent.nb_dof_per_node*_w])
                        D_val.extend(list(orient_uy@M_elem))
                        D_xx[1+_ent.nb_dof_per_node*_w, _ent.primal[1]+2*_ent.nb_dof_per_node*_w] = -_ent.period
                    else:
                        raise NameError("_ent.typ has no valid type")
            
            DD.append(D_xx)
            DD_xi.append(coo_matrix((np.conj(D_val), (dof_S_primal, dof_FEM)), shape=(_ent.nb_dof_per_node*self.nb_waves, n_dof)))

            # Application of periodicity to the columns of D_xi (D_ti and D_bi)
            for i_left, _dof_left in enumerate(self.dof_left):
                # Corresponding dof
                _dof_right = self.dof_right[i_left]-1
                index = [i for i,d in enumerate(dof_FEM) if d==_dof_right]
                for _i in index:
                    dof_FEM[_i] = _dof_left-1
                    D_val[_i] /= self.delta_periodicity*self.orientation_periodic_dofs[i_left]

            # Creation of the D_ix, minus sign <- transposition +normal 
            D_ix = coo_matrix((-_ent.ny*np.array(D_val), (dof_FEM, dof_S_dual)), shape=(n_dof, 2*_ent.nb_dof_per_node*self.nb_waves))
            # R_t and R_b
            RR.append(-linsolve.spsolve(D_ii, D_ix.todense()).reshape((n_dof, 2*_ent.nb_dof_per_node*self.nb_waves))) ##

        if any(self.plot):
            self.R_t = RR[1]
            self.R_b = RR[0]
            
        _s = _ent.nb_dof_per_node*self.nb_waves
        M_1 = np.zeros((2*_s, 2*_s), dtype=complex)
        M_2 = np.zeros((2*_s, 2*_s), dtype=complex)

        M_1[:_s,:] = DD_xi[1]@RR[0]# [D_ti][R_b]
        M_1[_s:,:] = DD[0]+DD_xi[0]@RR[0]# [D_bb]+[D_bi][R_b]

        M_2[:_s,:] = DD[1]+DD_xi[1]@RR[1]# [D_tt]+[D_ti][R_t]
        M_2[_s:,:] = DD_xi[0]@RR[1]# [D_bi][R_t]

        self.M_1 = M_1
        self.M_2 = M_2
        self.TM = -LA.solve(M_1, M_2)


    def update_Omega(self, Om, omega, method="Recursive Method"):
        self.Omega_minus = Om # To plot the solution
        if self.verbose: 
            print("Creation of the Transfer Matrix of the FEM layer")
        self.create_transfert_matrix()

        # print("TM=", np.diag(self.TM))


        m = self.nb_waves_in_medium*self.nb_waves

        Xi = np.eye(m)
      
        lambda_, Phi = LA.eig(self.TM)
        _index = np.argsort(np.abs(lambda_))[::-1]
        lambda_ = lambda_[_index]
  
        Phi = Phi[:, _index]
        Phi_inv = LA.inv(Phi)
        
        
        _list = [0.]*(m-1)+[1.] +[(lambda_[m+i]/lambda_[m-1]) for i in range(0, m)]
        Lambda = np.diag(np.array(_list))
        alpha_prime = Phi.dot(Lambda).dot(Phi_inv) # Eq (21)
        xi_prime = Phi_inv[:m,:] @ Om # Eq (23)
        _list = [(lambda_[m-1]/lambda_[i]) for i in range(m-1)] + [1.]
        xi_prime_lambda = LA.inv(xi_prime).dot(np.diag(_list))
        Om = alpha_prime.dot(Om).dot(xi_prime_lambda)
        Om[:,:m-1] += Phi[:, :m-1]
        Xi = (1/lambda_[m-1])*(xi_prime_lambda@Xi)
        return Om, Xi


    def update_Omegac(self, Om, omega):

        if self.verbose: 
            print("Creation of the Transfer Matrix of the FEM layer")

        # Initialisation of the lists
        self.A_i, self.A_j, self.A_v = [], [], []
        # Number of dof of the D_ii marix
        if self.condensation:
            n_dof = self.nb_dof_master-1
            self.T_i, self.T_j, self.T_v = [], [], []
        else:
            n_dof = self.nb_dof_FEM-1

        # Creation of the D_ii matrix (volumic term of the weak form) 
        for _ent in self.fem_entities:
            self.update_system(*_ent.update_system(self.omega))

        # Stabilisation terms 
        # Top interface
        _ent = self.pwfem_entities[1]
        Q, n_w = self.characteristics[1].Q, self.characteristics[1].n_w
        n_y = 1.
        # for _w, kx in enumerate(self.kx):
        for _elem in _ent.elements:
            M_elem = fsi_elementary_matrix(_elem)
            dof, orient, __ = _ent.method_dof(_elem)
            M = orient @ M_elem @ orient
            M = n_y*np.kron(LA.solve(Q[n_w:,:n_w],Q[n_w:,n_w:]), M)
            self.A_i.extend(list(chain.from_iterable([[_d]*len(dof) for _d in dof])))
            self.A_j.extend(list(dof)*len(dof))
            self.A_v.extend(M.flatten())

        # Bottom interface
        _ent = self.pwfem_entities[0]
        Q, n_w = self.characteristics[0].Q, self.characteristics[0].n_w
        n_y = -1;
        # for _w, kx in enumerate(self.kx):
        for _elem in _ent.elements:
            M_elem = fsi_elementary_matrix(_elem)
            dof, orient, __ = _ent.method_dof(_elem)
            M = orient @ M_elem @ orient
            M = n_y*np.kron(LA.solve(Q[:n_w,:n_w],Q[:n_w,n_w:]), M)
            self.A_i.extend(list(chain.from_iterable([[_d]*len(dof) for _d in dof])))
            self.A_j.extend(list(dof)*len(dof))
            self.A_v.extend(M.flatten())

        # Application of periodicity on Dii
        for i_left, dof_left in enumerate(self.dof_left):
            # Corresponding dof
            dof_right = self.dof_right[i_left]
            index = np.where(np.array(self.A_j) == dof_right)[0]
            for _i in index:
                self.A_j[_i] = dof_left
                self.A_v[_i] *= self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            # Summation of the rows for the Matrix
            index = np.where(np.array(self.A_i) == dof_right)[0]
            for _i in index:
                self.A_i[_i] = dof_left
                self.A_v[_i] /= self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            # Periodicity of the physical dofs
            self.A_i.append(dof_right)
            self.A_j.append(dof_left)
            self.A_v.append(self.delta_periodicity)
            self.A_i.append(dof_right)
            self.A_j.append(dof_right)
            self.A_v.append(-self.orientation_periodic_dofs[i_left])

        self.linear_system_2_numpy()
        index_A = np.where(((self.A_i*self.A_j) != 0) )

        D_ii = coo_matrix((self.A_v[index_A], (self.A_i[index_A]-1, self.A_j[index_A]-1)), shape=(n_dof, n_dof)).tocsr()        


        D_tt = np.zeros((_ent.nb_dof_per_node*self.nb_waves, 2*_ent.nb_dof_per_node*self.nb_waves), dtype=complex)
        D_bb = np.zeros((_ent.nb_dof_per_node*self.nb_waves, 2*_ent.nb_dof_per_node*self.nb_waves), dtype=complex)
        D_ti = np.zeros((_ent.nb_dof_per_node*self.nb_waves, n_dof), dtype=complex)
        D_bi = np.zeros((_ent.nb_dof_per_node*self.nb_waves, n_dof), dtype=complex)
        D_it = np.zeros((n_dof, 2*_ent.nb_dof_per_node*self.nb_waves), dtype=complex)
        D_ib = np.zeros((n_dof, 2*_ent.nb_dof_per_node*self.nb_waves), dtype=complex)


        # Top interface
        _ent = self.pwfem_entities[1]
        Q = self.characteristics[1].Q
        P = self.characteristics[1].P
        n_w = self.characteristics[1].n_w
        n_y = 1.
        for _w, kx in enumerate(self.kx):
            dof_q  = slice(2*_w*n_w, 2*(_w+1)*n_w)
            dof_qm = slice(2*n_w*_w+n_w, 2*(_w+1)*n_w)
            dof_eq = slice(_w*n_w, (_w+1)*n_w)
            for _elem in _ent.elements:
                M_elem = imposed_pw_elementary_vector(_elem, kx)
                dof, orient, __ = _ent.method_dof(_elem)
                dof =[d-1 for d in dof]
                M = orient @ M_elem
                MM = n_y*np.kron(LA.inv(Q[n_w:,:n_w]).T, M).T
                D_it[dof, dof_qm] -= MM
                D_ti[dof_eq , dof] += np.kron(np.eye(n_w), np.conj(M))
            D_tt[dof_eq, dof_q] -= P[n_w:, :] * _ent.period

        # Bottom interface
        _ent = self.pwfem_entities[0]
        Q = self.characteristics[0].Q
        P = self.characteristics[0].P
        n_w = self.characteristics[0].n_w
        n_y = -1.
        for _w, kx in enumerate(self.kx):
            dof_q  = slice(2*_w*n_w, 2*(_w+1)*n_w)
            dof_qp = slice(2*n_w*_w,2*n_w*_w+n_w)
            dof_eq = slice(_w*n_w, (_w+1)*n_w)
            for _elem in _ent.elements:
                M_elem = imposed_pw_elementary_vector(_elem, kx)
                dof, orient, __ = _ent.method_dof(_elem)
                dof =[d-1 for d in dof]
                M = orient @ M_elem
                MM = n_y*np.kron(LA.inv(Q[:n_w,:n_w]).T, M).T
                D_ib[dof, dof_qp] -= MM
                D_bi[dof_eq, dof] += np.kron(np.eye(n_w), np.conj(M))
            D_bb[dof_eq, dof_q] -= P[n_w:, :] * _ent.period


        # # Application of periodicity to the columns of D_xi (D_ti and D_bi)
        for i_left, dof_left in enumerate(self.dof_left):
            dof_right = self.dof_right[i_left]-1
            D_it[dof_left-1, :] += D_it[dof_right, :]/self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            D_ti[:, dof_left-1] += D_ti[:, dof_right]*self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            D_it[dof_right, :] = 0
            D_ti[:, dof_right] = 0
            D_ib[dof_left-1, :] += D_ib[dof_right, :]/self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            D_bi[:, dof_left-1] += D_bi[:, dof_right]*self.delta_periodicity*self.orientation_periodic_dofs[i_left]
            D_ib[dof_right, :] = 0
            D_bi[:, dof_right] = 0
            
        self.R_b = -linsolve.spsolve(D_ii, D_ib).reshape((n_dof, 2*_ent.nb_dof_per_node*self.nb_waves))
        self.R_t = -linsolve.spsolve(D_ii, D_it).reshape((n_dof, 2*_ent.nb_dof_per_node*self.nb_waves))

        _s = _ent.nb_dof_per_node*self.nb_waves
        M_1 = np.zeros((2*_s, 2*_s), dtype=complex)
        M_2 = np.zeros((2*_s, 2*_s), dtype=complex)

        M_1[:_s,:] = D_ti@self.R_b# [D_ti][R_b]
        M_1[_s:,:] = D_bb+D_bi@self.R_b# [D_bb]+[D_bi][R_b]

        M_2[:_s,:] = D_tt+D_ti@self.R_t# [D_tt]+[D_ti][R_t]
        M_2[_s:,:] = D_bi@self.R_t# [D_bi][R_t]

        self.M_1 = M_1
        self.M_2 = M_2
        
        self.TM = -LA.solve(M_1, M_2)
        # print("TM carac")
        # print(self.TM)

        m = self.nb_waves_in_medium*self.nb_waves

        Xi = np.eye(m)

        lambda_, Phi = LA.eig(self.TM)
        
        _index = np.argsort(np.abs(lambda_))[::-1]
        lambda_ = lambda_[_index]
  
        Phi = Phi[:, _index]
        Phi_inv = LA.inv(Phi)
        
        
        _list = [0.]*(m-1)+[1.] +[(lambda_[m+i]/lambda_[m-1]) for i in range(0, m)]
        Lambda = np.diag(np.array(_list))
        alpha_prime = Phi.dot(Lambda).dot(Phi_inv) # Eq (21)
        xi_prime = Phi_inv[:m,:] @ Om # Eq (23)
        _list = [(lambda_[m-1]/lambda_[i]) for i in range(m-1)] + [1.]
        xi_prime_lambda = LA.inv(xi_prime).dot(np.diag(_list))
        Om = alpha_prime.dot(Om).dot(xi_prime_lambda)
        Om[:,:m-1] += Phi[:, :m-1]
        Xi = (1/lambda_[m-1])*(xi_prime_lambda@Xi)
        return Om, Xi





    def update_system(self, _A_i, _A_j, _A_v, _F_i, _F_v, _T_i=None, _T_j=None, _T_v=None):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)
        if self.condensation:
            self.T_i.extend(_T_i)
            self.T_j.extend(_T_j)
            self.T_v.extend(_T_v)

    def linear_system_2_numpy(self):
        self.F_i = np.array(self.F_i)
        self.F_v = np.array(self.F_v, dtype=complex)
        self.A_i = np.array(self.A_i)
        self.A_j = np.array(self.A_j)
        self.A_v = np.array(self.A_v, dtype=complex)
        if self.condensation:
            self.T_i = np.array(self.T_i)-self.nb_dof_master
            self.T_j = np.array(self.T_j)
            self.T_v = np.array(self.T_v, dtype=complex)

    def plot_solution(self, S_b, S_t):

        X = self.R_b@S_b +self.R_t@S_t
        X = np.insert(X, 0, 0)
        # Concatenation of the slave dofs at the end of the vector
        self.nb_dof_condensed = self.nb_dof_FEM - self.nb_dof_master
        if self.condensation:
            T = coo_matrix((self.T_v, (self.T_i, self.T_j)), shape=(self.nb_dof_FEM-self.nb_dof_master, self.nb_dof_master)).tocsr()
            X = np.insert(T@X, 0, X)
        
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
        plot_fem_solution(self, self.kx)