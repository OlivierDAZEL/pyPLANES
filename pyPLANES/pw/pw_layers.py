#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_interfaces.py
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

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

from numpy import pi, sqrt

from pyPLANES.pw.pw_polarisation import fluid_waves_TMM, elastic_waves_TMM, PEM_waves_TMM
from scipy.linalg import block_diag

class PwLayer():
    """
    Base class for Plane Wave layer definition and manipulation
    
    Attributes :
    ------------------------

    medium : mediapack material 

    d : thickness of the layer 

    x : list with the abscissa of the layer

    interfaces : list with the interfaces of the layer
    
    """
    def __init__(self, mat, d, x_0=0):
        """
        Parameters
        ----------
        mat : mediapack medium 
            Name of the material 
        d : float
            thickness of the layer
        x_0 : position of the first interface        
        """
        self.medium = mat 
        self.d = d
        # pymls layer constructor 
        self.x = [x_0, x_0+self.d]  # 
        self.interfaces = [None, None]  # 
        self.nb_waves = None
        self.nb_fields_SV = None
        self.dofs = None
        self.lam = None
        self.SV = None
        self.SVp = None

    def __str__(self):
        pass

    def update_frequency(self, omega, kx):
        pass

    def transfert(self, Om):
        """
        Update the information matrix Omega, Implemented in derived classes

        Parameters
        ----------
        Om : ndarray
            Information matrix on the + side of the layer

        Returns
        ----------
        Om_ : ndarray
            Information matrix on the - side of the layer

        Xi : ndarray
            Back_propagation matrix (to be used only for transmission problems)
        """
        pass


    def update_Omega(self, Om):
        pass 

    def order_lam(self):
        _index_block = []
        for _w in range(self.nb_waves):
            index_block = range(_w*self.nb_fields_SV, (_w+1)*self.nb_fields_SV)
            index = np.argsort(self.lam[index_block].real)[::-1]
            _index_block += [index_block[i] for i in index]
        self.SV = self.SV[np.ix_(range(self.nb_fields_SV*self.nb_waves), _index_block)]
        self.lam = self.lam[np.ix_(_index_block)]

class FluidLayer(PwLayer):
    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)
        self.nb_fields_SV = 2

    def __str__(self):
        out = "\t Fluid Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.nb_waves = len(kx)
        self.SV, self.lam = fluid_waves_TMM(self.medium, kx)

    def transfert_matrix(self, om, ky):
        T = np.zeros((2, 2), dtype=complex)
        T[0, 0] = np.cos(ky*self.d)
        T[1, 0] = (om**2*self.medium.rho/ky)*np.sin(ky*self.d)
        T[0, 1] = -(ky/(om**2*self.medium.rho))*np.sin(ky*self.d)
        T[1, 1] = np.cos(ky*self.d)
        return T


    def update_Omega(self, om, Om, ky):
        T = np.zeros((2, 2), dtype=complex)
        T[0, 0] = np.cos(ky*self.d)
        T[1, 0] = (om**2*self.medium.rho/ky)*np.sin(ky*self.d)
        T[0, 1] = -(ky/(om**2*self.medium.rho))*np.sin(ky*self.d)
        T[1, 1] = np.cos(ky*self.d)
        return T@Om 

    def transfert(self, Om):
        self.order_lam()
        Om_ = np.zeros(Om.shape, dtype=complex)
        Xi = np.zeros((self.nb_waves, self.nb_waves), dtype=complex)
        for _w in range(self.nb_waves):
            index_0 = _w*self.nb_fields_SV
            index_1 = _w*self.nb_fields_SV+1
            alpha = self.SV[index_0, index_0]
            xi_prime = (Om[index_0, _w]/alpha+Om[index_1, _w])/2. #Ok
            Om_0 = (Om[index_0, _w]-alpha*Om[index_1, _w])/2. 
            Om_1 = -Om_0/alpha
            # Om_ = np.array([Om_0, Om_1])*np.exp(-2.*self.lam[index_0]*self.d)/xi_prime
            Om_[index_0, _w] = Om_0*np.exp(2.*self.lam[index_0]*self.d)/xi_prime+ alpha
            Om_[index_1, _w] = Om_1*np.exp(2.*self.lam[index_0]*self.d)/xi_prime+ 1.

            Xi[_w, _w] = np.exp(self.lam[index_0]*self.d)/xi_prime

        return Om_ , Xi

    def plot_solution_global(self, plot, X, nb_points=200):

        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        pr =  self.SV[1, 0]*np.exp(self.lam[0]*x_f)*X[0]
        pr += self.SV[1, 1]*np.exp(self.lam[1]*x_b)*X[1]
        ut =  self.SV[0, 0]*np.exp(self.lam[0]*x_f)*X[0]
        ut += self.SV[0, 1]*np.exp(self.lam[1]*x_b)*X[1]
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r')
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm')
            # plt.figure("ut")
            # plt.plot(self.x[0]+x_f, np.real(ut), 'r')
            # plt.plot(self.x[0]+x_f, np.imag(ut), 'm')

    def plot_solution_recursive(self, plot, X, nb_points=10):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        pr =  self.SV[1, 0]*np.exp(self.lam[0]*x_f)*X[0]
        pr += self.SV[1, 1]*np.exp(self.lam[1]*x_f)*X[1]
        ut =  self.SV[0, 0]*np.exp(self.lam[0]*x_f)*X[0]
        ut += self.SV[0, 1]*np.exp(self.lam[1]*x_f)*X[1]
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm.')


class PemLayer(PwLayer):

    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)
        self.nb_fields_SV = 6

    def __str__(self):
        out = "\t Poroelastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = PEM_waves_TMM(self.medium, kx)
        self.nb_waves = len(kx)

    def plot_solution_global(self, plot, X, nb_points=200):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        ux, uy, pr, ut = 0*1j*x_f, 0*1j*x_f, 0*1j*x_f, 0*1j*x_f
        for i_dim in range(3):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]  *x_f)*X[i_dim]
            ux += self.SV[1, i_dim+3]*np.exp(self.lam[i_dim+3]*x_b)*X[i_dim+3]
            uy += self.SV[5, i_dim  ]*np.exp(self.lam[i_dim]  *x_f)*X[i_dim]
            uy += self.SV[5, i_dim+3]*np.exp(self.lam[i_dim+3]*x_b)*X[i_dim+3]
            pr += self.SV[4, i_dim  ]*np.exp(self.lam[i_dim]  *x_f)*X[i_dim]
            pr += self.SV[4, i_dim+3]*np.exp(self.lam[i_dim+3]*x_b)*X[i_dim+3]
            ut += self.SV[2, i_dim  ]*np.exp(self.lam[i_dim]  *x_f)*X[i_dim]
            ut += self.SV[2, i_dim+3]*np.exp(self.lam[i_dim+3]*x_b)*X[i_dim+3]
        if plot[0]:
            plt.figure("Solid displacement along y")
            plt.plot(self.x[0]+x_f, np.abs(ux), 'r')
            plt.plot(self.x[0]+x_f, np.imag(ux), 'm')
        if plot[1]:
            plt.figure("Solid displacement along x")
            plt.plot(self.x[0]+x_f, np.abs(uy), 'r')
            plt.plot(self.x[0]+x_f, np.imag(uy), 'm')
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r')
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm')

    def plot_solution_recursive(self, plot, X, nb_points=10):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        ux, uy, pr, ut = 0*1j*x_f, 0*1j*x_f, 0*1j*x_f, 0*1j*x_f
        for i_dim in range(6):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            uy += self.SV[5, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            pr += self.SV[4, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
        if plot[0]:
            plt.figure("Solid displacement along y")
            plt.plot(self.x[0]+x_f, np.abs(ux), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(ux), 'm.')
        if plot[1]:
            plt.figure("Solid displacement along x")
            plt.plot(self.x[0]+x_f, np.abs(uy), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(uy), 'm.')
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm.')

    def transfert(self, Om):
        self.order_lam()
        Om_stack, Xi_stack = [], []
        for _w in range(self.nb_waves):
            index_w = list(range(6*_w, 6*(_w+1)))
            index_X = list(range(3*_w, 3*(_w+1)))

            Phi = self.SV[index_w,:][:,index_w]
            lambda_ = self.lam[index_w]
            Phi_inv = LA.inv(Phi)

            Lambda = np.diag([
                0,
                0,
                1,
                np.exp(-(lambda_[3]-lambda_[2])*self.d),
                np.exp(-(lambda_[4]-lambda_[2])*self.d),
                np.exp(-(lambda_[5]-lambda_[2])*self.d)
            ])

            alpha_prime = Phi.dot(Lambda).dot(Phi_inv)
            xi_prime = Phi_inv[:2,:] @ Om[index_w,:][:,index_X]
            xi_prime = np.concatenate([xi_prime, np.array([[0,0,1]])])  # TODO
            xi_prime_lambda = LA.inv(xi_prime).dot(np.diag([
                np.exp(-(lambda_[2]-lambda_[0])*self.d),
                np.exp(-(lambda_[2]-lambda_[1])*self.d),
                1
            ]))

            Om_ = alpha_prime.dot(Om[index_w,:][:,index_X]).dot(xi_prime_lambda)
            Om_[:,0] += Phi[:,0]
            Om_[:,1] += Phi[:,1]

            # eq. 24
            Xi = xi_prime_lambda*np.exp(lambda_[2]*self.d)

            Om_stack.append(Om_)
            Xi_stack.append(Xi)
        Om = block_diag(*Om_stack)
        Xi = block_diag(*Xi_stack)
        return Om, Xi


class ElasticLayer(PwLayer):

    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)
        self.nb_fields_SV = 4

    def __str__(self):
        out = "\t Elastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = elastic_waves_TMM(self.medium, kx)
        self.nb_waves = len(kx)

    def transfert(self, Om):
        self.order_lam()
        Om_stack, Xi_stack = [], [] 
        for _w in range(self.nb_waves):
            index_w = list(range(4*_w, 4*(_w+1)))
            index_X = list(range(2*_w, 2*(_w+1)))

            Phi = self.SV[index_w,:][:,index_w]
            lambda_ = self.lam[index_w]
            Phi_inv = LA.inv(Phi)

            Lambda = np.diag([
                0,
                1,
                np.exp(-(lambda_[2]-lambda_[1])*self.d),
                np.exp(-(lambda_[3]-lambda_[1])*self.d)
            ])

            alpha_prime = Phi.dot(Lambda).dot(Phi_inv)
            
            xi_prime = Phi_inv[:1,:] @ Om[index_w,:][:,index_X]
            xi_prime = np.concatenate([xi_prime, np.array([[0,1]])])  # TODO
            xi_prime_lambda = LA.inv(xi_prime).dot(np.diag([
                np.exp(-(lambda_[1]-lambda_[0])*self.d),
                1
            ]))

            Om_ = alpha_prime.dot(Om[index_w,:][:,index_X]).dot(xi_prime_lambda)
            Om_[:,0] += Phi[:,0]

            Xi = xi_prime_lambda*np.exp(lambda_[1]*self.d)

            Om_stack.append(Om_)
            Xi_stack.append(Xi)
        Om = block_diag(*Om_stack)
        Xi = block_diag(*Xi_stack)
        return Om, Xi

    def plot_solution_global(self, plot, X, nb_points=200):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        ux, uy = 0*1j*x_f, 0*1j*x_f
        for i_dim in range(2):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            ux += self.SV[1, i_dim+2]*np.exp(-self.lam[i_dim]*x_b)*X[i_dim+2]
            uy += self.SV[3, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            uy += self.SV[3, i_dim+2]*np.exp(-self.lam[i_dim]*x_b)*X[i_dim+2]
        if plot[0]:
            plt.figure("Solid displacement along y")
            plt.plot(self.x[0]+x_f, np.abs(ux), 'r')
            plt.plot(self.x[0]+x_f, np.imag(ux), 'm')
        if plot[1]:
            plt.figure("Solid displacement along x")
            plt.plot(self.x[0]+x_f, np.abs(uy), 'r')
            plt.plot(self.x[0]+x_f, np.imag(uy), 'm')

    def plot_solution_recursive(self, plot, X, nb_points=10):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        ux, uy = 0*1j*x_f, 0*1j*x_f
        for i_dim in range(4):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            uy += self.SV[3, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
        if plot[0]:
            plt.figure("Solid displacement along y")
            plt.plot(self.x[0]+x_f, np.abs(ux), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(ux), 'm.')
        if plot[1]:
            plt.figure("Solid displacement along x")
            plt.plot(self.x[0]+x_f, np.abs(uy), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(uy), 'm.')