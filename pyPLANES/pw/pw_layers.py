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
        self.nb_waves_medium = None
        self.nb_fields_SV = None
        self.nb_waves = None
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

        # import matplotlib.pyplot as plt 
        # plt.matshow(np.log10(np.abs(Om)))
        # plt.title(" Om entree transfer homogeneous")

        self.order_lam()

        Phi = self.SV
        lambda_ = self.lam

        Phi_inv = LA.inv(Phi)

        _index = self.nb_waves_medium*self.nb_waves
        _list = [0.]*(_index-1)+[1.] +[np.exp(-(lambda_[i]-lambda_[_index-1])*self.d) for i in range(_index, 2*_index)]
        Lambda = np.diag(np.array(_list))
        alpha_prime = Phi.dot(Lambda).dot(Phi_inv) # Eq (21)
        xi_prime = Phi_inv[:_index,:] @ Om
        # print("xi prime")
        # print(xi_prime)
        # print(xi_prime[:,-2:])
        # print(Phi_inv[:_index,:])
        # print(LA.det(xi_prime))
        # mlkmk
        # print(Phi_inv[:,0])
        # import matplotlib.pyplot as plt
        # plt.matshow(np.log10(np.abs(Phi_inv[:_index,:])))
        # plt.matshow(np.log10(np.abs(Om)))
        # plt.matshow(np.log10(np.abs(xi_prime)))
        # plt.show()



        _list = [np.exp(-(lambda_[_index-1]-lambda_[i])*self.d) for i in range(_index)]
        xi_prime_lambda = LA.inv(xi_prime).dot(np.diag(_list))

        # jkljkl

        Om = alpha_prime.dot(Om).dot(xi_prime_lambda)
        for i in range(_index-1):
            Om[:,i] += Phi[:, i]
        
        Xi = xi_prime_lambda*np.exp(lambda_[_index-1]*self.d)

        # import matplotlib.pyplot as plt 
        # plt.matshow(np.log10(np.abs(Om)))
        # plt.title(" Om sortie transfer homogeneous")

        return Om, Xi

    def update_Omega(self, Om):
        pass 

    def order_lam(self):
        # print(self.lam)
 
        _index = np.argsort(self.lam.real)[::-1]
        self.SV = self.SV[:, _index]
        self.lam = self.lam[_index]
        # print(self.lam)
        # sdf

class FluidLayer(PwLayer):
    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)
        self.nb_waves_medium = 1
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
        pr, ut = 0*1j*x_f, 0*1j*x_f
        for i_dim in range(2*self.nb_waves):        
            pr += self.SV[1, i_dim]*np.exp(self.lam[1]*x_f)*X[1]
            ut += self.SV[0, i_dim]*np.exp(self.lam[0]*x_f)*X[0]
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r.')
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm.')

class PemLayer(PwLayer):

    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)
        self.nb_waves_medium = 3
        self.nb_fields_SV = 6
        self.typ = "Biot98"

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
        for i_dim in range(6*self.nb_waves):
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

class ElasticLayer(PwLayer):

    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)
        self.nb_waves_medium = 2
        self.nb_fields_SV = 4

    def __str__(self):
        out = "\t Elastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = elastic_waves_TMM(self.medium, kx)
        self.nb_waves = len(kx)

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
        for i_dim in range(4*self.nb_waves):
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