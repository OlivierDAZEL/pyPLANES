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

class PwLayer():
    """
    Base class for Plane Wave layer definition and manipulation
    
    Attributes :
    ------------------------

    medium : mediapack material 

    d : thickness of the layer 

    x : list with the abscissa of the layers 

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
        index = np.argsort(self.lam.real)[::-1]
        self.SV = self.SV[:, index]
        self.lam = self.lam[index] 

class FluidLayer(PwLayer):
    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)

    def __str__(self):
        out = "\t Fluid Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = fluid_waves_TMM(self.medium, kx)

    def update_Omega(self, om, Om):
        T = np.zeros((2, 2), dtype=complex)
        T[0, 0] = np.cos(self.ky*self.d)
        T[1, 0] = (om**2*self.medium.rho/self.ky)*np.sin(self.ky*self.d)
        T[0, 1] = -(self.ky/(om**2*self.medium.rho))*np.sin(self.ky*self.d)
        T[1, 1] = np.cos(self.ky*self.d)
        return T@Om 

    def transfert(self, Om):
        self.order_lam()
        alpha = self.SV[0,0]
        xi_prime = (Om[0]/alpha+Om[1])/2.

        Om_0 = (Om[0]-alpha*Om[1])/2. 
        Om_1 = -Om_0/alpha
        Om_ = np.array([Om_0, Om_1])*np.exp(-2.*self.lam[0]*self.d)/xi_prime
        
        Om_[0] += alpha
        Om_[1] += 1.

        Xi = np.exp(-self.lam[0]*self.d)/xi_prime
        
        return Om_ , np.array([Xi])

    def plot_solution(self, plot, X, nb_points=200):
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

class PemLayer(PwLayer):

    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)

    def __str__(self):
        out = "\t Poroelastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = PEM_waves_TMM(self.medium, kx)

    def plot_solution(self, plot, X, nb_points=200):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        ux, uy, pr, ut = 0*1j*x_f, 0*1j*x_f, 0*1j*x_f, 0*1j*x_f
        for i_dim in range(3):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            ux += self.SV[1, i_dim+3]*np.exp(-self.lam[i_dim]*x_b)*X[i_dim+3]
            uy += self.SV[5, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            uy += self.SV[5, i_dim+3]*np.exp(-self.lam[i_dim]*x_b)*X[i_dim+3]
            pr += self.SV[4, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            pr += self.SV[4, i_dim+3]*np.exp(-self.lam[i_dim]*x_b)*X[i_dim+3]
            ut += self.SV[2, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            ut += self.SV[2, i_dim+3]*np.exp(-self.lam[i_dim]*x_b)*X[i_dim+3]
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


    def transfert(self, Om):
        index = np.argsort(self.lam.real)
        index = index[::-1]        

        Phi = self.SV[:,index]
        lambda_ = self.lam[index]

        Phi_inv = np.linalg.inv(Phi)

        Lambda = np.diag([
            0,
            0,
            1,
            np.exp((lambda_[3]-lambda_[2])*self.d),
            np.exp((lambda_[4]-lambda_[2])*self.d),
            np.exp((lambda_[5]-lambda_[2])*self.d)
        ])

        alpha_prime = Phi.dot(Lambda).dot(Phi_inv)
        xi_prime = Phi_inv[:2,:] @ Om
        xi_prime = np.concatenate([xi_prime, np.array([[0,0,1]])])  # TODO
        xi_prime_lambda = np.linalg.inv(xi_prime).dot(np.diag([
            np.exp((lambda_[2]-lambda_[0])*self.d),
            np.exp((lambda_[2]-lambda_[1])*self.d),
            1
        ]))

        Om_ = alpha_prime.dot(Om).dot(xi_prime_lambda)
        Om_[:,0] += Phi[:,0]
        Om_[:,1] += Phi[:,1]

        # eq. 24
        Xi = xi_prime_lambda*np.exp(-lambda_[2]*self.d)
    
        return Om_, Xi


class ElasticLayer(PwLayer):

    def __init__(self, _mat, d, _x = 0):
        PwLayer.__init__(self, _mat, d, _x)

    def __str__(self):
        out = "\t Elastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx):
        self.medium.update_frequency(omega)
        self.SV, self.lam = elastic_waves_TMM(self.medium, kx)

    def transfert(self, Om):
        index = np.argsort(self.lam.real)
        index = index[::-1]        
        Phi = self.SV[:,index]
        lambda_ = self.lam[index]

        Phi_inv = np.linalg.inv(Phi)

        Lambda = np.diag([
            0,
            1,
            np.exp((lambda_[2]-lambda_[1])*self.d),
            np.exp((lambda_[3]-lambda_[1])*self.d)
        ])

        alpha_prime = Phi.dot(Lambda).dot(Phi_inv)
        
        xi_prime = Phi_inv[:1,:] @ Om
        xi_prime = np.concatenate([xi_prime, np.array([[0,1]])])  # TODO
        xi_prime_lambda = np.linalg.inv(xi_prime).dot(np.diag([
            np.exp((lambda_[1]-lambda_[0])*self.d),
            1
        ]))

        Om_ = alpha_prime.dot(Om).dot(xi_prime_lambda)
        Om_[:,0] += Phi[:,0]

        Xi = xi_prime_lambda*np.exp(-lambda_[1]*self.d)

        return Om_, Xi

    def plot_solution(self, plot, X, nb_points=200):
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

