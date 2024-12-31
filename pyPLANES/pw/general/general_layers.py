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
from mediapack import Fluid, Air
from numpy import pi, sqrt

from pyPLANES.pw.general.polarisation_3D import fluid_waves_3D, elastic_waves_3D, PEM_waves_3D
from pyPLANES.pw.general.characteristics_3D import Characteristics_3D

from scipy.linalg import expm, block_diag


class PwGeneric_3D():
    def __init__(self, d, **kwargs):
        """
        Parameters
        ----------
        mat : mediapack medium 
            Name of the material 
        d : float
            thickness of the layer
        x_0 : position of the first interface        
        """
        self.d = d
        x_0 = kwargs.get("x_0", 0.0)
        self.method_TM = kwargs.get("method_TM", False)
        if self.method_TM in ["cheb_1", "cheb_2"]:
            self.order_chebychev = kwargs.get("order_chebychev", 20)
        # pymls layer constructor 
        self.x = [x_0, x_0+self.d]  # 
        self.dofs = None
        self.lam = None
        self.SV = None
        self.SVp = None

    def update_frequency(self, omega):
        pass

    def state_matrix(self, omega, x=0):
        pass

    def A2SV(self, S):
        pass 
        
    def transfert_matrix_analytic(self, omega, direction=1):
        pass

    def transfert_matrix(self, omega, direction=1):

        if self.method_TM == "expm":
            return expm(self.state_matrix(omega)*direction*self.d)
        elif self.method_TM == "diag":
            Phi = self.SV
            Phi_inv = LA.inv(Phi)
            return Phi.dot(np.diag(np.exp(direction*self.lam*self.d))).dot(Phi_inv)
        elif self.method_TM == "analytic":
            return self.transfert_matrix_analytic(omega, direction)
        elif self.method_TM == "cheb_1":
            return self.transfert_matrix_cheb_1(omega, direction)
        elif self.method_TM == "cheb_2":
            return self.transfert_matrix_cheb_2(omega, direction)
        else:
            raise NameError("self.method_TM incorrect")

    def update_Omega(self, Om, omega, method="Recursive Method"):
        """
        Update the information matrix Omega

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
        if method == "TMM":
            TM = self.transfert_matrix(omega, -1)
            Om = TM@Om 
            Xi = np.eye(Om.shape[1])
            return Om, Xi
        elif method == "Recursive Method":
            self.order_lam()
            Phi = self.SV
            lambda_ = self.lam

            Phi_inv = LA.inv(Phi)
            m = self.nb_waves_in_medium*self.nb_waves
            _list = [0.]*(m-1)+[1.] +[np.exp(-(lambda_[m+i]-lambda_[m-1])*self.d) for i in range(0, m)]
            Lambda = np.diag(np.array(_list))
            alpha_prime = Phi.dot(Lambda).dot(Phi_inv) # Eq (21)
            
            xi_prime = Phi_inv[:m,:] @ Om # Eq (23)
            _list = [np.exp(-(lambda_[m-1]-lambda_[i])*self.d) for i in range(m-1)] + [1.]
            xi_prime_lambda = LA.inv(xi_prime).dot(np.diag(_list))
            Om = alpha_prime.dot(Om).dot(xi_prime_lambda)

            Om[:,:m-1]  += Phi[:,:m-1]    
            
            Xi = xi_prime_lambda*np.exp(lambda_[m-1]*self.d)
            return Om, Xi

    def update_Omegac(self, Om, omega, method="Recursive Method"):
        """
        Update the information matrix Omega for characteristic method

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
        self.order_lam()
        
        Phi = np.kron(np.eye(self.nb_waves),self.carac.Q)@self.SV
        lambda_ = self.lam
        
        Phi_inv = LA.inv(Phi)
        
        m = self.nb_waves_in_medium*self.nb_waves
        _list = [0.]*(m-1)+[1.] +[np.exp(-(lambda_[m+i]-lambda_[m-1])*self.d) for i in range(0, m)]
        Lambda = np.diag(np.array(_list))
        alpha_prime = Phi.dot(Lambda).dot(Phi_inv) # Eq (21)
        
        xi_prime = Phi_inv[:m,:] @ Om # Eq (23)
        _list = [np.exp(-(lambda_[m-1]-lambda_[i])*self.d) for i in range(m-1)] + [1.]
        xi_prime_lambda = LA.inv(xi_prime).dot(np.diag(_list))
        Om = alpha_prime.dot(Om).dot(xi_prime_lambda)

        Om[:,:m-1]  += Phi[:,:m-1]
        
        Xi = xi_prime_lambda*np.exp(lambda_[m-1]*self.d)
        return Om, Xi

    def order_lam(self):
        _index = np.argsort(self.lam.real)
        self.SV = self.SV[:, _index]
        self.lam = self.lam[_index]

class PwLayer_3D(PwGeneric_3D):
    """
    Base class for Plane Wave layer definition and manipulation
    
    Attributes :
    ------------------------

    medium : mediapack material 

    d : thickness of the layer 

    x : list with the abscissa of the layer

    interfaces : list with the interfaces of the layer
    
    """
    def __init__(self, mat, d, **kwargs):
        """
        Parameters
        ----------
        mat : mediapack medium 
            Name of the material 
        d : float
            thickness of the layer
        x_0 : position of the first interface        
        """
        PwGeneric_3D.__init__(self, d, **kwargs)
        self.medium = mat
        self.carac = Characteristics_3D(self.medium)
        self.interfaces = [None, None]
        self.nb_waves_in_medium = None
        self.nb_fields_SV = None
        self.nb_waves_in_medium = None
        self.kx = None
        
    def __str__(self):
        pass

    def update_frequency(self, omega, kx, kz):
        self.kx = kx
        self.kz = kz
        self.medium.update_frequency(omega)
        self.carac.update_frequency(omega)
        
class FluidLayer_3D(PwLayer_3D):
    
    # S={0:u_y , 1:p}
    
    def __init__(self, mat, d, **kwargs):
        PwLayer_3D.__init__(self, mat, d, **kwargs)
        self.nb_waves_in_medium = 1
        self.nb_fields_SV = 2

    def __str__(self):
        out = "\t Fluid Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx=[0], kz=0):
        PwLayer_3D.update_frequency(self, omega, kx, kz)
        self.medium.update_frequency(omega)
        if isinstance(kx, np.ndarray):
            self.nb_waves = len(kx)
        else:
            self.nb_waves = 1
        self.SV, self.lam = fluid_waves_3D(self.medium, kx, kz)
        
    def state_matrix(self, omega):
        kx = self.kx
        if self.medium.MEDIUM_TYPE == 'eqf':
            K = self.medium.K_eq_til
            rho = self.medium.rho_eq_til
        elif self.medium.MEDIUM_TYPE == 'fluid':
            K = self.medium.K
            rho = self.medium.rho
         
        alpha = np.zeros((2, 2), dtype=complex)
        alpha[0, 1] = (kx**2/(rho*omega**2)) -1/K
        alpha[1, 0] = rho*omega**2
        return alpha
        
    def A2SV(self, omega, S):
        if self.medium.MEDIUM_TYPE == 'eqf':
            rho = self.medium.rho_eq_til
        elif self.medium.MEDIUM_TYPE == 'fluid':
            rho = self.medium.rho
        return np.array([(1/rho*omega**2)*S[:,:,1],S[:,:,0]]).reshape((2*self.order_chebychev,1))
         
    def transfert_matrix_analytic(self, om):
        kx = self.kx
        T = np.zeros((2, 2), dtype=complex)
        ky = np.sqrt((om/self.medium.c)**2-kx**2)
        alpha = ky/(self.medium.rho*om**2)
        T[0, 0] = np.cos(ky*self.d)
        T[1, 0] = np.sin(ky*self.d)/alpha
        T[0, 1] = -alpha*np.sin(ky*self.d)
        T[1, 1] = np.cos(ky*self.d)
        return T

    def plot_solution_global(self, plot, X, nb_points=200):

        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        x_b = x_f - (self.x[1]-self.x[0])
        pr =  self.SV[1, 0]*np.exp(self.lam[0]*x_f)*X[0]
        pr += self.SV[1, 1]*np.exp(self.lam[1]*x_b)*X[1]
        ut =  self.SV[0, 0]*np.exp(self.lam[0]*x_f)*X[0]
        ut += self.SV[0, 1]*np.exp(self.lam[1]*x_b)*X[1]
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r',label="abs(GM)")
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm',label="imag(GM)")
            # plt.figure("ut")
            # plt.plot(self.x[0]+x_f, np.real(ut), 'r')
            # plt.plot(self.x[0]+x_f, np.imag(ut), 'm')

    def plot_solution_recursive(self, plot, X, nb_points=25):
        x_f = np.linspace(0, self.x[1]-self.x[0], nb_points)
        pr, ut = 0*1j*x_f, 0*1j*x_f
        for i_dim in range(2*self.nb_waves):        
            pr += self.SV[1, i_dim]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            ut += self.SV[0, i_dim]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[0]+x_f, np.abs(pr), 'r.' ,label="abs(recursive)")
            plt.plot(self.x[0]+x_f, np.imag(pr), 'm.',label="imag(recursive)")

    def plot_solution_characteristics(self, plot, X, nb_points=25):
        x_f = np.linspace(-self.x[1]+self.x[0], 0, nb_points)
        pr, ut = 0*1j*x_f, 0*1j*x_f
        X = LA.inv(self.SV)@self.carac.P@X
        for i_dim in range(2*self.nb_waves):        
            pr += self.SV[1, i_dim]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            ut += self.SV[0, i_dim]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[1]+x_f, np.abs(pr), 'r+' ,label="abs(charac)")
            plt.plot(self.x[1]+x_f, np.imag(pr), 'm+',label="imag(charac)")

class PemLayer_3D(PwLayer_3D):

    def __init__(self, mat, d, **kwargs):
        PwLayer_3D.__init__(self, mat, d, **kwargs)
        self.nb_waves_in_medium = 4
        self.nb_fields_SV = 8
        self.typ = "Biot98"

    def __str__(self):
        out = "\t Poroelastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx=[0], kz=0):
        PwLayer_3D.update_frequency(self, omega, kx, kz)
        self.SV, self.lam = PEM_waves_3D(self.medium, self.kx, self.kz)


    def state_matrix(self, omega):
        # self.medium.update_frequency(omega)
        k_x =self.kx[0]
        m = self.medium
        alpha = np.array([
        [0, 0, 0, 1j*k_x*m.A_hat/m.P_hat, 1j*k_x*m.gamma_til, -(m.A_hat**2-m.P_hat**2)/m.P_hat*k_x**2-m.rho_til*omega**2],
        [0, 0, 0, 1/m.P_hat, 0, 1j*k_x*m.A_hat/m.P_hat],
        [0, 0, 0, 0, -1/m.K_eq_til+k_x**2/(m.rho_eq_til*omega**2), -1j*k_x*m.gamma_til],
        [1j*k_x, -m.rho_s_til*omega**2, -m.rho_eq_til*m.gamma_til*omega**2, 0, 0, 0],
        [0, m.rho_eq_til*m.gamma_til*omega**2, m.rho_eq_til*omega**2, 0, 0, 0],
        [1/m.N, 1j*k_x, 0, 0, 0, 0]])
        return alpha
    
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

    def plot_solution_characteristics(self, plot, X, nb_points=25):
        x_f = np.linspace(-self.x[1]+self.x[0], 0, nb_points)
        ux, uy, pr, ut = 0*1j*x_f, 0*1j*x_f, 0*1j*x_f, 0*1j*x_f
        X = LA.inv(self.SV)@self.carac.P@X
        for i_dim in range(6*self.nb_waves):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            uy += self.SV[5, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            pr += self.SV[4, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
        if plot[0]:
            plt.figure("Solid displacement along y")
            plt.plot(self.x[1]+x_f, np.abs(ux), 'r+')
            plt.plot(self.x[1]+x_f, np.imag(ux), 'm+')
        if plot[1]:
            plt.figure("Solid displacement along x")
            plt.plot(self.x[1]+x_f, np.abs(uy), 'r+')
            plt.plot(self.x[1]+x_f, np.imag(uy), 'm+')
        if plot[2]:
            plt.figure("Pressure")
            plt.plot(self.x[1]+x_f, np.abs(pr), 'r+')
            plt.plot(self.x[1]+x_f, np.imag(pr), 'm+')

class ElasticLayer_3D(PwLayer_3D):

    def __init__(self, mat, d, **kwargs):
        PwLayer_3D.__init__(self, mat, d, **kwargs)
        self.nb_waves_in_medium = 3
        self.nb_fields_SV = 6

    def __str__(self):
        out = "\t Elastic Layer / " + self.medium.name
        return out

    def update_frequency(self, omega, kx, kz):
        PwLayer_3D.update_frequency(self, omega, kx, kz)
        self.medium.update_frequency(omega)
        self.SV, self.lam = elastic_waves_3D(self.medium, kx, kz)


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

    def plot_solution_characteristics(self, plot, X, nb_points=25):
        x_f = np.linspace(-self.x[1]+self.x[0], 0, nb_points)
        ux, uy, = 0*1j*x_f, 0*1j*x_f
        X = LA.inv(self.SV)@np.kron(np.eye(self.nb_waves),self.carac.P)@X
        for i_dim in range(4*self.nb_waves):
            ux += self.SV[1, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
            uy += self.SV[3, i_dim  ]*np.exp(self.lam[i_dim]*x_f)*X[i_dim]
        if plot[0]:
            plt.figure("Solid displacement along y")
            plt.plot(self.x[1]+x_f, np.abs(ux), 'r+')
            plt.plot(self.x[1]+x_f, np.imag(ux), 'm+')
        if plot[1]:
            plt.figure("Solid displacement along x")
            plt.plot(self.x[1]+x_f, np.abs(uy), 'r+')
            plt.plot(self.x[1]+x_f, np.imag(uy), 'm+')



    def state_matrix(self, omega, x):
        alpha = np.zeros((2, 2), dtype=complex)
        r, r_prime = self.R(x)
        alpha[0, 0] = -2*self.m#r_prime/r
        alpha[0, 1] = -1/self.medium.K
        alpha[1, 0] = self.medium.rho*omega**2
        return alpha

    
    def transfert_matrix_analytic(self, omega, direction=1):
        T = np.zeros((2, 2), dtype=complex)
        k = omega/self.medium.c
        k_1 = np.sqrt(k**2-self.m**2+0*1j)
        Z_c1 = self.medium.rho*self.medium.c*k_1/k
        m =self.m
        L = self.d 
        
        T[0, 0] = np.cos(k_1*L)+(m/k_1)*np.sin(k_1*L)
        T[0, 1] = 1j*np.sin(k_1*L)/Z_c1
        T[1, 0] = 1j*Z_c1*(1+(m/k_1)**2)*np.sin(k_1*L)
        T[1, 1] = np.cos(k_1*L)-(m/k_1)*np.sin(k_1*L)
        
        T /= np.exp(-m*L)
        T[1,0] *= 1j*omega
        T[0,1] /= 1j*omega

        if direction == 1:
            return LA.inv(T)
        else:
            return T#LA.inv(T)    