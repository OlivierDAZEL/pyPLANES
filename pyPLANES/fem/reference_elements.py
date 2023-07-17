#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# reference_elements.py
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

import sys
sys.path.insert(0, "../..")

import numpy as np
import numpy.linalg as LA
import math as math
import importlib
import scipy.integrate as integrate

from pyPLANES.fem.lobatto_polynomials import lobatto as l
from pyPLANES.fem.lobatto_polynomials import lobatto_kernels as kernel
from pyPLANES.fem.utils_fem import create_legendre_table

import warnings
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from numpy.polynomial.legendre import leggauss
from pyPLANES.fem.quadlaum import quadlaum


class Ka:
    def __init__(self, order, p):
        """
        Parameters
        ----------
        order : int, polynomial order of the element
        p : int, order of the integration scheme
        """
        if order<8:
            self.order = order
        else:
            raise NameError("pyPLANES limited at order 7 (for the moment until integration schemes are validated for orders >7)")

        self.xi, self.w = leggauss(p)
        self.xi, self.w = quadlaum(p)
        # Number of Shape Functions
        self.nb_v = 2
        self.nb_e = order-1
        self.nb_SF = self.nb_v+self.nb_e
        # Initialisation of Shape Functions and derivatives
        self.Phi = np.zeros((self.nb_SF, len(self.w)))
        self.dPhi = np.zeros((self.nb_SF, len(self.w)))
        self.coefficients = []
        # Shape Functions
        for _o in range(order+1):
            self.Phi[_o, :], self.dPhi[_o, :], coeff = l(_o, self.xi)
            self.coefficients.append(coeff)


    def __str__(self):
        out = "K_a of order {}".format(self.order)
        return out

class KaPw(Ka):
    def __init__(self, order, p):
        Ka.__init__(self, order, p)
        # legendre table corresponds to values of L_n^(j)(1)
        self.legendre_table = create_legendre_table(self.order)

    def __str__(self):
        out = "KaPw of order {}".format(self.order)
        return out

    def int_monomial_exponential(self, k):

        eps = [0., 1., 0., -1.]*20
        gamma = [1, 0., -1., 0.]*20
        aa = [eps[o]/math.factorial(o) for o in range(50)]
        bb = [gamma[o]/math.factorial(o) for o in range(50)]

        I = np.zeros(self.order+1,dtype=np.complex128)
        I[0] = 2*np.sinc(k/np.pi)
        a_n, b_n = 0., 1.
        for o in range(1,self.order+1):
            a_n += aa[o]*k**o 
            b_n += bb[o]*k**o

            a_n_p = aa[o+1]+aa[o+2]*k+aa[o+3]*k**2+aa[o+4]*k**3
            b_n_p = bb[o+1]+bb[o+2]*k+aa[o+3]*k**2+aa[o+4]*k**3

            a_n_p = np.sum([aa[o+i+1]*k**i for i in range(15)])
            b_n_p = np.sum([bb[o+i+1]*k**i for i in range(15)])

            I[o] = b_n*a_n_p-a_n*b_n_p
            if math.isclose(np.sin(k),a_n_p*k**(o+1)+a_n,rel_tol=1e-10):
                # print("sin is close")
                r_s = 0.
            else:
                # print("sin is not close")
                r_s = (np.sin(k)-a_n_p*k**(o+1)-a_n)/k**(o+1)

            # print("cos")
            # print(np.cos(k))
            # print(b_n_p*k**(o+1)+b_n)

            if math.isclose(np.cos(k),b_n_p*k**(o+1)+b_n,rel_tol=1e-10):
                # print("cos is close")
                r_c = 0.
            else:
                # print("cos is not close")
                r_c = (np.cos(k)-b_n_p*k**(o+1)-b_n)/k**(o+1)
            if np.abs(k)>0.2:
                I[o] += b_n*r_s-a_n*r_c

            I[o] *=(2*math.factorial(o)*(-1j)**o)
        return I 

    def int_monomial_exponential_scipy(self, k):
        n, m = self.Phi.shape
        out = np.zeros(n, dtype=complex)
        for order in range(n):
            f_r = lambda x: x**order*np.cos(k*x)
            f_i = lambda x: x**order*np.sin(-k*x)
            out[order] = integrate.quad(f_r, -1, 1)[0]+1j*integrate.quad(f_i, -1, 1)[0]
        return out

    def int_monomial_exponential_pg(self, k):
        ''' Returns the vector of \dint{-1}{1} ln(xi)e^{-ik xi} dxi'''

        n, m = self.Phi.shape
        out = np.zeros(n, dtype=complex)
        # print(k)
        ik = 1j*k
        for order in range(n):
            for i_w, w in enumerate(self.w):
                out[order] += (np.exp(-ik*self.xi[i_w])*(self.xi[i_w]**order))*w
        return out

    def int_lobatto_exponential_scipy(self, k):
        n, m = self.Phi.shape
        out = np.zeros(n, dtype=complex)
        for order in range(n):
            f_r = lambda x: l(order, x)[0]*np.cos(k*x)
            f_i = lambda x: l(order, x)[0]*np.sin(-k*x)
            out[order] = integrate.quad(f_r, -1, 1)[0]+1j*integrate.quad(f_i, -1, 1)[0]
        return out

    def int_lobatto_exponential_analytic(self, k):
        ''' Returns the vector of \dint{-1}{1} ln(xi)e^{-ik xi} dxi'''
        I = self.int_monomial_exponential(k)
        I_scipy = self.int_monomial_exponential_scipy(k)
        out = np.zeros(self.order+1, dtype=np.complex128)
        out_scipy = np.zeros(self.order+1, dtype=np.complex128)
        out[0] = np.sum(self.coefficients[0]*I[:2])
        for _o in range(1, self.order+1):
            print(self.coefficients[_o]*I[:_o+1])
            print(np.sum(self.coefficients[_o]*I[:_o+1]))
            out[_o] = np.sum(self.coefficients[_o]*I[:_o+1])
        return out

    def int_lobatto_exponential(self, k):
        ''' Returns the vector of \dint{-1}{1} ln(xi)e^{-ik xi} dxi'''

        n, m = self.Phi.shape
        out = np.zeros(n, dtype=complex)
        ik = 1j*k

        if abs(k) < .1 :
            for i_w, w in enumerate(self.w):
                out += (np.exp(-ik*self.xi[i_w])*w)* self.Phi[:, i_w].reshape(n)
        else:
            out[0] = (np.exp(ik)/ik)+(1j*np.sin(k))/k**2
            out[1] =-(np.exp(-ik)/ik)-(1j*np.sin(k))/k**2
            for _n in range(2, n):
                _ = [self.legendre_table[_n-1, j]*(np.exp(-ik)+np.exp(ik)*(-1)**(_n+j))/(ik)**j for j in range(_n) ] 
                out[_n] = (np.sum(_)/(k**2*np.sqrt(2./(2*_n-1))))
        return out


    def int_lobatto_exponential_pg(self, k):
        ''' Returns the vector of \dint{-1}{1} ln(xi)e^{-ik xi} dxi'''
        n, m = self.Phi.shape
        out = np.zeros(n, dtype=complex)
        ik = 1j*k
        for i_w, w in enumerate(self.w):
            out += (np.exp(-ik*self.xi[i_w])*w)* self.Phi[:, i_w].reshape(n)
        return out

class Kt:
    def __init__(self, order=2):
        self.order = order
        # Integration scheme
        self.xi_1, self.xi_2, self.w = quadlaum(2*order, "Kt")
        
        
        # Number of Shape Functions
        self.nb_v = 3
        self.nb_e = 3*(order-1)
        self.nb_f = int(((order-1)*(order-2))/2)
        # Number of master shape functions
        self.nb_m_SF = self.nb_v+ self.nb_e

        # Number of slave shape functions (to condense)
        self.nb_s_SF = self.nb_f
        self.nb_SF = self.nb_m_SF+self.nb_s_SF

        self.Phi, self.dPhi = shape_functions_Kt(self.xi_1, self.xi_2, self.order)

        # For plots
        tri_Kt = mtri.Triangulation(np.asarray([-1,1,-1]), np.asarray([-1,-1,1]))
        tri_Kt = mtri.UniformTriRefiner(tri_Kt).refine_triangulation(subdiv=5)
        xi_1 = tri_Kt.x
        xi_2 = tri_Kt.y
        self.phi_plot = shape_functions_Kt(xi_1, xi_2, order)[0]


    def __str__(self):
        out = "K_t of order {}".format(self.order)
        return out

class PlotKt:
    def __init__(self, order=2):
        self.order = order
        x = np.asarray([-1,1,-1])
        y = np.asarray([-1,-1,1])
        tri = mtri.Triangulation(x, y)

        refiner = mtri.UniformTriRefiner(tri)
        fine_tri = refiner.refine_triangulation(subdiv=9)

        xi_1 = fine_tri.x
        xi_2 = fine_tri.y

        self.Phi, self.dPhi = shape_functions_Kt(xi_1, xi_2, self.order)

        for i_SF in range(self.Phi.shape[0]):
            plt.figure()
            plt.tricontourf(fine_tri, self.Phi[i_SF,:],cmap=cm.jet,levels=10)
            plt.colorbar()
        plt.show()

def shape_functions_Kt(xi_1, xi_2, order):
    ''' Return Lobatto Shape functions on Kt'''
    nb_SF = 3*order+int(((order-1)*(order-2))/2)

    lamda = np.zeros((3,len(xi_1)))
    dlamda = [np.zeros((3, len(xi_1))),np.zeros((3, len(xi_1)))]
    # Equations (2.17) of Solin 
    lamda[0,:] = (xi_2 +1.)/2.
    dlamda[1][0, :] = 1./2.
    lamda[1, :] = -(xi_1 +xi_2)/2.
    dlamda[0][1, :] = -1./2.
    dlamda[1][1, :] = -1./2.
    lamda[2, :] = (xi_1 +1.)/2.
    dlamda[0][2, :] = 1./2.

    # Initialisation of Shape Functions and derivatives
    Phi = np.zeros((nb_SF, len(xi_1)))
    dPhi = [np.zeros((nb_SF, len(xi_1))),np.zeros((nb_SF, len(xi_1)))]

    # Vertices Shape Functions Eq (2.20) of Solin
    Phi[0, :], Phi[1, :], Phi[2, :] = lamda[1, :], lamda[2, :], lamda[0, :]
    for _xi in range(2):
        dPhi[_xi][0, :], dPhi[_xi][1, :], dPhi[_xi][2, :] = dlamda[_xi][1, :], dlamda[_xi][2, :], dlamda[_xi][0, :]
    # Edge Shape Functions Eq (2.21) of Solin 
    for _o in range(order-1):
        _index = 3+_o
        Phi[_index, :] = lamda[1, :]*lamda[2, :]*kernel(_o, lamda[2, :]-lamda[1, :])[0]
        for _xi in range(2):
            dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[2, :]*kernel(_o, lamda[2, :]-lamda[1, :])[0]
            dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[1, :]*kernel(_o, lamda[2, :]-lamda[1, :])[0]
            dPhi[_xi][_index, :] += lamda[1, :]*lamda[2, :]*kernel(_o, lamda[2, :]-lamda[1,:])[1]*(dlamda[_xi][2, :]-dlamda[_xi][1, :])

        _index = 3+order-1 + _o
        Phi[_index, :] = lamda[2, :]*lamda[0, :]*kernel(_o, lamda[0, :]-lamda[2, :])[0]
        for _xi in range(2):
            dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[0, :]*kernel(_o, lamda[0, :]-lamda[2, :])[0]
            dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[2, :]*kernel(_o, lamda[0, :]-lamda[2, :])[0]
            dPhi[_xi][_index, :] += lamda[0, :]*lamda[2, :]*kernel(_o, lamda[0, :]-lamda[2,:])[1]*(dlamda[_xi][0, :]-dlamda[_xi][2, :])
        _index = 3+2*(order-1)+ _o
        Phi[_index, :] = lamda[0, :]*lamda[1, :]*kernel(_o, lamda[1, :]-lamda[0, :])[0]
        for _xi in range(2):
            dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[1, :]*kernel(_o, lamda[1, :]-lamda[0, :])[0]
            dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[0, :]*kernel(_o, lamda[1, :]-lamda[0, :])[0]
            dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*kernel(_o, lamda[1, :]-lamda[0,:])[1]*(dlamda[_xi][1, :]-dlamda[_xi][0, :])

    # Face Shape Functions Eq (2.23) of Solin
    _index = 3*order
    for n_1 in range(1, order):
        for n_2 in range(1, order-n_1):
            Phi[_index, :] = lamda[0, :]*lamda[1, :]*lamda[2, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
            for _xi in range(2):
                dPhi[_xi][_index, :] += dlamda[_xi][0, :]*lamda[1, :]*lamda[2, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += dlamda[_xi][1, :]*lamda[2, :]*lamda[0, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += dlamda[_xi][2, :]*lamda[0, :]*lamda[1, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*kernel(n_1-1, lamda[2, :]-lamda[1, :])[1]*(dlamda[_xi][2, :]-dlamda[_xi][1, :])*kernel(n_2-1, lamda[1, :]-lamda[0, :])[0]
                dPhi[_xi][_index, :] += lamda[0, :]*lamda[1, :]*lamda[2, :]*kernel(n_2-1, lamda[1, :]-lamda[0, :])[1]*(dlamda[_xi][1, :]-dlamda[_xi][0, :])*kernel(n_1-1, lamda[2, :]-lamda[1, :])[0]
            _index += 1

    return Phi, dPhi



if __name__ == "__main__":
    
    ref_elem = KaPw(7,20)
    vec_k=np.logspace(-20,0,100)

    vec_pg = np.zeros((ref_elem.order+1,len(vec_k)),dtype=np.complex128)
    vec_analytic = np.zeros((ref_elem.order+1,len(vec_k)),dtype=np.complex128)
    vec_scipy = np.zeros((ref_elem.order+1,len(vec_k)),dtype=np.complex128)
    mon_pg = np.zeros((ref_elem.order+1,len(vec_k)),dtype=np.complex128)
    mon_analytic = np.zeros((ref_elem.order+1,len(vec_k)),dtype=np.complex128)
    mon_scipy = np.zeros((ref_elem.order+1,len(vec_k)),dtype=np.complex128)

    for i, k in enumerate(vec_k):

        vec_pg[:, i]=ref_elem.int_lobatto_exponential_pg(k)
        vec_analytic[:, i]=ref_elem.int_lobatto_exponential_analytic(k)
        vec_scipy[:, i]=ref_elem.int_lobatto_exponential_scipy(k)
        mon_pg[:, i]=ref_elem.int_monomial_exponential_pg(k)
        mon_analytic[:, i]=ref_elem.int_monomial_exponential(k)
        mon_scipy[:, i]=ref_elem.int_monomial_exponential_scipy(k)



    for order in range(ref_elem.order+1):

        plt.figure()     
        plt.plot(vec_k,np.real(mon_analytic[order, :]),"r.",label="OD /real")
        plt.plot(vec_k,np.imag(mon_analytic[order, :]),"b.",label="OD/ imag")
        plt.plot(vec_k,np.real(mon_scipy[order, :]),"r",label="scipy")
        plt.plot(vec_k,np.imag(mon_scipy[order, :]),"b",label="scipy")
        plt.plot(vec_k,np.real(mon_pg[order, :]),"r+",label="pg")
        plt.plot(vec_k,np.imag(mon_pg[order, :]),"b+",label="pg")
        plt.title("M_{}".format(order))
        plt.legend()

    plt.show()