#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# pw_polarisations.py
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
from numpy import linalg as LA

def PEM_waves_TMM(mat, kx):
    ''' S={0: hat{sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:hat{sigma}_{yy}, 4:p, 5:u_x^s}'''
    
    n_w = len(kx)
    Phi = np.zeros((6*n_w, 6*n_w), dtype=complex)
    lam = np.zeros(6*n_w, dtype=complex)
    # ky_1 = np.sqrt(mat.delta_1**2-kx**2)
    # ky_2 = np.sqrt(mat.delta_2**2-kx**2)
    # ky_3 = np.sqrt(mat.delta_3**2-kx**2)
    
    lam_1 = np.sqrt(-mat.delta_1**2+kx**2)
    lam_2 = np.sqrt(-mat.delta_2**2+kx**2)
    lam_3 = np.sqrt(-mat.delta_3**2+kx**2)

    ky_1 = -1j*lam_1
    ky_2 = -1j*lam_2
    ky_3 = -1j*lam_3


    for _w, _kx in enumerate(kx):
        ky = np.array([ky_1[_w], ky_2[_w], ky_3[_w]], dtype=complex)
        delta = np.array([mat.delta_1, mat.delta_2, mat.delta_3], dtype=complex)

        alpha_1 = -1j*mat.A_hat*mat.delta_1**2-1j*2*mat.N*ky[0]**2
        alpha_2 = -1j*mat.A_hat*mat.delta_2**2-1j*2*mat.N*ky[1]**2
        alpha_3 = -2*1j*mat.N*ky[2]*_kx


        Phi[0+6*_w:6+6*_w, 0+6*_w] = np.array([-2*1j*mat.N*ky[0]*_kx, ky[0], mat.mu_1*ky[0], alpha_1, 1j*delta[0]**2*mat.K_eq_til*mat.mu_1, _kx], dtype=complex)
        
        Phi[0+6*_w:6+6*_w, 3+6*_w] = np.array([ 2*1j*mat.N*ky[0]*_kx,-ky[0],-mat.mu_1*ky[0], alpha_1, 1j*delta[0]**2*mat.K_eq_til*mat.mu_1, _kx], dtype=complex)

        Phi[0+6*_w:6+6*_w, 1+6*_w] = np.array([-2*1j*mat.N*ky[1]*_kx, ky[1], mat.mu_2*ky[1], alpha_2, 1j*delta[1]**2*mat.K_eq_til*mat.mu_2, _kx], dtype=complex)

        Phi[0+6*_w:6+6*_w, 4+6*_w] = np.array([ 2*1j*mat.N*ky[1]*_kx,-ky[1],-mat.mu_2*ky[1],alpha_2, 1j*delta[1]**2*mat.K_eq_til*mat.mu_2, _kx], dtype=complex)

        Phi[0+6*_w:6+6*_w, 2+6*_w] = np.array([1j*mat.N*(ky[2]**2-_kx**2), _kx, mat.mu_3*_kx, alpha_3, 0., -ky[2]], dtype=complex)
        
        Phi[0+6*_w:6+6*_w, 5+6*_w] = np.array([1j*mat.N*(ky[2]**2-_kx**2), _kx, mat.mu_3*_kx, -alpha_3, 0., ky[2]], dtype=complex)

        lam[0+6*_w:3+6*_w] =  -1j*ky
        lam[3+6*_w:6+6*_w] =  1j*ky


    return Phi, lam

def PEM_waves_PQ(mat, kx, omega):
    kx = kx[0]
    jom = 1j*omega


    k_1 = np.sqrt(mat.delta_1**2-kx**2)
    k_2 = np.sqrt(mat.delta_2**2-kx**2)
    k_3 = np.sqrt(mat.delta_3**2-kx**2)
    lam = np.array([-1j*k_1, -1j*k_2, -1j*k_3, 1j*k_1, 1j*k_2, 1j*k_3], dtype=complex)

    x_1 = -2*1j*mat.N*k_1*kx
    x_2 = -2*1j*mat.N*k_2*kx
    x_3 = 1j*mat.N*(k_3**2-kx**2)
    
    y_1 = -1j*mat.A_hat*mat.delta_1**2-1j*2*mat.N*k_1**2
    y_2 = -1j*mat.A_hat*mat.delta_2**2-1j*2*mat.N*k_2**2
    y_3 = -2*1j*mat.N*k_3*kx

    p_1 = 1j*mat.delta_1**2*mat.K_eq_til*mat.mu_1
    p_2 = 1j*mat.delta_2**2*mat.K_eq_til*mat.mu_2
    
    P = np.zeros((6, 6), dtype=complex)
    Q = np.zeros((6, 6), dtype=complex)

    P[0, :] = [jom*kx, jom*kx, -jom*k_3, jom*kx, jom*kx, jom*k_3]
    P[1, :] = [jom*k_1, jom*k_2, jom*kx, -jom*k_1, -jom*k_2, jom*kx]
    P[2, :] = [jom*mat.mu_1*k_1, jom*mat.mu_2*k_2, jom*mat.mu_3*kx, -jom*mat.mu_1*k_1, -jom*mat.mu_2*k_2, jom*mat.mu_3*kx]
    P[3,:] = [x_1, x_2, x_3, -x_1, -x_2, x_3]
    P[4,:] = [y_1, y_2, y_3, y_1, y_2, -y_3]
    P[5,:] = [p_1, p_2, 0, p_1, p_2, 0]


    A = mat.delta_1**2*mat.delta_2**2*k_3*mat.P_hat*mat.K_eq_til*(mat.mu_2-mat.mu_1)
    B = 1j*mat.N*mat.delta_3**2*k_1*k_2*(mat.mu_1-mat.mu_2)

    Q[0,:] = [-p_2*2*mat.N*kx*k_3/(2*A*omega), 
              -mat.N*k_2*(2*mat.mu_3*kx**2+mat.mu_2*(k_3**2-kx**2))/(2*B*omega), 
              k_2*mat.N*mat.delta_3**2/(2*B*omega),
              -(k_2*kx*(mat.mu_3-mat.mu_2)) /(2*B),
              k_3*p_2/(2*A), 1j*k_3*mat.delta_2**2*(2*mat.N+mat.A_hat)/(2*A)]
    Q[1,:] = [p_1*2*mat.N*kx*k_3/(2*A*omega), 
              mat.N*k_1*(2*mat.mu_3*kx**2+mat.mu_1*(k_3**2-kx**2))/(2*B*omega), 
              -k_1*mat.N*mat.delta_3**2/(2*B*omega),
              k_1*kx*(mat.mu_3-mat.mu_1) /(2*B),
              -k_3*p_1/(2*A), -1j*k_3*mat.delta_1**2*(2*mat.N+mat.A_hat)/(2*A)]
    Q[2,:] = [-1j*(p_1*y_2-p_2*y_1)/(2*A*omega), 
              -2*mat.N*k_1*k_2*kx*(mat.mu_2-mat.mu_1)/(2*B*omega), 
              0,
              -k_1*k_2*(mat.mu_2-mat.mu_1) /(2*B),
              kx*(p_2-p_1)/(2*A), -kx*(y_2-y_1)/(2*A)]
    Q[3,:] = [-1j*p_2*y_3/(2*A*omega), 
              mat.N*k_2*(2*mat.mu_3*kx**2+mat.mu_2*(k_3**2-kx**2))/(2*B*omega), 
              -k_2*mat.N*mat.delta_3**2/(2*B*omega),
              k_2*kx*(mat.mu_3-mat.mu_2) /(2*B),
              k_3*p_2/(2*A), 1j*k_3*mat.delta_2**2*(2*mat.N+mat.A_hat)/(2*A)]
    Q[4,:] = [1j*p_1*y_3/(2*A*omega), 
              -mat.N*k_1*(2*mat.mu_3*kx**2+mat.mu_1*(k_3**2-kx**2))/(2*B*omega), 
              k_1*mat.N*mat.delta_3**2/(2*B*omega),
              -k_1*kx*(mat.mu_3-mat.mu_1) /(2*B),
              -k_3*p_1/(2*A), -1j*k_3*mat.delta_1**2*(2*mat.N+mat.A_hat)/(2*A)]
    Q[5,:] = [1j*(p_1*y_2-p_2*y_1)/(2*A*omega), 
              -2*mat.N*k_1*k_2*kx*(mat.mu_2-mat.mu_1)/(2*B*omega), 
              0,
              -k_1*k_2*(mat.mu_2-mat.mu_1) /(2*B),
              -kx*(p_2-p_1)/(2*A), kx*(y_2-y_1)/(2*A)]

    return P, Q, lam

def elastic_waves_TMM(mat, kx):
    ''' S={0:sigma_{xy}, 1: u_y, 2 sigma_{yy}, 3 u_x}'''

    n_w = len(kx)
    Phi = np.zeros((4*n_w,4*n_w), dtype=complex)
    lam = np.zeros(4*n_w, dtype=complex)

    # ky_p = np.sqrt(mat.delta_p**2-kx**2)
    # ky_s = np.sqrt(mat.delta_s**2-kx**2)

    lam_p = np.sqrt(-mat.delta_p**2+kx**2)
    lam_s = np.sqrt(-mat.delta_s**2+kx**2)
    ky_p = -1j*lam_p
    ky_s = -1j*lam_s


    for _w, _kx in enumerate(kx):

        ky = np.array([ky_p[_w], ky_s[_w]])
        alpha_p = -1j*mat.lambda_*mat.delta_p**2 - 2j*mat.mu*ky[0]**2
        alpha_s = 2j*mat.mu*ky[1]*_kx

        Phi[4*_w:4*(_w+1), 0+4*_w] = np.array([-2.*1j*mat.mu*ky[0]*_kx,  ky[0], alpha_p, _kx]).T
        Phi[4*_w:4*(_w+1), 2+4*_w] = np.array([ 2.*1j*mat.mu*ky[0]*_kx, -ky[0], alpha_p, _kx]).T
        Phi[4*_w:4*(_w+1), 1+4*_w] = np.array([1j*mat.mu*(ky[1]**2-_kx**2), _kx,-alpha_s, -ky[1]]).T
        Phi[4*_w:4*(_w+1), 3+4*_w] = np.array([1j*mat.mu*(ky[1]**2-_kx**2), _kx, alpha_s, ky[1]]).T

        lam[0+4*_w:2+4*_w] =  -1j*ky
        lam[2+4*_w:4+4*_w] =  1j*ky

    return Phi, lam

def elastic_waves_PQ(mat, kx, omega):
    ''' 
    S={0: v_x, 1: v_y, 2: sigma_{xy}, 3: sigma_{yy}}
    '''

    kx = kx[0]
    jom = 1j*omega

    k_p = np.sqrt(mat.delta_p**2-kx**2)
    k_s = np.sqrt(mat.delta_s**2-kx**2)
    lam = np.array([-1j*k_p, -1j*k_s, 1j*k_p, 1j*k_s], dtype=complex)

    x_p = -2j*mat.mu*k_p*kx
    y_p = -1j*mat.lambda_*mat.delta_p**2 - 2j*mat.mu*k_p**2
    x_s = 1j*mat.mu*(k_s**2 - kx**2)
    y_s = -2j*mat.mu*k_s*kx

    P = np.zeros((4, 4), dtype=complex)
    Q = np.zeros((4, 4), dtype=complex)

    P[0,:] = [jom*kx, -jom*k_s, jom*kx, jom*k_s]
    P[1,:] = [jom*k_p, jom*kx, -jom*k_p, jom*kx]
    P[2,:] = [x_p, x_s, -x_p, x_s]
    P[3,:] = [y_p, y_s, y_p, -y_s]

    Q[0,:] = [-y_s/jom/k_s, x_s/jom/k_p, -kx/k_p, -1]
    Q[1,:] = [y_p/jom/k_s, -x_p/jom/k_p, 1, -kx/k_s]
    Q[2,:] = [-y_s/jom/k_s, -x_s/jom/k_p, +kx/k_p, -1]
    Q[3,:] = [-y_p/jom/k_s, -x_p/jom/k_p, 1, kx/k_s]

    Q = Q/(2j*omega**2*mat.rho)

    return P, Q, lam

def fluid_waves_TMM(mat, kx):
    """
    Polarisation  S={0:u_y , 1:p} and jky propagation terms

    Parameters
    ----------
    mat : mediapack medium 
    kx : numpy array of the transversal wave numbers. Its length corresponds to the number n_w of Bloch waves for the periodic medium (equal to 1 in the case of an infinite extend layers)

    Returns
    -------
    Phi : numpy matrix of dimension (2*n_w x 2*n_w) with the polarisation (block-matrix)
    lam : numpy vectors of length 2*n_w with the 1j*ky/ For each pair the first one is going in the positive y direction

    """
    if mat.MEDIUM_TYPE == 'eqf':
        K = mat.K_eq_til
        # ky = np.sqrt(mat.k**2-kx**2+0j)
        lamda = np.sqrt(-mat.k**2+kx**2+0j)
        ky = -1j*lamda
    elif mat.MEDIUM_TYPE == 'fluid':
        K = mat.K
        # ky = np.sqrt(mat.k**2-kx**2+0j)
        lamda = np.sqrt(-mat.k**2+kx**2+0j)
        ky = -1j*lamda
    else:
        raise ValueError('Provided material is not a fluid')



    if isinstance(kx, np.ndarray):
        n_w = len(kx)
        Phi = np.zeros((2*n_w, 2*n_w), dtype=complex)
        lam = np.zeros(2*n_w, dtype=complex)
        for _w, _ky in enumerate(ky):
            Phi[0+2*_w, 0+2*_w:2+2*_w] = np.array([-1j*_ky/(K*mat.k**2), 1j*_ky/(K*mat.k**2)])
            Phi[1+2*_w, 0+2*_w:2+2*_w] = np.array([1, 1])
    else:
        n_w = 1
        Phi = np.zeros((2*n_w, 2*n_w), dtype=complex)
        lam = np.zeros(2*n_w, dtype=complex)
        Phi[0, 0:2] = np.array([-1j*ky/(K*mat.k**2), 1j*ky/(K*mat.k**2)])
        Phi[1, 0:2] = np.array([1, 1])
    
    lam[::2] = -1j*ky
    lam[1::2] = 1j*ky
    return Phi, lam

def fluid_waves_PQ(mat, kx, omega):
    """
    Polarisation  S={0:v_y , 1:p} and jky propagation terms

    Parameters
    ----------
    mat : mediapack medium 
    kx : numpy array of the transversal wave numbers. Its length corresponds to the number n_w of Bloch waves for the periodic medium (equal to 1 in the case of an infinite extend layers)

    Returns
    -------
    Phi : numpy matrix of dimension (2*n_w x 2*n_w) with the polarisation (block-matrix)
    lam : numpy vectors of length 2*n_w with the 1j*ky/ For each pair the first one is going in the positive y direction

    """

    kx =kx[0]
    if mat.MEDIUM_TYPE == 'eqf':
        Z = np.sqrt(mat.K_eq_til*mat.rho_eq_til)
        lamda = np.sqrt(-mat.k**2+kx**2+0j)
        ky = -1j*lamda
    elif mat.MEDIUM_TYPE == 'fluid':
        Z = np.sqrt(mat.K*mat.rho)
        # ky = np.sqrt(mat.k**2-kx**2+0j)
        lamda = np.sqrt(-mat.k**2+kx**2+0j)
        ky = -1j*lamda
    else:
        raise ValueError('Provided material is not a fluid')

    Z_ = Z*mat.k/ky
    P = np.array([[1, 1], [Z_, -Z_]], dtype=complex)

    Q = np.array([[1/2, 1/(2*Z_)], [1/2, -1/(2*Z_)]], dtype=complex)

    lam = np.array([-1j*ky, 1j*ky], dtype=complex)
    return P, Q, lam