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

def PEM_waves_TMM(mat,ky):
    ''' S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
    
    n_w = len(ky)
    Phi = np.zeros((6*n_w, 6*n_w), dtype=complex)
    lam = np.zeros(6*n_w, dtype=complex)
    kx_1 = np.sqrt(mat.delta_1**2-ky**2)
    kx_2 = np.sqrt(mat.delta_2**2-ky**2)
    kx_3 = np.sqrt(mat.delta_3**2-ky**2)

    for _w, _ky in enumerate(ky):
    
        kx = np.array([kx_1[_w], kx_2[_w], kx_3[_w]], dtype=complex)
        delta = np.array([mat.delta_1, mat.delta_2, mat.delta_3], dtype=complex)

        alpha_1 = -1j*mat.A_hat*mat.delta_1**2-1j*2*mat.N*kx[0]**2
        alpha_2 = -1j*mat.A_hat*mat.delta_2**2-1j*2*mat.N*kx[1]**2
        alpha_3 = -2*1j*mat.N*kx[2]*_ky

        
        Phi[0+6*_w:6+6*_w, 0+6*_w] = np.array([-2*1j*mat.N*kx[0]*_ky, kx[0], mat.mu_1*kx[0], alpha_1, 1j*delta[0]**2*mat.K_eq_til*mat.mu_1, _ky], dtype=complex)
        Phi[0+6*_w:6+6*_w, 3+6*_w] = np.array([ 2*1j*mat.N*kx[0]*_ky,-kx[0],-mat.mu_1*kx[0], alpha_1, 1j*delta[0]**2*mat.K_eq_til*mat.mu_1, _ky], dtype=complex)

        Phi[0+6*_w:6+6*_w, 1+6*_w] = np.array([-2*1j*mat.N*kx[1]*_ky, kx[1], mat.mu_2*kx[1],alpha_2, 1j*delta[1]**2*mat.K_eq_til*mat.mu_2, _ky], dtype=complex)
        Phi[0+6*_w:6+6*_w, 4+6*_w] = np.array([ 2*1j*mat.N*kx[1]*_ky,-kx[1],-mat.mu_2*kx[1],alpha_2, 1j*delta[1]**2*mat.K_eq_til*mat.mu_2, _ky], dtype=complex)

        Phi[0+6*_w:6+6*_w, 2+6*_w] = np.array([1j*mat.N*(kx[2]**2-_ky**2), _ky, mat.mu_3*_ky, alpha_3, 0., -kx[2]], dtype=complex)
        Phi[0+6*_w:6+6*_w, 5+6*_w] = np.array([1j*mat.N*(kx[2]**2-_ky**2), _ky, mat.mu_3*_ky, -alpha_3, 0., kx[2]], dtype=complex)
        
        lam[0+6*_w:3+6*_w] =  -1j*kx
        lam[3+6*_w:6+6*_w] =  1j*kx


    return Phi, lam#np.concatenate((-1j*kx, 1j*kx))

def elastic_waves_TMM(mat,ky):
    ''' S={0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''

    n_w = len(ky)
    Phi = np.zeros((4*n_w, 4*n_w), dtype=complex)
    lam = np.zeros(4*n_w, dtype=complex)

    kx_p = np.sqrt(mat.delta_p**2-ky**2)
    kx_s = np.sqrt(mat.delta_s**2-ky**2)

    for _w, _ky in enumerate(ky):
        kx = np.array([kx_p[_w], kx_s[_w]])

        alpha_p = -1j*mat.lambda_*mat.delta_p**2 - 2j*mat.mu*kx[0]**2
        alpha_s = 2j*mat.mu*kx[1]*_ky

        Phi[0+4*_w:4+4*_w, 0+4*_w] = np.array([-2.*1j*mat.mu*kx[0]*_ky,  kx[0], alpha_p, _ky])
        Phi[0+4*_w:4+4*_w, 2+4*_w] = np.array([ 2.*1j*mat.mu*kx[0]*_ky, -kx[0], alpha_p, _ky])
        Phi[0+4*_w:4+4*_w, 1+4*_w] = np.array([1j*mat.mu*(kx[1]**2-_ky**2), _ky,-alpha_s, -kx[1]])
        Phi[0+4*_w:4+4*_w, 3+4*_w] = np.array([1j*mat.mu*(kx[1]**2-_ky**2), _ky, alpha_s, kx[1]])

        lam[0+4*_w:2+4*_w] =  -1j*kx
        lam[2+4*_w:4+4*_w] =  1j*kx

    return Phi, np.concatenate((-1j*kx, 1j*kx))

def fluid_waves_TMM(mat, ky):
    ''' S={0:u_y , 1:p}'''
    if mat.MEDIUM_TYPE == 'eqf':
        K = mat.K_eq_til
    elif mat.MEDIUM_TYPE == 'fluid':
        K = mat.K
    else:
        raise ValueError('Provided material is not a fluid')
    n_w = len(ky)
    kx = np.sqrt(mat.k**2-ky**2+0j)
    Phi = np.zeros((2*n_w, 2*n_w), dtype=complex)
    lam = np.zeros(2*n_w, dtype=complex)
    for _w, _kx in enumerate(kx):
        Phi[0+2*_w, 0+2*_w:2+2*_w] = np.array([-1j*_kx/(K*mat.k**2), 1j*_kx/(K*mat.k**2)])
        Phi[1+2*_w, 0+2*_w:2+2*_w] = np.array([1, 1])
    
    lam[::2] = -1j*kx
    lam[1::2] = 1j*kx
    return Phi, lam

# def fluid_waves(mat, nx,ny):
#     pass