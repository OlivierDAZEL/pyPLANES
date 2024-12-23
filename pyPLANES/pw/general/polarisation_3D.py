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

def PEM_waves_3D(mat, kx, kz):
    ''' S={0: u_x^s, 1:u_z^s, 2:u_y^s, 3:u_y_t, 4: hat{sigma}_{yy}, 5: hat{sigma}_{yz}, 6: hat{sigma}_{xy}, 7: p} 
        following Parra Martinez et al. J. Appl. Phys. 119, 084907 (2016)
        q = {q_1^+, q_1^-, q_2^+, q_2^-, q_3^+, q_3^-, q_4^+, q_4^-}
    '''
    

    
    n_w = len(kx)
    Phi = np.zeros((8*n_w, 8*n_w), dtype=complex)
    lam = np.zeros(8*n_w, dtype=complex)
    
    lam_1 = np.sqrt(-mat.delta_1**2+kx**2+kz**2)
    lam_2 = np.sqrt(-mat.delta_2**2+kx**2+kz**2)
    lam_3 = np.sqrt(-mat.delta_3**2+kx**2+kz**2)

    ky_1 = -1j*lam_1
    ky_2 = -1j*lam_2
    ky_3 = -1j*lam_3

    for _w, _kx in enumerate(kx):
        ky = np.array([ky_1[_w], ky_2[_w], ky_3[_w], ky_3[_w]], dtype=complex)

        delta = np.array([mat.delta_1, mat.delta_2, mat.delta_3, mat.delta_3], dtype=complex)
        _ = slice(0+8*_w,8+8*_w)
        Phi[_, 0+8*_w]=np.array([_kx,kz, ky[0], mat.mu_1*ky[0],-1j*(mat.A_hat*mat.delta_1**2+2*mat.N*ky[0]**2),-2*1j*mat.N*ky[0]*kz,-2*1j*mat.N*ky[0]*_kx,1j*mat.delta_1**2*mat.K_eq_til*mat.mu_1])
        Phi[_, 4+8*_w]=np.array([_kx,kz,-ky[0],-mat.mu_1*ky[0],-1j*(mat.A_hat*mat.delta_1**2+2*mat.N*ky[0]**2), 2*1j*mat.N*ky[0]*kz, 2*1j*mat.N*ky[0]*_kx,1j*mat.delta_1**2*mat.K_eq_til*mat.mu_1])
        Phi[_, 1+8*_w]=np.array([_kx,kz, ky[1], mat.mu_2*ky[1],-1j*(mat.A_hat*mat.delta_2**2+2*mat.N*ky[1]**2),-2*1j*mat.N*ky[1]*kz,-2*1j*mat.N*ky[1]*_kx,1j*mat.delta_2**2*mat.K_eq_til*mat.mu_2])
        Phi[_, 5+8*_w]=np.array([_kx,kz,-ky[1],-mat.mu_2*ky[1],-1j*(mat.A_hat*mat.delta_2**2+2*mat.N*ky[1]**2), 2*1j*mat.N*ky[1]*kz, 2*1j*mat.N*ky[1]*_kx,1j*mat.delta_2**2*mat.K_eq_til*mat.mu_2])
        Phi[_, 2+8*_w]=np.array([ky[2],0,-_kx,-mat.mu_3*_kx, 2*1j*mat.N*ky[2]*_kx, 1j*mat.N*_kx*kz,          -1j*mat.N*(ky[2]**2-_kx**2),0]) 
        Phi[_, 6+8*_w]=np.array([ky[2],0, _kx, mat.mu_3*_kx, 2*1j*mat.N*ky[2]*_kx,-1j*mat.N*_kx*kz,           1j*mat.N*(ky[2]**2-_kx**2),0]) 
        Phi[_, 3+8*_w]=np.array([0,ky[2],- kz,-mat.mu_3* kz, 2*1j*mat.N*ky[2]* kz,-1j*mat.N*(ky[2]**2-kz**2), 1j*mat.N*kz*_kx           ,0])
        Phi[_, 7+8*_w]=np.array([0,ky[2],  kz, mat.mu_3* kz, 2*1j*mat.N*ky[2]* kz, 1j*mat.N*(ky[2]**2-kz**2),-1j*mat.N*kz*_kx           ,0])

        lam[0+8*_w:4+8*_w] = -1j*ky
        lam[4+8*_w:8+8*_w] =  1j*ky

    return Phi, lam

def elastic_waves_3D(mat, kx, kz):
    ''' S={0: u_x, 1:u_z, 2:u_y, 3: sigma_{yy}, 4: sigma_{yz}, 5: hat{sigma}_{xy}} 
        extending Parra Martinez et al. J. Appl. Phys. 119, 084907 (2016)
        q={q_1^+, q_1^-, q_2^+, q_2^-, q_3^+, q_3^-}
    '''

    n_w = len(kx)
    Phi = np.zeros((6*n_w,6*n_w), dtype=complex)
    lam = np.zeros(6*n_w, dtype=complex)


    lam_p = np.sqrt(-mat.delta_p**2+kx**2+kz**2)
    lam_s = np.sqrt(-mat.delta_s**2+kx**2+kz**2)
    ky_p = -1j*lam_p
    ky_s = -1j*lam_s
    for _w, _kx in enumerate(kx):

        ky = np.array([ky_p[_w], ky_s[_w], ky_s[_w]])
        
        _ = slice(0+6*_w,6+6*_w)
        Phi[_, 0+6*_w] = np.array([_kx, kz, ky[0], -1j*(mat.lambda_*mat.delta_p**2+2*mat.mu*ky[0]**2),-2*1j*mat.mu*ky[0]*kz,-2*1j*mat.mu*ky[0]*_kx])
        Phi[_, 3+6*_w] = np.array([_kx, kz, -ky[0], -1j*(mat.lambda_*mat.delta_p**2+2*mat.mu*ky[0]**2), 2*1j*mat.mu*ky[0]*kz, 2*1j*mat.mu*ky[0]*_kx])
         
        Phi[_, 1+6*_w]=np.array([ky[1],0,-_kx, 2*1j*mat.mu*ky[1]*_kx, 1j*mat.mu*_kx*kz,          -1j*mat.mu*(ky[1]**2-_kx**2)]) 
        Phi[_, 4+6*_w]=np.array([ky[1],0, _kx, 2*1j*mat.mu*ky[1]*_kx,-1j*mat.mu*_kx*kz,           1j*mat.mu*(ky[1]**2-_kx**2)]) 
        Phi[_, 2+6*_w]=np.array([0,ky[1],- kz, 2*1j*mat.mu*ky[1]* kz,-1j*mat.mu*(ky[1]**2-kz**2), 1j*mat.mu*kz*_kx           ])
        Phi[_, 5+6*_w]=np.array([0,ky[1],  kz, 2*1j*mat.mu*ky[1]* kz, 1j*mat.mu*(ky[1]**2-kz**2), -1j*mat.mu*kz*_kx           ])





        lam[0+6*_w:3+6*_w] =  -1j*ky
        lam[3+6*_w:6+6*_w] =  1j*ky

    return Phi, lam

def fluid_waves_3D(mat, kx, kz):
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
        lamda = np.sqrt(-mat.k**2+kx**2+kz**2+0j)
        ky = -1j*lamda
    elif mat.MEDIUM_TYPE == 'fluid':
        K = mat.K
        # ky = np.sqrt(mat.k**2-kx**2+0j)
        lamda = np.sqrt(-mat.k**2+kx**2+kz**2+0j)
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