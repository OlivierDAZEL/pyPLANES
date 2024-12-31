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

    Phi = np.zeros((8, 8), dtype=complex)
    lam = np.zeros(8, dtype=complex)
    
    lam_1 = np.sqrt(-mat.delta_1**2+kx**2+kz**2)
    lam_2 = np.sqrt(-mat.delta_2**2+kx**2+kz**2)
    lam_3 = np.sqrt(-mat.delta_3**2+kx**2+kz**2)

    ky_1 = -1j*lam_1
    ky_2 = -1j*lam_2
    ky_3 = -1j*lam_3


    ky = np.array([ky_1, ky_2, ky_3, ky_3], dtype=complex)
    delta = np.array([mat.delta_1, mat.delta_2, mat.delta_3, mat.delta_3], dtype=complex)
    _ = slice(0,8)
    Phi[_, 0]=np.array([kx,kz, ky[0], mat.mu_1*ky[0],-1j*(mat.A_hat*mat.delta_1**2+2*mat.N*ky[0]**2),-2*1j*mat.N*ky[0]*kz,-2*1j*mat.N*ky[0]*kx,1j*mat.delta_1**2*mat.K_eq_til*mat.mu_1])
    Phi[_, 4]=np.array([kx,kz,-ky[0],-mat.mu_1*ky[0],-1j*(mat.A_hat*mat.delta_1**2+2*mat.N*ky[0]**2), 2*1j*mat.N*ky[0]*kz, 2*1j*mat.N*ky[0]*kx,1j*mat.delta_1**2*mat.K_eq_til*mat.mu_1])
    Phi[_, 1]=np.array([kx,kz, ky[1], mat.mu_2*ky[1],-1j*(mat.A_hat*mat.delta_2**2+2*mat.N*ky[1]**2),-2*1j*mat.N*ky[1]*kz,-2*1j*mat.N*ky[1]*kx,1j*mat.delta_2**2*mat.K_eq_til*mat.mu_2])
    Phi[_, 5]=np.array([kx,kz,-ky[1],-mat.mu_2*ky[1],-1j*(mat.A_hat*mat.delta_2**2+2*mat.N*ky[1]**2), 2*1j*mat.N*ky[1]*kz, 2*1j*mat.N*ky[1]*kx,1j*mat.delta_2**2*mat.K_eq_til*mat.mu_2])
    Phi[_, 2]=np.array([ky[2],0,-kx,-mat.mu_3*kx, 2*1j*mat.N*ky[2]*kx, 1j*mat.N*kx*kz,           -1j*mat.N*(ky[2]**2-kx**2),0]) 
    Phi[_, 6]=np.array([ky[2],0, kx, mat.mu_3*kx, 2*1j*mat.N*ky[2]*kx,-1j*mat.N*kx*kz,            1j*mat.N*(ky[2]**2-kx**2),0]) 
    Phi[_, 3]=np.array([0,ky[2],-kz,-mat.mu_3*kz, 2*1j*mat.N*ky[2]*kz,-1j*mat.N*(ky[2]**2-kz**2), 1j*mat.N*kz*kx           ,0])
    Phi[_, 7]=np.array([0,ky[2], kz, mat.mu_3*kz, 2*1j*mat.N*ky[2]*kz, 1j*mat.N*(ky[2]**2-kz**2),-1j*mat.N*kz*kx           ,0])
    lam[0:4] = -1j*ky
    lam[4:8] =  1j*ky
    return Phi, lam

def elastic_waves_3D(mat, kx, kz):
    ''' S={0: u_x, 1:u_z, 2:u_y, 3: sigma_{yy}, 4: sigma_{yz}, 5: hat{sigma}_{xy}} 
        extending Parra Martinez et al. J. Appl. Phys. 119, 084907 (2016)
        q={q_1^+, q_1^-, q_2^+, q_2^-, q_3^+, q_3^-}
    '''


    Phi = np.zeros((6,6), dtype=complex)
    lam = np.zeros(6, dtype=complex)

    lam_p = np.sqrt(-mat.delta_p**2+kx**2+kz**2)
    lam_s = np.sqrt(-mat.delta_s**2+kx**2+kz**2)
    ky_p = -1j*lam_p
    ky_s = -1j*lam_s
    

    ky = np.array([ky_p, ky_s, ky_s])
    _ = slice(0,6+6)
    Phi[_, 0] = np.array([kx, kz, ky[0], -1j*(mat.lambda_*mat.delta_p**2+2*mat.mu*ky[0]**2),-2*1j*mat.mu*ky[0]*kz,-2*1j*mat.mu*ky[0]*kx])
    Phi[_, 3] = np.array([kx, kz, -ky[0], -1j*(mat.lambda_*mat.delta_p**2+2*mat.mu*ky[0]**2), 2*1j*mat.mu*ky[0]*kz, 2*1j*mat.mu*ky[0]*kx])
    Phi[_, 1] = np.array([ky[1],0,-kx, 2*1j*mat.mu*ky[1]*kx, 1j*mat.mu*kx*kz,          -1j*mat.mu*(ky[1]**2-kx**2)]) 
    Phi[_, 4] = np.array([ky[1],0, kx, 2*1j*mat.mu*ky[1]*kx,-1j*mat.mu*kx*kz,           1j*mat.mu*(ky[1]**2-kx**2)]) 
    Phi[_, 2] = np.array([0,ky[1],-kz, 2*1j*mat.mu*ky[1]*kz,-1j*mat.mu*(ky[1]**2-kz**2), 1j*mat.mu*kz*kx           ])
    Phi[_, 5] = np.array([0,ky[1], kz, 2*1j*mat.mu*ky[1]*kz, 1j*mat.mu*(ky[1]**2-kz**2),-1j*mat.mu*kz*kx           ])
    lam[0:3] =  -1j*ky
    lam[3:6] =  1j*ky

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
    elif mat.MEDIUM_TYPE == 'fluid':
        K = mat.K
        # ky = np.sqrt(mat.k**2-kx**2+0j)
        
    else:
        raise ValueError('Provided material is not a fluid')
    lamda = np.sqrt(-mat.k**2+kx**2+kz**2+0j)
    ky = -1j*lamda
    Phi = np.zeros((2, 2), dtype=complex)
    lam = np.zeros(2, dtype=complex)
    Phi[0, 0:2] = np.array([-1j*ky/(K*mat.k**2), 1j*ky/(K*mat.k**2)])
    Phi[1, 0:2] = np.array([1, 1])

    lam[::2] = -1j*ky
    lam[1::2] = 1j*ky
    return Phi, lam