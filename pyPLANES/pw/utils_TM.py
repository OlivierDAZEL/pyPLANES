#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_fem.py
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
from scipy.linalg import expm
from numpy import sqrt
from pyPLANES.pw.transfert_matrices import TM_elastic, TM_fluid, TM_pem

from mediapack import Air
Air = Air()



def convert_Omega(Om_m, typ_minus, typ_plus):
    # fluid {0:u_y , 1:p}
    # elastic {0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
    # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
    # We have S = Omega @ [R,x,y]

    if typ_minus == typ_plus:
        Om_plus = Om_m
    else:
        if typ_minus == "pem":
            if typ_plus == "fluid":
                Om_plus = np.zeros((2, 1), dtype=complex)
                # The two in vacuo stresses are zero
                M = np.array([[Om_m[0, 1], Om_m[0, 2]], [Om_m[3, 1], Om_m[3, 2]]])
                F = np.array([Om_m[0, 0], Om_m[3, 0]])
                X = LA.solve(M, F)
                # print(Om_m[0,0]-x[0]*Om_m[0, 1]-x[1]*Om_m[0, 2])
                Om_plus[0, 0] = Om_m[2, 0]-x[0]*Om_m[2, 1]-x[1]*Om_m[2, 2]
                Om_plus[1, 0] = Om_m[4, 0]-x[0]*Om_m[4, 1]-x[1]*Om_m[4, 2]
            elif typ_plus == "elastic":
                Om_plus = np.zeros((4,2), dtype=complex)
                # We have to impose that u_y^s = u_y^t
                c_R = Om_m[2, 0] - Om_m[1, 0]
                c_1 = Om_m[2, 1] - Om_m[1, 1]
                c_2 = Om_m[2, 2] - Om_m[1, 2]
                if abs(c_2) == 0:
                    if abs(c_1) == 0:
                        raise ValueError("c_1 and c_2 are both zero")
                    else:
                        print("swapping c_1 and c_2")
                        c_1, c_2 = c_2, c_1
                        Om_m[:, [1, 2]]= Om_m[:, [2, 1]]
                Om_plus[0, 0] = Om_m[0, 0] - (c_R/c_2)*Om_m[0, 2]
                Om_plus[0, 1] = Om_m[0, 1] - (c_1/c_2)*Om_m[0, 2]
                Om_plus[1, 0] = Om_m[1, 0] - (c_R/c_2)*Om_m[1, 2]
                Om_plus[1, 1] = Om_m[1, 1] - (c_1/c_2)*Om_m[1, 2]
                Om_plus[2, 0] = (Om_m[3, 0]-Om_m[4,0]) - (c_R/c_2)*(Om_m[3, 2]-Om_m[4, 2])
                Om_plus[2, 1] = (Om_m[3, 1]-Om_m[4,1]) - (c_1/c_2)*(Om_m[3, 2]-Om_m[4, 2])
                Om_plus[3, 0] = Om_m[5, 0] - (c_R/c_2)*Om_m[5, 2]
                Om_plus[3, 1] = Om_m[5, 1] - (c_1/c_2)*Om_m[5, 2]
            elif typ_plus in ("Biot98", "Biot01"):
                Om_plus = Om_m
        elif typ_minus == "elastic":
            if typ_plus in ("pem", "Biot98", "Biot01"):
                Om_plus = np.zeros((6, 3), dtype=complex)
                Om_plus[0, :] = [Om_m[0, 0], Om_m[0, 1], 0]
                Om_plus[1, :] = [Om_m[1, 0], Om_m[1, 1], 0]
                Om_plus[2, :] = [Om_m[1, 0], Om_m[1, 1], 0]
                Om_plus[3, :] = [0 , 0, 1] # \hat{sigma}_yy is the new unknwon
                Om_plus[4, :] = [-Om_m[2, 0], -Om_m[2, 1], 1] # p = \hat{sigma}_yy-sigma^t_yy
                Om_plus[5, :] = [Om_m[3, 0], Om_m[3, 1], 0]
            elif typ_plus == "fluid":
                Om_plus = np.zeros((2, 1), dtype=complex)
                _ = Om_m[0, 0]/ Om_m[0, 1]
                Om_plus[0, 0] = Om_m[1, 0]-_*Om_m[1, 1]
                Om_plus[1, 0] = -(Om_m[2, 0]-_*Om_m[2, 1])
        elif typ_minus == "fluid":
            if typ_plus in ("pem", "Biot98", "Biot01"):
                Om_plus = np.zeros((6, 3), dtype=complex)
                Om_plus[1, :] = [0, 0, 1]
                Om_plus[2, :] = [Om_m[0, 0], 0, 0]
                Om_plus[4, :] = [Om_m[1, 0], 0, 0]
                Om_plus[5, :] = [0, 1, 0]
            elif typ_plus == "elastic":
                Om_plus = np.zeros((4,2), dtype=complex)
                Om_plus[1, :] = [Om_m[0, 0], 0]
                Om_plus[2, :] = [-Om_m[1, 0], 0] #sigma_yy = -p
                Om_plus[3, :] = [0, 1]
    return Om_plus


def weak_orth_terms(om, kx, Omega, layers, typ_end):
    # fluid {0:u_y , 1:p}
    # elastic {0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
    # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
    typ = "fluid"
    if layers:
        for _l in layers:
            # print("converting {} to {}".format(typ, _l.medium.MEDIUM_TYPE))
            Omega = convert_Omega(Omega, typ, _l.medium.MEDIUM_TYPE)
            if _l.medium.MEDIUM_TYPE == "fluid":
                Omega = TM_fluid(_l, kx, om)@Omega
            elif _l.medium.MEDIUM_TYPE == "elastic":
                Omega = TM_elastic(_l, kx, om)@Omega
            elif _l.medium.MEDIUM_TYPE == "pem":
                Omega = TM_pem(_l, kx, om)@Omega
        # print("converting {} to {}".format(layers[-1].medium.MEDIUM_TYPE, typ_end))
        Omega = convert_Omega(Omega, layers[-1].medium.MEDIUM_TYPE, typ_end)
    else:
        Omega = convert_Omega(Omega, "fluid", typ_end)
    if typ_end == "fluid":
        # u_y^t
        weak = np.array([Omega[0, :]])
        # p
        orth = np.array([Omega[1, :]])
    elif typ_end == "elastic":
        # sigma_xy and sigma_yy = -p
        weak = np.array([Omega[0, :], Omega[2, :]])
        # u_x^s and u_y
        orth = np.array([Omega[3, :], Omega[1, :]])
    elif typ_end == "Biot98":
        # sigma_xy, sigma_yy and u_y^t
        weak = np.array([Omega[0, :], Omega[3, :], Omega[2, :]])
        # u_x, u_y and p
        orth = np.array([Omega[5, :], Omega[1, :], Omega[4, :]])
    elif typ_end == "Biot01":
        # sigma_xy^t and sigma_yy^t and u_t-u_s
        weak = np.array([Omega[0, :], Omega[3, :]-Omega[4, :], Omega[2, :]-Omega[1, :]])
        # u_x^s and u_y^s and p
        orth = np.array([Omega[5, :], Omega[1, :], Omega[4, :]])
    else:
        raise ValueError("Unknown typ")
    return weak, orth

def ZOD_terms(om, kx, ml):
    # fluid {0:u_y , 1:p}
    # elastic {0:\sigma_{xy}, 1: u_y, 2 \sigma_{yy}, 3 u_x}'''
    # pem S={0:\hat{\sigma}_{xy}, 1:u_y^s, 2:u_y^t, 3:\hat{\sigma}_{yy}, 4:p, 5:u_x^s}'''
    typ = "fluid"
    typ_end = "fluid"
    Omega = np.eye(2)
    ml.update_frequency(om/(2*np.pi), om/Air.c, kx)

    for i_l, _l in enumerate(ml.layers):
        Omega =  ml.interfaces[i_l].update_Omega(Omega)
        Omega =  _l.update_Omega(om, Omega)
    Omega =  ml.interfaces[-1].update_Omega(Omega) 

    index_w_plus = [0] 
    index_f_plus = [1] 
    index_w_minus = [0] 
    index_f_minus = [1]

    A = Omega[index_w_plus, :][:, index_w_minus]
    B = Omega[index_w_plus, :][:,index_f_minus]
    C = Omega[index_f_plus, :][:,index_w_minus]
    D = Omega[index_f_plus, :][:,index_f_minus]

    Ci = LA.inv(C)
    
    wmfm = Ci@D
    wmfp = -Ci
    wpfm = (B-A@Ci@D) # - is for outgoing normal
    wpfp = (A@Ci) # - is for outgoing normal

    return wmfm, wmfp, wpfm, wpfp