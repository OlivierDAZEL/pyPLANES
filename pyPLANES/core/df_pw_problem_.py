#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# pw_classes.py
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
import json

from numpy import pi
import matplotlib.pyplot as plt
from mediapack import Air, Fluid
from alive_progress import alive_bar
from scipy import integrate

from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.utils.io import reference_frequencies, reference_curve, reference_C, reference_C_tr
from scipy.interpolate import lagrange
from pyPLANES.utils.quadrature_1d import Subdivision


class DfPwProblem(PwProblem):
    """
        Plane Wave Problem Class
    """
    def __init__(self, **kwargs):
        PwProblem.__init__(self, **kwargs)
        self.neval = []

    def resolution_kernel(self):
        """  Resolution of the problem """
        if self.alive_bar:
            with alive_bar(len(self.frequencies), title="pyPLANES Resolution") as bar:
                mlkkklmk
        else:
            tau = np.zeros(len(self.frequencies))
            D = 0.5 # Denominator in the TL 
            tol = 1e-3
            for i, f in enumerate(self.frequencies):

                self.f = f
                def func(theta):
                    self.theta_d = theta*180/pi
                    self.solve()
                    return np.sin(theta)*np.cos(theta)*self.result.tau[-1]

                theta_list = np.linspace(0,pi/2-0.01,200)
                f_list = 0*theta_list
                for i,theta in enumerate(theta_list):
                    f_list[i] = func(theta)
                plt.figure()
                plt.plot(theta_list, 1e3*f_list,"m",label= r"f($\theta)$")
                plt.xlabel(f"Incident ange (rad)")
                plt.ylabel("Transmission coeffficient (x$10^3$)")



                Tau, abserror, infodict,  = integrate.quad(func, 0, pi/2, full_output=1)
                # Tau, abserror, infodict,  = integrate.quad(func, 0, pi/2, full_output=1, epsabs=tol)
                print(f"Tau  scipy={Tau}")
                print("neval scipy =",infodict["neval"])

                Tau = self.diffuse_field_OD(func, tol)

                tau[i] = Tau
                self.result.R0 = []
                self.result.T0 = []
                self.result.Z_prime = []
                self.result.R = []
                self.result.T = []
                self.result.abs =[]
            self.result.tau = tau

    def resolution(self):
        """  Resolution of the problem """
        self.resolution_kernel()
        self.result.tau = list(self.result.tau)
        self.result.save(self.file_names, self.save_append)

    def compute_indicators(self):
        self.frequencies = reference_frequencies
        self.resolution_kernel()
        R = -10*np.log10(self.result.tau)
        ref = reference_curve
        diff = ref - R
        negative_difference = -np.sum(diff[diff<0])
        while negative_difference<32:
            ref -= 1
            diff = reference_curve - R
            negative_difference = -np.sum(diff[diff<0])
        R_w = ref[7] # Value of the reference curve at 500 Hz
        C = np.round(-10*np.log10(np.sum(np.power(10,(reference_C-R)/10))) - R_w)
        C_tr = np.round(-10*np.log10(np.sum(np.power(10,(reference_C_tr-R)/10))) - R_w)


        d = dict()
        d["R_w"] = R_w
        d["C"] = C
        d["C_tr"] = C_tr
        # Read the previous jsoln file
        with open(self.file_names+".json") as fp:
            d0 = json.load(fp)
        # Merge the two dictionaries
        d0.update(d)
        # Overwrite the previous json file
        with open(self.file_names+".json", "w") as json_file:
            json.dump(d0, json_file)
            json_file.write("\n")
        
    def diffuse_field_scipy(self, func):
        Tau, abserror, infodict,  = integrate.quad(func, 0, pi/2, full_output=1)
        self.neval.append(infodict["neval"])
        return Tau/0.5

    def diffuse_field_OD(self, func, tol):
        verbose = True
        if verbose: 
            print("diffuse_field_OD")
            # print("initialization with GK7-15 by default")
        subdivision = Subdivision()

        nb_it = 0
        print(subdivision)
        subdivision.update(func, nb_it)
        # print(subdivisions)

        print(f"I_c={subdivision.I_c}")
        print(f"I_r={subdivision.I_r}")
        print(f"error={subdivision.error}")
        print(f"error={subdivision.error_list}")

        quad_int = subdivision.interval_list[0]
        

        theta_list = np.linspace(0,pi/2-0.01,200)
        
        plt.plot(theta_list, 1e3*quad_int.poly_r(theta_list),"r--",label="polynomial, fine")
        plt.plot(theta_list, 1e3*quad_int.poly_c(theta_list),"b--",label="polynomial, coarse")
        plt.plot(quad_int.x_r, 1e3*quad_int.f_r,"r.")
        plt.plot(quad_int.x_c, 1e3*quad_int.f_c,"b.")
        
        plt.legend()

        plt.savefig("IN24_func.pdf")
        
        
        
        plt.figure()
        plt.bar(np.arange(1,7),100*quad_int.error_list)
        plt.xlabel("Interval")
        plt.ylabel("Percentage of error")
        plt.savefig("IN24_error.pdf")

        
        
        plt.show()
        kljhkjhk



        while np.sum(subdivision.error) > tol:
            nb_it +=1
            print(f"it #{nb_it}")
            subdivision.refine(nb_it)
            subdivision.update(func,nb_it)
            # print(f"I_c={subdivision.I_c}")
            # print(f"I_r={subdivision.I_r}")
            print(f"error={subdivision.error}")
            print(f"error={subdivision.error_list}")
            if nb_it == 5:
                plt.plot(subdivision.error_list)
                plt.show()                
        Tau = subdivision.I_r
        return Tau/0.5


        
        
        
# def func_c(s):
#     R = np.pi/4
#     theta = R+R*(np.cos(s)+1j*np.sin(s))
#     dtheta = R*(-np.sin(s)+1j*np.cos(s))
#     # print(f"s={s}")
#     # print(f"theta={theta}")
#     self.theta_d = theta*180/pi
#     self.solve()
#     self.result.tau[-1] =1
#     return np.sin(theta)*np.cos(theta)*self.result.tau[-1]*dtheta

# def func_r(s):
#     return np.real(func_c(s))

# def func_i(s):
#     return np.imag(func_c(s))

# lkjlkjlkjlkjlkjlkjlkjlkjlk
# plt.figure()
# theta_list = (np.pi/180)*np.linspace(1,89,200)
# tau_list = np.array([func(theta) for theta in theta_list])
# plt.plot(theta_list,tau_list,"m")

# # plt.figure()
# # s = np.linspace(0, np.pi , 200)
# # f_r = [func_r(ss) for ss in s]
# # f_i = [func_i(ss) for ss in s]
# # plt.plot(s, f_r,"b")
# # plt.plot(s, f_i,"m")

# plt.show()

# exit()


# Tau_r, abserror_r, infodict_r,  = integrate.quad(func_r, 0, pi, full_output=1)
# # neval.append(infodict["neval"])

# Tau_i, abserror_i, infodict_i,  = integrate.quad(func_i, 0, pi, full_output=1)
# # neval.append(infodict["neval"])

# print(f"scipy {Tau}")
# print(f"gauss {Tau_gauss}")
# print(f"kronr {Tau_kronrod}")
# print(neval)
# print(f"sci_r {Tau_r}")
# print(f"sci_i {Tau_i}")
# plt.show()                    

# Tau /= D