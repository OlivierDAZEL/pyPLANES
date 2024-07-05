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
from pyPLANES.utils.quadrature_1d import Integral


class DfPwProblem(PwProblem):
    """
        Plane Wave Problem Class
    """
    def __init__(self, **kwargs):
        PwProblem.__init__(self, **kwargs)
        self.neval = []



    def map_tau(self):
        """  Resolution of the problem """
        self.f = self.frequencies[0]
        print(self.f)
        self.update_frequency(2*np.pi*self.f)
        n_r, n_i = 80, 81
        theta_r = np.linspace(0, np.pi/2-0.00001, n_r)
        theta_i = 3*np.linspace(-np.pi/4, np.pi/4, n_i)
        R, I = np.meshgrid(theta_i, theta_r)
        TAU = np.zeros((n_r,n_i), dtype=complex)
        
        for i in range(n_r):
            for j in range(n_i):
                theta = (R[i,j]+I[i,j])
                self.theta_d = theta*180/pi
                self.update_frequency(2*np.pi*self.f)
                self.create_linear_system(2*np.pi*self.f)
                self.solve()
                TAU[i,j] = self.result.tau[-1]*np.sin(theta)*np.cos(theta)
                
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(R, I, TAU)
        # plt.figure()
        # plt.contourf(R,I,np.abs(TAU))
        # plt.colorbar()
        # plt.show()


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
                    self.update_frequency(2*np.pi*self.f)
                    self.create_linear_system(2*np.pi*self.f)
                    self.solve()
                    return np.sin(theta)*np.cos(theta)*self.result.tau[-1]
    
                # def func(theta):
                #     return theta
    

                Tau = self.diffuse_field_OD(func, tol)
                print(f"Tau     IN= {Tau}")

                # Tau_r,Tau_i = self.diffuse_field_cx(func, tol)
                # print(f"Tau     cx={complex(Tau_r,Tau_i)}")


                exit()
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
        
    def diffuse_field_OD(self, func, tol):
        verbose = True
        if verbose: 
            print("diffuse_field_OD")
            print("initialization with GK3-15 by default")
        integral = Integral(func)

        nb_it = 0
        # print(subdivision)
        integral.update(func, nb_it)
        print(integral)
        print(f"I_c={integral.I_c}")
        print(f"I_r={integral.I_r}")
        print(f"error={integral.error}")
        print(f"error={integral.interval_list[0].error_list}")

        # quad_int = integral.interval_list[0]
        # theta_list = np.linspace(0,pi/2-0.01,200)

        while np.sum(integral.error) > tol:
            nb_it +=1
            # print(f"it #{nb_it}")
            integral.refine(nb_it)
            integral.update(func,nb_it)
            # print(f"I_c={subdivision.I_c}")
            # print(f"I_r={subdivision.I_r}")
            # print(f"error={subdivision.error}")
            # print(f"error={subdivision.error_list}")
        Tau = integral.I_r
        return Tau


