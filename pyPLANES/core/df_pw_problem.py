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
from termcolor import colored
from numpy import pi
import matplotlib.pyplot as plt
from mediapack import Air, Fluid
from alive_progress import alive_bar
from scipy import integrate

from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.utils.io import reference_frequencies, reference_curve, reference_C, reference_C_tr
from scipy.interpolate import lagrange
from pyPLANES.quadrature.integral import Integral



class DfPwProblem(PwProblem):
    """
        Plane Wave Problem Class
    """
    def __init__(self, **kwargs):
        PwProblem.__init__(self, **kwargs)
        self.neval = 0# number of evaluation of the function
        self.DF_method = kwargs.get("DF_method", "scipy")
        self.verbose = kwargs.get("verbose", False)
        self.epsrel = kwargs.get("epsrel", 1.49e-8)
        self.epsabs = kwargs.get("epsabs", 1.49e-8)
        
        
    def __str__(self):
        return colored(f"I={Tau:.10E}","green") + " with " + colored(f"{infodict['neval']}", "red") + " evaluations"

        
    def update_frequency(self, omega):
        PwProblem.update_frequency(self, omega)
        self.f = omega/(2*np.pi)
        
        
    def resolution_kernel(self):
        """  Resolution of the problem """
        if self.alive_bar:
            with alive_bar(len(self.frequencies), title="pyPLANES Resolution") as bar:
                mlkkklmk
        else:
            tau = [None] * len(self.frequencies)
            neval = [None] * len(self.frequencies)
            final_error = [None] * len(self.frequencies)
            D = 0.5 # Denominator in the TL 
            for i, f in enumerate(self.frequencies):
                self.neval = 0
                self.f = f
                self.result.f.append(self.f)
                def func(theta):
                    self.theta_d = theta*180/pi
                    self.update_frequency(2*np.pi*self.f)
                    self.create_linear_system(2*np.pi*self.f)
                    self.solve()
                    return np.sin(theta)*np.cos(theta)*self.result.tau[-1]/D
    
                if self.DF_method == "scipy":
                    Tau, abserror, infodict = integrate.quad(func, 0, pi/2,full_output=1,epsrel=self.epsrel, epsabs=self.epsabs)
                    tau[i] = Tau
                    neval[i] = infodict['neval']
                    final_error[i] = abserror
                    if self.verbose: 
                        print(colored(f"I_scipy={Tau:.10E}","green") + " with " + colored(f"{infodict['neval']}", "red") + " evaluations")
                elif self.DF_method == "chebpy":
                    chebpy.chebfun(lambda theta: func(theta), [0, 10])
                    hjk
                
                elif self.DF_method == "quadLAUM":
                    # To be removed at the end                    
                    Tau, abserror, infodict = integrate.quad(func, 0, pi/2,full_output=1,epsrel=1e-8)
                    print(colored(f"I_ref  ={Tau:.10E}","green") + " with " + colored(f"{infodict['neval']}", "red") + " evaluations")

                    Tau, abserror, infodict = integrate.quad(func, 0, pi/2,full_output=1,epsrel=self.epsrel)
                    print(colored(f"I_scipy={Tau:.10E}","green") + " with " + colored(f"{infodict['neval']}", "red") + " evaluations")

                    integral = Integral(func, epsrel=self.epsrel, epsabs=self.epsabs)
                    integral.plot_error_on_intervals()
                    integral.plot_polynomials()
                    if not integral.test_convergence():
                        integral.step_1()

                        # print(colored(f"I      ={integral.I_r:.10E}","green") + " with " + colored(f"{integral.neval}", "red") + " evaluations")

                        plt.show()
                        exit()
                        integral.refine()

                    print(colored(f"I      ={integral.I_r:.10E}","green") + " with " + colored(f"{integral.neval}", "red") + " evaluations")
                    print(integral.I_c)
                    integral.plot_error_on_intervals()
                    integral.plot_polynomials()
                    plt.show()
                    exit()
                    tau[i], neval[i] = integral.I_r, integral.neval

                elif type(self.DF_method) is tuple:
                    integral = Integral(func, 0, np.pi/2,typ=self.DF_method[0],order=self.DF_method[1])
                    tau[i], neval[i], final_error[i] = abserror = integral.I_r, integral.neval, integral.I_c-integral.I_r
                else:
                    raise NotImplementedError(f"Method {self.DF_method} not implemented")

                        
                    



                self.result.R0 = []
                self.result.T0 = []
                self.result.Z_prime = []
                self.result.R = []
                self.result.T = []
                self.result.abs =[]
            self.result.tau = tau
            self.result.neval = neval
            self.result.final_error = final_error

    def resolution(self):
        """  Resolution of the problem """
        self.resolution_kernel()
        
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
        
    def map_tau(self):
        """  Resolution of the problem """
        self.f = self.frequencies[0]
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
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(R, I, TAU)
        plt.figure()
        plt.contourf(R,I,np.abs(TAU))
        plt.colorbar()
        plt.show()


    def integrand(self,f, theta_list, **kwargs):
        """  Resolution of the problem """
        self.f = f
        self.update_frequency(2*np.pi*self.f)
        
        rho_0 = self.layers[0].medium.rho
        c_0 = self.layers[0].medium.c
        rho = self.layers[1].medium.rho 
        E = np.real(self.layers[1].medium.E)
        nu = self.layers[1].medium.nu 
        h = self.layers[1].d


        D = E*h**3/(12*(1-nu**2))
        f_c = c_0**2*np.sqrt(rho*h/D)/(2*np.pi)
        s2 = c_0**2*np.sqrt(rho*h/D)/(2*np.pi*self.f)
        if s2<1:
            theta_c = np.arcsin(np.sqrt(s2))
        else:
            theta_c = None
            
        tau_list = []        
        for theta in theta_list:
                self.theta_d = theta*180/pi
                self.update_frequency(2*np.pi*self.f)
                self.create_linear_system(2*np.pi*self.f)
                self.solve()
                tau_list.append(self.result.tau[-1]*np.sin(theta)*np.cos(theta))
        return tau_list
        # if theta_c is not None:
        #     plt.plot([theta_c, theta_c], [0, 1.1*np.max(tau_list)],"r--")
        


