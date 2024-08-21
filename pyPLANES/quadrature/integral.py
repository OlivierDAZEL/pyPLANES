#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# quadrature_1d.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@univ-lemans.fr>

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
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.interpolate import lagrange
from scipy import integrate
# from .gauss_kronrod import gauss_kronrod, import_gauss_kronrod, 
from .interval import Interval
from .reference_scheme import ReferenceSchemes, ReferenceScheme, CC_autoadaptive
from termcolor import colored



class Integral(): 
    def __init__(self, f, a=0, b=pi/2, **kwargs):
        self.f = f
        self.a = a 
        self.b = b
        self.typ = kwargs.get("typ", "quadLAUM")
        self.order = kwargs.get("order", None)
        self.adaptation = kwargs.get("adaptation", True)
        
        
        if self.typ in ["GK", "CC"]:
            if self.order is not None:
                self.initial_reference_scheme = ReferenceScheme(self.typ, order=self.order)
        else:    
         self.initial_reference_scheme = kwargs.get("initial_reference_scheme", ReferenceScheme("CC", order=5))
        # self.reference_schemes = ReferenceSchemes(self.typ)
        # self.reference_scheme_for_adaptation = self.reference_schemes["3"]
        boundaries = kwargs.get("boundaries", False)
        if boundaries is False:
            self.intervals = [Interval(self.a, self.b, self.initial_reference_scheme)]
        else:
            self.intervals = []
            for i, x in enumerate(boundaries[:-1]):
                self.intervals.append(Interval(x, boundaries[i+1], CC_autoadaptive()))
            for quad_int in self.intervals:
                quad_int.status = "adapted"
                
        self.nb_refinements, self.nb_adaptations = 0, 0
        self.error_list = []
        self.I_c, self.I_r, self.error, self.neval = None, None, None, 0
        
        self.epsrel = kwargs.get("epsrel", 1.49e-8)
        self.epsabs = kwargs.get("epsabs", 1.49e-8)
        self.epsadapt = kwargs.get("epsadapt", 10)

        
        self.verbose = kwargs.get("verbose", False)
        self.update()
        
    def test_convergence(self):
        print(np.abs((self.I_c-self.I_r)/self.I_c))
        return np.abs((self.I_c-self.I_r)/self.I_c) < self.tol_refine
        
        
        
    def __str__(self):
        s = f"Integral at iteration #{self.nb_refinements}"
        for i, sub in enumerate(self.intervals):
            s += "\n" + str(i) +"/" + sub.__str__() 
        return s

    def plot_integrand(self):
        theta =np.linspace(self.a, self.b, 1000)
        f =np.array([self.f(xi) for xi in theta])
        # ff = f /(np.sin(theta)**4*np.cos(theta))

        plt.figure()
        plt.plot(theta,f,"b.")
        plt.figure()
        plt.semilogy(theta,f,"b.")

        plt.show()
        # exit()

    def number_of_nodes(self):
        n = np.sum([len(quad_int.x_r) for quad_int in self.intervals])
        return n-len(self.intervals)+1
        
    def update(self):
        """
            update the values of I_r, I_c, and the error
        """
        # Initialisation
        self.I_c, self.I_r, self.error, self.error_list = 0. , 0., 0, []
        for i_interval, quad_int in enumerate(self.intervals):
            quad_int.update(self.f)
            self.neval += quad_int.neval
            self.I_c += quad_int.I_c
            self.I_r += quad_int.I_r
            self.error_list.append(quad_int.error)
        self.error = np.sum(self.error_list)
        
        

    def determine_intervals(self):
        # Determine the intervals based on previous error

        test_interval, test_tol = True, True
        self.boundaries = []
        self.plot_polynomials()
        self.plot_error_on_intervals()
        while test_interval and test_tol:
            self.nb_adaptations += 1
            if self.verbose:
                print(f"Adaptation #{self.nb_adaptations}")

            # print([quad_int.status for quad_int in self.intervals])
            intervals = []
            for i, quad_int in enumerate(self.intervals):
                if quad_int.status == "pending":
                    intervals.extend(quad_int.adapt())
                else:
                    intervals.append(quad_int)
            self.intervals = intervals
            self.update()
            self.plot_polynomials()
            self.plot_error_on_intervals()
            plt.show()
            exit()

            # plt.show()
            # exit()
            # print(colored(f"I_r={self.I_r:.10E}","green"))
            # print(colored(f"error={self.error:.10E}", "yellow"))
            # print(np.abs(self.error/self.I_r))
            # for quad_int in self.intervals:
            #     print(quad_int)
            
            test_interval = any ([quad_int.status == "pending" for quad_int in self.intervals])
            test_tol = np.abs(self.error/self.I_r) > self.tol_adapt
            
        # self.plot_polynomials()
        # self.plot_error_on_intervals()
        # plt.show()
        
        # for quad_int in self.intervals:
        #         print(quad_int)

        if test_tol == False and test_interval == True:
            new_intervals = []
            # print("adaptin")
            for quad_int in self.intervals:
                if quad_int.status in ["converged", "adapted"]:
                    new_intervals.append(quad_int)
                elif quad_int.status == "pending":
                    self.boundaries  = (quad_int.x_r[0], quad_int.x_r[-1])
                    for i in range(quad_int.reference_scheme.n_r-1):
                        new_int = Interval(quad_int.x_r[i], quad_int.x_r[i+1], CC_autoadaptive(), quad_int.f[i], quad_int.f[i+1],status="adapted")
                        new_intervals.append(new_int)
                        # print(new_int)
                        
            self.intervals = new_intervals
            self.update()
        # for quad_int in self.intervals:
        #     print(quad_int)
        # plt.show()

    def refine(self):

        test = True
        while test:
            # plt.figure(33)
            self.nb_refinements += 1
            index_max = np.argmax(np.abs(self.error_list))
            x_0, x_1 = self.intervals[index_max].x_r[0], self.intervals[index_max].x_r[-1]
            if self.verbose:
                print(f"I_c={self.I_c}")
                print(colored(f"I_r={self.I_r:.10E}","green"))
                print(colored(f"reler={np.abs(self.error/self.I_r):.10E}", "yellow"))
                self.plot_error_on_intervals()
                self.plot_polynomials()
                plt.show()
            self.intervals[index_max].refine_reference_scheme()
            self.update()
            test = np.abs(self.error/self.I_r) > self.tol_refine
            if self.nb_refinements > 15:
                test = False
            if self.intervals[index_max].reference_scheme.order > 16:
                test = False

    def refine_old(self):
        # Refine the interval with the largest error
        self.nb_refinements += 1
        error = 0.
        # Determine on which interval the error is the largest
        for i_interval, quad_int in enumerate(self.intervals):
            if quad_int.status == "converged":
                intervals.append(quad_int) # Nothing is done
            else: # Determine the interval with the largest error
                max_abs_error = np.max(np.abs(quad_int.error))
                if max_abs_error > error:
                    index_interval_max = i_interval
                    error = max_abs_error
        # Refine the interval with the largest error
        self.intervals[index_interval_max].refine()


    def plot_polynomials(self):
        fig = plt.figure(100000+100*self.nb_adaptations+self.nb_refinements)
        x = np.linspace(self.a, self.b, 200)
        # x = np.linspace(0.85, 1., 1000)
        f = [self.f(xi) for xi in x]
        plt.plot(x,f, "b.-")
        # plt.show()
        # exit()
        for quad_int in self.intervals:
            quad_int.plot_polynomials(fig)
            f_r = [self.f(xi) for xi in quad_int.x_r]
            f_c = [self.f(xi) for xi in quad_int.x_c]
            plt.stem(quad_int.x_r, f_r, "r")
            plt.stem(quad_int.x_c, f_c, "y")
        for quad_int in self.intervals:            
            if quad_int.status == "pending":
                f_r = [self.f(xi) for xi in quad_int.x_r]
                f_c = [self.f(xi) for xi in quad_int.x_c]
                plt.stem(quad_int.x_r, f_r, "m")
                plt.stem(quad_int.x_c, f_c, "g")

        plt.title("Function and polynomial approximations")

    def plot_error_on_intervals(self):
        fig = plt.figure(200000+100*self.nb_adaptations+self.nb_refinements)
        ax = fig.add_subplot(111)
        ax.set_xlim(left=self.a, right=self.b)
        error = []
        for quad_int in self.intervals:
            quad_int.plot_error_on_intervals(ax)
            error.extend(quad_int.error_list)
        # error  = np.max(np.abs(error))
        
        for inter in self.intervals:
            inter.plot_grid(1.0)
        
        ax.set_ylim(bottom=0, top=1.1)
        plt.title("Repartition of the error function on intervals")
    
        # fig = plt.figure(4000+self.nb_refinements)
        # ax = fig.add_subplot(111)
        # ax.set_xlim(left=self.a, right=self.b)
        # error = []
        # for quad_int in self.intervals:
        #     # quad_int.plot_relative_error_on_intervals(ax)
        #     error.extend(quad_int.relative_error_list)
        # error  = np.max(np.abs(error))
        # ax.set_ylim(bottom=0, top=error)
        # plt.title("Relative error function on intervals")    

    def plot_error_function(self):
        fig = plt.figure(300000+100*self.nb_adaptations+self.nb_refinements)
        ax = fig.add_subplot(111)
        for quad_int in self.intervals:
            quad_int.plot_error_function(ax)
        plt.title("Error function   ")

    def plot_grid(self):
        plt.figure(4)
        for i, interval in enumerate(self.intervals):
            interval.plot_grid(10*self.nb_adaptations+self.nb_refinements)
            
            
