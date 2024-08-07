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
from .reference_scheme import ReferenceSchemes


class Integral(): 
    def __init__(self, f, a=0, b=pi/2, typ="CC"):

        self.f = f
        self.a = a 
        self.b = b
        self.typ = typ
        self.reference_schemes = ReferenceSchemes(typ).copy()
        self.intervals = [Interval(self.a, self.b, self.reference_schemes["6"])]
        self.nb_refinements = 0
        self.error_list = []
        self.I_c, self.I_r, self.error = None, None, None
        
    def __str__(self):
        s = f"Integral at iteration #{self.nb_refinements}"
        for i, sub in enumerate(self.intervals):
            s += "\n" + str(i) +"/" + sub.__str__() 
        return s

    def number_of_nodes(self):
        n = np.sum([len(quad_int.x_r) for quad_int in self.intervals])
        return n-len(self.intervals)+1
        
    def update(self):
        # Initialisation
        self.I_c, self.I_r, self.error, self.error_list = 0. , 0., 0, []
        for i_interval, quad_int in enumerate(self.intervals):
            quad_int.update(self.f)
            self.I_c += quad_int.I_c
            self.I_r += quad_int.I_r
            self.error_list.append(quad_int.error)
        self.error = np.sum(self.error_list)


    def adapt_intervals(self):
        index_max = np.argmax(np.abs(self.error_list))
        # for quad_int in self.intervals:
        #     print(quad_int)
        intervals = []
        for i, quad_int in enumerate(self.intervals):
            if i == index_max:
                intervals.extend(quad_int.adapt_intervals())
            else:
                intervals.append(quad_int)
        self.intervals = intervals
        # for quad_int in self.intervals:
        #     print(quad_int)
        self.update()

    def refine(self):
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
        fig = plt.figure(1000+self.nb_refinements)
        x = np.linspace(self.a, self.b, 100)
        f = [self.f(xi) for xi in x]
        plt.plot(x,f, "b")
        for quad_int in self.intervals:
            quad_int.plot_polynomials(fig)
            f_r = [self.f(xi) for xi in quad_int.x_r]
            f_c = [self.f(xi) for xi in quad_int.x_c]
            plt.stem(quad_int.x_r, f_r, "b")
            plt.stem(quad_int.x_c, f_c, "g")
        plt.title("Function and polynomial approximations")

    def plot_error_on_intervals(self):
        fig = plt.figure(2000+self.nb_refinements)
        ax = fig.add_subplot(111)
        ax.set_xlim(left=self.a, right=self.b)
        error = []
        for quad_int in self.intervals:
            quad_int.plot_error_on_intervals(ax)
            error.extend(quad_int.error_list)
        error  = np.max(np.abs(error))
        ax.set_ylim(bottom=0, top=error)
        plt.title("Repartition of the error function on intervals")
    
        # fig = plt.figure(4000+self.nb_refinements)
        # ax = fig.add_subplot(111)
        # ax.set_xlim(left=self.a, right=self.b)
        error = []
        for quad_int in self.intervals:
            # quad_int.plot_relative_error_on_intervals(ax)
            error.extend(quad_int.relative_error_list)
        # error  = np.max(np.abs(error))
        # ax.set_ylim(bottom=0, top=error)
        # plt.title("Relative error function on intervals")    

    def plot_error_function(self):
        fig = plt.figure(3000+self.nb_refinements)
        ax = fig.add_subplot(111)
        for quad_int in self.intervals:
            quad_int.plot_error_function(ax)
        plt.title("Error function   ")

    def plot_grid(self):
        plt.figure(10000)
        for i, interval in enumerate(self.intervals):
            interval.plot_grid(self.nb_refinements)
            
            
