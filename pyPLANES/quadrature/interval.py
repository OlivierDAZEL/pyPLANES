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
from termcolor import colored
# from .gauss_kronrod import gauss_kronrod, import_gauss_kronrod, 
# from .quadrature_schemes import import_scheme
from matplotlib.patches import Rectangle 
from .reference_scheme import ReferenceScheme, CC_autoadaptive

class Interval():
    def __init__(self, func, a, b, reference_scheme=None, f_a=None, f_b=None,**kwargs):
        # print(f"Interval [{a}, {b}]")
        self.epsrel = 1e-1
        self.func = func 
        self.status = kwargs.get("status","pending") # Other status, adapted // converged
        self.RelTol = 1e-3
        self.a = self.A = a
        self.b = self.B = b
        self.f_a, self.f_b = f_a, f_b
        self.reference_scheme = reference_scheme
        if reference_scheme is None:
            raise NameError("Reference scheme is missing")
        
        self.reference_scheme_for_step_1 = ReferenceScheme("GK", order=10)
        self.reference_scheme_for_peak = ReferenceScheme("CC", order=9)

        self.typ = reference_scheme.typ
        self.neval = 0
        self.x_c = self.A+(self.B-self.A)*(self.reference_scheme.xi_c+1)/2.
        self.x_r = self.A+(self.B-self.A)*(self.reference_scheme.xi_r+1)/2.
        self.x_r_with_boundaries = self.A+(self.B-self.A)*(self.reference_scheme.xi_r_with_boundaries+1)/2.
        if self.reference_scheme.typ == "PV":
            self.f = self.reference_scheme.f
            self.f_with_boundaries = self.reference_scheme.f
        else:
            self.f = np.array([None]*len(self.x_r))
            self.f_with_boundaries = np.array([self.f_a] +  [None]*len(self.x_r) +[self.f_b])
        
        self.P_c = ((self.B-self.A)/2)*self.reference_scheme.P_c
        self.P_r = ((self.B-self.A)/2)*self.reference_scheme.P_r

    def __str__(self):
        s = f"[{self.a:.3E} ; {self.b:.3E}]" + colored (f"{self.status} ", "green") + self.reference_scheme.__str__() 
        
        return s

    def x_of(self, i):
        # return the coordinates of the ith subinterval
        return self.x_r[2*i:2*i+3]


    def update_f_values(self):
        for i,f in enumerate(self.f):
            if f is None:
                self.f[i] = self.func(self.x_r[i])
                self.neval += 1
        if self.typ in ["CC", "PV"]:
            self.f_with_boundaries = self.f
        elif self.typ == "GK":
            self.f_with_boundaries = np.hstack([self.f_a, self.f, self.f_b])
        
    def update(self):
        self.update_f_values()
        

        self.coeff_c = self.P_c@self.f
        self.coeff_r = self.P_r@self.f
        
        # Computation of the integrals
        self.I_c = self.coeff_c@(self.reference_scheme.xib_powers-self.reference_scheme.xia_powers)
        self.I_r = self.coeff_r@(self.reference_scheme.xib_powers-self.reference_scheme.xia_powers)
    
        if self.typ in ["CC", "GK", "PV"]:
            # values_list = np.array([self.coeff_r@self.reference_scheme.xi_powers[i+1]- self.coeff_r@self.reference_scheme.xi_powers[i] for i in range(self.reference_scheme.n_c-1)])
            if self.typ in ["CC", "PV"]:
                self.error_list = np.array([ (self.coeff_c@self.reference_scheme.xi_powers[i+1]-self.coeff_r@self.reference_scheme.xi_powers[i+1])-(self.coeff_c@self.reference_scheme.xi_powers[i]-self.coeff_r@self.reference_scheme.xi_powers[i])for i in range(self.reference_scheme.n_c-1)])
            else:
                self.error_list = np.array([ (self.coeff_c@self.reference_scheme.xi_powers[i+1]-self.coeff_r@self.reference_scheme.xi_powers[i+1])-(self.coeff_c@self.reference_scheme.xi_powers[i]-self.coeff_r@self.reference_scheme.xi_powers[i])for i in range(self.reference_scheme.n_c+1)])
            
            self.error = np.sum(self.error_list)
            abs_error = np.abs(self.error_list)
            self.repartition_error = abs_error/np.sum(abs_error)
        else:
            raise NameError("Unknown quadrature scheme")
        
        

        # if np.abs(self.error/self.I_c) < self.RelTol:
        #     self.is_converged = True

    def split(self,refined_schemes):
        index_max = np.argmax(np.abs(self.error_list))
        # print(f"Splitting interval {index_max} on {len(self.error_list)-1}")
        refined_scheme = refined_schemes["3"]
        
        if index_max == 0:
            list_subintervals = []
            list_subintervals.append(Interval(self.x_r[0], self.x_r[1], refined_scheme,self.f_r[0], self.f_r[1]))
            list_subintervals.append(Interval(self.x_r[1], self.x_r[2], refined_scheme,self.f_r[1], self.f_r[2]))            
            x, f = self.x_r[2:], self.f_r[2:]
            if len(x) > 1:
                reference_scheme = ReferenceScheme("PV", x=x, f=f)
                list_subintervals.append(Interval(x[0], x[-1], reference_scheme))

        elif index_max == len(self.error_list)-1:
            list_subintervals = []
            x, f = self.x_r[:-2], self.f_r[:-2]
            if len(x) > 1:
                reference_scheme = ReferenceScheme("PV", x=x, f=f)
                list_subintervals.append(Interval(x[0], x[-1], reference_scheme))
            list_subintervals.append(Interval(self.x_r[-3], self.x_r[-2], refined_scheme,self.f_r[-3], self.f_r[-2]))
            list_subintervals.append(Interval(self.x_r[-2], self.x_r[-1], refined_scheme,self.f_r[-2], self.f_r[-1]))    
        else:
            list_subintervals = []
            indices = slice(1+2*index_max)
            x, f = self.x_r[indices], self.f_r[indices]
            reference_scheme = ReferenceScheme("PV", x=x, f=f)
            list_subintervals.append(Interval(x[0], x[-1], reference_scheme))
            index_a, index_b =  2*index_max , 2*index_max+1
            list_subintervals.append(Interval(self.x_r[index_a], self.x_r[index_b], refined_scheme,self.f_r[index_a], self.f_r[index_b]))
            index_a, index_b = 2*index_max+1 , 2*index_max+2
            list_subintervals.append(Interval(self.x_r[index_a], self.x_r[index_b], refined_scheme,self.f_r[index_a], self.f_r[index_b]))
            indices = slice(2*index_max+2, self.reference_scheme.n_r)
            x, f = self.x_r[indices], self.f_r[indices]
            reference_scheme = ReferenceScheme("PV", x=x, f=f)
            list_subintervals.append(Interval(x[0], x[-1], reference_scheme))

        return list_subintervals
            

            
            
            










        # if self.typ == "CC":
        #     nb_sub_intervals = len(self.x_c)-1
        #     list_subintervals = []
        #     for i in range(nb_sub_intervals):
        #         if i == index_max:
        #             previous_values = {"x_r" : self.x_r[:2], "f": self.f_r[:2], "xi" : self.xi_r[:2], "type" : "left"}
        #         else:
        #             indices = slice(1+2*i,1+2*i+3) 
        #             x, f = self.x_r[indices], self.f_r[indices]
        #             previous_values = ReferenceScheme("PV", x=x, f=f)
        #             exit()    
                
                
        #         list_subintervals.append(Interval(self.x_c[i], self.x_c[i+1], self.typ, self.reference_scheme, previous_values))
            
        #     for inter in list_subintervals:
        #         print(inter)
        #     exit()
        # elif self.typ == "GK":
        #     # detemine the number of nodes to add in each interval
        #     nb_sub_intervals = len(self.x_c)+1
        #     list_subintervals = [None]*nb_sub_intervals
            
            
        #     # First interval
        #     previous_values = {"x_r" : self.x_r[:2], "f": self.f_r[:2], "xi" : self.xi_r[:2], "type" : "left"}
        #     list_subintervals[0] = Interval(self.a, self.x_c[0], self.typ, self.reference_scheme, previous_values)
        #     # Intermediate intervals
        #     for i in range(nb_sub_intervals-2):
        #         previous_values = {"x_r" : self.x_r[1+2*i:1+2*i+3], "f": self.f_r[1+2*i:1+2*i+2], "xi" : self.xi_r[1+2*i:1+2*i+3], "type" : "middle"}
        #         list_subintervals[i+1] = Interval(self.x_c[i], self.x_c[i+1], self.typ, self.reference_scheme, previous_values)
        #     # Last interval
        #     previous_values = {"x_r" :  self.x_r[-2:], "f": self.f_r[-2:], "xi": self.xi_r[-2:], "type" : "right"}
        #     list_subintervals[nb_sub_intervals-1] = Interval(self.x_c[-1], self.b, self.typ, self.reference_scheme, previous_values)
        # else:
        #     raise NameError("Unknown quadrature scheme")

    def plot_polynomials(self, fig=None):
        x_plot =np.linspace(self.A, self.B, 500)
        xi_plot =np.linspace(-1, 1, 500)
        c_c = (self.reference_scheme.Vinverse_c@self.f)
        c_r = (self.reference_scheme.Vinverse_r@self.f)
        c_plot = (np.array([np.sum([c_c[i]*xi**i for i in range(self.reference_scheme.n_c)]) for xi in xi_plot]))
        r_plot = (np.array([np.sum([c_r[i]*xi**i for i in range(self.reference_scheme.n_r)]) for xi in xi_plot]))
        plt.plot(x_plot,c_plot,"b")
        plt.plot(x_plot,r_plot,"r")

    def plot_error_function(self, fig):
        x_plot =np.linspace(self.A, self.B, 500)
        xi_plot =np.linspace(-1, 1, len(x_plot))
        # plt.plot(x_plot, v_plot, "b")
        c_c = (self.matrix_poly_c@self.f_c)
        c_r = (self.matrix_poly_r@self.f_r)
        erreur_c_plot = (np.array([np.sum([c_c[i]*xi**i for i in range(self.n_c)]) for xi in xi_plot])-v_plot)
        erreur_r_plot = (np.array([np.sum([c_r[i]*xi**i for i in range(self.n_r)]) for xi in xi_plot])-v_plot)
        if self.status == "pending":
            plt.plot(x_plot,erreur_c_plot,"g")
            plt.plot(x_plot,erreur_r_plot,"m")
        else:
            plt.plot(x_plot,erreur_c_plot,"y")
            plt.plot(x_plot,erreur_r_plot,"r")

    def plot_error_on_intervals(self, ax):
        if self.typ in ["CC","PV","GK"]:
            if self.typ in ["CC", "PV"]:
                x_c = self.x_c
            elif self.typ == "GK":
                x_c = np.hstack([self.A, self.x_c, self.B])
            for i, error in enumerate(self.error_list):
                if self.error_list[i] > 0:
                    facecolor = "blue"
                else:
                    facecolor = "yellow"
                if self.status == "adapted":
                    facecolor = "red"
                    
                plt.plot(x_c[i:i+2], [ abs(error),  abs(error)], "k")
                ax.add_patch(Rectangle((x_c[i], 0), x_c[i+1]-x_c[i], abs(error), edgecolor = 'black', facecolor = facecolor, fill=True,lw=1))
        else:
            raise NameError("Unknown quadrature scheme")
        
    def plot_relative_error_on_intervals(self, ax):
        if self.typ in ["CC","PV"]: 
            error_list = self.error_list
            for i, error in enumerate(error_list):
                if error > 0:
                    facecolor = "blue"
                else:
                    facecolor = "red"
                ax.add_patch(Rectangle((self.x_c[i], 0), self.x_c[i+1]-self.x_c[i], abs(error), edgecolor = 'black', facecolor = facecolor, fill=True,lw=1))
        else:
            raise NameError("Unknown quadrature scheme")        

    def plot_grid(self,i):
        plt.plot(self.x_r,i*np.ones(self.reference_scheme.n_r),"r.")
        plt.plot(self.x_c,i*np.ones(self.reference_scheme.n_c),"b.")

    def identify_intervals(self,level=0.6):
        status_subinterval = ["peak"]*len(self.repartition_error)
        index_max = np.argmax(self.repartition_error)
        for i in range(index_max-1):
            status_subinterval[i] = "before"
        for i in range(index_max+2, self.reference_scheme.nb_subintervals):
            status_subinterval[i] = "after"
        mask = [int(status == "peak") for status in status_subinterval]
        if self.repartition_error.dot(mask) < level:
            status_subinterval = ["flat"]*len(self.repartition_error)
        return status_subinterval

    def subdivide_subintervals(self, x, f):
        reference_scheme = ReferenceScheme("PV", x=x, f=f)
        interval = Interval(self.func, x[0], x[-1], reference_scheme)
        interval.update()
        if np.abs(interval.error/interval.I_r) > 1e-2:
            interval  = Interval(self.func, x[0], x[-1], CC_autoadaptive(),status="adapted")
            interval.update()
            return interval
        else:
            interval.status = "converged"
            return interval

    def before(self):
        if "before" in self.status_subintervals:
            i_last = np.sum([status == "before" for status in self.status_subintervals])-1
            x, f = self.x_r_with_boundaries[:2*i_last+3], self.f_with_boundaries[:2*i_last+3]
            reference_scheme = ReferenceScheme("PV", x=x, f=f)
            interval = Interval(self.func, x[0], x[-1], reference_scheme)
            interval.update()
            if np.abs(interval.error/interval.I_r) < self.epsrel:
                interval.status = "converged"
                return [interval] # to be iterable 
            else:
                return [Interval(self.func, x[0], x[-1], self.reference_scheme_for_step_1)]
        else:
            return []
        
        
    def peak(self):
        indices_peak = [i for i, status in enumerate(self.status_subintervals) if status=="peak"]
        interval = Interval(self.func, self.x_r_with_boundaries[2*indices_peak[0]], self.x_r_with_boundaries[2*indices_peak[-1]+2], self.reference_scheme_for_peak)
        interval.update()
        
        return [interval]

    def after(self):
        if "after" in self.status_subintervals:
            i_first = self.status_subintervals.index("after")
            x, f = self.x_r_with_boundaries[2*i_first:], self.f_with_boundaries[2*i_first:]
            reference_scheme = ReferenceScheme("PV", x=x, f=f)
            interval = Interval(self.func, x[0], x[-1], reference_scheme)
            interval.update()
            if np.abs(interval.error/interval.I_r) < self.epsrel:
                interval.status = "converged"
                return [interval] # to be iterable 
            else: # In that, it is better to take new points 
                return [Interval(self.func, x[0], x[-1], self.reference_scheme_for_step_1)]
        else:
            return []

    def step_1(self):

        subintervals = []
        self.status_subintervals = self.identify_intervals()
        # print("--")
        # for i, status in enumerate(self.status_subintervals):
        #     print(f"[{self.x_r_with_boundaries[2*i]:.3f}, {self.x_r_with_boundaries[2*i+2]:.3f}] / {status}")
        
        subintervals.extend(self.before())
        subintervals.extend(self.peak())
        subintervals.extend(self.after())

        #             raise NameError("Unknown status")
        return subintervals

    def adapt(self):
        
        index_max = np.argmax(self.repartition_error)
        # print(index_max)
        subintervals = []
        if self.repartition_error[index_max] > 0.6: # The error is on a single subinterval
            if index_max ==0:
                indices = slice(2*index_max, 2*index_max+3)
                x, f = self.x_r[indices], self.f[indices]
                subintervals.append(Interval(x[0], x[2], self.reference_scheme))
                for i in range(index_max+1, self.reference_scheme.n_c-1):
                    subintervals.append(Interval(self.x_r[2*i], self.x_r[2*i+2], CC_autoadaptive(), self.f[2*i], self.f[2*i+2],status="adapted"))
                
                
            elif index_max == len(self.error_list)-1:
                for i in range(index_max):
                    subintervals.append(Interval(self.x_r[2*i], self.x_r[2*i+2], CC_autoadaptive(), self.f[2*i], self.f[2*i+2],status="adapted"))
                    # subintervals.append(Interval(self.x_r[2*i+1], self.x_r[2*i+2], CC_autoadaptive(), self.f_r[2*i+1], self.f_r[2*i+2],status="adapted"))
                indices = slice(2*index_max, 2*index_max+3)
                x, f = self.x_r[indices], self.f[indices]
                subintervals.append(Interval(x[0], x[2], self.reference_scheme))
            else: #error on an intermediate interval
                for i in range(index_max):
                    subintervals.append(Interval(self.x_r[2*i], self.x_r[2*i+2], CC_autoadaptive(), self.f[2*i], self.f[2*i+2],status="adapted"))
                    # subintervals.append(Interval(self.x_r[2*i+1], self.x_r[2*i+2], CC_autoadaptive(), self.f_r[2*i+1], self.f_r[2*i+2],status="adapted"))

                indices = slice(2*index_max, 2*index_max+3)
                x, f = self.x_r[indices], self.f[indices]
                subintervals.append(Interval(x[0], x[2], self.reference_scheme))
                # list_subintervals.append(Interval(x[1], x[2], reference_scheme))
                
                
                for i in range(index_max+1, self.reference_scheme.n_c-1):
                    subintervals.append(Interval(self.x_r[2*i], self.x_r[2*i+2], CC_autoadaptive(), self.f[2*i], self.f[2*i+2],status="adapted"))
                    # subintervals.append(Interval(self.x_r[2*i+1], self.x_r[2*i+2], CC_autoadaptive(), self.f_r[2*i+1], self.f_r[2*i+2],status="adapted"))
        else:
            for i in range(self.reference_scheme.n_c-1):
                    subintervals.append(Interval(self.x_r[2*i], self.x_r[2*i+2], CC_autoadaptive(), self.f[2*i], self.f[2*i+2],status="adapted"))
                    # subintervals.append(Interval(self.x_r[2*i+1], self.x_r[2*i+2], CC_autoadaptive(), self.f_r[2*i+1], self.f_r[2*i+2],status="adapted"))

        return subintervals

    def refine_reference_scheme(self):
        if isinstance(self.reference_scheme, CC_autoadaptive):
            self.x_c = self.A+(self.B-self.A)*(self.reference_scheme.xi_c+1)/2.
            self.x_r = self.A+(self.B-self.A)*(self.reference_scheme.xi_r+1)/2.
            self.P_c = ((self.B-self.A)/2)*self.reference_scheme.P_c
            self.P_r = ((self.B-self.A)/2)*self.reference_scheme.P_r