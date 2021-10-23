#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2021
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
from termcolor import colored
import platform

import platform
import json
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


class Result():
    """ pyPLANES Calculus 

    Attributes :
    ------------------------

    f : real or complex
        frequency of the simulation

    """

    def __init__(self, _in=None):
        if _in == None: # Creation of a void instance 
            self.f = []
            self.R0 = []
            self.T0 = []
            self.R = []
            self.T = []
            self.abs =[]
            self.Solver = None
            self.Server = platform.node()
        else: # Creation of a Result on either a dict or a file 
            if isinstance(_in, dict):
                d = _in
            elif isinstance(_in, str):
                d = json.load(open(_in+".json", 'r'))
            keys = [*d]
            if "f" in keys:
                self.f = d["f"]
            if "Solver" in keys:
                self.Solver = d["Solver"]
                if self.Solver == "FemProblem":
                    self.plot_symbol = ".-"
            if "real(R0)" in keys:
                if "imag(R0)" in keys:
                    self.R0 = np.array(d["real(R0)"])+1j*np.array(d["imag(R0)"])
            if "real(T0)" in keys:
                if "imag(T0)" in keys:
                    self.R0 = np.array(d["real(T0)"])+1j*np.array(d["imag(T0)"])
                else:
                    raise NameError("No imag(T0) field")
            if "abs" in keys:
                self.abs = np.array(d["abs"])
            if "order" in keys:
                self.order = d["order"]

    def save(self,file, append_file):
        d = dict()
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for m in members:
            if isinstance(self.__dict__[m], list):
                if len(self.__dict__[m]) != 0:
                    if m == "R0":
                        d["real(R0)"] = np.real(self.R0).tolist()
                        d["imag(R0)"] = np.imag(self.R0).tolist()
                    elif m == "T0":
                        d["real(T0)"] = np.real(self.T0).tolist()
                        d["imag(T0)"] = np.imag(self.T0).tolist()
                    else: 
                        d[m] = self.__dict__[m]
            else: 
                d[m] = self.__dict__[m]
        with open(file+".json", append_file) as json_file:
            json.dump(d, json_file)
            json_file.write("\n")


    def __str__(self):
        return "Result\n f={:+.15f}\t".format(self.f)

    def plot(self, *args, **kwargs):
        plt.plt(self.f, TL, *args, **kwargs)

class Test():
    def __init__(self, ref, result, indicator, **kwargs):
        eps = kwargs.get("eps", 1e-12)
        error = LA.norm(result.__dict__[indicator]-ref.__dict__[indicator])/len(result.__dict__[indicator])
        if error< eps:
            print("Overall error = {}".format(error) + "\t"*6 + "["+ colored("OK", "green")  +"]")
        else:
            print("Overall error = {}".format(error) + "\t"*6 + "["+ colored("Fail", "red")  +"]")

class Results():
    def __init__(self, file=False, **kwargs):
        self.list = []
        self.f = []
        if file:
            for l in open(file, 'r'):
                self.list.append(Result(json.loads(l)))

    # def create_convergence_curve(self):
    #     reference = [r for r in self.list if r.Solver == ]
    #     index_reference = [i for i, x in enumerate(list_solvers) if x]
    #     if len(index_reference) == 1:
    #         reference = self.list[index_reference[0]]
    #         self.list.pop(index_reference[0])
    #         for r in self.list:
    #             r.error = np.abs(r.R0-reference.R0)
    #         #     print(r.error)
    #         # qdssqddsq
            
    #         plt.figure()
    #         if len(reference.f) == 1:
    #             # Computation of the errors 
    #             list_orders = set([r.order for r in self.list])
    #             convergence_curve = dict()
    #             for order in list_orders:
    #                 n_dof_list = [r.n_dof for r in self.list if r.order == order]
    #                 error_list = [r.error[-1] for r in self.list if r.order == order]
    #                 convergence_curve[str(order)] = (n_dof_list, error_list)
    #             plt.figure()
    #             for key in convergence_curve.keys():
    #                 cc = convergence_curve[key]
    #                 plt.loglog(cc[0],cc[1],".-", label=key)
    #         else: 
    #             for r in self.list:
    #                 # print("order={}, error={}".format(r.order,r.error))
    #                 plt.loglog(r.f*(0.1/340), r.error, r.plot_symbol, label=r.order)
    #             # ord=1;plt.loglog(r.f*(0.1/340),1e-8*r.f**(2*ord+1),"b--",label="{}".format(ord))
    #             # ord=2;plt.loglog(r.f,1e-19*r.f**(2*ord+1+2),"m--",label="{}".format(ord))
    #             # ord=3;plt.loglog(r.f,2e-25*r.f**(2*ord+1),"g--",label="{}".format(ord))
    #             # ord=4;plt.loglog(r.f,2e-28*r.f**(2*ord+1),"m--",label="{}".format(ord))
    #             # plt.loglog(r.f,1e-19*r.f**(2*3+1),"r--")
    #     elif len(index_reference) > 1:
    #         raise NameError("Several reference results")
    #     else: # Eigen problem 
    #             list_orders = set([r.order for r in self.list])
    #             convergence_curve = dict()
    #             for order in list_orders:
    #                 n_elem_list = [r.h for r in self.list if r.order == order]
    #                 error_list = [r.error[-1] for r in self.list if r.order == order]
    #                 convergence_curve[str(order)] = (n_elem_list, error_list)
    #             plt.figure()
    #             for key in convergence_curve.keys():
    #                 cc = convergence_curve[key]
    #                 plt.loglog(cc[0],cc[1],".-", label=key)
    #             h= np.array(cc[0])
    #             plt.loglog(h, h**(-2.),"b--",label="2")
    #             plt.loglog(h, h**(-3.),"b--",label="3")
        

    #     plt.legend()
    #     plt.savefig("cc.pdf")
    #     plt.show()






class ConvergenceCurve():
    def __init__(self, ref, results, indicator, param):
        ref = ref 
        results = results
        indicator = indicator
        param = param 






# class PwResult(Result):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.R0 = kwargs.get("R0", None)
#         self.T0 = kwargs.get("T0", None)
#         self.abs = None

#     def __str__(self):
#         _text = "R0={:+.15f}".format(self.R0)
#         if self.T0:
#             _text += " / T0={:+.15f}".format(self.T0)
#         return _text

#     def plot_TL(self,*args, **kwargs):
#         TL = -20*np.log10(np.abs(self.T0))
#         plt.semilogx(self.f, TL, *args, **kwargs)

# class FEMResult(PwResult):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.order = kwargs.get("order", None)
#         self.n_dof = kwargs.get("n_dof", None)
#         self.D_lambda = kwargs.get("D_lambda", None)
#         self.R = None
#         self.T = None
#         self.plot_symbol = ".-"

# class PeriodicPwResult(PwResult):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.order = kwargs.get("order", None)
#         self.n_dof = kwargs.get("n_dof", None)
#         self.D_lambda = kwargs.get("D_lambda", None)
#         self.R = None
#         self.T = None
#         self.plot_symbol = "+--"

# class EigenResult(PwResult):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.order = kwargs.get("order", None)
#         self.n_elem = kwargs.get("n_elem", None)
#         self.h = np.sqrt(self.n_elem)
#         self.error = kwargs.get("error", None)



# class Results():

#     def __init__(self, file=False, **kwargs):
#         self.list = []
#         self.f = []
#         if file:
#             for l in open(file, 'r'):
#                 self.list.append(Result(json.loads(l)))



                



#                 f = np.array(d["f"])
#                 if d["Solver"] == "PwProblem":
#                     R0 = np.array(d["real(R0)"])+1j*np.array(d["imag(R0)"])
#                     T0 = np.array(d["real(T0)"])+1j*np.array(d["imag(T0)"])
#                     self.list.append(PwResult(f=f,R0=R0,T0=T0))
#                 elif d["Solver"] == "PeriodicPwProblem":
#                     R0 = np.array(d["real(R0)"])+1j*np.array(d["imag(R0)"])
#                     T0 = np.array(d["real(T0)"])+1j*np.array(d["imag(T0)"])
#                     self.list.append(PeriodicPwResult(f=f,R0=R0,T0=T0,n_dof=d["n_dof"], order =d["order"]))
#                 elif d["Solver"] == "FemProblem":
#                     R0 = np.array(d["real(R0)"])+1j*np.array(d["imag(R0)"])
#                     self.list.append(FEMResult(f=f,R0=R0,n_dof=d["n_dof"], order =d["order"],D_lambda=d["D_lambda"]))
#                 elif d["Solver"] == "EigenProblem":
#                     self.list.append(EigenResult(n_elem= d["n_elem"], order =d["order"], error = d["eigen"]))
    
#     def create_convergence_curve(self):
#         list_solvers = [ not(isinstance(r, (PeriodicPwResult, FEMResult, EigenResult))) for r in self.list]
#         index_reference = [i for i, x in enumerate(list_solvers) if x]
#         if len(index_reference) == 1:
#             reference = self.list[index_reference[0]]
#             self.list.pop(index_reference[0])
#             for r in self.list:
#                 r.error = np.abs(r.R0-reference.R0)
#             #     print(r.error)
#             # qdssqddsq
            
#             plt.figure()
#             if len(reference.f) == 1:
#                 # Computation of the errors 
#                 list_orders = set([r.order for r in self.list])
#                 convergence_curve = dict()
#                 for order in list_orders:
#                     n_dof_list = [r.n_dof for r in self.list if r.order == order]
#                     error_list = [r.error[-1] for r in self.list if r.order == order]
#                     convergence_curve[str(order)] = (n_dof_list, error_list)
#                 plt.figure()
#                 for key in convergence_curve.keys():
#                     cc = convergence_curve[key]
#                     plt.loglog(cc[0],cc[1],".-", label=key)
#             else: 
#                 for r in self.list:
#                     # print("order={}, error={}".format(r.order,r.error))
#                     plt.loglog(r.f*(0.1/340), r.error, r.plot_symbol, label=r.order)
#                 # ord=1;plt.loglog(r.f*(0.1/340),1e-8*r.f**(2*ord+1),"b--",label="{}".format(ord))
#                 # ord=2;plt.loglog(r.f,1e-19*r.f**(2*ord+1+2),"m--",label="{}".format(ord))
#                 # ord=3;plt.loglog(r.f,2e-25*r.f**(2*ord+1),"g--",label="{}".format(ord))
#                 # ord=4;plt.loglog(r.f,2e-28*r.f**(2*ord+1),"m--",label="{}".format(ord))
#                 # plt.loglog(r.f,1e-19*r.f**(2*3+1),"r--")
#         elif len(index_reference) > 1:
#             raise NameError("Several reference results")
#         else: # Eigen problem 
#                 list_orders = set([r.order for r in self.list])
#                 convergence_curve = dict()
#                 for order in list_orders:
#                     n_elem_list = [r.h for r in self.list if r.order == order]
#                     error_list = [r.error[-1] for r in self.list if r.order == order]
#                     convergence_curve[str(order)] = (n_elem_list, error_list)
#                 plt.figure()
#                 for key in convergence_curve.keys():
#                     cc = convergence_curve[key]
#                     plt.loglog(cc[0],cc[1],".-", label=key)
#                 h= np.array(cc[0])
#                 plt.loglog(h, h**(-2.),"b--",label="2")
#                 plt.loglog(h, h**(-3.),"b--",label="3")
        

#         plt.legend()
#         plt.savefig("cc.pdf")
#         plt.show()



    