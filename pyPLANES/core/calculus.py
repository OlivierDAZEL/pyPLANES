#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
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

import platform
import socket
import datetime

from os import path, mkdir, rename

import time, timeit
import numpy as np
import numpy.linalg as LA
from numpy import pi

from termcolor import colored
import matplotlib.pyplot as plt

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla

from mediapack import Air
from pyPLANES.core.result import Result
from alive_progress import alive_bar

Air = Air()

class Calculus():
    """ pyPLANES Calculus 

    Attributes :
    ------------------------

    frequencies : ndarray
        list of calculation frequencies

    current_frequency : real or complex
        current frequency

    omega : real or complex
        current circular frequency

    theta_d : real or False
        Incident angle in degree 

    name_project : str
        Incident angle in degree 

    outfiles_directory : str or False
        Directory for out files

    plot : Boolean
        True/False if plots are on/off

    """

    def __init__(self, **kwargs):
        self.name_server = platform.node()
        self.txt_file_extension = False
        self.verbose = kwargs.get("verbose", False)
        self.save_format = kwargs.get("save_format", "json")        
        self.save_append = kwargs.get("save_append", "w")
        self.Result = Result() 
        label = kwargs.get("label", False)
        if label:
            self.Result["label"] = label
            
        self.init_vec_frequencies(kwargs.get("frequencies", False))
       
        h = kwargs.get("h", None)
        if h:
            self.Result.h = h

        self.print_result = kwargs.get("print_result", False)    
        self.plot = kwargs.get("plot_solution", [False]*6)   
        
        self.export_plots = kwargs.get("export_plots", [False]*6)   
        self.export_paraview = kwargs.get("export_paraview", False)  
        self.outfiles_directory = kwargs.get("outfiles_directory", False)

        if self.outfiles_directory:
            if not path.exists(self.outfiles_directory):
                    mkdir(self.outfiles_directory) 
        else: 
            self.outfiles_directory = "out"
            if not path.exists("out"):
                mkdir("out")
        if self.export_paraview:
            self.paraview_directory = "vtk"
            if not path.exists("vtk"):
                mkdir("vtk")
            self.export_paraview = 0 # To count the animation

        self.name_project = kwargs.get("name_project", "unnamed_project")
        self.sub_project = kwargs.get("sub_project", False)
        self.file_names = self.outfiles_directory + "/" + self.name_project
        if self.sub_project:
            self.file_names += "_" + self.sub_project
        # self.info_file_name = self.file_names + ".info.txt"
        # self.open_info_file()
        self.start_time = time.time()

    def resolution(self):
        """  Resolution of the problem """
        if self.verbose == False:
            if len(self.frequencies) == 1:
                self.f = self.frequencies[0]
                self.solve()
                self.plot_solutions()
            else:
                with alive_bar(len(self.frequencies), title="pyPLANES Resolution") as bar:
                    for f in self.frequencies:
                        bar()
                        self.f = f 
                        self.solve()
                        self.plot_solutions()
        else:
            print("%%%%%%%%%%%%% Resolution of PLANES %%%%%%%%%%%%%%%%%")
            for f in self.frequencies:
                self.f = f 
                self.solve()
                self.plot_solutions()

        self.Result.save(self.file_names,self.save_append)

    def results_to_json(self):
        pass

    def save_json(self):
        name_file = self.file_names + ".json"
        with open(name_file, self.save_append) as json_file:
                json.dump(self.Result, json_file)
                json_file.write("\n")
        
    def open_info_file(self):
        """  Initialise out files """    
        # self.txt_file = open(self.txt_file_name, 'w')
        self.info_file = open(self.info_file_name, 'w')
        self.info_file.write("Output File from pyPLANES\n")
        self.info_file.write("Generated on {}\n".format(self.name_server))
        self.info_file.write("Calculus started at %s.\n"%(datetime.datetime.now()))
        self.start_time = time.time()

    def create_linear_system(self, omega):
        """
        Create the linear system
        """
        if self.verbose:
            print("Creation of the linear system for f={}".format(omega/(2*pi)))

    def solve(self):
        """ Resolution of the linear system"""
        if self.verbose:
            print("Resolution of the linear system")

        self.Result.f.append(self.f)
        omega = 2*pi*self.f
        self.update_frequency(omega)
        self.create_linear_system(omega)

    def plot_solutions(self):
        if any(self.plot):
            self.plot_solution()
        if any(self.export_plots):
            if self.export_plots[5]:
                plt.figure("Pressure map")
                plt.savefig("Pressure")

    def close_info_file(self):
        """  Close out files at the end of the calculus """
        # self.txt_file.close()
        self.info_file.close()
        # rename the info file so as to include the name of the method in the name of the text part
        new_name = self.info_file_name.split(".")
        new_name.insert(1, self.out_file_method)
        rename(self.info_file_name, ".".join(new_name))

    def plot_solution(self):
        pass

    def init_vec_frequencies(self, frequency):
        """
        Assignate to  self.frequency the frequency vector that will be used in the calculations

        Parameters
        ----------
        # frequency : can be of different types 
        #       if frequency is a ndarray, self.frequency is frequency
        #       if frequency is a scalar self.frequency is a ndarray with this single frequency
        #       if frequency is a list of 3 numbers corresponding to the
        #           f_bounds[0] is the first frequency
        #           f_bounds[1] is the last frequency
        #           f_bounds[2] corresponds to the number of frequency steps. If positive (resp. negative), a linear (resp. logarithmic) step is chosen.

        Returns
        -------
        # ndarray of the frequencies
        """

        if isinstance(frequency, np.ndarray):
            self.frequencies = frequency
        elif np.isscalar(frequency):
             self.frequencies = np.array([frequency], dtype=np.float)
        elif len(frequency) == 3: 
            if frequency[2] > 0:
                self.frequencies = np.linspace(frequency[0], frequency[1], frequency[2], dtype=np.float)
            elif frequency[2]<0:
                self.frequencies = np.logspace(np.log10(frequency[0]),np.log10(frequency[1]),abs(frequency[2]), dtype=np.float)
        elif frequency == None:
            self.frequencies = np.array([1e3], dtype=np.float)
        
        return frequency

    def update_frequency(self, omega):
        pass


