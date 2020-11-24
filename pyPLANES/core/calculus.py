#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# problem.py
#
# This file is part of pymls, a software distributed under the MIT license.
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

from os import path, mkdir

import time, timeit
import numpy as np
import numpy.linalg as LA
from numpy import pi

from termcolor import colored
import matplotlib.pyplot as plt

from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, linalg as sla


from pyPLANES.utils.io import display_sol

from mediapack import Air

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
        self.verbose = kwargs.get("verbose", False)
        _ = kwargs.get("frequencies", False)
        __ = kwargs.get("f_bounds", False)
        if _ is not False:
            self.frequencies = _
        elif f_bounds is not False:
            self.frequencies = self.init_vec_frequencies(__)
        else:
            self.frequencies = np.array([1e3])
        self.plot = kwargs.get("plot_solution", [False]*6)        
        self.outfiles_directory = kwargs.get("outfiles_directory", False)
        if self.outfiles_directory:
            if not path.exists(self.outfiles_directory):
                    mkdir(self.outfiles_directory) 
        else: 
            self.outfiles_directory = "out"
            if not path.exists("out"):
                mkdir("out")
        self.name_project = kwargs.get("name_project", "unnamed_project")
        self.sub_project = kwargs.get("sub_project", False)
        self.file_names = self.outfiles_directory + "/" + self.name_project
        if self.sub_project:
            self.file_names += "_" + self.sub_project

    def resolution(self):
        """  Resolution of the problem """        
        if self.verbose:
            print("%%%%%%%%%%%%% Resolution of PLANES %%%%%%%%%%%%%%%%%")
        self.preprocess()
        for f in self.frequencies:
            self.f = f
            omega = 2*pi*f
            self.update_frequency(omega)
            self.create_linear_system(omega)
            self.solve()
            self.write_out_files()
            if any(self.plot):
                self.display_sol()
        self.close_out_files()

    def preprocess(self):
        self.initialisation_out_files()

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

    def initialisation_out_files(self):
        """  Initialise out files """    
            # Creation of the directory if it .oes not exists
 
        self.out_file = open(self.out_file_name, 'w')
        self.info_file = open(self.info_file_name, 'w')

        self.info_file.write("Output File from pyPLANES\n")
        self.info_file.write("Generated on {}\n".format(self.name_server))
        self.info_file.write("Calculus started at %s.\n"%(datetime.datetime.now()))
        self.start_time = time.time()

    def write_out_files(self):
        """  Write out files at current frequency"""    
        pass

    def close_out_files(self):
        """  Close out files at the end of the calculus """
        self.out_file.close()
        self.info_file.close()

    def display_sol(self):
        pass

    def init_vec_frequencies(self, f_bounds):
        """
        Create the frequency vector that will be used in the calculations

        Parameters
        ----------
        f_bounds : array_like 
            a list of 3 numbers corresponding to the
            f_bounds[0] is the first frequency
            f_bounds[1] is the last frequency
            f_bounds[2] corresponds to the number of frequency steps. If positive (resp. negative), a linear (resp. logarithmic) step is chosen.

        Returns
        -------
        ndarray of the frequencies
        """
        if frequency[2] > 0:
                frequencies = np.linspace(f_bounds[0], f_bounds[1], f_bounds[2])
        elif frequency[2]<0:
            frequencies = np.logspace(np.log10(f_bounds[0]),np.log10(f_bounds[1]),abs(f_bounds[2]))
        # else % Case of complex frequency
        #     temp_1=linspace(frequency.min,frequency.max,frequency.nb(1));
        #     temp_2=linspace(frequency.min_imag,frequency.max_imag,frequency.nb(2));
        #     frequency.vec=[];
        #     for ii=1:frequency.nb(2)
        #         frequency.vec=[frequency.vec temp_1+1j*temp_2(ii)];
        #     end
        #     frequency.nb=frequency.nb(1)*frequency.nb(2);
        return frequencies

    def update_frequency(self, omega):
        pass

