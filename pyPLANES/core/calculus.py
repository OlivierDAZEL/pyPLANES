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
import time, timeit
import numpy as np
from numpy import pi
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
        self.verbose = kwargs.get("verbose", True)
        self.frequencies = self.init_vec_frequencies(kwargs.get("frequencies", np.array([440])))
        self.omega = None
        self.name_project = kwargs.get("name_project", "unnamed_project")
        self.sub_project = kwargs.get("sub_project", False)
        self.outfiles_directory = kwargs.get("outfiles_directory", False)
        self.plot = kwargs.get("plot_results", [False]*6)


    def create_linear_system(self, omega):
        """
        Create the linear system at circular frequency f

        Parameters
        ----------
        f : real or complex number
            frequency of resolution 
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
        if self.outfiles_directory:
            if self.outfiles_directory != "":
                directory = self.outfiles_directory
                if not path.exists(directory):
                    mkdir(directory)
                self.out_file = directory + "/" + self.out_file_name
                self.info_file = directory + "/"+ self.info_file_name

        self.out_file_name = self.out_file
        self.info_file_name = self.info_file

        self.out_file = open(self.out_file, 'w')
        self.info_file = open(self.info_file, 'w')

        name_server = socket.gethostname()
        self.info_file.write("Output File from pyPLANES\n")
        self.info_file.write("Generated on {}\n".format(name_server))
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

    def resolution(self):
        """  Resolution of the problem """        
        if self.verbose:
            print("%%%%%%%%%%%%% Resolution of PLANES %%%%%%%%%%%%%%%%%")
        self.initialisation_out_files()
        for f in self.frequencies:
            omega = 2*pi*f
            self.update_frequency(omega)
            self.create_linear_system(omega)
            self.solve()
            self.write_out_files()
            if any(self.plot):
                self.display_sol()
        self.close_out_files()

    def init_vec_frequencies(self, frequency):
        """
        Create the frequency vector that will be used in the calculations

        Parameters
        ----------
        frequency : array_like 
            a list of 3 numbers corresponding to the
            frequency[0] is the first frequency
            frequency[1] is the last frequency
            frequency[2] corresponds to the number of frequency steps. If positive (resp. negative), a linear (resp. logarithmic) step is chosen.

        Returns
        -------
        ndarray of the frequencies
        """
        if frequency[2] > 0:
                frequencies = np.linspace(frequency[0], frequency[1], frequency[2])
        elif frequency[2]<0:
            frequencies = np.logspace(np.log10(frequency[0]),np.log10(frequency[1]),abs(frequency[2]))
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
        """  Update frequency  """
        self.omega = omega

class PwCalculus(Calculus):
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.termination = kwargs.get("termination", "Rigid")
        self.method = kwargs.get("method", "global")
        self.kx, self.ky, self.k = None, None, None
        self.X = None
        self.R = None
        self.T = None
        if self.method == "global":
            self.out_file = self.name_project + ".GM.txt"
            self.info_file = self.name_project + ".info.GM.txt"
        elif self.method == "recursive":
            self.out_file = self.name_project + ".RM.txt"
            self.info_file = self.name_project + ".info.RM.txt"         

    def initialisation_out_files(self):
        Calculus.initialisation_out_files(self)
        if self.method == "global":
            self.info_file.write("Plane wave solver // Global method\n")        
        elif self.method == "recursive":
            self.info_file.write("Plane wave solver // Recursive method\n")

    def update_frequency(self, omega):
        Calculus.update_frequency(self, omega)
        self.kx = self.omega*np.sin(self.theta_d*np.pi/180)/Air.c
        self.ky = self.omega*np.cos(self.theta_d*np.pi/180)/Air.c
        self.k = self.omega/Air.c

    def write_out_files(self):
        self.out_file.write("{:.12e}\t".format(self.omega/(2*pi)))
        self.out_file.write("{:.12e}\t".format(self.R.real))
        self.out_file.write("{:.12e}\t".format(self.R.imag))
        self.out_file.write("{:.12e}\t".format(self.T.real))
        self.out_file.write("{:.12e}\t".format(self.T.imag))
        self.out_file.write("\n")
        
    def plot_results(self):
        data = np.loadtxt(self.out_file_name)
        plt.figure(1)
        if self.method == "recursive":
            plt.plot(data[:,0],data[:,1],'r.',label="Re(R) RM")
            plt.plot(data[:,0],data[:,2],'b.',label="Im(R) RM")
        elif self.method == "global":
            plt.plot(data[:,0],data[:,1],'r+',label="Re(R) GM")
            plt.plot(data[:,0],data[:,2],'b+',label="Im(R) GM")
        if self.termination == "transmission":
            plt.figure(2)
            if self.method == "recursive":
                plt.plot(data[:,0],data[:,3],'r.',label="Re(R) RM")
                plt.plot(data[:,0],data[:,4],'b.',label="Im(R) RM")
            elif self.method == "global":
                plt.plot(data[:,0],data[:,3],'r+',label="Re(R) GM")
                plt.plot(data[:,0],data[:,4],'b+',label="Im(R) GM")