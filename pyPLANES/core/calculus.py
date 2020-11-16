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
import time

import numpy as np
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
        current frequency

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
        if self.name_server in ["ODs-macbook-pro.home", "ODs-MacBook-Pro.local"]:
            self.verbose = True
        else:
            self.verbose = False
        self.verbose = kwargs.get("verbose", True)
        self.frequencies = self.init_vec_frequencies(kwargs.get("frequencies", np.array([440])))
        self.current_frequency = None
        self.omega = None
        self.name_project = kwargs.get("name_project", "unnamed_project")
        self.outfiles_directory = kwargs.get("outfiles_directory", False)
        self.plot = kwargs.get("plot_results", [False]*6)

    def create_linear_system(self, f):
        """
        Create the linear system at frequency f

        Parameters
        ----------
        f : real or complex number
            frequency of resolution 
        """
        if self.verbose:
            print("Creation of the linear system for f={}".format(f))
        self.update_frequency(f)

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
        pass 

    def display_sol(self):
        pass

    def resolution(self):
        """  Resolution of the problem """        
        if self.verbose:
            print("%%%%%%%%%%%%% Resolution of PLANES %%%%%%%%%%%%%%%%%")
        for f in self.frequencies:
            self.update_frequency(f)
            self.create_linear_system(f)
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

    def update_frequency(self, f):
        """  Update frequency  """
        self.current_frequency = f
        self.omega = 2*np.pi*f


class FemCalculus(Calculus):  
    """
    Finite-Element Calculus
    """
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        self.out_file = self.name_project + ".FEM.txt"
        self.info_file = self.name_project + ".info.FEM.txt"
        self.dim = 2
        self.edges = []
        self.faces = []
        self.bubbles = []
        self.reference_elements = dict() # dictionary of reference_elements

        self.nb_edges = self.nb_faces = self.nb_bubbles = 0
        self.F_i, self.F_v = None, None
        self.A_i, self.A_j, self.A_v = None, None, None
        self.A_i_c, self.A_j_c, self.A_v_c = None, None, None
        self.T_i, self.T_j, self.T_v = None, None, None
        self.order = kwargs.get("order", 2)
        self.interface_zone = kwargs.get("interface_zone", 0.01)
        self.incident_ml = kwargs.get("incident_ml", False)
        self.interface_ml = kwargs.get("interface_ml", False)


    def update_frequency(self, f):
        Calculus.update_frequency(self, f)
        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.A_i_c, self.A_j_c, self.A_v_c = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []
        for _ent in self.model_entities:
            _ent.update_frequency(self.omega)


        
    def extend_AF(self, _A_i, _A_j, _A_v, _F_i, _F_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def extend_F(self, _F_i, _F_v):
        self.F_i.extend(_F_i)
        self.F_v.extend(_F_v)

    def extend_A(self, _A_i, _A_j, _A_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)

    def extend_A_F_from_coo(self, AF):
        self.A_i.extend(list(AF[0].row))
        self.A_j.extend(list(AF[0].col))
        self.A_v.extend(list(AF[0].data))
        self.F_i.extend(list(AF[1].row))
        self.F_v.extend(list(AF[1].data))

    def extend_AT(self, _A_i, _A_j, _A_v, _T_i, _T_j, _T_v):
        self.A_i.extend(_A_i)
        self.A_j.extend(_A_j)
        self.A_v.extend(_A_v)
        self.T_i.extend(_T_i)
        self.T_j.extend(_T_j)
        self.T_v.extend(_T_v)

    def linear_system_2_numpy(self):
        self.F_i = np.array(self.F_i)
        self.F_v = np.array(self.F_v, dtype=complex)
        self.A_i = np.array(self.A_i)
        self.A_j = np.array(self.A_j)
        self.A_v = np.array(self.A_v, dtype=complex)
        self.T_i = np.array(self.T_i)-self.nb_dof_master
        self.T_j = np.array(self.T_j)
        self.T_v = np.array(self.T_v, dtype=complex)







    def resolution(self):
        Calculus.resolution(self)
        if self.name_server == "il-calc1":
            mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
            os.system(mail)

class PeriodicFemCalculus(FemCalculus):  
    """
    Periodic Finite-Element Calculus
    """
    def __init__(self, **kwargs):
        FemCalculus.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.modulus_reflex, self.modulus_trans, self.abs = None, None, None

    def update_frequency(self, f):
        FemCalculus.update_frequency(self, f)
        self.F_i, self.F_v = [], []
        self.A_i, self.A_j, self.A_v = [], [], []
        self.A_i_c, self.A_j_c, self.A_v_c = [], [], []
        self.T_i, self.T_j, self.T_v = [], [], []

        self.kx = (self.omega/Air.c)*np.sin(self.theta_d*np.pi/180)
        self.ky = (self.omega/Air.c)*np.cos(self.theta_d*np.pi/180)
        self.delta_periodicity = np.exp(-1j*self.kx*self.period)

        self.nb_dofs = self.nb_dof_FEM
        for _ent in self.model_entities:
            _ent.update_frequency(self.omega)
        self.modulus_reflex, self.modulus_trans, self.abs = 0, 0, 1

    def resolution(self):
        Calculus.resolution(self)
        if self.name_server == "il-calc1":
            mail = " mailx -s \"FEM pyPLANES Calculation of " + self.name_project + " over on \"" + self.name_server + " olivier.dazel@univ-lemans.fr < " + self.info_file.name
            os.system(mail)





class PwCalculus(Calculus):
    def __init__(self, **kwargs):
        Calculus.__init__(self, **kwargs)
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.X = None
        self.out_file = self.name_project + ".PW.txt"
        self.info_file = self.name_project + ".info.PW.txt"

    def update_frequency(self, f):
        Calculus.update_frequency(self, f)
        self.kx = self.omega*np.sin(self.theta_d*np.pi/180)/Air.c
        self.ky = self.omega*np.cos(self.theta_d*np.pi/180)/Air.c
        self.k = self.omega/Air.c

        