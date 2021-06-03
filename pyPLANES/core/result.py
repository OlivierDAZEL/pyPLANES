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
import pickle

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

Air = Air()


class Result():
    """ pyPLANES Calculus 

    Attributes :
    ------------------------

    f : real or complex
        frequency of the simulation

    """

    def __init__(self, **kwargs):
        self.f = kwargs.get("f", None)

    def write_as_txt(self, f):
        f.write("{:.12e}\t".format(self.f))

    def __str__(self):
        return "f={:+.15f}\t".format(self.f)


class PwResult(Result):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.R0 = None
        self.T0 = False
        self.abs = None

    def __str__(self):
        _text = "R0={:+.15f}".format(self.R0)
        if self.T0:
            _text += " / T0={:+.15f}".format(self.T0)
        return _text

    def write_as_txt(self, f):
        Result.write_as_txt(self, f)
        f.write("{:.12e}\t".format(self.R0.real))
        f.write("{:.12e}\t".format(self.R0.imag))
        if self.T0:
            f.write("{:.12e}\t".format(self.T0.real))
            f.write("{:.12e}\t".format(self.T0.imag))
        f.write("\n")


class PeriodicPwResult(PwResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.R = None
        self.T = None


class ListOfResults():

    def __init__(self, pkl_file):
        self.f = []
        with open(pkl_file, "rb") as f:
            self.res = pickle.load(f)
            # self.f.append(self.res[-1].f)
        f.close()


        if isinstance(self.res[0], PwResult):
            self.f, self.R0, self.T0 = [], [], []
            for r in self.res:
                self.f.append(r.f)
                self.R0.append(r.R0)
                self.T0.append(r.T0)

    def plot_TL(self, *args):
        TL = -20*np.log10(np.abs(self.T0))
        plt.semilogx(self.f, TL, args)

    