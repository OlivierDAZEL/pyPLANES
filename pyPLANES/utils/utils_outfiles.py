#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_outfiles.py
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

from os import path, mkdir
from pyPLANES.classes.entity_classes import IncidentPwFem, TransmissionPwFem


def initialisation_out_files(self, p):
    # Creation of the directory if it .oes not exists
    if hasattr(p,"outfiles_directory"):
        if p.outfiles_directory != "":
            directory = p.outfiles_directory
            if not path.exists(directory):
                mkdir(directory)
            self.outfile = open(directory + "/pyPLANES.out", "w")
            self.resfile = open(directory + "/pyPLANES.res", "w")
        else:
            self.outfile = open("pyPLANES.out", "w")
            self.resfile = open("pyPLANES.res", "w")
    else :
        self.outfile = open("pyPLANES.out", "w")
        self.resfile = open("pyPLANES.res", "w")


    self.resfile.write("Frequency [Hz]\n")
    if [isinstance(_ent, (IncidentPwFem, TransmissionPwFem)) for _ent in self.model_entities]:
            self.resfile.write("absorption [no unity]\n")
    if [isinstance(_ent, (IncidentPwFem)) for _ent in self.model_entities]:
            self.resfile.write("|R| [no unity]\n")
    if [isinstance(_ent, (TransmissionPwFem)) for _ent in self.model_entities]:
            self.resfile.write("|T| [no unity]\n")

def write_out_files(self):

    self.outfile.write("{:.12e}\t".format(self.current_frequency))
    if [isinstance(_ent, (IncidentPwFem, TransmissionPwFem)) for _ent in self.model_entities]:
        self.outfile.write("{:.12e}\t".format(self.abs))
    if [isinstance(_ent, (IncidentPwFem)) for _ent in self.model_entities]:
        self.outfile.write("{:.12e}\t".format(self.modulus_reflex))
    if [isinstance(_ent, (TransmissionPwFem)) for _ent in self.model_entities]:
        self.outfile.write("{:.12e}\t".format(self.modulus_reflex))
    self.outfile.write("\n")

