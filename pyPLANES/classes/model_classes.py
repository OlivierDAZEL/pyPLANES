#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# model_classes.py
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
from numpy import pi

class ModelParameter:
    '''Class ModelParameter'''
    def __init__(self):
        pass
    def __str__(self):
        out = "Parameters of the FEM Model{}\n"
        out += "\t f = {}\n".format(self.f)

        return out
        
if __name__ == "__main__":
    model_parameter = Model_Parameter()