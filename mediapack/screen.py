#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# screen.py
#
# This file is part of pymls, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2017
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

from .pem import PEM
from .air import Air
from numpy.lib.scimath import sqrt
import numpy as np


class Screen(PEM):

    MEDIUM_TYPE = 'screen'

    def _compute_missing(self):
        self.lambda_ = self.E*self.nu/((1+self.nu)*(1-2*self.nu))

        self.rho_12 = 0  # Simplification on alpha ~= 1
        self.rho_11 = self.rho_1-self.rho_12
        self.rho_2 = self.phi*Air.rho
        self.rho_22 = self.rho_2-self.rho_12

    def update_frequency(self, omega):

        #  Simplified model for rho_eq_til & K_eq_til
        self.rho_eq_til = (Air.rho/self.phi)+self.sigma/(1j*omega)
        self.alpha_til = self.phi*self.rho_eq_til/Air.rho
        self.K_eq_til = Air.P/self.phi

        self.c_eq_til = sqrt(self.K_eq_til/self.rho_eq_til)

        self.rho_22_til = self.phi**2*self.rho_eq_til
        self.rho_12_til = self.rho_2-self.rho_22_til
        self.rho_11_til = self.rho_1-self.rho_12_til
        self.rho_til = self.rho_11_til-((self.rho_12_til**2)/self.rho_22_til)

        self.gamma_til = self.phi*(self.rho_12_til/self.rho_22_til-(1-self.phi)/self.phi)
        self.rho_s_til = self.rho_til+self.gamma_til**2*self.rho_eq_til

        if self.loss_type == 'anelastic':
            raise NotImplementedError("Anelastic losses not implemented yet")
            # *HAS* to be in update_frequency() if anelastic is considered
            # self.structural_loss=1+(b_hat*(1j*omega/beta_hat)**alpha_hat)/(1+(1j*omega/beta_hat)**alpha_hat)
        elif self.loss_type == 'structural':
            self.structural_loss = 1+1j*self.eta
        elif self.loss_type == 'none':
            self.structural_loss = 1
        else:
            raise ValueError('Unknown type of losses')

        self.N = self.E/(2*(1+self.nu))*self.structural_loss
        self.A_hat = (self.E*self.nu)/((1+self.nu)*(1-2*self.nu))*self.structural_loss
        self.P_hat = self.A_hat+2*self.N

        # Biot 1956 elastic coefficients
        self.R_til = self.K_eq_til*self.phi**2
        self.Q_til = ((1-self.phi)/self.phi)*self.R_til
        self.P_til = self.P_hat+self.Q_til**2/self.R_til

        delta_eq = omega*sqrt(self.rho_eq_til/self.K_eq_til)
        delta_s_1 = omega*sqrt(self.rho_til/self.P_hat)
        delta_s_2 = omega*sqrt(self.rho_s_til/self.P_hat)

        Psi = ((delta_s_2**2+delta_eq**2)**2-4*delta_eq**2*delta_s_1**2)
        sdelta_total = sqrt(Psi)

        delta_1 = sqrt(0.5*(delta_s_2**2+delta_eq**2+sdelta_total))
        delta_2 = sqrt(0.5*(delta_s_2**2+delta_eq**2-sdelta_total))
        delta_3 = omega*sqrt(self.rho_til/self.N)

        if np.abs(delta_1-delta_eq) < np.abs(delta_2-delta_eq):
            delta_1, delta_2 = delta_2, delta_1

        mu_1 = self.gamma_til*delta_eq**2/(delta_1**2-delta_eq**2)
        mu_2 = self.gamma_til*delta_eq**2/(delta_2**2-delta_eq**2)
        mu_3 = -self.gamma_til

        self.delta_s_1 = delta_s_1
        self.delta_s_2 = delta_s_2
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_3 = delta_3
        self.delta_eq = delta_eq
        self.sdelta_total = sdelta_total
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
