import numpy as np 
import matplotlib.pyplot as plt 
from scipy import integrate
from scipy.special import jv
from numpy import exp, cos, sin, sqrt, pi
from mediapack import Air


class PWSolver():
    def __init__(self, **kwargs):
        self.DF_method = kwargs.get("DF_method", "scipy")
        # Angles
        self.theta_d = kwargs.get("theta_d", 0.0)
        self.angles = kwargs.get("angles", None)
        if self.angles is not None:
            self.theta_d = self.angles[0]
            self.phi_d = self.angles[1]
        self.epsrel = kwargs.get("epsrel", 1.49e-1)
        self.epsabs = kwargs.get("epsabs", 1.49e-1)
        self.diffuse_field = True if self.theta_d == 90 else False
        self.method = kwargs.get("method","Global Method")
        # Computation method
        self.method = kwargs.get("method", "Global Method")            
        if self.method.lower() in ["recursive", "jap", "recursive method"]:
            self.method = "Recursive Method"
        elif self.method.lower() in ["tmm", "transfer matrix method"]:
            self.method = "TMM"
        elif self.method.lower() in ["characteristics", "characteristic", "carac"]:
            self.method = "characteristics"
        else: 
            self.method = "Global Method"
        # Put a non zero angle for some methods
        if self.method in [ "TMM", "Recursive Method"]:
            if self.theta_d == 0:
                self.theta_d = 1e-12



