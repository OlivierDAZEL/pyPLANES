import numpy as np 
import matplotlib.pyplot as plt 
from scipy import integrate
from scipy.special import jv
from numpy import exp, cos, sin, sqrt, pi
from mediapack import Air

        # Ks = 5E06
        # Ms = 0
        # hs= 0.5
        # Lp = 0.1
        # Lh = 0.51
        # Lv = 1.22



class Stud():
    def __init__(self,layer, ml, **kwargs):
        self.connection_type = kwargs.get("connection_type", "Point")
        self.layer = layer
        self.nb_PW = None
        # test if the stud connect two elastic layers
        if ml[self.layer-1].medium.MODEL != "elastic" and ml[self.layer+1].medium.MODEL != "elastic":
            raise NameError("The stud must connect two elastic layers")
        self.layer_b = ml[self.layer-1]
        self.layer_t = ml[self.layer-1]
        self.K = np.zeros((6,6), dtype=complex)
        K =1e9
        self.K[0,2] = -1  
        self.K[0,5] = 1
        self.K[3,2] = -self.K[0,2]  
        self.K[3,5] = self.K[0,5]
        self.K *= K
        
    def __str__(self):
        str = f"Stud associated to layer {self.layer}"
        return str
    
    def update_frequency(self, kx):
        pass
    
    def compute_Uw(self):
        self.Uw = np.zeros((6,self.nb_PW-1), dtype=complex)
        SV = self.layer_b.SV
        d = [self.layer_b.d]*self.layer_b.nb_waves_in_medium+[0]*self.layer_b.nb_waves_in_medium
        delta = np.diag(np.exp(self.layer_b.lam*d))
        self.Uw[0:3,self.layer_b.dofs-1] = SV[:3,:]@delta
        
        SV = self.layer_t.SV
        d = [0]*self.layer_t.nb_waves_in_medium +[-self.layer_t.d]*self.layer_t.nb_waves_in_medium
        delta = np.diag(np.exp(self.layer_t.lam*d))
        self.Uw[3:6,self.layer_t.dofs-1] = SV[:3,:]@delta
        return self.Uw
    

    def update_frequency(self, kx):
        self.kx = kx 



