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



class Studs():
    def __init__(self, **kwargs):
        self.studs = []
        self.period = kwargs.get("period", None)
        studs = kwargs.get("studs", [])
        # Studs
        self.studs = []
        if studs != []:
            for st in studs:
                st = Stud(st[0]+1,self.layers) # +1 Because of incident layer
                self.studs.append(st)
            self.homogeneous = False
            self.method = "Stud"
            for st in self.studs:
                i_eq = 0
                for i, _int in enumerate(self.interfaces):
                    if i == st.layer-1: # -1 Because of incident layer
                        st.sigma_b = i_eq + np.array(_int.relations_sigma)
                        st.sigma_t = i_eq + _int.number_relations+np.array(self.interfaces[i+1].relations_sigma)
                        st.B = np.zeros((self.nb_PW-1, 6), dtype=complex)
                        st.B[st.sigma_t, :3] = np.eye(3)
                        st.B[st.sigma_b, 3:] = -np.eye(3)
                    i_eq += _int.number_relations
                st.nb_PW = self.nb_PW
        
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
        
        K =1e18
        self.K[0,2] = -K
        self.K[0,5] = K
        self.K[3,2] = -self.K[0,2]
        self.K[3,5] = -self.K[0,5]


    def __str__(self):
        str = f"Stud associated to layer {self.layer}"
        return str

    def update_frequency(self, kx):
        pass
    
    def compute_Uw(self):
        self.Uw = np.zeros((6,self.nb_PW-1), dtype=complex)
        SV = self.layer_t.SV
        d = np.array([0]*self.layer_t.nb_waves_in_medium +[-self.layer_t.d]*self.layer_t.nb_waves_in_medium)
        delta = np.diag(np.exp(self.layer_t.lam*d))
        self.Uw[:3,self.layer_t.dofs-1] = SV[:3,:]@delta #-1 for dofs as the first column will be removed
                
        SV = self.layer_b.SV
        d = np.array([self.layer_b.d]*self.layer_b.nb_waves_in_medium+[0]*self.layer_b.nb_waves_in_medium)
        delta = np.diag(np.exp(self.layer_b.lam*d))
        self.Uw[3:,self.layer_b.dofs-1] = SV[:3,:]@delta #-1 for dofs as the first column will be removed
        

        return self.Uw



