import sys, os, platform
from datetime import datetime
sys.path.insert(0, "../../..")

import numpy as np
import matplotlib.pyplot as plt
import json 


from pyPLANES.gmsh.templates.inclusions import one_inclusion, one_inclusion_bicomposite, one_inclusion_square
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem
from pyPLANES.core.result import Result



class MetaporousSimulation():
    def __init__(self, **kwargs):
        self.name_project = "metaporous_simulation_"+ datetime.now().strftime("%Y%m%d%H%M%S")
        self.server = kwargs.get("server",None)
        self.method = kwargs.get("method", None)
        self.order = kwargs.get("order", None)
        self.nb_bloch_waves = kwargs.get("nb_bloch_waves", None)
        self.lcar = kwargs.get("lcar", None)
        self.theta_d = kwargs.get("theta_d", 0)
        self.frequencies = kwargs.get("frequencies", None)
        
        self.case = kwargs.get("case", None)

        if self.case in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            self.L = 0.02
            self.d = 0.02
            self.a = 8e-3
            
            if self.case in [1, 2, 3, 4]:
                self.ml = [["mesh" , None]]
            elif self.case in [5, 6, 7, 8]:
                self.ml = [["mesh" , None]]
            elif self.case in [9, 10, 11, 12]:
                self.ml = [("rubber",0.2e-3), ["mesh" , None]]
            elif self.case in [13, 14, 15, 16]:
                self.ml = [("rubber",0.2e-3), ["mesh" , None], ("rubber",0.2e-3)]
            
            if self.case in [1, 2, 3, 4, 9, 10, 11, 12]:
                self.termination = "rigid"
            else:
                self.termination = "transmission"
            
            
            if self.case in [1, 5, 9, 13]:
                one_inclusion("mesh", self.L, self.d, self.a, self.lcar, "pem_benchmark_1", "Steel")
            elif self.case in [2, 6, 10, 14]:
                r_i = 7.8e-3
                one_inclusion_bicomposite("mesh", self.L, self.d, self.a, r_i, self.lcar, "pem_benchmark_1", "rubber", "Air")
            elif self.case in [3, 7, 11, 15]:
                one_inclusion("mesh", self.L, self.d, self.a, self.lcar, "pem_benchmark_1", "pem_benchmark_2")
            elif self.case in [4, 8, 12, 16]:
                r_i = 2e-3
                one_inclusion_bicomposite("mesh", self.L, self.d, self.a, r_i, self.lcar, "pem_benchmark_2", "pem_benchmark_1", "Steel")
            if hasattr(self.frequencies, '__iter__'):
                self.frequencies = np.array(self.frequencies)
            elif self.frequencies == None:
                if self.case in [1, 5,6,9, 13]:
                    self.frequencies = np.linspace(10, 5010, 201)
                else:
                    self.frequencies = np.linspace(1, 5000, 200)

        elif self.case == "square":
            self.L = 0.02
            self.d = 0.02
            self.a = 8e-3
            self.ml = [("rubber",0.2e-3), ["mesh" , None], ("rubber",0.2e-3)]
            self.termination = "transmission"
            one_inclusion_square("mesh", self.L, self.d, self.a, self.lcar, "pem_benchmark_1", "Steel")

        self.pb = PeriodicPwProblem(ml=self.ml, theta_d=self.theta_d, name_project=self.name_project, order=self.order, nb_bloch_waves=self.nb_bloch_waves, frequencies=self.frequencies,termination=self.termination, method=self.method)

    def export_name(self):
        name_file = f"case={self.case}"
        name_file += f"_theta={self.theta_d}"
        name_file += f"_method={self.method}"
        name_file += f"_order={self.order}"
        name_file += f"_nb_bloch_waves={self.nb_bloch_waves}"
        name_file += f"_lcar={self.lcar}"
        return name_file

    def json_file_export_name(self):
        if self.server is None:
            if sys.platform == "darwin":
                directory = f"out/mac/case={self.case}/method={self.method}/"
            elif sys.platform == "linux":
                directory = f"out/helmholtz/case={self.case}/method={self.method}/"
            else:
                raise NameError("Wrong OS")
        if not os.path.exists(directory):
            os.makedirs(directory)



        return directory+self.export_name()

    def run(self):
        self.pb.resolution()

    def save(self):
        self.run()
        json_file = "out/" + self.name_project + ".json"
        with open(json_file, 'r') as file:
            dic = json.load(file)
        # Add informations 
        dic["case"] = self.case
        dic["nb_bloch_waves"] = self.nb_bloch_waves
        dic["lcar"] = self.lcar
        dic["theta_d"] = self.theta_d
        dic["L"] = self.L
        dic["d"] = self.d
        dic["a"] = self.a
        # Save the file
        with open(json_file, 'w') as fp:
            json.dump(dic, fp)
 
        new_file = self.json_file_export_name()
        print("Renaming " + json_file + " to " + new_file+".json")
        os.rename(json_file, new_file+".json")



class MultipleScatteringSimulation():
    def __init__(self, case, theta_d, pre_directory=""):
        if case in [1, 5, 6, 7, 9, 10, 13]:
            ms = np.loadtxt(pre_directory+"ms/ABS_CASE{}_{}.dat".format(case, 90-theta_d))
            self.f = ms[:,0]
            self.abs = ms[:,1]
        else:
            ms = np.loadtxt(pre_directory+"ms/metaporous_{}_theta_d={}_ms.txt".format(case, theta_d))
            self.f = ms[:,0]
            self.abs = ms[:,2]
