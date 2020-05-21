import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pymls import from_yaml, Solver, Layer, backing
from mediapack import Air

from pyPLANES.model.model import Model
from pyPLANES.classes.model_classes import ModelParameter
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh

from pyPLANES.utils.utils_io import print_entities


param = ModelParameter()
param.theta_d = 60.0000001
param.frequencies = (300., 5010., 1)
param.name_project = "one_layer"

pem = from_yaml('foam2.yaml')
Wwood = from_yaml('Wwood.yaml')

L = 0.01
d = 0.1
lcar = 0.005

param.order = 5
param.plot = [True, True, True, False, False, False]
# param.plot = [False]*6

G = Gmsh(param.name_project)

p_0 = G.new_point(0, 0, lcar)
p_1 = G.new_point(L, 0, lcar)
p_2 = G.new_point(L, d, lcar)
p_3 = G.new_point(0, d, lcar)
l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
matrice = G.new_surface([ll_0.tag])
G.new_physical(l_2, "condition=Transmission")
# G.new_physical(l_2, "condition=Rigid Wall")
G.new_physical([l_1, l_3], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
# G.new_physical(matrice, "mat=Air")
G.new_physical(matrice, "mat=foam2")
# G.new_physical(matrice, "mat=Wwood")
G.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
G.new_physical([matrice], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))
option = "-2 -v 0 "
G.run_gmsh(option)

param.incident_ml = [Layer(Wwood, d)] ; param.shift_pw = -param.incident_ml[0].thickness
param.transmission_ml = [Layer(Air, d)]


model = Model(param)
result_pyPLANES = model.resolution(param)

param.solver_pymls = Solver()
param.solver_pymls.layers = [
    Layer(Wwood, d),
    # Layer(Air,d),#,Layer(Air,d)
    Layer(pem, d),
    Layer(Air,d)
    # Layer(pem, d),
    # Layer(Wwood,d),
    # Layer(Wwood,d/10),
]

# param.solver_pymls.backing = backing.rigid
param.solver_pymls.backing = backing.transmission


param.S_PW = Solver_PW(param.solver_pymls, param)
result_pyPLANESPW = param.S_PW.resolution(param.theta_d)

# result_pymls =  param.solver_pymls.solve(param.frequencies[0], param.theta_d)

# print("result_pymls     = {}".format(result_pymls["R"][0]))
print("result_pyPLANESPW= {}".format(result_pyPLANESPW["R"]))
print("result_pyPLANES  = {}".format(result_pyPLANES["R"]))
# print("result_pymls  T  = {}".format(result_pymls["T"][0]))

# rho = param.solver_pymls.layers[0].medium.rho
# lam = param.solver_pymls.layers[0].medium.lambda_
# mu = param.solver_pymls.layers[0].medium.mu
# omega =2*np.pi*param.frequencies[0]
# k = omega*np.sqrt(rho/(lam+2.*mu))
# Z = np.sqrt(rho*(lam+2.*mu))
# Z_s = -1j*Z/np.tan(k*param.solver_pymls.layers[0].thickness)
# R_a = (Z_s-Air.Z)/(Z_s+Air.Z)
# print(R_a)

if any(param.plot):
    plt.show()