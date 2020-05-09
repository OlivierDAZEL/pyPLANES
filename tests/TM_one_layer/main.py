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

name_server = platform.node()

param = ModelParameter()
param.theta_d = 30.000000
param.frequencies = (200., 5010., 1)
param.name_project = "one_layer"

L = 0.01
d = 0.1
lcar = 0.01

param.order = 4
param.plot = [True, True, True, False, False, False]
# param.plot = [False]*6

if name_server in ["oliviers-macbook-pro.home", "Oliviers-MacBook-Pro.local"]:
    param.verbose = True

G = Gmsh(param.name_project)

p_0 = G.new_point(0, 0, lcar)
p_1 = G.new_point(L, 0,lcar)
p_2 = G.new_point(L, d, lcar)
p_3 = G.new_point(0, d, lcar)
l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
matrice = G.new_surface([ll_0.tag])
# G.new_physical(l_2, "condition=Transmission")
G.new_physical(l_2, "condition=Rigid Wall")
G.new_physical([l_1, l_3], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
G.new_physical(matrice, "mat=foam2")
# G.new_physical(matrice, "mat=Wwood")
G.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
G.new_physical([matrice], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))
option = "-2 -v 0 "
G.run_gmsh(option)
model = Model(param)
print_entities(model)
model.resolution(param)

pem = from_yaml('foam2.yaml')
Wwood = from_yaml('Wwood.yaml')

param.solver_pymls = Solver()
param.solver_pymls.layers = [
    # Layer(Wwood, d),
    Layer(pem, d),
]
param.solver_pymls.backing = backing.rigid


# param.incident_ml =  [
#     Layer(Wwood, d),
# ]


# result_pymls =  param.solver_pymls.solve(param.frequencies[0], param.theta_d)
# print("result_pymls     = {}".format(result_pymls["R"][0]))
param.S_PW = Solver_PW(param.solver_pymls, param)
param.S_PW.resolution(param.theta_d)



c = np.sqrt((Wwood.lambda_+2*Wwood.mu)/Wwood.rho)
omega = 2*np.pi*param.frequencies[0]
k = omega/c
Z_c = Wwood.rho*c
Z_s = -1j*Z_c/np.tan(k*d)

R = (Z_s-Air.Z)/(Z_s+Air.Z)
print(R)
print((1+R)/(1j*omega*Z_s))

if any(param.plot):
    plt.show()